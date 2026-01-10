"""数据集读取模块 - 支持多种格式"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod
import yaml

from .formats import DatasetFormat, FORMAT_INFO


# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}


@dataclass
class DatasetItem:
    """单个数据集样本"""
    image_path: Path
    label_path: Optional[Path]
    classes: list[int] = field(default_factory=list)  # 该样本包含的类别索引


@dataclass
class ClassConfig:
    """类别配置"""
    nc: int  # 类别数量
    names: list[str]  # 类别名称列表


@dataclass
class AttributeFieldConfig:
    """单个属性字段配置"""
    name: str  # 属性名称，如 "color", "size"
    values: list[str]  # 属性值名称列表，如 ["red", "blue"] 或 ["small", "big"]


@dataclass
class OBBFormatConfig:
    """OBB 格式配置

    YAML 配置文件示例:
    ```yaml
    format_name: "armor_obb"
    description: "装甲板 OBB 格式"

    # 属性字段定义 (按标注文件中的顺序)
    attributes:
      - name: color
        values: ['red', 'blue']
      - name: size
        values: ['small', 'big']

    # 类别索引在属性之后的位置 (0-based, 相对于属性字段之后)
    # 如果 class_position 为 0，表示类别索引紧跟在所有属性之后
    # 默认值 0 表示: [attr1, attr2, ..., class, x1, y1, ...]
    class_position: 0

    # 类别配置 (可选)
    classes:
      nc: 8
      names: ['0', '1', '2', '3', '4', '5', 'negative', 'guard']
    ```

    标注文件格式: <attr1> <attr2> ... <class> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
    """
    format_name: str = "custom_obb"
    description: str = "自定义 OBB 格式"
    attributes: list[AttributeFieldConfig] = field(default_factory=list)
    class_position: int = 0  # 类别索引位置 (相对于属性字段之后)
    classes: Optional[ClassConfig] = None  # 内嵌的类别配置

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> 'OBBFormatConfig':
        """从 YAML 文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        attributes = []
        for attr_data in data.get('attributes', []):
            attributes.append(AttributeFieldConfig(
                name=attr_data['name'],
                values=attr_data.get('values', [])
            ))

        classes = None
        if 'classes' in data:
            classes_data = data['classes']
            classes = ClassConfig(
                nc=classes_data.get('nc', len(classes_data.get('names', []))),
                names=classes_data.get('names', [])
            )

        return cls(
            format_name=data.get('format_name', 'custom_obb'),
            description=data.get('description', '自定义 OBB 格式'),
            attributes=attributes,
            class_position=data.get('class_position', 0),
            classes=classes
        )

    @classmethod
    def armor_obb_default(cls) -> 'OBBFormatConfig':
        """返回装甲板 OBB 格式的默认配置 (向后兼容)"""
        return cls(
            format_name="armor_obb",
            description="装甲板 OBB 格式 (color size class x1 y1 x2 y2 x3 y3 x4 y4)",
            attributes=[
                AttributeFieldConfig(name="color", values=['red', 'blue']),
                AttributeFieldConfig(name="size", values=['small', 'big']),
            ],
            class_position=0
        )

    def get_total_fields(self) -> int:
        """计算标注文件每行的总字段数"""
        # 属性数 + 类别(1) + 坐标(8)
        return len(self.attributes) + 1 + 8

    def get_class_index(self) -> int:
        """获取类别字段在行中的索引 (0-based)"""
        return len(self.attributes) + self.class_position

    def get_points_start_index(self) -> int:
        """获取坐标字段的起始索引 (0-based)"""
        return self.get_class_index() + 1


@dataclass
class DatasetInfo:
    """数据集信息"""
    root_path: Path
    format: DatasetFormat
    items: list[DatasetItem]
    class_config: Optional[ClassConfig]

    # 目录信息
    images_dir: Optional[Path] = None
    labels_dir: Optional[Path] = None

    # 统计信息
    total_images: int = 0
    total_labels: int = 0
    matched_pairs: int = 0
    missing_labels: list[Path] = field(default_factory=list)
    orphan_labels: list[Path] = field(default_factory=list)
    class_distribution: dict[int, int] = field(default_factory=dict)


class BaseDatasetReader(ABC):
    """数据集读取器基类"""

    def __init__(self, dataset_path: str | Path):
        self.root_path = Path(dataset_path).resolve()

    @abstractmethod
    def validate_structure(self) -> tuple[bool, str]:
        """验证数据集目录结构"""
        pass

    @abstractmethod
    def read(self, class_config_path: Optional[str | Path] = None) -> DatasetInfo:
        """读取数据集"""
        pass

    def load_class_config(self, config_path: Optional[str | Path] = None) -> Optional[ClassConfig]:
        """加载类别配置文件"""
        if config_path is None:
            # 尝试在数据集根目录查找默认配置文件
            default_paths = [
                self.root_path / "classes.yaml",
                self.root_path / "classes.yml",
            ]
            for p in default_paths:
                if p.exists():
                    config_path = p
                    break

        if config_path is None:
            return None

        config_path = Path(config_path)
        if not config_path.exists():
            return None

        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if 'nc' not in data or 'names' not in data:
            raise ValueError(f"配置文件格式错误，需要包含 'nc' 和 'names' 字段: {config_path}")

        return ClassConfig(nc=data['nc'], names=data['names'])

    def _get_image_files(self, directory: Path) -> dict[str, Path]:
        """获取目录下所有图片文件，返回 {stem: path} 字典"""
        images = {}
        for ext in IMAGE_EXTENSIONS:
            for img_path in directory.glob(f"*{ext}"):
                images[img_path.stem] = img_path
            for img_path in directory.glob(f"*{ext.upper()}"):
                images[img_path.stem] = img_path
        return images


class YOLOReader(BaseDatasetReader):
    """标准 YOLO 格式读取器 (images/ + labels/ 分离)"""

    def __init__(self, dataset_path: str | Path):
        super().__init__(dataset_path)
        self.images_dir = self.root_path / "images"
        self.labels_dir = self.root_path / "labels"

    def validate_structure(self) -> tuple[bool, str]:
        """验证数据集目录结构"""
        if not self.root_path.exists():
            return False, f"数据集路径不存在: {self.root_path}"

        if not self.images_dir.exists():
            return False, f"images 目录不存在: {self.images_dir}"

        if not self.labels_dir.exists():
            return False, f"labels 目录不存在: {self.labels_dir}"

        return True, "目录结构验证通过"

    def _get_label_files(self) -> dict[str, Path]:
        """获取所有标签文件"""
        labels = {}
        for label_path in self.labels_dir.glob("*.txt"):
            labels[label_path.stem] = label_path
        return labels

    def _parse_label(self, label_path: Path) -> list[int]:
        """解析 YOLO 格式标签文件，返回类别索引列表"""
        classes = set()
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if parts:
                            class_idx = int(parts[0])
                            classes.add(class_idx)
        except (ValueError, IndexError):
            pass
        return sorted(classes)

    def read(self, class_config_path: Optional[str | Path] = None) -> DatasetInfo:
        """读取数据集"""
        valid, msg = self.validate_structure()
        if not valid:
            raise ValueError(msg)

        class_config = self.load_class_config(class_config_path)

        images = self._get_image_files(self.images_dir)
        labels = self._get_label_files()

        items = []
        missing_labels = []
        class_distribution: dict[int, int] = {}

        for stem, img_path in images.items():
            label_path = labels.get(stem)
            classes = []

            if label_path:
                classes = self._parse_label(label_path)
                for cls in classes:
                    class_distribution[cls] = class_distribution.get(cls, 0) + 1
            else:
                missing_labels.append(img_path)

            items.append(DatasetItem(
                image_path=img_path,
                label_path=label_path,
                classes=classes
            ))

        orphan_labels = [labels[stem] for stem in labels if stem not in images]

        if class_config is None and class_distribution:
            max_class = max(class_distribution.keys())
            class_config = ClassConfig(
                nc=max_class + 1,
                names=[f"class_{i}" for i in range(max_class + 1)]
            )

        return DatasetInfo(
            root_path=self.root_path,
            format=DatasetFormat.YOLO,
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            items=items,
            class_config=class_config,
            total_images=len(images),
            total_labels=len(labels),
            matched_pairs=len(images) - len(missing_labels),
            missing_labels=missing_labels,
            orphan_labels=orphan_labels,
            class_distribution=class_distribution
        )


class YOLOOBBReader(BaseDatasetReader):
    """YOLO OBB 格式读取器 (images/ + labels/ 分离，标签为四点坐标)"""

    def __init__(self, dataset_path: str | Path):
        super().__init__(dataset_path)
        self.images_dir = self.root_path / "images"
        self.labels_dir = self.root_path / "labels"

    def validate_structure(self) -> tuple[bool, str]:
        """验证数据集目录结构"""
        if not self.root_path.exists():
            return False, f"数据集路径不存在: {self.root_path}"

        if not self.images_dir.exists():
            return False, f"images 目录不存在: {self.images_dir}"

        if not self.labels_dir.exists():
            return False, f"labels 目录不存在: {self.labels_dir}"

        return True, "目录结构验证通过"

    def _get_label_files(self) -> dict[str, Path]:
        """获取所有标签文件"""
        labels = {}
        for label_path in self.labels_dir.glob("*.txt"):
            labels[label_path.stem] = label_path
        return labels

    def _parse_label(self, label_path: Path) -> list[int]:
        """解析 YOLO OBB 格式标签文件，返回类别索引列表"""
        classes = set()
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if parts:
                            class_idx = int(parts[0])
                            classes.add(class_idx)
        except (ValueError, IndexError):
            pass
        return sorted(classes)

    def read(self, class_config_path: Optional[str | Path] = None) -> DatasetInfo:
        """读取数据集"""
        valid, msg = self.validate_structure()
        if not valid:
            raise ValueError(msg)

        class_config = self.load_class_config(class_config_path)

        images = self._get_image_files(self.images_dir)
        labels = self._get_label_files()

        items = []
        missing_labels = []
        class_distribution: dict[int, int] = {}

        for stem, img_path in images.items():
            label_path = labels.get(stem)
            classes = []

            if label_path:
                classes = self._parse_label(label_path)
                for cls in classes:
                    class_distribution[cls] = class_distribution.get(cls, 0) + 1
            else:
                missing_labels.append(img_path)

            items.append(DatasetItem(
                image_path=img_path,
                label_path=label_path,
                classes=classes
            ))

        orphan_labels = [labels[stem] for stem in labels if stem not in images]

        if class_config is None and class_distribution:
            max_class = max(class_distribution.keys())
            class_config = ClassConfig(
                nc=max_class + 1,
                names=[f"class_{i}" for i in range(max_class + 1)]
            )

        return DatasetInfo(
            root_path=self.root_path,
            format=DatasetFormat.YOLO_OBB,
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            items=items,
            class_config=class_config,
            total_images=len(images),
            total_labels=len(labels),
            matched_pairs=len(images) - len(missing_labels),
            missing_labels=missing_labels,
            orphan_labels=orphan_labels,
            class_distribution=class_distribution
        )


class YOLOSegReader(BaseDatasetReader):
    """YOLO Segmentation 格式读取器 (images/ + labels/ 分离，标签为多边形点)"""

    def __init__(self, dataset_path: str | Path):
        super().__init__(dataset_path)
        self.images_dir = self.root_path / "images"
        self.labels_dir = self.root_path / "labels"

    def validate_structure(self) -> tuple[bool, str]:
        """验证数据集目录结构"""
        if not self.root_path.exists():
            return False, f"数据集路径不存在: {self.root_path}"

        if not self.images_dir.exists():
            return False, f"images 目录不存在: {self.images_dir}"

        if not self.labels_dir.exists():
            return False, f"labels 目录不存在: {self.labels_dir}"

        return True, "目录结构验证通过"

    def _get_label_files(self) -> dict[str, Path]:
        """获取所有标签文件"""
        labels = {}
        for label_path in self.labels_dir.glob("*.txt"):
            labels[label_path.stem] = label_path
        return labels

    def _parse_label(self, label_path: Path) -> list[int]:
        """解析 YOLO Seg 格式标签文件，返回类别索引列表"""
        classes = set()
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if parts:
                            class_idx = int(parts[0])
                            classes.add(class_idx)
        except (ValueError, IndexError):
            pass
        return sorted(classes)

    def read(self, class_config_path: Optional[str | Path] = None) -> DatasetInfo:
        """读取数据集"""
        valid, msg = self.validate_structure()
        if not valid:
            raise ValueError(msg)

        class_config = self.load_class_config(class_config_path)

        images = self._get_image_files(self.images_dir)
        labels = self._get_label_files()

        items = []
        missing_labels = []
        class_distribution: dict[int, int] = {}

        for stem, img_path in images.items():
            label_path = labels.get(stem)
            classes = []

            if label_path:
                classes = self._parse_label(label_path)
                for cls in classes:
                    class_distribution[cls] = class_distribution.get(cls, 0) + 1
            else:
                missing_labels.append(img_path)

            items.append(DatasetItem(
                image_path=img_path,
                label_path=label_path,
                classes=classes
            ))

        orphan_labels = [labels[stem] for stem in labels if stem not in images]

        if class_config is None and class_distribution:
            max_class = max(class_distribution.keys())
            class_config = ClassConfig(
                nc=max_class + 1,
                names=[f"class_{i}" for i in range(max_class + 1)]
            )

        return DatasetInfo(
            root_path=self.root_path,
            format=DatasetFormat.YOLO_SEG,
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            items=items,
            class_config=class_config,
            total_images=len(images),
            total_labels=len(labels),
            matched_pairs=len(images) - len(missing_labels),
            missing_labels=missing_labels,
            orphan_labels=orphan_labels,
            class_distribution=class_distribution
        )


class CustomOBBReader(BaseDatasetReader):
    """自定义属性 OBB 格式读取器

    支持通过 YAML 配置文件自定义标注格式，可灵活定义：
    - 任意数量的属性字段（如颜色、大小、状态等）
    - 每个属性的可选值列表
    - 类别索引的位置

    标注格式: <attr1> <attr2> ... <class> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
    """

    def __init__(self, dataset_path: str | Path, format_config: Optional[OBBFormatConfig] = None):
        """初始化读取器

        Args:
            dataset_path: 数据集根目录路径
            format_config: OBB 格式配置，如果为 None 则需要在 read() 时提供配置文件路径
        """
        super().__init__(dataset_path)
        self.images_dir = self.root_path / "images"
        self.labels_dir = self.root_path / "labels"
        self.format_config = format_config

        # 属性统计 {属性名: {属性值索引: 计数}}
        self.attribute_distributions: dict[str, dict[int, int]] = {}

    def validate_structure(self) -> tuple[bool, str]:
        """验证数据集目录结构"""
        if not self.root_path.exists():
            return False, f"数据集路径不存在: {self.root_path}"

        if not self.images_dir.exists():
            return False, f"images 目录不存在: {self.images_dir}"

        if not self.labels_dir.exists():
            return False, f"labels 目录不存在: {self.labels_dir}"

        return True, "目录结构验证通过"

    def _get_label_files(self) -> dict[str, Path]:
        """获取所有标签文件"""
        labels = {}
        for label_path in self.labels_dir.glob("*.txt"):
            labels[label_path.stem] = label_path
        return labels

    def _parse_label(self, label_path: Path) -> tuple[list[int], dict[str, set[int]]]:
        """解析自定义 OBB 格式标签文件

        Returns:
            (classes, attribute_values): 类别索引列表, 各属性的值集合
        """
        if self.format_config is None:
            raise ValueError("格式配置未设置，请先加载格式配置")

        classes = set()
        attribute_values: dict[str, set[int]] = {
            attr.name: set() for attr in self.format_config.attributes
        }

        min_fields = self.format_config.get_total_fields()
        class_idx = self.format_config.get_class_index()

        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= min_fields:
                            # 解析属性值
                            for i, attr in enumerate(self.format_config.attributes):
                                attr_val = int(parts[i])
                                attribute_values[attr.name].add(attr_val)

                            # 解析类别
                            class_val = int(parts[class_idx])
                            classes.add(class_val)
        except (ValueError, IndexError):
            pass

        return sorted(classes), attribute_values

    def load_format_config(self, config_path: Optional[str | Path] = None) -> OBBFormatConfig:
        """加载格式配置

        Args:
            config_path: 配置文件路径，如果为 None 则尝试在数据集根目录查找

        Returns:
            OBBFormatConfig 配置对象
        """
        if config_path is not None:
            self.format_config = OBBFormatConfig.from_yaml(config_path)
            return self.format_config

        # 尝试在数据集根目录查找默认配置文件
        default_paths = [
            self.root_path / "format.yaml",
            self.root_path / "format.yml",
            self.root_path / "obb_format.yaml",
            self.root_path / "obb_format.yml",
        ]
        for p in default_paths:
            if p.exists():
                self.format_config = OBBFormatConfig.from_yaml(p)
                return self.format_config

        raise FileNotFoundError(
            f"未找到格式配置文件，请在数据集根目录创建 format.yaml 或指定配置文件路径"
        )

    def read(
        self,
        class_config_path: Optional[str | Path] = None,
        format_config_path: Optional[str | Path] = None
    ) -> DatasetInfo:
        """读取数据集

        Args:
            class_config_path: 类别配置文件路径 (可选，可在 format_config 中指定)
            format_config_path: 格式配置文件路径 (如果未在构造函数中提供)

        Returns:
            DatasetInfo 数据集信息
        """
        valid, msg = self.validate_structure()
        if not valid:
            raise ValueError(msg)

        # 加载格式配置
        if self.format_config is None:
            self.load_format_config(format_config_path)

        # 加载类别配置 (优先使用格式配置中的类别，其次使用外部配置)
        class_config = None
        if self.format_config.classes is not None:
            class_config = self.format_config.classes
        elif class_config_path is not None:
            class_config = self.load_class_config(class_config_path)

        images = self._get_image_files(self.images_dir)
        labels = self._get_label_files()

        items = []
        missing_labels = []
        class_distribution: dict[int, int] = {}

        # 初始化属性统计
        self.attribute_distributions = {
            attr.name: {} for attr in self.format_config.attributes
        }

        for stem, img_path in images.items():
            label_path = labels.get(stem)
            classes = []

            if label_path:
                classes, attr_values = self._parse_label(label_path)

                # 更新类别统计
                for cls in classes:
                    class_distribution[cls] = class_distribution.get(cls, 0) + 1

                # 更新属性统计
                for attr_name, values in attr_values.items():
                    for val in values:
                        if attr_name not in self.attribute_distributions:
                            self.attribute_distributions[attr_name] = {}
                        self.attribute_distributions[attr_name][val] = \
                            self.attribute_distributions[attr_name].get(val, 0) + 1
            else:
                missing_labels.append(img_path)

            items.append(DatasetItem(
                image_path=img_path,
                label_path=label_path,
                classes=classes
            ))

        orphan_labels = [labels[stem] for stem in labels if stem not in images]

        # 如果没有类别配置，自动生成
        if class_config is None and class_distribution:
            max_class = max(class_distribution.keys())
            class_config = ClassConfig(
                nc=max_class + 1,
                names=[f"class_{i}" for i in range(max_class + 1)]
            )

        return DatasetInfo(
            root_path=self.root_path,
            format=DatasetFormat.CUSTOM_OBB,
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            items=items,
            class_config=class_config,
            total_images=len(images),
            total_labels=len(labels),
            matched_pairs=len(images) - len(missing_labels),
            missing_labels=missing_labels,
            orphan_labels=orphan_labels,
            class_distribution=class_distribution
        )

    def get_attribute_stats(self) -> dict:
        """获取属性统计信息

        Returns:
            包含每个属性的分布统计和值名称的字典
        """
        if self.format_config is None:
            return {}

        stats = {}
        for attr in self.format_config.attributes:
            stats[attr.name] = {
                'distribution': self.attribute_distributions.get(attr.name, {}),
                'value_names': attr.values
            }
        return stats

    def get_format_info(self) -> dict:
        """获取当前格式配置信息"""
        if self.format_config is None:
            return {}

        return {
            'format_name': self.format_config.format_name,
            'description': self.format_config.description,
            'attributes': [
                {'name': attr.name, 'values': attr.values}
                for attr in self.format_config.attributes
            ],
            'total_fields': self.format_config.get_total_fields(),
            'class_index': self.format_config.get_class_index(),
            'points_start_index': self.format_config.get_points_start_index()
        }


class COCOReader(BaseDatasetReader):
    """COCO JSON 格式读取器"""

    def __init__(self, dataset_path: str | Path):
        super().__init__(dataset_path)
        self.images_dir = self.root_path / "images"
        self.annotations_dir = self.root_path / "annotations"
        self.annotation_file: Optional[Path] = None

    def validate_structure(self) -> tuple[bool, str]:
        """验证数据集目录结构"""
        if not self.root_path.exists():
            return False, f"数据集路径不存在: {self.root_path}"

        if not self.images_dir.exists():
            return False, f"images 目录不存在: {self.images_dir}"

        if not self.annotations_dir.exists():
            return False, f"annotations 目录不存在: {self.annotations_dir}"

        # 查找 annotation 文件
        json_files = list(self.annotations_dir.glob("*.json"))
        if not json_files:
            return False, f"annotations 目录中没有 JSON 文件"

        # 优先使用 instances_*.json
        for f in json_files:
            if f.name.startswith("instances"):
                self.annotation_file = f
                break
        if self.annotation_file is None:
            self.annotation_file = json_files[0]

        return True, "目录结构验证通过"

    def read(self, class_config_path: Optional[str | Path] = None) -> DatasetInfo:
        """读取数据集"""
        valid, msg = self.validate_structure()
        if not valid:
            raise ValueError(msg)

        class_config = self.load_class_config(class_config_path)

        # 读取 COCO JSON
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        # 构建类别映射
        categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
        cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(sorted(categories.keys()))}

        # 如果没有提供类别配置，从 COCO 数据中提取
        if class_config is None and categories:
            sorted_cat_ids = sorted(categories.keys())
            class_config = ClassConfig(
                nc=len(categories),
                names=[categories[cat_id] for cat_id in sorted_cat_ids]
            )

        # 构建图片 ID 到文件名的映射
        image_id_to_info = {img['id']: img for img in coco_data.get('images', [])}

        # 构建图片 ID 到标注的映射
        image_annotations: dict[int, list] = {}
        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)

        # 获取实际存在的图片文件
        existing_images = self._get_image_files(self.images_dir)

        items = []
        missing_labels = []
        class_distribution: dict[int, int] = {}

        for img_id, img_info in image_id_to_info.items():
            file_name = img_info.get('file_name', '')
            stem = Path(file_name).stem

            img_path = existing_images.get(stem)
            if img_path is None:
                continue

            annotations = image_annotations.get(img_id, [])
            classes = []

            if annotations:
                for ann in annotations:
                    cat_id = ann.get('category_id')
                    if cat_id in cat_id_to_idx:
                        cls_idx = cat_id_to_idx[cat_id]
                        if cls_idx not in classes:
                            classes.append(cls_idx)
                            class_distribution[cls_idx] = class_distribution.get(cls_idx, 0) + 1
            else:
                missing_labels.append(img_path)

            items.append(DatasetItem(
                image_path=img_path,
                label_path=self.annotation_file,  # COCO 使用单一 JSON 文件
                classes=sorted(classes)
            ))

        return DatasetInfo(
            root_path=self.root_path,
            format=DatasetFormat.COCO,
            images_dir=self.images_dir,
            labels_dir=self.annotations_dir,
            items=items,
            class_config=class_config,
            total_images=len(existing_images),
            total_labels=len(image_annotations),
            matched_pairs=len(items) - len(missing_labels),
            missing_labels=missing_labels,
            orphan_labels=[],  # COCO 格式不存在孤立标签
            class_distribution=class_distribution
        )


class VOCReader(BaseDatasetReader):
    """Pascal VOC XML 格式读取器"""

    def __init__(self, dataset_path: str | Path):
        super().__init__(dataset_path)
        # VOC 标准目录结构
        self.images_dir = self.root_path / "JPEGImages"
        self.annotations_dir = self.root_path / "Annotations"

        # 也支持简化目录结构
        if not self.images_dir.exists():
            self.images_dir = self.root_path / "images"
        if not self.annotations_dir.exists():
            self.annotations_dir = self.root_path / "annotations"

    def validate_structure(self) -> tuple[bool, str]:
        """验证数据集目录结构"""
        if not self.root_path.exists():
            return False, f"数据集路径不存在: {self.root_path}"

        if not self.images_dir.exists():
            return False, f"图片目录不存在: 需要 JPEGImages/ 或 images/"

        if not self.annotations_dir.exists():
            return False, f"标注目录不存在: 需要 Annotations/ 或 annotations/"

        return True, "目录结构验证通过"

    def _get_annotation_files(self) -> dict[str, Path]:
        """获取所有 XML 标注文件"""
        annotations = {}
        for ann_path in self.annotations_dir.glob("*.xml"):
            annotations[ann_path.stem] = ann_path
        return annotations

    def _parse_voc_xml(self, xml_path: Path) -> tuple[list[str], list[int]]:
        """解析 VOC XML 标注文件，返回 (类别名称列表, 类别索引列表)"""
        class_names = []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                name_elem = obj.find('name')
                if name_elem is not None and name_elem.text:
                    class_names.append(name_elem.text)
        except ET.ParseError:
            pass

        return class_names, []

    def read(self, class_config_path: Optional[str | Path] = None) -> DatasetInfo:
        """读取数据集"""
        valid, msg = self.validate_structure()
        if not valid:
            raise ValueError(msg)

        class_config = self.load_class_config(class_config_path)

        images = self._get_image_files(self.images_dir)
        annotations = self._get_annotation_files()

        # 第一遍：收集所有类别名称
        all_class_names: set[str] = set()
        for ann_path in annotations.values():
            class_names, _ = self._parse_voc_xml(ann_path)
            all_class_names.update(class_names)

        # 构建类别名称到索引的映射
        sorted_class_names = sorted(all_class_names)
        name_to_idx = {name: idx for idx, name in enumerate(sorted_class_names)}

        # 如果没有提供类别配置，从数据中构建
        if class_config is None and sorted_class_names:
            class_config = ClassConfig(
                nc=len(sorted_class_names),
                names=sorted_class_names
            )

        items = []
        missing_labels = []
        class_distribution: dict[int, int] = {}

        for stem, img_path in images.items():
            ann_path = annotations.get(stem)
            classes = []

            if ann_path:
                class_names, _ = self._parse_voc_xml(ann_path)
                seen_classes = set()
                for name in class_names:
                    if name in name_to_idx:
                        cls_idx = name_to_idx[name]
                        if cls_idx not in seen_classes:
                            classes.append(cls_idx)
                            seen_classes.add(cls_idx)
                            class_distribution[cls_idx] = class_distribution.get(cls_idx, 0) + 1
            else:
                missing_labels.append(img_path)

            items.append(DatasetItem(
                image_path=img_path,
                label_path=ann_path,
                classes=sorted(classes)
            ))

        orphan_labels = [annotations[stem] for stem in annotations if stem not in images]

        return DatasetInfo(
            root_path=self.root_path,
            format=DatasetFormat.VOC,
            images_dir=self.images_dir,
            labels_dir=self.annotations_dir,
            items=items,
            class_config=class_config,
            total_images=len(images),
            total_labels=len(annotations),
            matched_pairs=len(images) - len(missing_labels),
            missing_labels=missing_labels,
            orphan_labels=orphan_labels,
            class_distribution=class_distribution
        )


def create_reader(dataset_path: str | Path, format: DatasetFormat) -> BaseDatasetReader:
    """工厂函数：根据格式创建对应的读取器"""
    readers = {
        DatasetFormat.YOLO: YOLOReader,
        DatasetFormat.YOLO_OBB: YOLOOBBReader,
        DatasetFormat.YOLO_SEG: YOLOSegReader,
        DatasetFormat.CUSTOM_OBB: CustomOBBReader,
        DatasetFormat.COCO: COCOReader,
        DatasetFormat.VOC: VOCReader,
    }

    reader_class = readers.get(format)
    if reader_class is None:
        raise ValueError(f"不支持的数据集格式: {format}")

    return reader_class(dataset_path)


# 保持向后兼容
class DatasetReader(YOLOReader):
    """兼容旧版本的读取器（默认 YOLO 格式）"""
    pass
