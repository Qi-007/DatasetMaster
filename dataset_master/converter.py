"""数据集格式转换模块 - 支持 COCO/VOC 转 YOLO"""

import json
import shutil
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .formats import DatasetFormat
from .reader import ClassConfig, IMAGE_EXTENSIONS


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class BoundingBox:
    """边界框数据 (归一化坐标)"""
    class_idx: int
    x_center: float  # 0-1
    y_center: float  # 0-1
    width: float     # 0-1
    height: float    # 0-1

    def to_yolo_line(self) -> str:
        """转换为 YOLO 标签行"""
        return f"{self.class_idx} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


@dataclass
class Segmentation:
    """分割多边形数据 (归一化坐标)"""
    class_idx: int
    points: list[tuple[float, float]]  # [(x1,y1), (x2,y2), ...]

    def to_yolo_line(self) -> str:
        """转换为 YOLO-Seg 标签行"""
        points_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in self.points)
        return f"{self.class_idx} {points_str}"


@dataclass
class AnnotationData:
    """完整标注数据 (用于转换器)"""
    image_path: Path
    image_width: int
    image_height: int
    bboxes: list[BoundingBox] = field(default_factory=list)
    segmentations: list[Segmentation] = field(default_factory=list)


@dataclass
class ConversionResult:
    """转换结果"""
    success: bool
    output_path: Path
    total_images: int
    total_annotations: int
    class_config: Optional[ClassConfig]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ============================================================================
# 坐标转换函数
# ============================================================================

def coco_bbox_to_yolo(
    bbox: list,  # [x, y, width, height] 左上角+宽高
    img_w: int,
    img_h: int
) -> tuple[float, float, float, float]:
    """COCO bbox 转 YOLO 格式 (归一化中心点+宽高)"""
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    norm_w = w / img_w
    norm_h = h / img_h
    # Clamp 到 [0, 1] 范围
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))
    return x_center, y_center, norm_w, norm_h


def voc_bbox_to_yolo(
    xmin: float, ymin: float, xmax: float, ymax: float,
    img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """VOC bbox 转 YOLO 格式 (归一化中心点+宽高)"""
    x_center = (xmin + xmax) / 2 / img_w
    y_center = (ymin + ymax) / 2 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    # Clamp 到 [0, 1] 范围
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    return x_center, y_center, width, height


def coco_segmentation_to_yolo(
    segmentation: list,  # [[x1,y1,x2,y2,...]] 或 RLE
    img_w: int,
    img_h: int
) -> Optional[list[tuple[float, float]]]:
    """COCO segmentation 转 YOLO 格式 (归一化多边形点)"""
    # 只处理多边形格式，跳过 RLE 格式
    if not segmentation or not isinstance(segmentation, list):
        return None
    if not isinstance(segmentation[0], list):
        # RLE 格式，暂不支持
        return None

    # 取第一个多边形（COCO 可能有多个多边形表示同一对象）
    polygon = segmentation[0]
    if len(polygon) < 6:  # 至少需要 3 个点
        return None

    points = []
    for i in range(0, len(polygon), 2):
        x = polygon[i] / img_w
        y = polygon[i + 1] / img_h
        # Clamp 到 [0, 1] 范围
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        points.append((x, y))

    return points


# ============================================================================
# 转换器基类
# ============================================================================

class BaseConverter(ABC):
    """格式转换器基类"""

    def __init__(
        self,
        source_path: str | Path,
        output_path: str | Path,
        target_format: DatasetFormat = DatasetFormat.YOLO
    ):
        self.source_path = Path(source_path).resolve()
        self.output_path = Path(output_path).resolve()
        self.target_format = target_format
        self.class_config: Optional[ClassConfig] = None

    @abstractmethod
    def extract_annotations(self) -> list[AnnotationData]:
        """提取标注数据"""
        pass

    @abstractmethod
    def build_class_config(self) -> ClassConfig:
        """构建类别配置"""
        pass

    def convert(
        self,
        copy_images: bool = True,
        progress_callback: Optional[callable] = None
    ) -> ConversionResult:
        """执行转换"""
        errors = []
        warnings = []
        total_annotations = 0

        try:
            # 1. 构建类别配置
            self.class_config = self.build_class_config()

            # 2. 创建输出目录结构
            self._create_output_structure()

            # 3. 提取标注
            annotations = self.extract_annotations()

            # 4. 转换并保存
            for i, ann in enumerate(annotations):
                try:
                    # 保存标签文件
                    label_saved = self._save_label(ann)
                    if label_saved:
                        total_annotations += len(ann.bboxes) + len(ann.segmentations)

                    # 复制/链接图片
                    if copy_images:
                        self._copy_image(ann.image_path)

                    if progress_callback:
                        progress_callback(i + 1, len(annotations))

                except Exception as e:
                    errors.append(f"处理 {ann.image_path.name} 失败: {e}")

            # 5. 生成类别配置文件
            self._save_class_config()

            return ConversionResult(
                success=len(errors) == 0,
                output_path=self.output_path,
                total_images=len(annotations),
                total_annotations=total_annotations,
                class_config=self.class_config,
                errors=errors,
                warnings=warnings
            )

        except Exception as e:
            return ConversionResult(
                success=False,
                output_path=self.output_path,
                total_images=0,
                total_annotations=0,
                class_config=None,
                errors=[str(e)],
                warnings=warnings
            )

    def _create_output_structure(self):
        """创建 YOLO 输出目录结构"""
        (self.output_path / "images").mkdir(parents=True, exist_ok=True)
        (self.output_path / "labels").mkdir(parents=True, exist_ok=True)

    def _save_label(self, annotation: AnnotationData) -> bool:
        """保存标签文件"""
        label_file = self.output_path / "labels" / f"{annotation.image_path.stem}.txt"

        lines = []

        # 根据目标格式选择输出内容
        if self.target_format == DatasetFormat.YOLO_SEG:
            # 优先使用分割标注
            for seg in annotation.segmentations:
                lines.append(seg.to_yolo_line())
            # 如果没有分割标注，使用边界框
            if not lines:
                for bbox in annotation.bboxes:
                    lines.append(bbox.to_yolo_line())
        else:
            # 标准 YOLO 检测格式
            for bbox in annotation.bboxes:
                lines.append(bbox.to_yolo_line())

        if lines:
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
            return True
        return False

    def _copy_image(self, image_path: Path):
        """复制图片到输出目录"""
        dest = self.output_path / "images" / image_path.name
        if not dest.exists():
            shutil.copy2(image_path, dest)

    def _save_class_config(self):
        """保存类别配置文件"""
        if self.class_config is None:
            return

        config_file = self.output_path / "classes.yaml"
        config_data = {
            'nc': self.class_config.nc,
            'names': self.class_config.names
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False)


# ============================================================================
# COCO 转换器
# ============================================================================

class COCOToYOLOConverter(BaseConverter):
    """COCO → YOLO 转换器"""

    def __init__(
        self,
        source_path: str | Path,
        output_path: str | Path,
        target_format: DatasetFormat = DatasetFormat.YOLO
    ):
        super().__init__(source_path, output_path, target_format)
        self.coco_data: Optional[dict] = None
        self.category_mapping: dict[int, int] = {}  # COCO category_id → YOLO class_idx
        self.images_dir = self.source_path / "images"
        self.annotation_file: Optional[Path] = None

    def _load_coco_data(self):
        """加载 COCO JSON 数据"""
        if self.coco_data is not None:
            return

        # 查找标注文件
        annotations_dir = self.source_path / "annotations"
        if annotations_dir.exists():
            json_files = list(annotations_dir.glob("*.json"))
            # 优先使用 instances_*.json
            for f in json_files:
                if f.name.startswith("instances"):
                    self.annotation_file = f
                    break
            if self.annotation_file is None and json_files:
                self.annotation_file = json_files[0]

        if self.annotation_file is None:
            raise FileNotFoundError("未找到 COCO 标注文件")

        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)

    def build_class_config(self) -> ClassConfig:
        """从 COCO 数据构建类别配置"""
        self._load_coco_data()

        categories = {cat['id']: cat['name'] for cat in self.coco_data.get('categories', [])}
        sorted_cat_ids = sorted(categories.keys())

        # 构建类别映射 (COCO ID → 连续索引)
        self.category_mapping = {cat_id: idx for idx, cat_id in enumerate(sorted_cat_ids)}

        return ClassConfig(
            nc=len(categories),
            names=[categories[cat_id] for cat_id in sorted_cat_ids]
        )

    def extract_annotations(self) -> list[AnnotationData]:
        """从 COCO 数据提取标注"""
        self._load_coco_data()

        # 构建图像信息映射
        image_info = {img['id']: img for img in self.coco_data.get('images', [])}

        # 构建图像 ID → 标注列表映射
        image_annotations: dict[int, list] = {}
        for ann in self.coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)

        # 获取实际存在的图片文件
        existing_images = self._get_existing_images()

        annotations = []
        for img_id, img_info in image_info.items():
            file_name = img_info.get('file_name', '')
            stem = Path(file_name).stem
            img_path = existing_images.get(stem)

            if img_path is None:
                continue

            img_w = img_info.get('width', 0)
            img_h = img_info.get('height', 0)

            if img_w <= 0 or img_h <= 0:
                continue

            bboxes = []
            segmentations = []

            for ann in image_annotations.get(img_id, []):
                cat_id = ann.get('category_id')
                if cat_id not in self.category_mapping:
                    continue

                class_idx = self.category_mapping[cat_id]

                # 提取边界框
                if 'bbox' in ann:
                    bbox = ann['bbox']
                    if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                        x_c, y_c, w, h = coco_bbox_to_yolo(bbox, img_w, img_h)
                        bboxes.append(BoundingBox(
                            class_idx=class_idx,
                            x_center=x_c,
                            y_center=y_c,
                            width=w,
                            height=h
                        ))

                # 提取分割标注
                if 'segmentation' in ann and self.target_format == DatasetFormat.YOLO_SEG:
                    points = coco_segmentation_to_yolo(ann['segmentation'], img_w, img_h)
                    if points:
                        segmentations.append(Segmentation(
                            class_idx=class_idx,
                            points=points
                        ))

            if bboxes or segmentations:
                annotations.append(AnnotationData(
                    image_path=img_path,
                    image_width=img_w,
                    image_height=img_h,
                    bboxes=bboxes,
                    segmentations=segmentations
                ))

        return annotations

    def _get_existing_images(self) -> dict[str, Path]:
        """获取实际存在的图片文件"""
        images = {}
        if not self.images_dir.exists():
            return images

        for ext in IMAGE_EXTENSIONS:
            for img_path in self.images_dir.glob(f"*{ext}"):
                images[img_path.stem] = img_path
            for img_path in self.images_dir.glob(f"*{ext.upper()}"):
                images[img_path.stem] = img_path
        return images


# ============================================================================
# VOC 转换器
# ============================================================================

class VOCToYOLOConverter(BaseConverter):
    """VOC → YOLO 转换器"""

    def __init__(
        self,
        source_path: str | Path,
        output_path: str | Path,
        target_format: DatasetFormat = DatasetFormat.YOLO
    ):
        super().__init__(source_path, output_path, target_format)
        self.class_name_to_idx: dict[str, int] = {}  # class_name → YOLO class_idx
        self.images_dir: Optional[Path] = None
        self.annotations_dir: Optional[Path] = None
        self._detect_directories()

    def _detect_directories(self):
        """检测 VOC 目录结构"""
        # 标准 VOC 结构
        if (self.source_path / "JPEGImages").exists():
            self.images_dir = self.source_path / "JPEGImages"
        elif (self.source_path / "images").exists():
            self.images_dir = self.source_path / "images"

        if (self.source_path / "Annotations").exists():
            self.annotations_dir = self.source_path / "Annotations"
        elif (self.source_path / "annotations").exists():
            self.annotations_dir = self.source_path / "annotations"

        if self.images_dir is None or self.annotations_dir is None:
            raise ValueError("无法识别 VOC 目录结构，需要 JPEGImages/images 和 Annotations/annotations 目录")

    def build_class_config(self) -> ClassConfig:
        """收集所有类别并构建配置"""
        all_class_names: set[str] = set()

        # 扫描所有 XML 文件收集类别名称
        for xml_path in self.annotations_dir.glob("*.xml"):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall('object'):
                    name_elem = obj.find('name')
                    if name_elem is not None and name_elem.text:
                        all_class_names.add(name_elem.text.strip())
            except ET.ParseError:
                continue

        # 按字母顺序排序
        sorted_names = sorted(all_class_names)
        self.class_name_to_idx = {name: idx for idx, name in enumerate(sorted_names)}

        return ClassConfig(
            nc=len(sorted_names),
            names=sorted_names
        )

    def extract_annotations(self) -> list[AnnotationData]:
        """从 VOC XML 提取标注"""
        # 获取实际存在的图片文件
        existing_images = self._get_existing_images()

        annotations = []

        for xml_path in self.annotations_dir.glob("*.xml"):
            try:
                ann = self._parse_voc_xml(xml_path, existing_images)
                if ann:
                    annotations.append(ann)
            except Exception:
                continue

        return annotations

    def _parse_voc_xml(
        self,
        xml_path: Path,
        existing_images: dict[str, Path]
    ) -> Optional[AnnotationData]:
        """解析单个 VOC XML 文件"""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取文件名
        filename_elem = root.find('filename')
        if filename_elem is None or not filename_elem.text:
            return None

        stem = Path(filename_elem.text).stem
        img_path = existing_images.get(stem)

        # 如果按 filename 找不到，尝试用 XML 文件名
        if img_path is None:
            img_path = existing_images.get(xml_path.stem)

        if img_path is None:
            return None

        # 获取图像尺寸
        size_elem = root.find('size')
        if size_elem is None:
            return None

        width_elem = size_elem.find('width')
        height_elem = size_elem.find('height')

        if width_elem is None or height_elem is None:
            return None

        try:
            img_w = int(width_elem.text)
            img_h = int(height_elem.text)
        except (ValueError, TypeError):
            return None

        if img_w <= 0 or img_h <= 0:
            return None

        # 解析所有 object
        bboxes = []
        for obj in root.findall('object'):
            name_elem = obj.find('name')
            if name_elem is None or not name_elem.text:
                continue

            class_name = name_elem.text.strip()
            if class_name not in self.class_name_to_idx:
                continue

            class_idx = self.class_name_to_idx[class_name]

            # 获取边界框
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue

            try:
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
            except (ValueError, TypeError, AttributeError):
                continue

            # 验证边界框
            if xmax <= xmin or ymax <= ymin:
                continue

            x_c, y_c, w, h = voc_bbox_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
            bboxes.append(BoundingBox(
                class_idx=class_idx,
                x_center=x_c,
                y_center=y_c,
                width=w,
                height=h
            ))

        if not bboxes:
            return None

        return AnnotationData(
            image_path=img_path,
            image_width=img_w,
            image_height=img_h,
            bboxes=bboxes,
            segmentations=[]
        )

    def _get_existing_images(self) -> dict[str, Path]:
        """获取实际存在的图片文件"""
        images = {}
        if not self.images_dir.exists():
            return images

        for ext in IMAGE_EXTENSIONS:
            for img_path in self.images_dir.glob(f"*{ext}"):
                images[img_path.stem] = img_path
            for img_path in self.images_dir.glob(f"*{ext.upper()}"):
                images[img_path.stem] = img_path
        return images


# ============================================================================
# 工厂函数
# ============================================================================

def create_converter(
    source_format: DatasetFormat,
    target_format: DatasetFormat,
    source_path: str | Path,
    output_path: str | Path
) -> BaseConverter:
    """创建格式转换器

    Args:
        source_format: 源数据集格式 (COCO 或 VOC)
        target_format: 目标格式 (YOLO 或 YOLO_SEG)
        source_path: 源数据集路径
        output_path: 输出路径

    Returns:
        对应的转换器实例
    """
    if target_format not in (DatasetFormat.YOLO, DatasetFormat.YOLO_SEG):
        raise ValueError(f"不支持的目标格式: {target_format}，仅支持 YOLO 和 YOLO_SEG")

    converters = {
        DatasetFormat.COCO: COCOToYOLOConverter,
        DatasetFormat.VOC: VOCToYOLOConverter,
    }

    converter_class = converters.get(source_format)
    if converter_class is None:
        raise ValueError(f"不支持从 {source_format} 格式转换")

    return converter_class(source_path, output_path, target_format)
