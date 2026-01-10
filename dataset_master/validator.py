"""数据完整性检查模块 - 支持多种格式"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field

from .reader import DatasetInfo, DatasetItem
from .formats import DatasetFormat, FORMAT_INFO


@dataclass
class LabelError:
    """标签错误信息"""
    file_path: Path
    line_number: int
    error_type: str
    message: str


@dataclass
class ImageError:
    """图片错误信息"""
    file_path: Path
    error_type: str
    message: str


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    label_errors: list[LabelError] = field(default_factory=list)
    image_errors: list[ImageError] = field(default_factory=list)
    corrupted_images: list[Path] = field(default_factory=list)


class DatasetValidator:
    """数据集验证器"""

    def __init__(self, dataset_info: DatasetInfo):
        self.dataset_info = dataset_info
        self.format_info = FORMAT_INFO.get(dataset_info.format)

    def _validate_yolo_label_line(self, parts: list[str], line_num: int, label_path: Path) -> list[LabelError]:
        """验证标准 YOLO 格式标签行"""
        errors = []

        # 检查字段数量（YOLO格式: class x y w h）
        if len(parts) < 5:
            errors.append(LabelError(
                file_path=label_path,
                line_number=line_num,
                error_type="format",
                message=f"字段数量不足，期望至少5个，实际{len(parts)}个"
            ))
            return errors

        # 检查类别索引
        try:
            class_idx = int(parts[0])
            if class_idx < 0:
                errors.append(LabelError(
                    file_path=label_path,
                    line_number=line_num,
                    error_type="class_index",
                    message=f"类别索引不能为负数: {class_idx}"
                ))
        except ValueError:
            errors.append(LabelError(
                file_path=label_path,
                line_number=line_num,
                error_type="class_index",
                message=f"类别索引必须是整数: {parts[0]}"
            ))

        # 检查坐标值范围 (0-1)
        for i, name in enumerate(['x', 'y', 'w', 'h'], 1):
            try:
                val = float(parts[i])
                if not (0 <= val <= 1):
                    errors.append(LabelError(
                        file_path=label_path,
                        line_number=line_num,
                        error_type="coordinate",
                        message=f"{name} 坐标值超出范围 [0,1]: {val}"
                    ))
            except ValueError:
                errors.append(LabelError(
                    file_path=label_path,
                    line_number=line_num,
                    error_type="coordinate",
                    message=f"{name} 必须是数字: {parts[i]}"
                ))

        return errors

    def _validate_yolo_obb_label_line(self, parts: list[str], line_num: int, label_path: Path) -> list[LabelError]:
        """验证 YOLO OBB 格式标签行"""
        errors = []

        # 检查字段数量（OBB格式: class x1 y1 x2 y2 x3 y3 x4 y4）
        if len(parts) < 9:
            errors.append(LabelError(
                file_path=label_path,
                line_number=line_num,
                error_type="format",
                message=f"字段数量不足，期望9个，实际{len(parts)}个"
            ))
            return errors

        # 检查类别索引
        try:
            class_idx = int(parts[0])
            if class_idx < 0:
                errors.append(LabelError(
                    file_path=label_path,
                    line_number=line_num,
                    error_type="class_index",
                    message=f"类别索引不能为负数: {class_idx}"
                ))
        except ValueError:
            errors.append(LabelError(
                file_path=label_path,
                line_number=line_num,
                error_type="class_index",
                message=f"类别索引必须是整数: {parts[0]}"
            ))

        # 检查8个坐标值范围 (0-1)
        coord_names = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
        for i, name in enumerate(coord_names, 1):
            try:
                val = float(parts[i])
                if not (0 <= val <= 1):
                    errors.append(LabelError(
                        file_path=label_path,
                        line_number=line_num,
                        error_type="coordinate",
                        message=f"{name} 坐标值超出范围 [0,1]: {val}"
                    ))
            except ValueError:
                errors.append(LabelError(
                    file_path=label_path,
                    line_number=line_num,
                    error_type="coordinate",
                    message=f"{name} 必须是数字: {parts[i]}"
                ))

        return errors

    def _validate_yolo_seg_label_line(self, parts: list[str], line_num: int, label_path: Path) -> list[LabelError]:
        """验证 YOLO Segmentation 格式标签行"""
        errors = []

        # 检查字段数量（Seg格式: class x1 y1 x2 y2 ... xn yn，至少需要 7 个字段 = 1 class + 3 points）
        if len(parts) < 7:
            errors.append(LabelError(
                file_path=label_path,
                line_number=line_num,
                error_type="format",
                message=f"字段数量不足，分割格式至少需要7个字段（1类别+3点），实际{len(parts)}个"
            ))
            return errors

        # 检查点数是否为偶数（x,y 成对）
        coord_count = len(parts) - 1
        if coord_count % 2 != 0:
            errors.append(LabelError(
                file_path=label_path,
                line_number=line_num,
                error_type="format",
                message=f"坐标点数量必须成对（x,y），当前坐标数: {coord_count}"
            ))

        # 检查类别索引
        try:
            class_idx = int(parts[0])
            if class_idx < 0:
                errors.append(LabelError(
                    file_path=label_path,
                    line_number=line_num,
                    error_type="class_index",
                    message=f"类别索引不能为负数: {class_idx}"
                ))
        except ValueError:
            errors.append(LabelError(
                file_path=label_path,
                line_number=line_num,
                error_type="class_index",
                message=f"类别索引必须是整数: {parts[0]}"
            ))

        # 检查坐标值范围 (0-1)
        for i in range(1, len(parts)):
            try:
                val = float(parts[i])
                if not (0 <= val <= 1):
                    errors.append(LabelError(
                        file_path=label_path,
                        line_number=line_num,
                        error_type="coordinate",
                        message=f"坐标值超出范围 [0,1]: {val}"
                    ))
                    break  # 只报告第一个错误
            except ValueError:
                errors.append(LabelError(
                    file_path=label_path,
                    line_number=line_num,
                    error_type="coordinate",
                    message=f"坐标必须是数字: {parts[i]}"
                ))
                break

        return errors

    def _validate_armor_obb_label_line(self, parts: list[str], line_num: int, label_path: Path) -> list[LabelError]:
        """验证装甲板 OBB 格式标签行 (color size class x1 y1 x2 y2 x3 y3 x4 y4)"""
        errors = []

        # 检查字段数量（Armor OBB格式: color size class + 8个坐标 = 11个字段）
        if len(parts) < 11:
            errors.append(LabelError(
                file_path=label_path,
                line_number=line_num,
                error_type="format",
                message=f"字段数量不足，期望11个，实际{len(parts)}个"
            ))
            return errors

        # 检查颜色属性 (应为非负整数)
        try:
            color = int(parts[0])
            if color < 0:
                errors.append(LabelError(
                    file_path=label_path,
                    line_number=line_num,
                    error_type="attribute",
                    message=f"颜色属性不能为负数: {color}"
                ))
        except ValueError:
            errors.append(LabelError(
                file_path=label_path,
                line_number=line_num,
                error_type="attribute",
                message=f"颜色属性必须是整数: {parts[0]}"
            ))

        # 检查大小属性 (应为非负整数)
        try:
            size = int(parts[1])
            if size < 0:
                errors.append(LabelError(
                    file_path=label_path,
                    line_number=line_num,
                    error_type="attribute",
                    message=f"大小属性不能为负数: {size}"
                ))
        except ValueError:
            errors.append(LabelError(
                file_path=label_path,
                line_number=line_num,
                error_type="attribute",
                message=f"大小属性必须是整数: {parts[1]}"
            ))

        # 检查类别索引
        try:
            class_idx = int(parts[2])
            if class_idx < 0:
                errors.append(LabelError(
                    file_path=label_path,
                    line_number=line_num,
                    error_type="class_index",
                    message=f"类别索引不能为负数: {class_idx}"
                ))
        except ValueError:
            errors.append(LabelError(
                file_path=label_path,
                line_number=line_num,
                error_type="class_index",
                message=f"类别索引必须是整数: {parts[2]}"
            ))

        # 检查8个坐标值范围 (0-1)
        coord_names = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
        for i, name in enumerate(coord_names, 3):
            try:
                val = float(parts[i])
                if not (0 <= val <= 1):
                    errors.append(LabelError(
                        file_path=label_path,
                        line_number=line_num,
                        error_type="coordinate",
                        message=f"{name} 坐标值超出范围 [0,1]: {val}"
                    ))
            except ValueError:
                errors.append(LabelError(
                    file_path=label_path,
                    line_number=line_num,
                    error_type="coordinate",
                    message=f"{name} 必须是数字: {parts[i]}"
                ))

        return errors

    def _validate_coco_annotation(self, coco_data: dict, label_path: Path) -> list[LabelError]:
        """验证 COCO JSON 标注格式"""
        errors = []

        # 检查必需字段
        required_fields = ['images', 'annotations', 'categories']
        for field in required_fields:
            if field not in coco_data:
                errors.append(LabelError(
                    file_path=label_path,
                    line_number=0,
                    error_type="format",
                    message=f"缺少必需字段: {field}"
                ))

        if errors:
            return errors

        # 检查 annotations 格式
        for i, ann in enumerate(coco_data.get('annotations', [])):
            if 'image_id' not in ann:
                errors.append(LabelError(
                    file_path=label_path,
                    line_number=i + 1,
                    error_type="format",
                    message=f"annotation[{i}] 缺少 image_id"
                ))
            if 'category_id' not in ann:
                errors.append(LabelError(
                    file_path=label_path,
                    line_number=i + 1,
                    error_type="format",
                    message=f"annotation[{i}] 缺少 category_id"
                ))
            if 'bbox' not in ann and 'segmentation' not in ann:
                errors.append(LabelError(
                    file_path=label_path,
                    line_number=i + 1,
                    error_type="format",
                    message=f"annotation[{i}] 缺少 bbox 或 segmentation"
                ))

            # 只报告前10个错误
            if len(errors) >= 10:
                break

        return errors

    def _validate_voc_xml(self, xml_path: Path) -> list[LabelError]:
        """验证 VOC XML 标注格式"""
        errors = []

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 检查根元素
            if root.tag != 'annotation':
                errors.append(LabelError(
                    file_path=xml_path,
                    line_number=0,
                    error_type="format",
                    message=f"根元素应为 'annotation'，实际为 '{root.tag}'"
                ))
                return errors

            # 检查 object 元素
            objects = root.findall('object')
            for i, obj in enumerate(objects):
                name = obj.find('name')
                if name is None or not name.text:
                    errors.append(LabelError(
                        file_path=xml_path,
                        line_number=i + 1,
                        error_type="format",
                        message=f"object[{i}] 缺少 name 元素"
                    ))

                bndbox = obj.find('bndbox')
                if bndbox is None:
                    errors.append(LabelError(
                        file_path=xml_path,
                        line_number=i + 1,
                        error_type="format",
                        message=f"object[{i}] 缺少 bndbox 元素"
                    ))
                else:
                    for coord in ['xmin', 'ymin', 'xmax', 'ymax']:
                        elem = bndbox.find(coord)
                        if elem is None or elem.text is None:
                            errors.append(LabelError(
                                file_path=xml_path,
                                line_number=i + 1,
                                error_type="format",
                                message=f"object[{i}] bndbox 缺少 {coord}"
                            ))

        except ET.ParseError as e:
            errors.append(LabelError(
                file_path=xml_path,
                line_number=0,
                error_type="parse_error",
                message=f"XML 解析错误: {e}"
            ))

        return errors

    def validate_labels(self) -> list[LabelError]:
        """验证标签文件格式"""
        errors = []
        format_type = self.dataset_info.format

        # COCO 格式特殊处理（单一 JSON 文件）
        if format_type == DatasetFormat.COCO:
            if self.dataset_info.labels_dir:
                json_files = list(self.dataset_info.labels_dir.glob("*.json"))
                for json_file in json_files:
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            coco_data = json.load(f)
                        errors.extend(self._validate_coco_annotation(coco_data, json_file))
                    except json.JSONDecodeError as e:
                        errors.append(LabelError(
                            file_path=json_file,
                            line_number=0,
                            error_type="parse_error",
                            message=f"JSON 解析错误: {e}"
                        ))
            return errors

        # VOC 格式特殊处理（每个图片一个 XML）
        if format_type == DatasetFormat.VOC:
            for item in self.dataset_info.items:
                if item.label_path and item.label_path.exists():
                    errors.extend(self._validate_voc_xml(item.label_path))
            return errors

        # YOLO 系列格式（逐行验证）
        for item in self.dataset_info.items:
            if item.label_path is None:
                continue

            try:
                with open(item.label_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()

                        if format_type == DatasetFormat.YOLO:
                            errors.extend(self._validate_yolo_label_line(parts, line_num, item.label_path))
                        elif format_type == DatasetFormat.YOLO_OBB:
                            errors.extend(self._validate_yolo_obb_label_line(parts, line_num, item.label_path))
                        elif format_type == DatasetFormat.YOLO_SEG:
                            errors.extend(self._validate_yolo_seg_label_line(parts, line_num, item.label_path))
                        elif format_type == DatasetFormat.CUSTOM_OBB:
                            errors.extend(self._validate_armor_obb_label_line(parts, line_num, item.label_path))

            except Exception as e:
                errors.append(LabelError(
                    file_path=item.label_path,
                    line_number=0,
                    error_type="read_error",
                    message=f"读取文件失败: {e}"
                ))

        return errors

    def validate_images(self, check_corrupted: bool = False) -> list[ImageError]:
        """验证图片文件"""
        errors = []

        if not check_corrupted:
            return errors

        try:
            from PIL import Image
        except ImportError:
            return errors

        for item in self.dataset_info.items:
            try:
                with Image.open(item.image_path) as img:
                    img.verify()
            except Exception as e:
                errors.append(ImageError(
                    file_path=item.image_path,
                    error_type="corrupted",
                    message=f"图片损坏或无法读取: {e}"
                ))

        return errors

    def validate(self, check_corrupted_images: bool = False) -> ValidationResult:
        """执行完整验证"""
        result = ValidationResult()

        # 检查缺失标签
        if self.dataset_info.missing_labels:
            result.warnings.append(
                f"发现 {len(self.dataset_info.missing_labels)} 个图片缺少对应标签"
            )

        # 检查孤立标签
        if self.dataset_info.orphan_labels:
            result.warnings.append(
                f"发现 {len(self.dataset_info.orphan_labels)} 个孤立标签（无对应图片）"
            )

        # 验证标签格式
        label_errors = self.validate_labels()
        result.label_errors = label_errors
        if label_errors:
            result.errors.append(f"发现 {len(label_errors)} 个标签格式错误")
            result.is_valid = False

        # 验证图片（可选）
        if check_corrupted_images:
            image_errors = self.validate_images(check_corrupted=True)
            result.image_errors = image_errors
            result.corrupted_images = [e.file_path for e in image_errors]
            if image_errors:
                result.warnings.append(f"发现 {len(image_errors)} 个损坏的图片")

        # 检查类别配置一致性
        if self.dataset_info.class_config and self.dataset_info.class_distribution:
            max_class_in_data = max(self.dataset_info.class_distribution.keys())
            if max_class_in_data >= self.dataset_info.class_config.nc:
                result.errors.append(
                    f"数据中存在超出配置范围的类别索引: {max_class_in_data} >= {self.dataset_info.class_config.nc}"
                )
                result.is_valid = False

        return result
