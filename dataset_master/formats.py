"""数据集格式定义"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class DatasetFormat(Enum):
    """支持的数据集格式"""
    YOLO = "yolo"                # 标准 YOLO 格式 (images/ + labels/ 分离)
    YOLO_OBB = "yolo_obb"        # YOLO OBB 旋转框格式
    YOLO_SEG = "yolo_seg"        # YOLO 分割格式
    CUSTOM_OBB = "custom_obb"    # 自定义属性 OBB 格式 (通过 YAML 配置)
    COCO = "coco"                # COCO JSON 格式
    VOC = "voc"                  # Pascal VOC XML 格式


@dataclass
class FormatInfo:
    """格式信息"""
    name: str
    description: str
    label_fields: int  # 每行字段数 (0 表示不适用，如 JSON/XML 格式)
    separate_dirs: bool  # 图片和标签是否分离目录
    label_ext: str = ".txt"  # 标签文件扩展名


FORMAT_INFO = {
    DatasetFormat.YOLO: FormatInfo(
        name="YOLO",
        description="标准 YOLO 格式 (class x_center y_center width height)",
        label_fields=5,
        separate_dirs=True
    ),
    DatasetFormat.YOLO_OBB: FormatInfo(
        name="YOLO-OBB",
        description="YOLO OBB 旋转框格式 (class x1 y1 x2 y2 x3 y3 x4 y4)",
        label_fields=9,
        separate_dirs=True
    ),
    DatasetFormat.YOLO_SEG: FormatInfo(
        name="YOLO-Seg",
        description="YOLO 分割格式 (class x1 y1 x2 y2 ... xn yn)",
        label_fields=0,  # 可变长度
        separate_dirs=True
    ),
    DatasetFormat.CUSTOM_OBB: FormatInfo(
        name="Custom-OBB",
        description="自定义属性 OBB 格式 (通过 YAML 配置定义字段)",
        label_fields=0,  # 可变，由配置决定
        separate_dirs=True
    ),
    DatasetFormat.COCO: FormatInfo(
        name="COCO",
        description="COCO JSON 格式 (annotations/*.json)",
        label_fields=0,
        separate_dirs=True,
        label_ext=".json"
    ),
    DatasetFormat.VOC: FormatInfo(
        name="Pascal VOC",
        description="Pascal VOC XML 格式 (Annotations/*.xml)",
        label_fields=0,
        separate_dirs=True,
        label_ext=".xml"
    ),
}


def get_format_choices() -> list[tuple[str, DatasetFormat]]:
    """获取格式选择列表"""
    return [
        (f"{info.name} - {info.description}", fmt)
        for fmt, info in FORMAT_INFO.items()
    ]
