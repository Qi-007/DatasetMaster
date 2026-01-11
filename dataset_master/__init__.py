"""DatasetMaster - 数据集划分与管理工具"""

__version__ = "0.1.0"

from .formats import DatasetFormat, FORMAT_INFO
from .reader import (
    create_reader,
    DatasetInfo,
    DatasetItem,
    ClassConfig,
    YOLOReader,
    COCOReader,
    VOCReader,
)
from .converter import (
    create_converter,
    COCOToYOLOConverter,
    VOCToYOLOConverter,
    ConversionResult,
    BoundingBox,
    Segmentation,
    AnnotationData,
)

__all__ = [
    # 版本
    "__version__",
    # 格式
    "DatasetFormat",
    "FORMAT_INFO",
    # 读取器
    "create_reader",
    "DatasetInfo",
    "DatasetItem",
    "ClassConfig",
    "YOLOReader",
    "COCOReader",
    "VOCReader",
    # 转换器
    "create_converter",
    "COCOToYOLOConverter",
    "VOCToYOLOConverter",
    "ConversionResult",
    "BoundingBox",
    "Segmentation",
    "AnnotationData",
]
