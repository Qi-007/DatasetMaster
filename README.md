# DatasetMaster

目标检测数据集划分与管理工具，支持多种主流数据集格式，提供交互式命令行界面。

## 特性

- **多格式支持**: YOLO、YOLO-OBB、YOLO-Seg、COCO、Pascal VOC、自定义 OBB 格式
- **格式转换**: 支持 COCO/VOC 转换为 YOLO 格式
- **交互式 CLI**: 美观的命令行界面，引导式操作流程
- **智能划分**: 支持分层抽样，确保各类别比例一致
- **数据验证**: 检查标签格式错误、缺失标签、孤立标签、损坏图片
- **配置生成**: 自动生成 YOLO `data.yaml` 配置文件
- **灵活配置**: 支持自定义 OBB 格式，通过 YAML 定义属性字段

## 一键运行

```bash
wget -qO- https://raw.githubusercontent.com/Qi-007/DatasetMaster/main/dataset_master.sh | bash
```

或者下载脚本后运行：

```bash
wget https://raw.githubusercontent.com/Qi-007/DatasetMaster/main/dataset_master.sh
bash dataset_master.sh && rm dataset_master.sh
```

## 安装

```bash
# 克隆项目
git clone https://github.com/Qi-007/DatasetMaster.git
cd DatasetMaster

# 安装依赖
pip install -r requirements.txt

# 运行
python main.py
```

### 依赖

- Python >= 3.10
- rich >= 13.0.0
- questionary >= 2.0.0
- PyYAML >= 6.0
- Pillow >= 9.0.0

## 快速开始

```bash
python main.py
```

启动后，按照交互式提示操作：

1. 选择数据集格式
2. 指定数据集目录
3. (可选) 导入类别配置文件
4. 验证数据集
5. 设置划分比例
6. 选择输出方式
7. 执行划分

## 支持的数据集格式

| 格式 | 说明 | 标签格式 |
|------|------|----------|
| **YOLO** | 标准 YOLO 格式 | `class x_center y_center width height` |
| **YOLO-OBB** | YOLO 旋转框格式 | `class x1 y1 x2 y2 x3 y3 x4 y4` |
| **YOLO-Seg** | YOLO 分割格式 | `class x1 y1 x2 y2 ... xn yn` |
| **COCO** | COCO JSON 格式 | `annotations/*.json` |
| **Pascal VOC** | VOC XML 格式 | `Annotations/*.xml` |
| **Custom-OBB** | 自定义属性 OBB | 通过 YAML 配置定义 |

## 目录结构要求

### YOLO 系列格式

```
dataset/
├── images/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── labels/
│   ├── img001.txt
│   ├── img002.txt
│   └── ...
└── classes.yaml  # 可选
```

### COCO 格式

```
dataset/
├── images/
│   ├── img001.jpg
│   └── ...
└── annotations/
    └── instances.json
```

### Pascal VOC 格式

```
dataset/
├── JPEGImages/  # 或 images/
│   ├── img001.jpg
│   └── ...
└── Annotations/  # 或 annotations/
    ├── img001.xml
    └── ...
```

## 类别配置文件

`classes.yaml` 示例：

```yaml
nc: 3  # 类别数量
names: ['cat', 'dog', 'bird']  # 类别名称列表
```

## 自定义 OBB 格式

对于包含额外属性的 OBB 格式（如颜色、大小等），可通过 `format.yaml` 定义：

```yaml
# 格式名称（可选）
format_name: "my_custom_obb"

# 格式描述（可选）
description: "我的自定义 OBB 格式"

# 属性字段定义（按标注文件中的顺序）
attributes:
  - name: color
    values: ['red', 'blue']
  - name: size
    values: ['small', 'big']

# 类别索引位置（相对于属性字段之后，默认 0）
class_position: 0

# 类别配置（可选，也可使用单独的 classes.yaml）
classes:
  nc: 8
  names: ['0', '1', '2', '3', '4', '5', 'negative', 'guard']
```
当 `attributes` 不为空时，**标注行格式为**：

`<attr1> <attr2> ... <class> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>`

## 输出结构

划分后的数据集结构：

```
output/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml  # YOLO 配置文件
```

## 划分选项
·
| 选项 | 说明 |
|------|------|
| **划分比例** | 支持 8:1:1、7:2:1、6:2:2 或自定义 |
| **分层抽样** | 确保各类别在 train/val/test 中比例一致 |
| **随机种子** | 设置种子确保结果可复现 |
| **操作方式** | 复制文件（保留原始数据）或移动文件（节省空间） |
| **Dry-run** | 预览划分结果，不执行实际操作 |

## 验证功能

- **标签格式验证**: 检查字段数量、数值范围、坐标合法性
- **匹配检查**: 检测缺失标签、孤立标签
- **图片验证**: 可选检测损坏的图片文件
- **类别一致性**: 验证数据中的类别索引是否与配置匹配

## 使用示例

### 基本使用

```bash
python main.py
```

### 作为模块使用

```python
from dataset_master.reader import create_reader
from dataset_master.validator import DatasetValidator
from dataset_master.splitter import DatasetSplitter, SplitConfig
from dataset_master.formats import DatasetFormat

# 读取数据集
reader = create_reader("/path/to/dataset", DatasetFormat.YOLO)
dataset_info = reader.read()

# 验证数据集
validator = DatasetValidator(dataset_info)
result = validator.validate()

# 划分数据集
config = SplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
splitter = DatasetSplitter(dataset_info, config)
split_result = splitter.execute("/path/to/output")
```

## 格式转换

支持将 COCO 和 Pascal VOC 格式转换为 YOLO 格式。

### 支持的转换

| 源格式 | 目标格式 | 说明 |
|--------|----------|------|
| COCO | YOLO | 检测框转换 |
| COCO | YOLO-Seg | 实例分割转换 |
| Pascal VOC | YOLO | 检测框转换 |

### 坐标转换

**COCO → YOLO:**
- COCO: `[x, y, width, height]` (左上角 + 宽高，绝对像素)
- YOLO: `class x_center y_center width height` (中心点 + 宽高，归一化 0-1)

**VOC → YOLO:**
- VOC: `<xmin><ymin><xmax><ymax>` (左上角 + 右下角，绝对像素)
- YOLO: `class x_center y_center width height` (中心点 + 宽高，归一化 0-1)

### 转换示例

```python
from dataset_master import create_converter, DatasetFormat

# 创建转换器
converter = create_converter(
    source_format=DatasetFormat.COCO,
    target_format=DatasetFormat.YOLO,
    source_path="/path/to/coco_dataset",
    output_path="/path/to/yolo_output"
)

# 执行转换
result = converter.convert(copy_images=True)
print(f"转换完成: {result.total_images} 张图片, {result.total_annotations} 个标注")
```

### 转换输出结构

```
output/
├── images/
│   ├── 000001.jpg
│   └── ...
├── labels/
│   ├── 000001.txt
│   └── ...
└── classes.yaml
```
