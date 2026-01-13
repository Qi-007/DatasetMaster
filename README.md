# DatasetMaster

目标检测数据集划分与格式转换工具，支持 YOLO、COCO、Pascal VOC 等主流格式。

## 主要功能

- **数据集划分**：将数据集按比例划分为 train/val/test，支持分层抽样
- **格式转换**：COCO/Pascal VOC → YOLO 格式转换

## 快速开始

### 安装运行

```bash
# 克隆项目
git clone https://github.com/Qi-007/DatasetMaster.git
cd DatasetMaster

# 安装依赖
pip install -r requirements.txt

# 运行
python main.py
```

或使用一键脚本（自动安装依赖）：

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/Qi-007/DatasetMaster/main/dataset_master.sh)
```

### 系统要求

- Python >= 3.10
- pip

---

## 详细使用说明

### 功能一：数据集划分

将数据集划分为训练集、验证集和测试集。

#### 支持的格式

| 格式 | 标签格式 |
|------|----------|
| YOLO | `class x_center y_center width height` |
| YOLO-OBB | `class x1 y1 x2 y2 x3 y3 x4 y4` |
| YOLO-Seg | `class x1 y1 x2 y2 ... xn yn` |
| Custom-OBB | 通过 YAML 配置定义 |

#### 输入目录结构

```
dataset/
├── images/
│   ├── img001.jpg
│   └── img002.jpg
├── labels/
│   ├── img001.txt
│   └── img002.txt
└── classes.yaml      # 可选
```

#### 操作步骤

1. **选择数据集格式** - 根据标签文件格式选择
2. **选择数据集目录** - 包含 images/ 和 labels/ 的目录
3. **导入类别配置（可选）** - 加载 `classes.yaml` 定义类别名称
4. **验证数据集** - 检查标签格式、缺失文件、损坏图片
5. **设置划分比例** - 预设 8:1:1、7:2:1、6:2:2 或自定义
6. **配置选项**
   - 分层抽样：确保各类别比例一致
   - 随机种子：保证结果可复现
   - 操作方式：复制（保留原数据）或移动
7. **预览并执行** - 支持 dry-run 模式仅预览

#### 输出结构

```
output/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml         # YOLO 配置文件（可选生成）
```

#### 类别配置文件

`classes.yaml` 格式：

```yaml
nc: 3
names:
  - cat
  - dog
  - bird
```

---

### 功能二：格式转换

将 COCO 或 Pascal VOC 格式转换为 YOLO 格式。

#### 支持的转换

| 源格式 | 目标格式 |
|--------|----------|
| COCO | YOLO、YOLO-Seg |
| Pascal VOC | YOLO |

#### COCO 格式输入

```
coco_dataset/
├── images/
│   └── *.jpg
└── annotations/
    └── instances.json
```

#### Pascal VOC 格式输入

```
voc_dataset/
├── JPEGImages/       # 或 images/
│   └── *.jpg
└── Annotations/      # 或 annotations/
    └── *.xml
```

#### 操作步骤

1. 选择源格式（COCO / Pascal VOC）
2. 选择目标格式（YOLO / YOLO-Seg）
3. 选择源数据集目录
4. 选择输出目录
5. 选择是否复制图片到输出目录
6. 确认执行

#### 转换输出

```
output/
├── images/
│   └── *.jpg
├── labels/
│   └── *.txt
└── classes.yaml
```

#### 坐标转换说明

| 源格式 | 原始坐标 | YOLO 坐标 |
|--------|----------|-----------|
| COCO | `[x, y, w, h]` 左上角+宽高，像素 | `x_center y_center w h` 归一化 |
| VOC | `xmin ymin xmax ymax` 像素 | `x_center y_center w h` 归一化 |

---

### 自定义 OBB 格式

对于包含额外属性的 OBB 格式，可通过 `format.yaml` 定义：

```yaml
format_name: "my_custom_obb"
description: "自定义 OBB 格式"

# 属性字段（按标注文件顺序）
attributes:
  - name: color
    values: ['red', 'blue']
  - name: size
    values: ['small', 'big']

# 类别索引位置
class_position: 0

# 类别配置
classes:
  nc: 3
  names: ['cat', 'dog', 'bird']
```

标注格式：`<attr1> <attr2> ... <class> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>`

---

### 编程接口

```python
from dataset_master.reader import create_reader
from dataset_master.validator import DatasetValidator
from dataset_master.splitter import DatasetSplitter, SplitConfig
from dataset_master.converter import create_converter
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
splitter.execute("/path/to/output")

# 格式转换
converter = create_converter(
    source_format=DatasetFormat.COCO,
    target_format=DatasetFormat.YOLO,
    source_path="/path/to/coco",
    output_path="/path/to/output"
)
result = converter.convert(copy_images=True)
```
