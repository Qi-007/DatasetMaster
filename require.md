我想做一个
- 能够读取dataset的images和labels并且根据数目自己划分好训练集，验证集，测试集
- 如果是 YOLO 项目，可以顺手生成一个 data.yaml（自动写入 train/val/test 路径和类别数）**data.yaml 示例：**         
	`path: dataset
	`train: images/train`
	`val: images/val`
	`test: images/test`
	
	`nc: 3`
	`names: ['red', 'blue', 'green']`

- 支持"按类别分层抽样"（避免某些类别全跑到 train 或 test 里）
- 支持用户通过 YAML 文件导入类别配置（nc 和 names），若未提供则自动统计
- 能支持不同格式的数据集（可以先写标准的YOLO格式）
- 可交互的终端脚本，供用户选择

## 补充需求

### 数据完整性检查
- 检测图片与标签是否一一对应（报告缺失标签/孤立标签）
- 验证标签文件格式是否正确
- 可选：检测损坏的图片

### 划分预览 & 可复现性
- 划分前显示统计预览（各集合样本数、各类别分布）
- 支持设置随机种子（--seed），确保划分结果可复现

### 安全性考虑
- 默认使用"复制"而非"移动"，避免误操作丢失原始数据
- 移动操作前需二次确认
- 支持 --dry-run 模式，仅预览不执行

### 划分后统计报告
- 输出各集合的样本数量
- 输出各类别在 train/val/test 中的分布情况
- 示例：
```
Dataset Split Report
━━━━━━━━━━━━━━━━━━━━
Train: 800 images
  - class_0: 320
  - class_1: 280
  - class_2: 200
Val:   100 images
Test:  100 images
```

# `format.yaml` 配置文件说明

`format.yaml` 用于描述**自定义标注文件的字段结构**，支持 **多属性 + OBB（四点框）** 的目标检测格式。

---

## 一、基本结构

```
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



---

## 二、对应的标注文件格式

当 `attributes` 不为空时，**标注行格式为**：

`<attr1> <attr2> ... <class> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>`

### 示例

`0 1 3 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4`

含义：

- `0` → color = red
  
- `1` → size = big
  
- `3` → class = '3'
  
- `(x1, y1) ~ (x4, y4)` → 归一化四点坐标（OBB）

---

## 三、示例配置

### 1️⃣ 只有类别的 OBB 格式（无属性字段）

```
format_name: "simple_obb"
description: "简单 OBB 格式，无属性"

attributes: []  # 空列表表示无属性
```

对应标注格式：

`<class> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>`

---

### 2️⃣ 双属性 OBB 格式

```
format_name: "armor_obb"
description: "装甲板 OBB 格式"

attributes:
  - name: color
    values: ['red', 'blue']
  - name: size
    values: ['small', 'big']

classes:
  nc: 8
  names: ['0', '1', '2', '3', '4', '5', 'negative', 'guard']

```

对应标注格式：

`<color> <size> <class> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>`

---

### 3️⃣ 多属性 OBB 格式

```
format_name: "robot_detection"
description: "机器人检测格式"

attributes:
  - name: team
    values: ['red', 'blue']
  - name: type
    values: ['infantry', 'hero', 'engineer', 'sentry', 'drone']
  - name: status
    values: ['normal', 'damaged', 'offline']

```

对应标注格式：

`<team> <type> <status> <class> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>`