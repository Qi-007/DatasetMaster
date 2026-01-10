"""数据集划分模块 - 支持分层抽样（纯Python实现）"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
import shutil
import random

from .reader import DatasetInfo, DatasetItem


@dataclass
class SplitConfig:
    """划分配置"""
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: Optional[int] = None
    stratify: bool = True  # 分层抽样
    copy_files: bool = True  # True=复制, False=移动


@dataclass
class SplitResult:
    """划分结果"""
    train_items: list[DatasetItem] = field(default_factory=list)
    val_items: list[DatasetItem] = field(default_factory=list)
    test_items: list[DatasetItem] = field(default_factory=list)

    # 统计信息
    train_class_dist: dict[int, int] = field(default_factory=dict)
    val_class_dist: dict[int, int] = field(default_factory=dict)
    test_class_dist: dict[int, int] = field(default_factory=dict)


def stratified_split(
    items: list,
    labels: list,
    ratios: tuple[float, float, float],
    seed: Optional[int] = None
) -> tuple[list, list, list]:
    """
    分层抽样划分

    Args:
        items: 待划分的样本列表
        labels: 每个样本对应的标签
        ratios: (train_ratio, val_ratio, test_ratio)
        seed: 随机种子

    Returns:
        (train_items, val_items, test_items)
    """
    if seed is not None:
        random.seed(seed)

    train_ratio, val_ratio, test_ratio = ratios

    # 按类别分组
    groups: dict[int, list] = defaultdict(list)
    for item, label in zip(items, labels):
        groups[label].append(item)

    train_items = []
    val_items = []
    test_items = []

    # 对每个类别分别划分
    for label, group_items in groups.items():
        random.shuffle(group_items)
        n = len(group_items)

        # 计算各集合的数量
        n_test = max(1, round(n * test_ratio)) if test_ratio > 0 else 0
        n_val = max(1, round(n * val_ratio)) if val_ratio > 0 else 0
        n_train = n - n_test - n_val

        # 确保至少有训练样本
        if n_train < 1 and n > 0:
            n_train = 1
            if n_val > 0:
                n_val = max(0, n - n_train - n_test)
            if n_val < 0:
                n_val = 0
                n_test = n - n_train

        # 划分
        train_items.extend(group_items[:n_train])
        val_items.extend(group_items[n_train:n_train + n_val])
        test_items.extend(group_items[n_train + n_val:])

    # 打乱顺序
    random.shuffle(train_items)
    random.shuffle(val_items)
    random.shuffle(test_items)

    return train_items, val_items, test_items


def random_split(
    items: list,
    ratios: tuple[float, float, float],
    seed: Optional[int] = None
) -> tuple[list, list, list]:
    """
    随机划分（不分层）

    Args:
        items: 待划分的样本列表
        ratios: (train_ratio, val_ratio, test_ratio)
        seed: 随机种子

    Returns:
        (train_items, val_items, test_items)
    """
    if seed is not None:
        random.seed(seed)

    train_ratio, val_ratio, test_ratio = ratios

    items = list(items)
    random.shuffle(items)
    n = len(items)

    n_test = round(n * test_ratio)
    n_val = round(n * val_ratio)
    n_train = n - n_test - n_val

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]

    return train_items, val_items, test_items


class DatasetSplitter:
    """数据集划分器"""

    def __init__(self, dataset_info: DatasetInfo, config: SplitConfig):
        self.dataset_info = dataset_info
        self.config = config

        # 验证比例
        total = config.train_ratio + config.val_ratio + config.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"划分比例之和必须为1，当前为 {total}")

    def _get_stratify_labels(self, items: list[DatasetItem]) -> list[int]:
        """获取用于分层抽样的标签（使用主类别）"""
        labels = []
        for item in items:
            if item.classes:
                # 使用第一个类别作为分层依据
                labels.append(item.classes[0])
            else:
                # 无标签的样本标记为 -1
                labels.append(-1)
        return labels

    def _calculate_class_distribution(self, items: list[DatasetItem]) -> dict[int, int]:
        """计算类别分布"""
        dist: dict[int, int] = {}
        for item in items:
            for cls in item.classes:
                dist[cls] = dist.get(cls, 0) + 1
        return dist

    def split(self) -> SplitResult:
        """执行数据集划分"""
        items = self.dataset_info.items

        if len(items) == 0:
            return SplitResult()

        ratios = (self.config.train_ratio, self.config.val_ratio, self.config.test_ratio)

        if self.config.stratify:
            labels = self._get_stratify_labels(items)
            train_items, val_items, test_items = stratified_split(
                items, labels, ratios, self.config.seed
            )
        else:
            train_items, val_items, test_items = random_split(
                items, ratios, self.config.seed
            )

        return SplitResult(
            train_items=train_items,
            val_items=val_items,
            test_items=test_items,
            train_class_dist=self._calculate_class_distribution(train_items),
            val_class_dist=self._calculate_class_distribution(val_items),
            test_class_dist=self._calculate_class_distribution(test_items)
        )

    def preview(self) -> SplitResult:
        """预览划分结果（不执行文件操作）"""
        return self.split()

    def execute(self, output_dir: str | Path, dry_run: bool = False) -> SplitResult:
        """执行划分并复制/移动文件"""
        result = self.split()
        output_path = Path(output_dir).resolve()

        splits = [
            ("train", result.train_items),
            ("val", result.val_items),
            ("test", result.test_items)
        ]

        # 创建目录结构 (images/ + labels/ 分离)
        if not dry_run:
            for split_name, _ in splits:
                (output_path / "images" / split_name).mkdir(parents=True, exist_ok=True)
                (output_path / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        # 复制/移动文件
        file_op = shutil.copy2 if self.config.copy_files else shutil.move

        for split_name, items in splits:
            for item in items:
                if dry_run:
                    continue

                # 处理图片
                img_dest = output_path / "images" / split_name / item.image_path.name
                file_op(str(item.image_path), str(img_dest))

                # 处理标签
                if item.label_path:
                    label_dest = output_path / "labels" / split_name / item.label_path.name
                    file_op(str(item.label_path), str(label_dest))

        return result
