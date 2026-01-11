"""交互式 CLI 模块"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypeVar

import questionary
from questionary import Style as QStyle
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

from .formats import DatasetFormat, FORMAT_INFO
from .reader import create_reader, DatasetInfo, ClassConfig
from .validator import DatasetValidator, ValidationResult
from .splitter import DatasetSplitter, SplitConfig, SplitResult
from .config import YAMLConfigGenerator
from .converter import create_converter, ConversionResult


# ============================================================================
# 主题配置
# ============================================================================

@dataclass
class Theme:
    """CLI 主题配置"""
    # 主色调
    primary: str = "cyan"
    secondary: str = "magenta"
    accent: str = "yellow"

    # 状态颜色
    success: str = "green"
    warning: str = "yellow"
    error: str = "red"
    muted: str = "dim"

    # Emoji 图标
    icon_app: str = "📦"
    icon_folder: str = "📁"
    icon_image: str = "🖼️"
    icon_label: str = "🏷️"
    icon_check: str = "✅"
    icon_warn: str = "⚠️"
    icon_error: str = "❌"
    icon_info: str = "💡"
    icon_rocket: str = "🚀"
    icon_sparkles: str = "✨"
    icon_chart: str = "📊"
    icon_gear: str = "⚙️"
    icon_save: str = "💾"
    icon_bye: str = "👋"
    icon_thinking: str = "🤔"
    icon_party: str = "🎉"

    # 表格样式
    table_header: str = "bold cyan"
    table_border: str = "dim"

    # questionary 样式
    @property
    def questionary_style(self) -> QStyle:
        return QStyle([
            ('qmark', 'fg:cyan bold'),
            ('question', 'bold'),
            ('answer', 'fg:cyan'),
            ('pointer', 'fg:cyan bold'),
            ('highlighted', 'fg:cyan bold'),
            ('selected', 'fg:green'),
        ])


# 全局主题实例
theme = Theme()
console = Console()

T = TypeVar('T')


# ============================================================================
# 输出辅助函数
# ============================================================================

def msg(text: str, style: str = "", icon: str = "", end: str = "\n"):
    """统一的消息输出"""
    prefix = f"{icon} " if icon else ""
    if style:
        console.print(f"{prefix}{text}", style=style, end=end)
    else:
        console.print(f"{prefix}{text}", end=end)


def msg_info(text: str):
    """信息消息"""
    msg(text, theme.primary, theme.icon_info)


def msg_success(text: str):
    """成功消息"""
    msg(text, theme.success, theme.icon_check)


def msg_warning(text: str):
    """警告消息"""
    msg(text, theme.warning, theme.icon_warn)


def msg_error(text: str):
    """错误消息"""
    msg(text, theme.error, theme.icon_error)


def msg_muted(text: str):
    """次要消息"""
    msg(text, theme.muted)


def msg_step(text: str):
    """步骤消息"""
    msg(text, theme.primary, theme.icon_gear)


# ============================================================================
# 异常类
# ============================================================================

class UserCancelled(Exception):
    """用户取消操作异常"""
    pass


def ask(question: questionary.Question) -> T:
    """包装 questionary 调用，统一处理用户取消"""
    result = question.ask()
    if result is None:
        raise UserCancelled()
    return result


# ============================================================================
# 界面组件
# ============================================================================

def print_banner():
    """打印欢迎横幅"""
    banner_text = Text()
    banner_text.append("╭───────────────────────────────────────────╮\n", style=theme.primary)
    banner_text.append("│", style=theme.primary)
    banner_text.append(f"  {theme.icon_app} ", style="")
    banner_text.append("DatasetMaster", style=f"bold {theme.primary}")
    banner_text.append(" v0.1.0", style=theme.muted)
    banner_text.append("                  │\n", style=theme.primary)
    banner_text.append("│", style=theme.primary)
    banner_text.append("     数据集划分与管理工具", style="")
    banner_text.append("                  │\n", style=theme.primary)
    banner_text.append("╰───────────────────────────────────────────╯", style=theme.primary)

    console.print()
    console.print(banner_text)
    console.print()


def print_dataset_info(info: DatasetInfo):
    """打印数据集信息"""
    format_info = FORMAT_INFO.get(info.format)
    format_name = format_info.name if format_info else "Unknown"

    table = Table(
        title=f"{theme.icon_chart} 数据集概览",
        show_header=True,
        header_style=theme.table_header,
        border_style=theme.table_border,
        title_style=f"bold {theme.primary}"
    )
    table.add_column("项目", style=theme.muted)
    table.add_column("数值", justify="right")

    table.add_row(f"{theme.icon_folder} 数据集格式", format_name)
    table.add_row(f"{theme.icon_image} 总图片数", str(info.total_images))
    table.add_row(f"{theme.icon_label} 总标签数", str(info.total_labels))
    table.add_row(f"{theme.icon_check} 匹配样本数", f"[{theme.success}]{info.matched_pairs}[/]")

    if info.missing_labels:
        table.add_row(f"{theme.icon_warn} 缺失标签", f"[{theme.warning}]{len(info.missing_labels)}[/]")
    else:
        table.add_row(f"  缺失标签", f"[{theme.muted}]0[/]")

    if info.orphan_labels:
        table.add_row(f"{theme.icon_warn} 孤立标签", f"[{theme.warning}]{len(info.orphan_labels)}[/]")
    else:
        table.add_row(f"  孤立标签", f"[{theme.muted}]0[/]")

    console.print(table)

    # 类别分布
    if info.class_distribution:
        console.print()
        class_table = Table(
            title=f"{theme.icon_label} 类别分布",
            show_header=True,
            header_style=theme.table_header,
            border_style=theme.table_border,
            title_style=f"bold {theme.secondary}"
        )
        class_table.add_column("索引", style=theme.muted, justify="center")
        class_table.add_column("类别名称")
        class_table.add_column("样本数", justify="right")

        for cls_idx in sorted(info.class_distribution.keys()):
            name = info.class_config.names[cls_idx] if info.class_config and cls_idx < len(info.class_config.names) else f"class_{cls_idx}"
            count = info.class_distribution[cls_idx]
            class_table.add_row(str(cls_idx), name, str(count))

        console.print(class_table)


def print_validation_result(result: ValidationResult):
    """打印验证结果"""
    console.print()

    if result.is_valid and not result.warnings:
        msg_success("数据集验证通过，一切正常！")
        return

    if result.warnings:
        for warning in result.warnings:
            msg_warning(warning)

    if result.errors:
        for error in result.errors:
            msg_error(error)

    if result.label_errors[:5]:
        console.print()
        msg(f"标签格式错误示例:", theme.error)
        for err in result.label_errors[:5]:
            console.print(f"   {err.file_path.name}:{err.line_number} - {err.message}", style=theme.muted)
        if len(result.label_errors) > 5:
            console.print(f"   ... 还有 {len(result.label_errors) - 5} 个错误", style=theme.muted)


def print_split_preview(result: SplitResult, class_config: Optional[ClassConfig] = None):
    """打印划分预览"""
    console.print()

    # 统计面板
    train_count = len(result.train_items)
    val_count = len(result.val_items)
    test_count = len(result.test_items)
    total = train_count + val_count + test_count

    def make_bar(count: int, max_width: int = 20) -> str:
        ratio = count / total if total > 0 else 0
        filled = int(ratio * max_width)
        return "█" * filled + "░" * (max_width - filled)

    preview_content = f"""
[bold]Train[/]  {make_bar(train_count)} [cyan]{train_count:>5}[/] 张 ({train_count/total*100:.1f}%)
[bold]Val[/]    {make_bar(val_count)} [cyan]{val_count:>5}[/] 张 ({val_count/total*100:.1f}%)
[bold]Test[/]   {make_bar(test_count)} [cyan]{test_count:>5}[/] 张 ({test_count/total*100:.1f}%)
"""

    console.print(Panel(
        preview_content,
        title=f"{theme.icon_chart} 划分预览",
        border_style=theme.primary,
        title_align="left"
    ))

    # 类别分布详情
    all_classes = set(result.train_class_dist.keys()) | set(result.val_class_dist.keys()) | set(result.test_class_dist.keys())

    if all_classes:
        table = Table(
            title=f"{theme.icon_sparkles} 类别分布详情",
            show_header=True,
            header_style=theme.table_header,
            border_style=theme.table_border,
            title_style=f"bold {theme.secondary}"
        )
        table.add_column("类别", style=theme.muted)
        table.add_column("Train", justify="right", style=theme.success)
        table.add_column("Val", justify="right", style=theme.primary)
        table.add_column("Test", justify="right", style=theme.accent)

        for cls_idx in sorted(all_classes):
            name = class_config.names[cls_idx] if class_config and cls_idx < len(class_config.names) else f"class_{cls_idx}"
            table.add_row(
                name,
                str(result.train_class_dist.get(cls_idx, 0)),
                str(result.val_class_dist.get(cls_idx, 0)),
                str(result.test_class_dist.get(cls_idx, 0))
            )

        console.print(table)


def print_final_report(result: SplitResult, format_name: str, output_dir: str, dry_run: bool = False):
    """打印最终报告"""
    console.print()

    if dry_run:
        title = f"{theme.icon_thinking} Dry-run 模式"
        border_style = theme.accent
        status_msg = "[dim]未执行实际操作，仅预览结果[/dim]"
    else:
        title = f"{theme.icon_party} 划分完成"
        border_style = theme.success
        status_msg = f"[{theme.success}]数据集已成功划分！[/]"

    report_content = f"""
{status_msg}

{theme.icon_folder} 格式: [bold]{format_name}[/bold]
{theme.icon_image} Train: [cyan]{len(result.train_items)}[/cyan] 张图片
{theme.icon_image} Val:   [cyan]{len(result.val_items)}[/cyan] 张图片
{theme.icon_image} Test:  [cyan]{len(result.test_items)}[/cyan] 张图片

{theme.icon_save} 输出目录: [underline]{output_dir}[/underline]
"""

    console.print(Panel(
        report_content,
        title=title,
        border_style=border_style,
        title_align="left"
    ))


def print_conversion_result(result: ConversionResult):
    """打印转换结果"""
    console.print()

    if result.success:
        title = f"{theme.icon_party} 转换完成"
        border_style = theme.success
        status_msg = f"[{theme.success}]数据集格式转换成功！[/]"
    else:
        title = f"{theme.icon_error} 转换失败"
        border_style = theme.error
        status_msg = f"[{theme.error}]转换过程中发生错误[/]"

    class_info = ""
    if result.class_config:
        class_info = f"\n{theme.icon_label} 类别数: [cyan]{result.class_config.nc}[/cyan]"

    report_content = f"""
{status_msg}

{theme.icon_image} 图片数: [cyan]{result.total_images}[/cyan]
{theme.icon_label} 标注数: [cyan]{result.total_annotations}[/cyan]{class_info}

{theme.icon_save} 输出目录: [underline]{result.output_path}[/underline]
"""

    console.print(Panel(
        report_content,
        title=title,
        border_style=border_style,
        title_align="left"
    ))

    # 显示警告
    if result.warnings:
        console.print()
        for warning in result.warnings:
            msg_warning(warning)

    # 显示错误
    if result.errors:
        console.print()
        for error in result.errors[:5]:
            msg_error(error)
        if len(result.errors) > 5:
            console.print(f"   ... 还有 {len(result.errors) - 5} 个错误", style=theme.muted)


# ============================================================================
# 主流程
# ============================================================================

def run_convert_workflow():
    """格式转换工作流"""
    # 1. 选择源格式
    source_format = ask(questionary.select(
        f"{theme.icon_folder} 选择源数据集格式:",
        choices=[
            questionary.Choice("COCO - COCO JSON 格式", value=DatasetFormat.COCO),
            questionary.Choice("Pascal VOC - Pascal VOC XML 格式", value=DatasetFormat.VOC),
        ],
        style=theme.questionary_style
    ))

    # 2. 选择目标格式
    target_choices = [
        questionary.Choice("YOLO - 标准 YOLO 检测格式", value=DatasetFormat.YOLO),
    ]
    # COCO 支持分割格式转换
    if source_format == DatasetFormat.COCO:
        target_choices.append(
            questionary.Choice("YOLO-Seg - YOLO 实例分割格式", value=DatasetFormat.YOLO_SEG)
        )

    target_format = ask(questionary.select(
        f"{theme.icon_folder} 选择目标格式:",
        choices=target_choices,
        style=theme.questionary_style
    ))

    # 3. 选择源数据集目录
    console.print()
    if source_format == DatasetFormat.COCO:
        msg_muted(f"{theme.icon_info} COCO 格式需要 images/ 和 annotations/ 目录")
    else:
        msg_muted(f"{theme.icon_info} VOC 格式需要 JPEGImages/ 和 Annotations/ 目录 (或 images/ 和 annotations/)")

    source_path = ask(questionary.path(
        f"{theme.icon_folder} 请选择源数据集目录:",
        only_directories=True,
        style=theme.questionary_style
    ))

    # 4. 选择输出目录
    default_output = str(Path(source_path).parent / f"{Path(source_path).name}_yolo")
    output_path = ask(questionary.path(
        f"{theme.icon_folder} 请选择输出目录:",
        default=default_output,
        style=theme.questionary_style
    ))

    # 5. 是否复制图片
    console.print()
    copy_images = ask(questionary.confirm(
        f"{theme.icon_image} 是否复制图片到输出目录?",
        default=True,
        style=theme.questionary_style
    ))

    # 6. 确认转换
    console.print()
    source_format_name = FORMAT_INFO[source_format].name
    target_format_name = FORMAT_INFO[target_format].name

    if not ask(questionary.confirm(
        f"{theme.icon_rocket} 确认将 {source_format_name} 转换为 {target_format_name}?",
        default=True,
        style=theme.questionary_style
    )):
        raise UserCancelled()

    # 7. 执行转换
    console.print()
    msg_step("正在转换格式...")

    try:
        converter = create_converter(
            source_format=source_format,
            target_format=target_format,
            source_path=source_path,
            output_path=output_path
        )

        with Progress(
            SpinnerColumn(style=theme.primary),
            TextColumn(f"[{theme.primary}]{{task.description}}[/]"),
            console=console
        ) as progress:
            task = progress.add_task(f"{theme.icon_rocket} 正在转换...", total=None)
            result = converter.convert(copy_images=copy_images)
            progress.update(task, description=f"{theme.icon_check} 转换完成!")

        # 8. 显示结果
        print_conversion_result(result)

    except Exception as e:
        msg_error(f"转换失败: {e}")
        return


def run_split_workflow():
    """数据集划分工作流"""
    # 1. 选择数据集格式
    format_choices = [
        questionary.Choice(f"{info.name} - {info.description}", value=fmt)
        for fmt, info in FORMAT_INFO.items()
    ]

    dataset_format = ask(questionary.select(
        f"{theme.icon_folder} 选择数据集格式:",
        choices=format_choices,
        style=theme.questionary_style
    ))

    format_info = FORMAT_INFO[dataset_format]

    # 2. 选择数据集目录
    console.print()
    if format_info.separate_dirs:
        msg_muted(f"{theme.icon_info} 该格式需要分离的 images/ 和 labels/ 目录")
    else:
        msg_muted(f"{theme.icon_info} 该格式的图片和标签在同一目录")

    dataset_path = ask(questionary.path(
        f"{theme.icon_folder} 请选择数据集目录:",
        only_directories=True,
        style=theme.questionary_style
    ))

    # 读取数据集
    console.print()
    msg_step("正在读取数据集...")

    try:
        reader = create_reader(dataset_path, dataset_format)
    except Exception as e:
        msg_error(f"创建读取器失败: {e}")
        return

    # 3. 询问是否导入类别配置
    use_class_config = ask(questionary.confirm(
        f"{theme.icon_gear} 是否导入类别配置文件 (classes.yaml)?",
        default=False,
        style=theme.questionary_style
    ))

    class_config_path = None
    if use_class_config:
        class_config_path = ask(questionary.path(
            f"{theme.icon_folder} 请选择类别配置文件:",
            default=str(Path(dataset_path) / "classes.yaml"),
            style=theme.questionary_style
        ))

    try:
        dataset_info = reader.read(class_config_path)
    except Exception as e:
        msg_error(f"读取数据集失败: {e}")
        return

    # 显示数据集信息
    console.print()
    print_dataset_info(dataset_info)

    if not use_class_config and dataset_info.class_config:
        console.print()
        msg_warning("未导入类别配置，类别名称将使用默认格式 (class_0, class_1...)")

    # 4. 验证数据集
    console.print()
    check_images = ask(questionary.confirm(
        f"{theme.icon_image} 是否检查损坏的图片？(可能较慢)",
        default=False,
        style=theme.questionary_style
    ))

    msg_step("正在验证数据集...")
    validator = DatasetValidator(dataset_info)
    validation_result = validator.validate(check_corrupted_images=check_images)
    print_validation_result(validation_result)

    if not validation_result.is_valid:
        console.print()
        if not ask(questionary.confirm(
            f"{theme.icon_thinking} 存在错误，是否仍要继续?",
            default=False,
            style=theme.questionary_style
        )):
            raise UserCancelled()

    # 5. 设置划分比例
    console.print()
    ratio_choice = ask(questionary.select(
        f"{theme.icon_chart} 选择划分比例:",
        choices=[
            questionary.Choice("8:1:1 (推荐)", value="8:1:1"),
            questionary.Choice("7:2:1", value="7:2:1"),
            questionary.Choice("6:2:2", value="6:2:2"),
            questionary.Choice("自定义...", value="custom")
        ],
        style=theme.questionary_style
    ))

    if ratio_choice == "8:1:1":
        train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
    elif ratio_choice == "7:2:1":
        train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
    elif ratio_choice == "6:2:2":
        train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
    else:
        train_ratio = float(ask(questionary.text("  Train 比例 (0-1):", default="0.8", style=theme.questionary_style)))
        val_ratio = float(ask(questionary.text("  Val 比例 (0-1):", default="0.1", style=theme.questionary_style)))
        test_ratio = float(ask(questionary.text("  Test 比例 (0-1):", default="0.1", style=theme.questionary_style)))

    # 6. 分层抽样
    use_stratify = ask(questionary.confirm(
        f"{theme.icon_sparkles} 是否启用分层抽样？(确保各类别比例一致)",
        default=True,
        style=theme.questionary_style
    ))

    # 7. 随机种子
    use_seed = ask(questionary.confirm(
        f"{theme.icon_gear} 是否设置随机种子？(确保结果可复现)",
        default=True,
        style=theme.questionary_style
    ))

    seed = None
    if use_seed:
        seed_str = ask(questionary.text("  随机种子:", default="42", style=theme.questionary_style))
        seed = int(seed_str)

    # 8. 操作方式
    console.print()
    copy_files = ask(questionary.select(
        f"{theme.icon_save} 选择操作方式:",
        choices=[
            questionary.Choice(f"复制文件 (推荐，保留原始数据)", value=True),
            questionary.Choice(f"移动文件 (节省空间)", value=False)
        ],
        style=theme.questionary_style
    ))

    # 9. 输出目录
    output_dir = ask(questionary.path(
        f"{theme.icon_folder} 输出目录:",
        default=str(Path(dataset_path).parent / "dataset_split"),
        only_directories=True,
        style=theme.questionary_style
    ))

    # 创建划分配置
    split_config = SplitConfig(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratify=use_stratify,
        copy_files=copy_files
    )

    # 10. 预览划分结果
    console.print()
    msg_step("正在计算划分结果...")
    splitter = DatasetSplitter(dataset_info, split_config)
    preview_result = splitter.preview()
    print_split_preview(preview_result, dataset_info.class_config)

    # 11. Dry-run 或执行
    console.print()
    action = ask(questionary.select(
        f"{theme.icon_rocket} 选择操作:",
        choices=[
            questionary.Choice(f"执行划分", value="execute"),
            questionary.Choice(f"仅预览 (dry-run)", value="dry_run"),
            questionary.Choice(f"取消", value="cancel")
        ],
        style=theme.questionary_style
    ))

    if action == "cancel":
        raise UserCancelled()

    dry_run = action == "dry_run"

    if not dry_run and not copy_files:
        # 移动操作二次确认
        console.print()
        if not ask(questionary.confirm(
            f"{theme.icon_warn} 警告: 移动操作将改变原始数据位置，确认继续?",
            default=False,
            style=theme.questionary_style
        )):
            raise UserCancelled()

    # 12. 执行划分
    console.print()
    with Progress(
        SpinnerColumn(style=theme.primary),
        TextColumn(f"[{theme.primary}]{{task.description}}[/]"),
        console=console
    ) as progress:
        task = progress.add_task(f"{theme.icon_rocket} 正在处理...", total=None)
        result = splitter.execute(output_dir, dry_run=dry_run)
        progress.update(task, description=f"{theme.icon_check} 处理完成!")

    # 13. 生成 data.yaml
    if not dry_run and dataset_info.class_config:
        console.print()
        generate_yaml = ask(questionary.confirm(
            f"{theme.icon_save} 是否生成 YOLO data.yaml?",
            default=True,
            style=theme.questionary_style
        ))

        if generate_yaml:
            generator = YAMLConfigGenerator(
                output_dir=output_dir,
                class_config=dataset_info.class_config,
                format=dataset_format,
                has_train=len(result.train_items) > 0,
                has_val=len(result.val_items) > 0,
                has_test=len(result.test_items) > 0
            )
            yaml_path = generator.generate()
            msg_success(f"已生成配置文件: {yaml_path}")

    # 最终报告
    format_name = FORMAT_INFO[dataset_format].name
    print_final_report(result, format_name, output_dir, dry_run)


def run_interactive():
    """运行交互式 CLI 主菜单"""
    print_banner()

    # 主功能选择
    action = ask(questionary.select(
        f"{theme.icon_rocket} 请选择功能:",
        choices=[
            questionary.Choice(f"数据集划分 - 将数据集划分为 train/val/test", value="split"),
            questionary.Choice(f"格式转换 - 将 COCO/VOC 转换为 YOLO 格式", value="convert"),
            questionary.Choice(f"退出", value="exit")
        ],
        style=theme.questionary_style
    ))

    if action == "split":
        run_split_workflow()
    elif action == "convert":
        run_convert_workflow()
    elif action == "exit":
        raise UserCancelled()


def main():
    """主入口"""
    try:
        run_interactive()
    except (KeyboardInterrupt, UserCancelled):
        console.print()
        msg(f"已取消操作，下次再见！", theme.accent, theme.icon_bye)
        sys.exit(0)
    except Exception as e:
        console.print()
        msg_error(f"发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
