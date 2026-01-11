#!/bin/bash
# DatasetMaster 一键运行脚本
# 用法: wget -qO- https://raw.githubusercontent.com/Qi-007/DatasetMaster/main/dataset_master.sh | bash

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}"
echo "╭───────────────────────────────────────────╮"
echo "│  📦 DatasetMaster 一键运行脚本           │"
echo "│     数据集划分与格式转换工具             │"
echo "╰───────────────────────────────────────────╯"
echo -e "${NC}"

# 检查 Python
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}❌ 未找到 Python，请先安装 Python 3.10+${NC}"
        exit 1
    fi

    # 检查版本
    PY_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo -e "${GREEN}✓ 检测到 Python $PY_VERSION${NC}"
}

# 检查并安装依赖
install_deps() {
    echo -e "${YELLOW}📥 检查依赖...${NC}"

    DEPS=("rich" "questionary" "pyyaml" "pillow")
    MISSING=()

    for dep in "${DEPS[@]}"; do
        if ! $PYTHON_CMD -c "import $dep" 2>/dev/null; then
            MISSING+=("$dep")
        fi
    done

    if [ ${#MISSING[@]} -gt 0 ]; then
        echo -e "${YELLOW}📦 安装缺失依赖: ${MISSING[*]}${NC}"
        $PYTHON_CMD -m pip install -q rich questionary pyyaml pillow
    fi

    echo -e "${GREEN}✓ 依赖检查完成${NC}"
}

# 克隆或更新仓库
setup_repo() {
    INSTALL_DIR="${HOME}/.local/share/DatasetMaster"

    if [ -d "$INSTALL_DIR" ]; then
        echo -e "${YELLOW}📁 更新 DatasetMaster...${NC}"
        cd "$INSTALL_DIR"
        git pull -q origin main 2>/dev/null || true
    else
        echo -e "${YELLOW}📥 下载 DatasetMaster...${NC}"
        mkdir -p "$(dirname "$INSTALL_DIR")"
        git clone -q https://github.com/Qi-007/DatasetMaster.git "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi

    echo -e "${GREEN}✓ 仓库准备完成${NC}"
}

# 运行程序
run_app() {
    echo ""
    echo -e "${CYAN}🚀 启动 DatasetMaster...${NC}"
    echo ""
    $PYTHON_CMD main.py
}

# 本地运行模式（当前目录已有代码）
run_local() {
    if [ -f "main.py" ] && [ -d "dataset_master" ]; then
        check_python
        install_deps
        echo ""
        echo -e "${CYAN}🚀 启动 DatasetMaster...${NC}"
        echo ""
        $PYTHON_CMD main.py
    else
        echo -e "${RED}❌ 当前目录未找到 DatasetMaster 代码${NC}"
        echo -e "${YELLOW}💡 请在项目根目录运行此脚本，或使用在线安装方式${NC}"
        exit 1
    fi
}

# 主流程
main() {
    # 如果当前目录有 main.py，直接本地运行
    if [ -f "main.py" ] && [ -d "dataset_master" ]; then
        run_local
    else
        # 否则下载并运行
        check_python
        install_deps
        setup_repo
        run_app
    fi
}

# 清理函数（用于一次性运行后删除）
cleanup() {
    if [ "$1" = "--cleanup" ]; then
        rm -f "$0"
    fi
}

# 执行
main "$@"
