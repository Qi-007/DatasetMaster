#!/bin/bashm 
# DatasetMaster 一键运行脚本

set -euo pipefail

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

# -------- 参数解析 --------
DO_CLEANUP=0
APP_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--cleanup" ]]; then
    DO_CLEANUP=1
  else
    APP_ARGS+=("$arg")  # 透传给 main.py
  fi
done

# -------- 工具检查 --------
need_cmd() {
  local cmd="$1"
  local hint="${2:-}"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo -e "${RED}❌ 未找到命令: ${cmd}${NC}"
    [[ -n "$hint" ]] && echo -e "${YELLOW}💡 参考: ${hint}${NC}"
    exit 1
  fi
}

# 检查 Python
check_python() {
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
  else
    echo -e "${RED}❌ 未找到 Python，请先安装 Python 3.10+${NC}"
    exit 1
  fi

  # 检查版本（至少 3.10）
  local ver
  ver=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  local major minor
  major=${ver%%.*}
  minor=${ver##*.}
  if [[ "$major" -lt 3 || ( "$major" -eq 3 && "$minor" -lt 10 ) ]]; then
    echo -e "${RED}❌ Python 版本过低: $ver（需要 3.10+）${NC}"
    exit 1
  fi
  echo -e "${GREEN}✓ 检测到 Python $ver${NC}"
}

# 检查 pip
check_pip() {
  if ! $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
    echo -e "${RED}❌ 未检测到 pip（$PYTHON_CMD -m pip 不可用）${NC}"
    echo -e "${YELLOW}💡 Ubuntu/Debian 可尝试：sudo apt install python3-pip${NC}"
    echo -e "${YELLOW}💡 Arch 可尝试：sudo pacman -S python-pip${NC}"
    exit 1
  fi
}

# 检查并安装依赖
install_deps() {
  echo -e "${YELLOW}📥 检查依赖...${NC}"

  # pip 包名 vs import 名（pyyaml 的 import 名是 yaml；pillow 的 import 名是 PIL）
  local PIP_DEPS=("rich" "questionary" "pyyaml" "pillow")
  local IMPORT_DEPS=("rich" "questionary" "yaml" "PIL")

  local missing=()

  for i in "${!PIP_DEPS[@]}"; do
    local pkg="${PIP_DEPS[$i]}"
    local mod="${IMPORT_DEPS[$i]}"
    if ! $PYTHON_CMD -c "import ${mod}" >/dev/null 2>&1; then
      missing+=("$pkg")
    fi
  done

  if [[ "${#missing[@]}" -gt 0 ]]; then
    echo -e "${YELLOW}📦 安装缺失依赖: ${missing[*]}${NC}"

    # 优先不吵：失败时给可读提示
    if ! $PYTHON_CMD -m pip install -q --upgrade "${missing[@]}"; then
      echo -e "${RED}❌ pip 安装依赖失败${NC}"
      echo -e "${YELLOW}可能原因：网络/代理问题、权限AUR/系统包冲突、权限不足。${NC}"
      echo -e "${YELLOW}你可以尝试：${NC}"
      echo -e "  1) 使用代理后重试（例如设置 http_proxy/https_proxy）"
      echo -e "  2) 使用用户安装：${CYAN}$PYTHON_CMD -m pip install --user ${missing[*]}${NC}"
      echo -e "  3) 或在 venv 里安装"
      exit 1
    fi
  fi

  echo -e "${GREEN}✓ 依赖检查完成${NC}"
}

# 克隆或更新仓库
setup_repo() {
  need_cmd git "请先安装 git：Ubuntu/Debian: sudo apt install git | Arch: sudo pacman -S git"

  INSTALL_DIR="${HOME}/.local/share/DatasetMaster"

  if [[ -d "$INSTALL_DIR/.git" ]]; then
    echo -e "${YELLOW}📁 更新 DatasetMaster...${NC}"
    cd "$INSTALL_DIR"
    # 拉取失败不要中断（比如离线），但后面运行可能会报错
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
  $PYTHON_CMD main.py "${APP_ARGS[@]:-}"
}

# 本地运行模式（当前目录已有代码）
run_local() {
  if [[ -f "main.py" && -d "dataset_master" ]]; then
    check_python
    check_pip
    install_deps
    run_app
  else
    echo -e "${RED}❌ 当前目录未找到 DatasetMaster 代码${NC}"
    echo -e "${YELLOW}💡 请在项目根目录运行此脚本，或使用在线安装方式${NC}"
    exit 1
  fi
}

# 主流程
main() {
  # 如果当前目录有 main.py，直接本地运行
  if [[ -f "main.py" && -d "dataset_master" ]]; then
    run_local
  else
    # 否则下载并运行
    check_python
    check_pip
    install_deps
    setup_repo
    run_app
  fi
}

# 执行主逻辑
main

# 清理：仅当 --cleanup 且脚本本身是一个真实文件时删除
# 注意：管道执行（wget -qO- ... | bash）时 $0 通常不是脚本文件，不能删
if [[ "$DO_CLEANUP" == "1" ]]; then
  if [[ -f "$0" ]]; then
    rm -f "$0"
  fi
fi
