#!/usr/bin/env bash
# GOAI FlashRT — 独立 torch venv 创建脚本（torch 前端推理）
#
# 用法: bash vendor/venv-torch/install_torch_venv.sh [venv_dir]
# 默认 venv: <project>/vendor/venv-torch/.venv
#
# 说明:
#   - aarch64 (Thor) 的 torch CUDA wheel 官方不完整 (2.5.1 _C.so 是 stub),
#     因此复用系统 NVIDIA NGC torch (/usr/local/lib/python3.12/dist-packages/torch)。
#   - 用 PURE venv (不加 --system-site-packages) 隔离系统 jax (numpy 2.x),
#     只把 NGC torch 及其依赖符号链接进 venv, 并装 numpy 1.x。
#   - x86 机器可直接 pip install torch (完整 wheel), 无需此链接逻辑。
#
# 依赖: 系统已安装 NVIDIA NGC torch (nv25.08+), python3.12
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${1:-${SCRIPT_DIR}/.venv}"
SYS_DIST="/usr/local/lib/python3.12/dist-packages"

echo "==== Torch venv 安装 (aarch64 + NGC torch) ===="
echo "  venv: ${VENV_DIR}"

# 1. 纯 venv (不继承系统 site-packages, 隔离 jax/numpy2)
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip

# 2. 链接系统 NGC torch 及依赖到 venv (排除 jax / numpy 2)
VENV_SITE="${VENV_DIR}/lib/python3.12/site-packages"
if [ -d "${SYS_DIST}/torch" ]; then
  echo "== 链接系统 NGC torch =="
  for p in torch functorch torchgen torchprofile torchvision torchvision.libs \
           torchao torch_tensorrt torch_c_dlpack_ext \
           filelock fsspec jinja2 networkx sympy mpmath MarkupSafe \
           typing_extensions pyparsing; do
    ln -sfn "${SYS_DIST}/${p}" "${VENV_SITE}/${p}" 2>/dev/null || true
  done
  # dist-info 链接
  for f in "${SYS_DIST}"/torch-*.dist-info "${SYS_DIST}"/torchgen-*.dist-info \
           "${SYS_DIST}"/filelock-*.dist-info "${SYS_DIST}"/fsspec-*.dist-info \
           "${SYS_DIST}"/jinja2-*.dist-info "${SYS_DIST}"/networkx-*.dist-info \
           "${SYS_DIST}"/sympy-*.dist-info "${SYS_DIST}"/mpmath-*.dist-info \
           "${SYS_DIST}"/MarkupSafe-*.dist-info "${SYS_DIST}"/typing_extensions-*.dist-info; do
    [ -e "$f" ] && ln -sfn "$f" "${VENV_SITE}/$(basename "$f")" 2>/dev/null || true
  done
else
  echo "[WARN] 系统 NGC torch 未找到, 尝试 pip 安装 torch (x86 或官方 wheel)"
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
fi

# 3. numpy 1.x + 运行时依赖 (纯 venv 内, 覆盖)
echo "== 安装 numpy 1.x + 运行时依赖 =="
pip install -i https://mirrors.aliyun.com/pypi/simple/ \
  "numpy==1.26.4" ml_dtypes safetensors pillow pyyaml msgpack msgpack-numpy \
  websockets sentencepiece --timeout 120

# 4. 验证
echo "== 验证 =="
python - <<'PY'
import warnings; warnings.filterwarnings('ignore')
import numpy as np, torch
print("numpy:", np.__version__)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(2,3).cuda()
    print("gpu OK:", tuple(x.shape))
# 隔离验证: jax 应不可见
try:
    import jax
    print("WARN: jax 可见 (隔离失败)")
except ImportError:
    print("jax isolated: OK (not visible)")
PY

echo "==== 完成 ===="
echo "使用: source ${VENV_DIR}/bin/activate"
echo "权重需为 LeRobot safetensors (model.safetensors + preprocessor/postprocessor)"
