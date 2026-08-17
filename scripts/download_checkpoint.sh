#!/usr/bin/env bash
# GOAI FlashRT — 模型权重下载脚本
#
# 用法:
#   bash scripts/download_checkpoint.sh <model> <dest_dir> [--format orbax|safetensors]
#
# 参数:
#   model      模型标识（见下方 MODEL 映射表）
#   dest_dir   目标目录（会下载到 ${dest_dir}/<model>/）
#   --format   权重格式: orbax (jax, 默认) / safetensors (torch)
#
# 示例:
#   bash scripts/download_checkpoint.sh pi05-arx-x5 /data/ckpts
#   bash scripts/download_checkpoint.sh pi05-arx-x5 /data/ckpts --format safetensors
#
# 结果目录:
#   /data/ckpts/pi05-arx-x5/          <- orbax: {params, assets}
#   /data/ckpts/pi05-arx-x5/          <- safetensors: model.safetensors
set -euo pipefail

# 模型来源映射（HuggingFace）
# <name>|<repo>|<remote_path>
MODELS=(
  "pi05-arx-x5|RoboDojo-Benchmark/RoboDojo|ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999"
)

MODEL="${1:?usage: download_checkpoint.sh <model> <dest_dir> [--format orbax|safetensors]}"
DEST_DIR="${2:?usage: download_checkpoint.sh <model> <dest_dir> [--format orbax|safetensors]}"
FORMAT="orbax"
shift 2
while [[ $# -gt 0 ]]; do
  case "$1" in
    --format) FORMAT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# 查找模型源
SRC=""
for m in "${MODELS[@]}"; do
  name="${m%%|*}"; rest="${m#*|}"
  if [ "${name}" == "${MODEL}" ]; then
    REPO="${rest%%|*}"; REMOTE="${rest#*|}"
    SRC="1"; break
  fi
done
if [ -z "${SRC}" ]; then
  echo "[ERROR] unknown model: ${MODEL}. Available:" >&2
  for m in "${MODELS[@]}"; do echo "  ${m%%|*}" >&2; done
  exit 1
fi

DEST="${DEST_DIR}/${MODEL}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

echo "=== 下载 pi0.5 模型 ==="
echo "  model:  ${MODEL}"
echo "  repo:   ${REPO}   path: ${REMOTE}"
echo "  target: ${DEST}"
echo "  format: ${FORMAT}   endpoint: ${HF_ENDPOINT}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[需要] huggingface_hub"
  python3 -m pip install "huggingface_hub[hf_transfer]"
fi

mkdir -p "${DEST}"
huggingface-cli download \
  --repo-type dataset \
  --endpoint "${HF_ENDPOINT}" \
  "${REPO}" "${REMOTE}" \
  --local-dir "${DEST}"

echo "=== 完成 ==="
echo "  checkpoint: ${DEST}"
du -sh "${DEST}"
