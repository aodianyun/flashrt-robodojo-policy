#!/usr/bin/env bash
# GOAI FlashRT — 权重路径解析（自动查找，缺失时可选下载）
#
# 用法:
#   bash scripts/resolve_checkpoint.sh <model> <ckpt_dir> [--download] [--format orbax|safetensors]
#
# 逻辑:
#   1. 查找 <ckpt_dir>/<model>/ 下权重
#       - orbax: 存在 params/ 或 assets/
#       - safetensors: 存在 model.safetensors
#   2. 缺失时: 若 --download 则调用 download_checkpoint.sh，否则报错退出
#
# 输出: 解析后的 checkpoint 绝对路径（print 到 stdout）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${1:?usage: resolve_checkpoint.sh <model> <ckpt_dir> [--download] [--format]}"
CKPT_DIR="${2:?usage: resolve_checkpoint.sh <model> <ckpt_dir> [--download] [--format]}"
DOWNLOAD="0"
FORMAT="orbax"
shift 2
while [[ $# -gt 0 ]]; do
  case "$1" in
    --download) DOWNLOAD="1"; shift ;;
    --format)   FORMAT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

CKPT="${CKPT_DIR}/${MODEL}"

# 已存在？
FOUND="0"
if [ -d "${CKPT}/params" ] || [ -d "${CKPT}/assets" ]; then
  FOUND="1"
elif [ -f "${CKPT}/model.safetensors" ]; then
  FOUND="1"
fi

if [ "${FOUND}" == "1" ]; then
  echo "${CKPT}"
  exit 0
fi

# 缺失 → 下载或报错
if [ "${DOWNLOAD}" == "1" ]; then
  echo "[resolve] 权重缺失，下载 ${MODEL} -> ${CKPT_DIR}" >&2
  bash "${SCRIPT_DIR}/download_checkpoint.sh" "${MODEL}" "${CKPT_DIR}" --format "${FORMAT}"
  echo "${CKPT}"
else
  echo "[ERROR] checkpoint not found: ${CKPT}" >&2
  echo "  运行: bash scripts/resolve_checkpoint.sh ${MODEL} ${CKPT_DIR} --download" >&2
  exit 1
fi
