#!/usr/bin/env bash
# GOAI 2026 快速部署 — 一键构建并启动 flashrt-robodojo-policy 策略服务器 (Docker)
#
# 用法:
#   bash docker/deploy_docker.sh \
#       --model-zip /path/to/RoboDojo-...-30000.zip \
#       [--model-dir /models/model] [--port 3101] [--no-build]
#
# 流程:
#   1. 构建镜像 (首次; 已有镜像加 --no-build 跳过)
#   2. 解压模型 zip 到挂载目录
#   3. 以 --gpus all 启动 WS 策略服务器 (端口 3101)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ZIP=""
MODEL_DIR="/models/model"
PORT="3101"
NO_BUILD="0"
IMAGE="flashrt-robodojo:latest"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-zip)  MODEL_ZIP="$2"; shift 2 ;;
    --model-dir)  MODEL_DIR="$2"; shift 2 ;;
    --port)       PORT="$2"; shift 2 ;;
    --no-build)   NO_BUILD="1"; shift ;;
    -h|--help)
      echo "Usage: $0 --model-zip <zip> [--model-dir DIR] [--port N] [--no-build]"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

[ -n "${MODEL_ZIP}" ] || { echo "错误: 需要 --model-zip <模型zip路径>"; exit 1; }
[ -f "${MODEL_ZIP}" ] || { echo "错误: 模型zip不存在: ${MODEL_ZIP}"; exit 1; }

# ── 1. 构建镜像 ──
if [ "${NO_BUILD}" = "0" ]; then
  echo "== [1/3] 构建镜像 ${IMAGE} =="
  docker build -t "${IMAGE}" "${SCRIPT_DIR}"
else
  echo "== [1/3] 跳过构建 (--no-build) =="
fi

# ── 2. 解压模型 ──
echo "== [2/3] 解压模型到 ${MODEL_DIR} =="
mkdir -p "$(dirname "${MODEL_DIR}")"
if [ ! -d "${MODEL_DIR}/params" ]; then
  # zip 内是 <name>/ 目录, 提取第一个子目录
  TMP=$(mktemp -d)
  unzip -q "${MODEL_ZIP}" -d "${TMP}"
  INNER=$(find "${TMP}" -maxdepth 1 -mindepth 1 -type d | head -1)
  if [ -z "${INNER}" ]; then echo "错误: zip 内无模型目录"; exit 1; fi
  mv "${INNER}" "${MODEL_DIR}"
  rm -rf "${TMP}"
  echo "  模型已解压: ${MODEL_DIR}"
else
  echo "  模型已存在, 跳过解压"
fi

# ── 3. 启动容器 ──
echo "== [3/3] 启动策略服务器 (端口 ${PORT}) =="
docker rm -f flashrt-robodojo-server >/dev/null 2>&1 || true
docker run -d --name flashrt-robodojo-server \
  --gpus all --shm-size=8g \
  -p "${PORT}:${PORT}" \
  -v "${MODEL_DIR}:/models/model" \
  "${IMAGE}" \
  --checkpoint /models/model --port "${PORT}"

echo ""
echo "==== 部署完成 ===="
echo "  策略服务器: ws://<本机IP>:${PORT}"
echo "  日志:       docker logs -f flashrt-robodojo-server"
echo "  评测端连接: bash scripts/robodojo.sh client --policy-host <本机IP> --policy-port ${PORT} ..."
