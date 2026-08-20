#!/usr/bin/env bash
# GOAI 2026 快速部署 — 一键构建并启动 flashrt-robodojo-policy 策略服务器 (Docker)
#
# 用法:
#   bash docker/deploy_docker.sh \
#       --model-dir /models/model \
#       [--port 3101] [--no-build]
#
# 模型目录可用魔塔官方命令下载:
#   modelscope download --model cpadyun/RoboDojo-goai2026-arx_x5-joint-0-pi05-flashrt-30000 \
#       --local_dir /models/model
#
# 流程:
#   1. 构建镜像 (首次; 已有镜像加 --no-build 跳过)
#   2. 以 --gpus all 挂载模型目录并启动 WS 策略服务器 (端口 3101)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/models/model}"
PORT="${PORT:-3101}"
NO_BUILD="${NO_BUILD:-0}"
IMAGE="flashrt-robodojo:latest"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)  MODEL_DIR="$2"; shift 2 ;;
    --port)       PORT="$2"; shift 2 ;;
    --no-build)   NO_BUILD="1"; shift ;;
    -h|--help)
      echo "Usage: $0 --model-dir <dir> [--port N] [--no-build]"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

[ -n "${MODEL_DIR}" ] || { echo "错误: 需要 --model-dir <模型目录>"; exit 1; }
[ -d "${MODEL_DIR}/params" ] || { echo "错误: 目录无效(缺少 params/): ${MODEL_DIR}"; exit 1; }

# ── 1. 构建镜像 ──
if [ "${NO_BUILD}" = "0" ]; then
  echo "== [1/3] 构建镜像 ${IMAGE} =="
  docker build -t "${IMAGE}" "${SCRIPT_DIR}"
else
  echo "== [1/3] 跳过构建 (--no-build) =="
fi

# ── 2. 校验模型目录 ──
echo "== [2/3] 校验模型目录 ${MODEL_DIR} =="
for f in config.json norm_stats.json params; do
  [ -e "${MODEL_DIR}/${f}" ] || { echo "错误: 缺少 ${MODEL_DIR}/${f}"; exit 1; }
done
echo "  模型目录完整"

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
