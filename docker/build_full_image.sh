#!/usr/bin/env bash
# GOAI 2026 完整镜像构建 — 在物理机执行
#
# 生成一个"代码 + 预编译内核 + 模型"完全自足的镜像, 赛事方 docker run 即用。
#
# 前置:
#   - 本机已编译 FlashRT 内核 .so (vendor/FlashRT/flash_rt/*.so)
#   - 参赛模型目录 (默认 /app/models/30000, 含 params/config/norm_stats)
#   - 物理机可访问 /app (挂载) 且有 docker + nvidia-container-toolkit
#
# 用法:
#   bash docker/build_full_image.sh [--model-dir DIR] [--image NAME] [--tag TAG]
#
# 产物: flashrt-robodojo:full (可 docker save / push 到公共仓库)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/app/models/30000}"
IMAGE="${IMAGE:-flashrt-robodojo}"
TAG="${TAG:-full}"
ARCH_TAG="cpython-$(python3 -c 'import sys; print(sys.version_info.minor)')-$(uname -m)-linux-gnu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --image)     IMAGE="$2"; shift 2 ;;
    --tag)       TAG="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--model-dir DIR] [--image NAME] [--tag TAG]"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

echo "==== GOAI 2026 完整镜像构建 ===="
echo "  model-dir: ${MODEL_DIR}"
echo "  image:     ${IMAGE}:${TAG}"
echo "  arch:      ${ARCH_TAG}"

# ── 1. 校验模型 ──
[ -d "${MODEL_DIR}/params" ] || { echo "错误: 模型目录缺少 params/"; exit 1; }
[ -f "${MODEL_DIR}/norm_stats.json" ] || { echo "错误: 缺少 norm_stats.json"; exit 1; }
[ -f "${MODEL_DIR}/config.json" ] || { echo "错误: 缺少 config.json"; exit 1; }

# ── 2. 校验预编译内核 ──
KERNELS_OK=0
for so in "${PROJECT_ROOT}"/vendor/FlashRT/flash_rt/flash_rt_kernels*"${ARCH_TAG}".so \
          "${PROJECT_ROOT}"/vendor/FlashRT/flash_rt/flash_rt_fa2*"${ARCH_TAG}".so; do
  [ -f "$so" ] && [ -s "$so" ] && KERNELS_OK=$((KERNELS_OK+1))
done
if [ "${KERNELS_OK}" -lt 2 ]; then
  echo "错误: 缺少匹配架构的预编译内核 (需 flash_rt_kernels + flash_rt_fa2, ${ARCH_TAG})"
  echo "  请先按 vendor/FlashRT/README.md 编译, 或确认本机架构与内核一致"
  exit 1
fi
echo "  预编译内核 OK (${KERNELS_OK} 个)"

# ── 3. 准备构建上下文 (临时目录, 保留 .so + 模型) ──
echo "== 准备构建上下文 =="
CTX=$(mktemp -d)
trap 'rm -rf "${CTX}"' EXIT

# 代码: 用 git archive 取干净源码 (无 .so, 无 .git)
(cd "${PROJECT_ROOT}" && git archive HEAD | tar -x -C "${CTX}")

# 把预编译 .so 放回构建上下文的 flash_rt/ 目录 (Dockerfile COPY . . 需要)
mkdir -p "${CTX}/vendor/FlashRT/flash_rt"
cp "${PROJECT_ROOT}"/vendor/FlashRT/flash_rt/*.so "${CTX}/vendor/FlashRT/flash_rt/" 2>/dev/null || true

# 模型: 拷贝进构建上下文
mkdir -p "${CTX}/models/model"
cp -r "${MODEL_DIR}"/params "${MODEL_DIR}"/norm_stats.json "${MODEL_DIR}"/config.json \
      "${MODEL_DIR}"/assets "${MODEL_DIR}"/README.md "${CTX}/models/model/" 2>/dev/null || true
ls "${CTX}/models/model/" >/dev/null

# ── 4. 构建镜像 ──
echo "== docker build ${IMAGE}:${TAG} =="
docker build -f "${CTX}/docker/Dockerfile.full" -t "${IMAGE}:${TAG}" "${CTX}"

echo ""
echo "==== 完成 ===="
echo "  镜像: ${IMAGE}:${TAG}"
echo "  导出: docker save ${IMAGE}:${TAG} | gzip > flashrt-robodojo-full.tar.gz"
echo "  推送: docker tag ${IMAGE}:${TAG} <registry>/flashrt-robodojo:${TAG} && docker push <registry>/flashrt-robodojo:${TAG}"
echo "  运行: docker run --gpus all --shm-size=8g -p 3101:3101 ${IMAGE}:${TAG}"
