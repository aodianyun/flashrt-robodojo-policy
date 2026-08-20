#!/usr/bin/env bash
# 上传参赛模型目录到魔塔社区 (ModelScope SDK)
#
# 用法:
#   export MODELSCOPE_TOKEN=ms-xxxx
#   bash scripts/upload_model_modelscope.sh \
#       --model-id <user>/<model> \
#       --src /path/to/model_dir
set -euo pipefail

MODEL_ID=""
SRC=""
TOKEN="${MODELSCOPE_TOKEN:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --src)      SRC="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --model-id <user>/<model> --src <model_dir>"
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

[ -n "${MODEL_ID}" ] || { echo "错误: 需要 --model-id"; exit 1; }
[ -n "${SRC}" ] && [ -d "${SRC}" ] || { echo "错误: 目录不存在: ${SRC}"; exit 1; }
[ -n "${TOKEN}" ] || { echo "错误: 请设置 MODELSCOPE_TOKEN"; exit 1; }

# 解释器: 优先 PYTHON_BIN, 否则 PATH 中的 python3 (需已安装 modelscope)
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "== 上传目录 ${SRC} -> ${MODEL_ID} =="
MODELSCOPE_API_TOKEN="${TOKEN}" "${PYTHON_BIN}" - <<PY
import os
from modelscope.hub.api import HubApi

token = "${TOKEN}"
model_id = "${MODEL_ID}"
src = "${SRC}"

api = HubApi(token=token)
print("认证: OK")

# 遍历 src 下所有文件, 保持相对路径上传
uploaded = 0
for root, dirs, files in os.walk(src):
    for fname in files:
        local = os.path.join(root, fname)
        rel = os.path.relpath(local, src)
        print(f"上传 {rel} ...", flush=True)
        api.upload_file(
            repo_id=model_id,
            path_or_fileobj=local,
            path_in_repo=rel,
        )
        uploaded += 1

print(f"完成: 上传 {uploaded} 个文件")
PY

echo ""
echo "==== 完成 ===="
echo "  模型页: https://www.modelscope.cn/models/${MODEL_ID}"
