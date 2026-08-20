#!/usr/bin/env bash
# GOAI 2026 参赛模型打包脚本
#
# 将 RoboDojo ARX-X5 双臂 Pi0.5 模型（Orbax 格式，30000 步）打包为可直接
# 被 flashrt-robodojo-policy 推理的 zip：
#
#   <model>.zip/
#   ├── config.json          # 模型配置（action_dim/horizon/paligemma 等）
#   ├── norm_stats.json      # 动作/状态归一化统计（openpi schema）
#   ├── assets/
#   │   └── arx_x5_sim/
#   │       └── norm_stats.json
#   └── params/              # Orbax checkpoint 权重（原样）
#       ├── _METADATA
#       ├── _sharding
#       ├── manifest.ocdbt
#       ├── array_metadatas/
#       ├── d/
#       └── ocdbt.process_0/
#
# 用法: bash scripts/package_model.sh <src_dir> <out_dir>
set -euo pipefail

SRC="${1:-/app/models/30000}"
OUT="${2:-/app/models/dist}"
MODEL_NAME="RoboDojo-goai2026-arx_x5-joint-0-pi05-flashrt-30000"

command -v zip >/dev/null || { echo "zip 未安装"; apt-get install -y zip >/dev/null 2>&1; }
mkdir -p "${OUT}"

echo "== 校验源模型 =="
[ -d "${SRC}/params" ] || { echo "错误: ${SRC}/params 不存在"; exit 1; }
[ -f "${SRC}/norm_stats.json" ] || { echo "错误: ${SRC}/norm_stats.json 缺失"; exit 1; }
[ -f "${SRC}/config.json" ] || { echo "错误: ${SRC}/config.json 缺失"; exit 1; }

echo "== 打包 ${MODEL_NAME} =="
STAGE=$(mktemp -d)
mkdir -p "${STAGE}/${MODEL_NAME}"
cp -r "${SRC}/params" "${STAGE}/${MODEL_NAME}/params"
cp "${SRC}/config.json" "${STAGE}/${MODEL_NAME}/config.json"
cp "${SRC}/norm_stats.json" "${STAGE}/${MODEL_NAME}/norm_stats.json"
mkdir -p "${STAGE}/${MODEL_NAME}/assets/arx_x5_sim"
cp "${SRC}/assets/arx_x5_sim/norm_stats.json" \
   "${STAGE}/${MODEL_NAME}/assets/arx_x5_sim/norm_stats.json"

ZIP="${OUT}/${MODEL_NAME}.zip"
cd "${STAGE}" && zip -q -r "${ZIP}" "${MODEL_NAME}"
rm -rf "${STAGE}"

echo "== 生成 MD5 =="
md5sum "${ZIP}" | tee "${OUT}/${MODEL_NAME}.zip.md5"

echo "== 完成 =="
ls -la "${ZIP}" "${OUT}/${MODEL_NAME}.zip.md5"
