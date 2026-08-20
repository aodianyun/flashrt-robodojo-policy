# FlashRT RoboDojo Policy — GOAI 2026

面向 **GOAI 2026 双臂协同挑战赛** 的 Pi0.5 双臂操作策略服务。
基于 RoboDojo ARX-X5 双臂数据训练, 通过 WebSocket 为评测端提供实时关节动作。

- **机器人**: ARX X5 双臂 (14 关节)
- **模型**: 魔塔自训练模型 (见 [COMPETE.md](./COMPETE.md))
- **协议**: XPolicyLab WebSocket
- **硬件**: RTX 5090/5060 Ti (SM120) 或 RTX 4090 (SM89), 16GB+ 显存

## 快速启动

```bash
git clone https://github.com/aodianyun/flashrt-robodojo-policy.git
cd flashrt-robodojo-policy

# 1. 下载模型 (魔塔)
pip install modelscope
modelscope download \
  --model cpadyun/RoboDojo-goai2026-arx_x5-joint-0-pi05-flashrt-30000 \
  --local_dir /models/model

# 2. 启动策略服务器
bash server/start_server.sh \
  --checkpoint /models/model --port 3101 \
  --framework jax --quantization fp8 --hardware auto
```

就绪后评测端连接 `ws://<SERVER_IP>:3101`。

## 部署

- **Docker**: [`docker/build_full_image.sh`](./docker/build_full_image.sh)(构建含模型的完整镜像)
- **完整部署文档**: [`DEPLOY.md`](./DEPLOY.md)

## 参赛说明

- [`COMPETE.md`](./COMPETE.md) — 提交信息、模型、配置、评测命令

## 目录结构

```
├── COMPETE.md               # 参赛说明
├── DEPLOY.md                # 部署文档
├── server/                  # WS 策略服务器
├── docker/                  # Docker 镜像构建
├── scripts/                 # 模型下载/上传脚本
└── vendor/
    ├── FlashRT/             # 推理引擎
    └── xp_lib/              # XPolicyLab 策略适配层
```
