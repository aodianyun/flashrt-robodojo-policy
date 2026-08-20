# FlashRT RoboDojo Policy — GOAI 2026

面向 **GOAI 2026 双臂协同挑战赛** 的 Pi0.5 双臂操作策略服务。
基于 RoboDojo ARX-X5 双臂数据训练, 通过 WebSocket 为评测端提供实时关节动作。

- **机器人**: ARX X5 双臂 (14 关节)
- **模型**: 魔塔自训练模型 (见 [COMPETE.md](./COMPETE.md))
- **协议**: XPolicyLab WebSocket
- **GPU 要求**: NVIDIA RTX 50 系 (已验证 RTX 5060 Ti), 16GB+ 显存

## 快速开始 (Docker)

### 1. 运行策略服务器

```bash
docker run --gpus all --shm-size=8g -p 3101:3101 \
  registry.cn-hangzhou.aliyuncs.com/adpub/flashrt-goai-robodojo-wsserver:v1.2
```

就绪标志:

```
INFO:client_server.ws.model_server:websocket policy server listening on ws://0.0.0.0:3101
```

> 镜像已内置代码 + 内核 + 模型, 拉取即用, 无需构建或下载模型。

### 2. 评测端命令 (RoboDojo 端, 12 任务批量)

```bash
bash scripts/robodojo.sh client \
  --policy-name goai_flashrt \
  --policy-host <SERVER_IP> \
  --policy-port 3101 \
  --action-type joint \
  --ckpt goai_stack \
  --eval-num 5 \
  --only stack_bowls,push_T,pack_objects_into_box,fold_clothes,hang_mugs,sweep_blocks,pour_liquid_into_cup,make_toast,arrange_largest_number,sort_nesting_dolls_by_size,store_laptop_and_headphones,stack_blocks
```

## 环境要求

| 项 | 要求 |
|---|---|
| GPU | NVIDIA RTX 50 系 (已验证 RTX 5060 Ti), 16GB+ 显存 |
| 驱动 | NVIDIA 545+ |
| Docker | 20.10+, nvidia-container-toolkit |

## 更多文档

- [`COMPETE.md`](./COMPETE.md) — 参赛说明 (提交信息、模型、配置)
- [`DEPLOY.md`](./DEPLOY.md) — 部署流程

## 目录结构

```
├── COMPETE.md               # 参赛说明
├── DEPLOY.md                # 部署文档
├── server/                  # WS 策略服务器
├── docker/Dockerfile        # 完整构建流程 (git + 模型 + FlashRT 编译 + 依赖 + 运行)
├── scripts/                 # 模型下载/上传脚本
└── vendor/
    ├── FlashRT/             # 推理引擎
    └── xp_lib/              # XPolicyLab 策略适配层
```
