# 部署文档 — GOAI 2026 策略服务器

部署策略服务器, 使 RoboDojo 评测端通过 WebSocket 获取动作。

## 环境要求

- GPU: NVIDIA RTX 50 系 (已验证 RTX 5060 Ti), 16GB+ 显存
- 驱动: NVIDIA 545+
- Docker: 20.10+, nvidia-container-toolkit

## Docker 部署 (推荐)

> 基础镜像 `nvcr.io/nvidia/pytorch:25.10-py3`, 镜像已内置代码 + 内核 + 模型。

### 1. 拉取并运行

```bash
docker run --gpus all --shm-size=8g -p 3101:3101 \
  registry.cn-hangzhou.aliyuncs.com/adpub/flashrt-goai-robodojo-wsserver:v1.2
```

就绪标志:

```
INFO:client_server.ws.model_server:websocket policy server listening on ws://0.0.0.0:3101
```

### 2. 评测端连接 (RoboDojo 端)

```bash
bash scripts/robodojo.sh client \
  --policy-dir XPolicyLab/policy/goai_flashrt \
  --task stack_bowls \
  --policy-host <SERVER_IP> \
  --policy-port 3101 \
  --action-type joint \
  --ckpt goai_stack \
  --eval-num 10
```

评测端在容器内时, `--policy-host` 用 `host.docker.internal` (加 `--add-host=host.docker.internal:host-gateway`)。

## 故障排查

| 现象 | 处理 |
|---|---|
| 端口被占 | 换 `--port`, 并保持 client `--policy-port` 一致 |
| 显存不足 | 关闭其他 GPU 进程; 模型需 16GB+ |
