# 部署文档 — GOAI 2026 策略服务器

在物理机上基于 `nvcr.io/nvidia/pytorch:25.10-py3` 完整部署策略服务器。

## 环境要求

- GPU: NVIDIA RTX 50 系 (已验证 RTX 5060 Ti), 16GB+ 显存
- 驱动: NVIDIA 545+
- Docker: 20.10+, nvidia-container-toolkit

## 步骤 1 — 拉取基础镜像

```bash
docker pull nvcr.io/nvidia/pytorch:25.10-py3
```

## 步骤 2 — 获取项目代码

```bash
git clone https://github.com/aodianyun/flashrt-robodojo-policy.git
cd flashrt-robodojo-policy
```

## 步骤 3 — 准备模型

```bash
pip install modelscope
modelscope download \
  --model cpadyun/RoboDojo-goai2026-arx_x5-joint-0-pi05-flashrt-30000 \
  --local_dir /models/model
```

## 步骤 4 — 构建镜像

```bash
bash docker/build_full_image.sh \
  --model-dir /models/model \
  --image flashrt-goai-robodojo-wsserver \
  --tag v1.2
```

> 脚本基于 `nvcr.io/nvidia/pytorch:25.10-py3`, 将代码 + 预编译内核 + 模型
> 一并打入镜像。产物: `flashrt-goai-robodojo-wsserver:v1.2`。

## 步骤 5 — 运行策略服务器

```bash
docker run --gpus all --shm-size=8g -p 3101:3101 \
  flashrt-goai-robodojo-wsserver:v1.2
```

就绪标志:

```
INFO:client_server.ws.model_server:websocket policy server listening on ws://0.0.0.0:3101
```

## 评测端连接 (RoboDojo 端)

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
