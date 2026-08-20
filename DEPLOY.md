# 部署文档 — GOAI 2026 策略服务器

本文档面向评测方/运维: 部署 `flashrt-robodojo-policy` 策略服务器,
使 RoboDojo 评测端 (`robodojo.sh client`) 通过 WebSocket 获取动作。

## 环境要求

| 项 | 要求 |
|---|---|
| GPU | RTX 5090/5060 Ti (SM120) 或 RTX 4090 (SM89), 16GB+ 显存 |
| 驱动 | NVIDIA 545+ |
| 磁盘 | 模型 ~12GB |
| Docker | 20.10+, nvidia-container-toolkit (Docker 方式) |

## 方式 A — Docker (推荐)

> 镜像 `flashrt-goai-robodojo-wsserver:v1.0` **已内置代码 + 内核 + 模型**,
> 拉取即可运行, 无需构建或下载模型。

### 1. 拉取并运行

```bash
docker run --gpus all --shm-size=8g -p 3101:3101 flashrt-goai-robodojo-wsserver:v1.0
```

就绪标志:

```
INFO:client_server.ws.model_server:websocket policy server listening on ws://0.0.0.0:3101
```

### 2. (可选) 自行构建镜像

如需重打包(例如更新代码或换模型):

```bash
cd flashrt-robodojo-policy
bash docker/build_full_image.sh --model-dir /models/model --image flashrt-goai-robodojo-wsserver --tag v1.0
```

> 构建脚本在物理机执行, 将代码 + 预编译内核 + 模型一并打入镜像。

## 方式 B — 裸机

### 1. 准备模型 (同上)

```bash
modelscope download \
  --model cpadyun/RoboDojo-goai2026-arx_x5-joint-0-pi05-flashrt-30000 \
  --local_dir /models/model
```

### 2. 编译 FlashRT 内核

```bash
cd vendor/FlashRT
# 见 vendor/FlashRT/README.md: clone 源码 7fd75d2 + 应用补丁 + CUTLASS + cmake
```

### 3. 启动

```bash
cd flashrt-robodojo-policy
bash server/start_server.sh \
  --checkpoint /models/model --port 3101 \
  --framework jax --quantization fp8 --hardware auto
```

## 评测端连接

```bash
cd /path/to/RoboDojo
bash scripts/robodojo.sh client \
  --policy-dir XPolicyLab/policy/goai_flashrt \
  --task stack_bowls \
  --policy-host <SERVER_IP> \
  --policy-port 3101 \
  --action-type joint \
  --ckpt goai_stack \
  --eval-num 10
```

### 网络 (评测端在容器内)

| 场景 | `--policy-host` |
|---|---|
| 同机 bridge 容器 | `host.docker.internal` (加 `--add-host=host.docker.internal:host-gateway`) |
| 同机 host 网络 | `127.0.0.1` |
| 远程机器 | 策略服务器 IP |

## 故障排查

| 现象 | 处理 |
|---|---|
| `paligemma_tokenizer.model not found` | 确保容器内设置了 `FLASH_RT_PALIGEMMA_TOKENIZER` (Dockerfile 已内置) |
| `norm_stats not found` | 确保 checkpoint 含 `norm_stats.json` (魔塔模型已带) |
| 端口被占 | 换 `--port`, 并保持 client `--policy-port` 一致 |
| 显存不足 | 关闭其他 GPU 进程; 模型需 16GB+ |
