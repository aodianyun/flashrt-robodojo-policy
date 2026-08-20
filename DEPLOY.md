# 快速部署文档 (GOAI 2026 参赛)

本文档说明如何把 `flashrt-robodojo-policy`(Pi0.5 双臂策略服务器)部署起来,
让 RoboDojo / GOAI 2026 评测端 (XPolicyLab `robodojo.sh client`) 通过
WebSocket 连接并实时获取动作。

## 0. 交付物清单

| 交付物 | 说明 |
|---|---|
| 代码仓库 | `https://github.com/aodianyun/flashrt-robodojo-policy.git` |
| 模型 | `RoboDojo-goai2026-arx_x5-joint-0-pi05-flashrt-30000.zip`(见 `COMPETE.md` 的 MD5) |
| 内核 | 预编译 SM120/RTX x86_64 内核(`vendor/FlashRT/flash_rt/*.so`)已入库/随镜像 |

## 1. 环境要求

- NVIDIA GPU:
  - 运行用: RTX 5090 / 5060 Ti (SM120) 或 RTX 4090 (SM89), 16GB+ 显存
  - 构建用: 同 SM 或更高, CUDA 13 工具链
- 驱动 545+, CUDA 13.0+ 工具链(仅编译/镜像内构建需要)
- Python 3.12, Docker 20.10+ (nvidia-container-toolkit)
- 磁盘: 模型 12GB + 镜像/依赖若干

> 预编译 `.so` 为 **x86_64 / Python 3.12 / CUDA 13 / SM120 (Blackwell)**。
> 换架构或 Python 版本需按 `vendor/FlashRT/README.md` 重新编译。

## 2. 方式 A — Docker 快速部署 (推荐)

### 2.1 准备模型

```bash
# 解压参赛模型 zip (或挂载现成目录)
unzip RoboDojo-goai2026-arx_x5-joint-0-pi05-flashrt-30000.zip -d /models/
mv /models/RoboDojo-goai2026-arx_x5-joint-0-pi05-flashrt-30000 /models/model
# 校验 MD5 (见 COMPETE.md)
md5sum /models/model/*  # 或对 zip 校验
```

### 2.2 一键部署

```bash
cd flashrt-robodojo-policy
bash docker/deploy_docker.sh \
  --model-zip /path/to/RoboDojo-...-30000.zip \
  --port 3101
```

首次会构建镜像(内含预编译内核,无需容器内编译),随后解压模型并启动容器。

等价手动命令:

```bash
# 构建 (已有镜像可省略)
docker build -t flashrt-robodojo:latest .

# 启动策略服务器
docker run -d --name flashrt-robodojo-server \
  --gpus all --shm-size=8g \
  -p 3101:3101 \
  -v /models/model:/models/model \
  flashrt-robodojo:latest \
  --checkpoint /models/model --port 3101

# 查看日志, 出现 "websocket policy server listening on ws://0.0.0.0:3101" 即就绪
docker logs -f flashrt-robodojo-server
```

## 3. 方式 B — 裸机部署

### 3.1 环境

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip
bash vendor/venv-jax/install_jax_venv.sh .venv   # 或按需
bash vendor/venv-torch/install_torch_venv.sh .venv
```

### 3.2 启动

```bash
source .venv/bin/activate
export PYTHONPATH="$(pwd)/vendor/FlashRT:$(pwd)/vendor/xp_lib:$(pwd)"
export FLASHRT_PI05_STATE_PROMPT_MODE=fixed
export FLASH_RT_PALIGEMMA_TOKENIZER="$(pwd)/vendor/FlashRT/assets/paligemma_tokenizer.model"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python3 -u server/run_server.py \
  --framework jax --quantization fp8 --hardware auto \
  --checkpoint /models/model --num-views 3 --action-dim 14 \
  --port 3101 --host 0.0.0.0
```

或使用包装脚本:

```bash
bash server/start_server.sh --checkpoint /models/model --port 3101 \
  --framework jax --quantization fp8 --hardware auto
```

就绪日志:

```
[SERVER] framework=jax quant=fp8 model loaded, starting on 0.0.0.0:3101
INFO:websockets.server:server listening on 0.0.0.0:3101
INFO:client_server.ws.model_server:websocket policy server listening on ws://0.0.0.0:3101
```

## 4. 评测端连接 (RoboDojo / XPolicyLab)

策略服务器运行在 GPU 机上, 评测端 (RoboDojo `robodojo.sh client`, 可在
simulator Docker 容器或另一台机器) 通过 WebSocket 连接:

```bash
cd /path/to/RoboDojo
bash scripts/robodojo.sh client \
  --policy-dir XPolicyLab/policy/goai_flashrt \
  --task stack_bowls \
  --policy-host <策略服务器IP> \
  --policy-port 3101 \
  --action-type joint \
  --ckpt goai_stack \
  --eval-num 10
```

关键参数必须与 server 一致:

| 参数 | server | client |
|---|---|---|
| 端口 | `--port 3101` | `--policy-port 3101` |
| 动作类型 | (固定 joint) | `--action-type joint` |
| env 配置 | arx_x5(默认) | `--env-cfg arx_x5` |

### 4.1 网络 (Docker 容器内评测)

| 场景 | `--policy-host` |
|---|---|
| 同机 bridge 容器 | `host.docker.internal`(加 `--add-host=host.docker.internal:host-gateway`) |
| 同机 host 网络 | `127.0.0.1` |
| 远程机器 | 策略服务器公网/局域网 IP |

## 5. 协议与动作格式

- 协议: XPolicyLab WebSocket (`protocol: ws`), 见 `vendor/xp_lib/client_server/ws/`
- 观测: 标准 Observation Data Format v1.0
  - `vision/cam_head/color`, `vision/cam_left_wrist/color`, `vision/cam_right_wrist/color` (RGB)
  - `state/left_arm_joint_state(6)`, `left_ee_joint_state(1)`, `right_arm_joint_state(6)`, `right_ee_joint_state(1)`
- 动作: 14 维双臂关节目标 (joint), 返回 `{left_arm_joint_state(6), left_ee_joint_state(1), right_arm_joint_state(6), right_ee_joint_state(1)}`
- 模型配置: `action_dim=32`, `action_horizon=50`, Pi0.5 (Gemma 2B + 300M action expert)

## 6. 故障排查

| 现象 | 处理 |
|---|---|
| `cuBLAS error ... code=15` | 误用 Thor 前端; 确保 `--hardware auto` 检测到 SM120/89 走 RTX 前端 |
| `norm_stats not found` | 确保 checkpoint 含 `norm_stats.json`(模型 zip 已带) |
| 端口被占 | 换 `--port`, 并保持 client `--policy-port` 一致 |
| 显存不足 | 检查无其他进程占 GPU (`nvidia-smi`); 模型需 16GB+ |
| `flash_rt_fa2` 缺失 | 预编译 .so 未随镜像/仓库; 按 `vendor/FlashRT/README.md` 重新编译 |

## 7. 参考

- 项目 README: `README.md`
- 参赛说明: `COMPETE.md`
- FlashRT 内核说明: `vendor/FlashRT/README.md`
