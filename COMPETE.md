# GOAI 2026 参赛说明 — FlashRT RoboDojo 双臂策略

> **比赛**: GOAI 2026 双臂协同挑战赛 (RoboDojo / XPolicyLab 评测)
> **任务示例**: stack_bowls (叠碗), fold_clothes (叠衣服)
> **机器人**: ARX X5 双臂, 14 关节 (每臂 6 关节 + 1 夹爪)

## 1. 提交信息总览

| 项目 | 值 |
|---|---|
| 队伍 / 提交方 | aodianyun |
| 代码仓库 | `https://github.com/aodianyun/flashrt-robodojo-policy.git` |
| 策略服务器端口 | **3101** |
| 协议 | XPolicyLab WebSocket (`protocol: ws`) |
| 动作类型 | `joint` (双臂关节) |
| 机器人配置 | `arx_x5` / `dual_x5` (`arm_dim=[6,6]`, `ee_dim=[1,1]`) |
| 推理框架 | JAX (Orbax 权重) / PyTorch (safetensors) 均可 |
| 量化 | FP8 (默认), NVFP4 (fp4, 仅 Thor) |

## 2. 模型

### 2.1 模型文件

| 字段 | 值 |
|---|---|
| 文件名 | `RoboDojo-goai2026-arx_x5-joint-0-pi05-flashrt-30000.zip` |
| 大小 | 12,443,251,034 字节 (12.4 GB) |
| **MD5** | `c749512b472bc85c1a34f44d69a77c4f` |
| 下载地址 | (待填: Hugging Face / 网盘链接) |

> 上传后请在下方"下载地址"处补充链接。校验: `md5sum RoboDojo-...-30000.zip`。

### 2.2 模型内容与结构

解压后目录 `RoboDojo-goai2026-arx_x5-joint-0-pi05-flashrt-30000/`:

```
├── config.json                 # 模型配置
├── norm_stats.json             # 动作/状态归一化统计 (openpi schema)
├── assets/arx_x5_sim/
│   └── norm_stats.json
└── params/                     # Orbax checkpoint 权重 (51 组)
```

- 格式: Orbax (JAX)
- 配置 (`config.json`): `action_dim=32`, `action_horizon=50`,
  `paligemma_variant=gemma_2b`, `action_expert_variant=gemma_300m`,
  `precision=bfloat16`
- 权重校验: 加载后 FlashRT 识别 51 个权重组
- 归一化: RoboDojo ARX-X5 双臂统计 (openpi quantile 语义 + DeltaActions)

### 2.3 模型来源

- 上游: RoboDojo ARX-X5 joint 策略, 30000 训练步
- 训练数据: RoboDojo-Benchmark ARX X5 双臂数据
- 推理适配: FlashRT (commit `7fd75d2`) + `flashrt_goai_dualarm.patch` (14 维双臂)

## 3. 策略服务器

### 3.1 部署

完整部署文档: [`DEPLOY.md`](./DEPLOY.md)

| 方式 | 命令 | 说明 |
|---|---|---|
| **Docker (推荐)** | `bash docker/deploy_docker.sh --model-zip <zip> --port 3101` | 镜像内置预编译内核, 快速 |
| 裸机 | `bash vendor/xp_lib/XPolicyLab/policy/goai_flashrt/setup_eval_policy_server.sh --checkpoint <dir> --port 3101` | 需 Python 3.12 + CUDA 13 |
| 包装脚本 | `bash server/start_server.sh --checkpoint <dir> --port 3101 --hardware auto` | |

就绪日志:

```
INFO:client_server.ws.model_server:websocket policy server listening on ws://0.0.0.0:3101
```

### 3.2 环境要求

- GPU: RTX 5090 / 5060 Ti (SM120) 或 RTX 4090 (SM89), 16GB+ 显存
- 驱动 545+, CUDA 13.0, Python 3.12
- 预编译内核为 **x86_64 / Py3.12 / CUDA13 / SM120**; 换架构需重编 (见 `vendor/FlashRT/README.md`)

### 3.3 硬件路由

`--hardware auto`(默认)按 GPU 计算能力自动选择前端:

| Compute Capability | GPU | 前端 |
|---|---|---|
| SM110 | Jetson AGX Thor | Thor (`Pi05*FrontendThor`) |
| SM120 | RTX 5090 / 5060 Ti | RTX (`Pi05*FrontendRtx`) |
| SM89 | RTX 4090 | RTX (`Pi05*FrontendRtx`) |

也可显式 `--hardware thor|rtx_sm120|rtx_sm89`。NVFP4 (fp4) 仅 Thor 可用, RTX 自动退化为 FP8。

## 4. 评测命令 (RoboDojo 端)

### 4.1 单任务评测 (split: 远程策略服务器 + 本地 simulator)

策略服务器侧 (GPU 机):

```bash
bash vendor/xp_lib/XPolicyLab/policy/goai_flashrt/setup_eval_policy_server.sh \
  --checkpoint /models/model --port 3101 --framework jax --quantization fp8
```

评测端 (RoboDojo / Isaac Sim, 可 Docker):

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

### 4.2 Docker client (simulator 容器)

```bash
docker run --rm --gpus all \
  --add-host=host.docker.internal:host-gateway \
  <robodojo-sim-image> \
  bash scripts/robodojo.sh client \
    --policy-name goai_flashrt \
    --policy-host host.docker.internal \
    --policy-port 3101 \
    --task stack_bowls --action-type joint --ckpt goai_stack --eval-num 10
```

### 4.3 关键参数一致性

| 参数 | server | client |
|---|---|---|
| 端口 | `--port 3101` | `--policy-port 3101` |
| 动作类型 | joint (固定) | `--action-type joint` |
| env 配置 | arx_x5 | `--env-cfg arx_x5` (默认) |
| 任务 | - | `--task stack_bowls` / `fold_clothes` 等 |

### 4.4 本地 smoke (无 simulator 离线验证)

```bash
# XPolicyLab debug 模式验证适配层 (形状/动作键/批处理)
export EVAL_ENV_TYPE=debug
cd vendor/xp_lib/XPolicyLab/policy/goai_flashrt
bash setup_eval_policy_server.sh --checkpoint /models/model --port 3101
```

## 5. 协议与数据格式

- **协议**: XPolicyLab WebSocket, 见 `vendor/xp_lib/client_server/ws/`
- **观测** (Observation Data Format v1.0):
  - `vision/cam_head/color` (H,W,3 RGB)
  - `vision/cam_left_wrist/color`, `vision/cam_right_wrist/color`
  - `state/left_arm_joint_state(6)`, `state/left_ee_joint_state(1)`,
    `state/right_arm_joint_state(6)`, `state/right_ee_joint_state(1)`
- **动作** (返回, joint): `{left_arm_joint_state(6), left_ee_joint_state(1), right_arm_joint_state(6), right_ee_joint_state(1)}`
- **动作语义**: openpi Pi0.5 标准 (quantile 反归一化 + DeltaActions → 绝对目标), 与 Thor-JAX 参考实现一致

## 6. 代码仓库结构

```
flashrt-robodojo-policy/
├── COMPETE.md                    # 本文件 (参赛说明)
├── DEPLOY.md                     # 部署文档
├── README.md                     # 项目说明
├── deploy.sh                     # 一键部署 (venv + 权重 + 启动)
├── scripts/
│   ├── package_model.sh          # 模型打包脚本 (生成 zip + md5)
│   ├── download_checkpoint.sh
│   └── resolve_checkpoint.sh
├── server/                       # WS 策略服务器
│   ├── run_server.py
│   └── start_server.sh
├── docker/
│   ├── Dockerfile                # 参赛镜像 (预编译内核)
│   └── deploy_docker.sh          # 一键 Docker 部署
└── vendor/
    ├── FlashRT/                  # 推理引擎 (7fd75d2 + 补丁 + 预编译 .so)
    └── xp_lib/XPolicyLab/policy/goai_flashrt/  # 策略适配层
        ├── model.py
        ├── deploy.yml
        └── setup_eval_policy_server.sh
```

## 7. 模型打包复现

```bash
# 从源模型目录生成参赛 zip + md5
bash scripts/package_model.sh /path/to/model_dir /output/dir
```

## 8. 变更记录

| 日期 | 内容 |
|---|---|
| 2026-08-20 | 打包 30000 模型 (MD5 `c749512b472bc85c1a34f44d69a77c4f`), 完成 Docker 部署 + 参赛文档 |
