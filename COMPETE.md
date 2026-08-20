# GOAI 2026 参赛说明

> GOAI 2026 双臂协同挑战赛 (RoboDojo / XPolicyLab 评测)
> 机器人: ARX X5 双臂, 14 关节 (每臂 6 关节 + 1 夹爪)

## 1. 提交信息

| 项目 | 值 |
|---|---|
| 代码仓库 | `https://github.com/aodianyun/flashrt-robodojo-policy.git` |
| 策略服务器端口 | **3101** |
| 协议 | XPolicyLab WebSocket (`protocol: ws`) |
| 动作类型 | `joint` (双臂关节) |
| 机器人配置 | `arx_x5` / `dual_x5` (`arm_dim=[6,6]`, `ee_dim=[1,1]`) |

## 2. 模型

本策略使用**自训练模型**(基于 RoboDojo ARX-X5 双臂数据微调),非官方预训练模型。

### 2.1 模型获取

| 字段 | 值 |
|---|---|
| 魔塔仓库 | [cpadyun/RoboDojo-goai2026-arx_x5-joint-0-pi05-flashrt-30000](https://www.modelscope.cn/models/cpadyun/RoboDojo-goai2026-arx_x5-joint-0-pi05-flashrt-30000) |
| 大小 | ~12.4 GB |

```bash
pip install modelscope
modelscope download \
  --model cpadyun/RoboDojo-goai2026-arx_x5-joint-0-pi05-flashrt-30000 \
  --local_dir /models/model
```

### 2.2 模型目录结构

下载后 `/models/model/`:

```
├── config.json                 # 模型配置
├── norm_stats.json             # 动作/状态归一化统计
├── assets/arx_x5_sim/
│   └── norm_stats.json
└── params/                     # Orbax checkpoint 权重
```

### 2.3 模型配置

| 字段 | 值 |
|---|---|
| 格式 | Orbax (JAX) |
| 动作维度 | 14 (每臂 6 关节 + 1 夹爪) |
| 动作类型 | joint |
| 量化 | FP8 |
| action_dim | 32 |
| action_horizon | 50 |
| 基础模型 | Pi0.5 (Gemma 2B + 300M action expert) |

## 3. 部署

完整部署流程 (逐步) 见 [`DEPLOY.md`](./DEPLOY.md)。

### 环境要求

- GPU: NVIDIA RTX 50 系 (已验证 RTX 5060 Ti), 16GB+ 显存
- 部署基座: `nvcr.io/nvidia/pytorch:25.10-py3`

## 4. 评测命令 (RoboDojo 端)

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

参数一致性:

| 参数 | server | client |
|---|---|---|
| 端口 | `--port 3101` | `--policy-port 3101` |
| 动作类型 | joint (固定) | `--action-type joint` |
| env 配置 | arx_x5 (默认) | `--env-cfg arx_x5` |

## 5. 观测与动作格式

- **观测** (Observation Data Format v1.0):
  - `vision/cam_head/color` (H,W,3 RGB)
  - `vision/cam_left_wrist/color`, `vision/cam_right_wrist/color`
  - `state/left_arm_joint_state(6)`, `state/left_ee_joint_state(1)`,
    `state/right_arm_joint_state(6)`, `state/right_ee_joint_state(1)`
- **动作** (joint):
  `{left_arm_joint_state(6), left_ee_joint_state(1), right_arm_joint_state(6), right_ee_joint_state(1)}`

## 6. 参考

- 部署文档: [`DEPLOY.md`](./DEPLOY.md)
- 项目说明: [`README.md`](./README.md)
