# FlashRT RoboDojo Policy

基于 [FlashRT](https://github.com/flashrt-project/FlashRT) 的 pi0.5 双臂操作策略服务，
面向 RoboDojo / GOAI 2026 双臂挑战赛（ARX X5 双臂，14 关节）。

在 NVIDIA Thor SM110 上以 **FP8 / NVFP4** 量化推理，通过 WebSocket 协议为
RoboDojo 评估器提供实时动作，支持叠碗（stack_bowls）、叠衣服（fold_clothes）等任务。

## ✨ 亮点

- **双前端**：JAX（orbax 权重）和 PyTorch（safetensors 权重）均可推理，独立 venv 隔离
- **多量化**：FP8（默认）、NVFP4、NVFP4+AWQ，一行参数切换
- **自包含**：FlashRT 源码 + 编译内核 + tokenizer 全部随项目携带，拉取即用
- **权重自动管理**：`<ckpt-dir>/<model>/` 规范目录，缺失自动下载
- **全 CLI 参数**：框架/量化/权重/端口全部命令行控制，环境变量仅用于系统层
- **已实测**：stack_bowls 叠碗 ✅、fold_clothes 叠衣服（FP8: 叠碗 5/10、叠衣服 4/10）

## 🔗 相关项目

| 项目 | 说明 |
|---|---|
| [FlashRT](https://github.com/flashrt-project/FlashRT) | 底层推理引擎（commit `7fd75d2`） |
| [RoboDojo-Benchmark/RoboDojo](https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo) | 模型权重来源 |
| [openpi](https://github.com/Physical-Intelligence/openpi) | 参考实现（动作语义/归一化对齐） |

## 🚀 Quickstart

### 环境要求

- NVIDIA Thor SM110（aarch64）或 SM80+ 系列 GPU
- CUDA 13.0+ toolkit，NVIDIA 驱动 545+
- Python 3.12

### 一键部署

```bash
git clone https://github.com/<you>/flashrt-robodojo-policy.git
cd flashrt-robodojo-policy

# JAX 推理（独立 venv + 权重自动下载）
bash deploy.sh --framework jax --quantization fp8 \
  --model pi05-arx-x5 --ckpt-dir /data/ckpts --download
```

### 手动部署

```bash
# 1. 创建独立 venv
bash vendor/venv-jax/install_jax_venv.sh            # JAX
bash vendor/venv-torch/install_torch_venv.sh        # Torch（可选）

# 2. 下载模型权重
bash scripts/download_checkpoint.sh pi05-arx-x5 /data/ckpts

# 3. 启动 server（--hardware 默认 auto，按 GPU 自动选择 Thor/RTX 后端）
bash server/start_server.sh --quantization fp8        # FP8（默认）
bash server/start_server.sh --quantization fp4-awq    # NVFP4 + AWQ（仅 Thor）
bash server/start_server.sh --framework torch --quantization bf16 \
  --model pi05-arx-x5 --ckpt-dir /data/ckpts          # Torch 前端
bash server/start_server.sh --hardware rtx_sm120 --framework torch \
  --quantization fp8 --model pi05-arx-x5 --ckpt-dir /data/ckpts  # RTX (SM120) 显式指定
```

### 评测端

```bash
# 任务可选 stack_bowls / fold_clothes
cd /data/RoboDojo && source /opt/miniconda3/etc/profile.d/conda.sh && conda activate RoboDojo
bash scripts/robodojo.sh client \
  --policy-dir XPolicyLab/policy/Pi_05 \
  --task stack_bowls \
  --policy-host <SERVER_IP> \
  --policy-port <PUBLIC_PORT> \
  --action-type joint \
  --ckpt goai_stack --eval-num 10
```

## 📦 项目结构

```
flashrt-robodojo-policy/
├── deploy.sh                  # 一键部署（venv + 权重解析 + 启动）
├── requirements.txt           # 依赖版本清单
├── scripts/                   # 权重下载/解析
├── server/                    # WS server（run_server.py + start_server.sh）
└── vendor/                    # FlashRT 源码(+编译内核) + venv 脚本 + XPolicyLab client
    ├── FlashRT/               # 推理引擎（含 flashrt_goai_dualarm.patch 补丁）
    └── xp_lib/XPolicyLab/policy/goai_flashrt/   # 适配层（jax/torch 双前端）
```

## ⚙️ 参数说明

```text
--framework jax|torch           推理框架
--quantization fp8|fp4|fp4-awq|bf16   量化方案
--hardware auto|thor|rtx_sm120|rtx_sm89   GPU 后端（默认 auto 自动检测）
--model NAME --ckpt-dir DIR     权重定位（<dir>/<name>/）
--checkpoint PATH               显式权重路径（覆盖 --model）
--download-missing              权重缺失时自动下载
--port N --host IP --num-views N --action-dim N
```

## 📝 注意事项

1. **JAX 前端仅 FP8/NVFP4**：无真 FP16 路径（权重固定 FP8 量化）；BF16 需 torch 前端 + safetensors 权重
2. **`.so` 内核已随项目入库**（aarch64 / Py3.12，见 `vendor/FlashRT/README.md`）；换架构或 Python 版本需按该文档重新编译，`*.so` 仅豁免这 3 个内核文件
3. **硬件自动路由**：`--hardware auto`（默认）按 GPU 计算能力选择后端——sm_110 → Thor 前端，sm_120（RTX 5090/5060 Ti 等 Blackwell）/ sm_89（RTX 4090 等 Ada）→ RTX 前端；也可用 `--hardware thor|rtx_sm120|rtx_sm89` 显式指定。NVFP4（fp4/fp4-awq）仅 Thor 可用，RTX 上自动退化为 FP8
4. **模型权重不入库**（~12GB），用 `scripts/download_checkpoint.sh` 下载
5. **固定 state prompt**：`FLASHRT_PI05_STATE_PROMPT_MODE=fixed` 已默认，避免 CUDA Graph 重捕漂移
6. **量化切换需重启**：权重在初始化时加载，运行时不可切换
7. 环境变量仅用于系统层（XLA 内存、tokenizer 路径），业务参数一律 CLI

## 📊 评测结果

| 任务 | FP8 | NVFP4+AWQ |
|---|---|---|
| stack_bowls（叠碗） | 5/10 | 相近 |
| fold_clothes（叠衣服） | 4/10 | 相近 |

## 📄 License

[Apache License 2.0](./LICENSE)

FlashRT 上游同为 Apache 2.0。模型权重版权归 RoboDojo-Benchmark 所有，请遵循其使用条款。
