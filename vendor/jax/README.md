# JAX 环境依赖说明

## 版本清单（当前运行时）

| 包 | 版本 | 说明 |
|---|---|---|
| jax | 0.11.0 | 核心 |
| jax-cuda13-pjrt | 0.11.0 | CUDA 13 PJRT |
| jax-cuda13-plugin | 0.11.0 | CUDA 13 插件 (xla_cuda13) |
| ml-dtypes | 0.5.4 | bf16/fp8 dtype 支持 |
| orbax-checkpoint | 0.12.2 | Orbax checkpoint 加载 |
| flax | 0.12.8 | NNX 模型 |
| numpy | 2.5.1 | |

## 为什么必须锁版本

Orbax / jaxlib / PJRT plugin 的 ABI 跨 minor 版本不稳定。升级任何一个
而不同步其他三个，会出现 "PJRT device not registered" 或
"Unable to initialize backend" 错误。

## 安装方式

### 方式 A：脚本安装（推荐）

```bash
bash vendor/venv-jax/install_jax_venv.sh
```

脚本创建独立 venv（默认 `<project>/vendor/venv-jax/.venv`），安装上表
锁定的 JAX 栈 + WS server 依赖，并验证 `jax.devices()`。

> jax 栈约 1.1G（含 68 个包：jax/jaxlib/cuda plugin/orbax/flax/numpy 等）。

### 方式 B：从现有 venv 复用

直接使用已安装的 `vendor/venv-jax/.venv`，并在启动 server 时用其
`bin/python`（`deploy.sh` 默认复用，无需重装）。

## 平台说明

- 当前运行时在 **aarch64 (Thor SM110), CUDA 13.2 驱动 / 13.0 toolkit**。
- jax-cuda13-plugin 的 wheel 是平台相关的，换平台需重新安装。
- 系统还需: CUDA 13.0+ toolkit, NVIDIA 驱动 545+。

## 运行时 PYTHONPATH（启动 server 前）

JAX 栈已装入独立 venv，无需再手工设置 PYTHONPATH；
`server/start_server.sh` 会自动导出 FlashRT / xp_lib / 项目根路径。
