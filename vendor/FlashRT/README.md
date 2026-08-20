# FlashRT 版本标注

## 版本信息

- **Git commit**: `7fd75d20c0528f4c3ba70e23b9fbf579dbc68211`
- **提交说明**: `7fd75d2 pi05 thor update: NVFP4 encoder/decoder tier + FA4 attention (#164)`
- **上游**: https://github.com/flashrt-project/FlashRT （与 origin/main 一致）
- **本目录内容**: `flash_rt/` Python 源码 + 补丁 `flashrt_goai_dualarm.patch`

## 本目录说明

`flash_rt/` 是已应用补丁 `flashrt_goai_dualarm.patch` 的 **Python 源码目录**。

**编译内核 `.so` 不入库**（平台/架构相关，且 `flash_rt_fa2.so` 超过 GitHub 100MB 限制）。获取方式：

1. **推荐：使用项目提供的 Docker 镜像**（`docker/Dockerfile` 在镜像内完整编译内核），拉取即用；
2. **裸机自行编译**：见下文"从源码重新编译"。

## 编译内核（.so）清单

镜像内编译后 `flash_rt/` 下会生成：

| 文件 | 作用 |
|---|---|
| `flash_rt_kernels.cpython-312-<arch>-linux-gnu.so` | 手写内存核（norm/activation/FP8/注意力等），必需 |
| `flash_rt_fa2.cpython-312-<arch>-linux-gnu.so` | RTX 前端 vendored Flash-Attention 2（SM120/SM89 必需） |
| `flash_rt_fp4.cpython-312-<arch>-linux-gnu.so` | NVFP4 GEMM（`--quantization fp4`/`fp4-awq` 时必需，仅 Thor） |
| `libfmha_fp16_strided.so` | FP16 FMHA（仅 Thor） |

> SM120 运行时经 `api.py` / 前端 `has_nvfp4()` 守卫自动退化为 FP8。

## 量化方案

| 部分 | 量化 |
|---|---|
| SigLIP 视觉 | FP8 E4M3 静态 |
| Encoder (Gemma 18层) | FP8 E4M3 |
| Decoder (action expert) | FP8 E4M3 |
| FP4 可选 | `fp4`/`fp4-awq`：Encoder FFN 用 NVFP4 GEMM（`flash_rt_fp4.so`） |

> 注：表格描述的是 **FP8 默认配置**；本项目 `--quantization` 支持 `fp8|fp4|fp4-awq|bf16`，
> fp4 路径由 `Pi05JaxFrontendThorFP4`（`frontends/jax/pi05_thor_fp4.py`）加载 `flash_rt_fp4`。

## 从源码重新编译（仅当换机器/换架构时需要）

```bash
# 系统要求: Python 3.12, CUDA 13.0+, CMake 3.24+, GCC 11+, NVIDIA 驱动 545+
python3.12 -m venv .venv && source .venv/bin/activate

# 依赖
pip install torch --index-url https://download.pytorch.org/whl/cu124  # Thor
pip install pybind11 cmake "numpy>=1.24" safetensors
pip install "transformers<4.56" pandas pillow pyarrow
pip install jax==0.5.3 jax-cuda12-pjrt==0.5.3 jax-cuda12-plugin==0.5.3 ml_dtypes==0.5.3

# 获取源码 + CUTLASS
git clone https://github.com/flashrt-project/FlashRT.git
cd FlashRT && git checkout 7fd75d2
git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git third_party/cutlass

# 应用本项目补丁（双臂支持）
git apply flashrt_goai_dualarm.patch   # 本目录自带

# 编译
pip install -e ".[jax]"   # editable 模式必需，.so 会落入 flash_rt/ 目录
cd build && cmake .. && make -j4
```

> 注意：本机（Thor SM110, aarch64）编译产物是
> `*.cpython-312-aarch64-linux-gnu.so`。若在 x86 或其他 Python 版本上
> 运行，必须重新编译。
