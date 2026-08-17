# FlashRT 版本标注

## 版本信息

- **Git commit**: `7fd75d20c0528f4c3ba70e23b9fbf579dbc68211`
- **提交说明**: `7fd75d2 pi05 thor update: NVFP4 encoder/decoder tier + FA4 attention (#164)`
- **上游**: https://github.com/flashrt-project/FlashRT （与 origin/main 一致）
- **本目录内容**: `flash_rt/` Python 源码 + 已编译内核 .so（aarch64 / Thor SM110）

## 本目录说明

`flash_rt/` 是**编译后的可运行目录**（含 `.so`），已应用补丁 `flashrt_goai_dualarm.patch`。
本目录中的 3 个 `.so` 内核（aarch64 / Py3.12）已随项目入库（见根 `.gitignore` 豁免项）。
拉取本项目后，直接 `sys.path.insert(0, ".../vendor/FlashRT")` 即可 `import flash_rt`，无需重新编译。

## 已编译内核

| 文件 | 作用 |
|---|---|
| `flash_rt/flash_rt_kernels.cpython-312-aarch64-linux-gnu.so` | 手写内存核（norm/activation/FP8/注意力等），必需 |
| `flash_rt/flash_rt_fp4.cpython-312-aarch64-linux-gnu.so` | NVFP4 GEMM（`--quantization fp4`/`fp4-awq` 时必需） |
| `flash_rt/libfmha_fp16_strided.so` | FP16 FMHA |

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
