# GOAI 2026 参赛镜像 — flashrt-robodojo-policy 策略服务器 (完全自足)
#
# 镜像内完整编译 FlashRT CUDA 内核 (flash_rt_kernels + flash_rt_fa2),
# 不依赖仓库中的预编译 .so, 任何有 NVIDIA CUDA 工具链的主机都能构建。
#
# 基础: NVIDIA NGC PyTorch 25.10 (CUDA 13.0, PyTorch 2.9, Python 3.12, SM120)
#
# 构建 (建议在目标 GPU 相同或更高 SM 的主机上, 需 CUDA 工具链):
#   docker build -t flashrt-robodojo:latest .
#   # GPU_ARCH 可显式指定目标架构: 120=5090/5060Ti, 89=4090
#   docker build --build-arg GPU_ARCH=120 -t flashrt-robodojo:latest .
#
# 运行 (策略服务器, WebSocket):
#   docker run --gpus all --shm-size=8g \
#     -p 3101:3101 \
#     -v /path/to/model:/models/model \
#     flashrt-robodojo:latest
#   # 覆盖 checkpoint 路径:  --checkpoint /models/model --port 3101

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE_IMAGE}

# 目标 GPU 架构 (120=RTX5090/5060Ti Blackwell, 89=RTX4090 Ada, 110=Thor)
# 留空则由 CMake 在构建时通过 nvidia-smi 自动检测
ARG GPU_ARCH=""
ARG FLASHRT_COMMIT=7fd75d20c0528f4c3ba70e23b9fbf579dbc68211
ARG CUTLASS_REF=v4.4.2
ARG SMOKE=1

# ── 基础工具 (NGC 自带 cmake/ninja/git, 保险起见再装) ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        ccache git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── 项目代码 ──
WORKDIR /opt/flashrt-robodojo-policy
COPY . .

# ── FlashRT 完整源码 (含 C++ csrc) + CUTLASS ──
RUN git clone https://github.com/flashrt-project/FlashRT.git /tmp/FlashRT \
    && cd /tmp/FlashRT \
    && git checkout ${FLASHRT_COMMIT} \
    && git clone --depth 1 --branch ${CUTLASS_REF} \
        https://github.com/NVIDIA/cutlass.git third_party/cutlass \
    && rm -rf third_party/cutlass/.git \
    # 应用双臂补丁 (本项目携带)
    && git apply /opt/flashrt-robodojo-policy/vendor/FlashRT/flashrt_goai_dualarm.patch \
    # 用项目内的 Python 源码覆盖 (含补丁后前端)
    && cp -r /opt/flashrt-robodojo-policy/vendor/FlashRT/flash_rt . \
    # CUTLASS 符号链接
    && mkdir -p third_party

WORKDIR /tmp/FlashRT

# ── 编译 CUDA 内核 ──
RUN cmake -B build -S . \
        $( [ -n "${GPU_ARCH}" ] && echo "-DGPU_ARCH=${GPU_ARCH}" ) \
    && cmake --build build -j"$(nproc)" \
    && rm -rf build/CMakeFiles \
    # 将编译产物 .so 合并回项目目录
    && cp flash_rt/*.so* /opt/flashrt-robodojo-policy/vendor/FlashRT/flash_rt/ 2>/dev/null || true

WORKDIR /opt/flashrt-robodojo-policy

# ── 运行时依赖 (JAX/Orbax 前端 + WS server) ──
RUN python3 -m pip install --no-cache-dir \
        "jax==0.11.0" \
        "ml-dtypes==0.5.4" \
        "orbax-checkpoint==0.12.2" \
        "flax==0.12.8" \
        "numpy==2.5.1" \
        websockets msgpack msgpack-numpy pyyaml sentencepiece pillow \
        pydantic opencv-python-headless h5py \
    && python3 -m pip install --no-cache-dir --no-build-isolation -e "/opt/flashrt-robodojo-policy/vendor/FlashRT[torch]"

# ── 自检 ──
RUN if [ "${SMOKE}" = "1" ]; then \
      PYTHONPATH=/opt/flashrt-robodojo-policy/vendor/FlashRT:/opt/flashrt-robodojo-policy/vendor/xp_lib:/opt/flashrt-robodojo-policy \
      python3 -c "import flash_rt; from flash_rt import flash_rt_kernels; from flash_rt import flash_rt_fa2; print('flash_rt kernels+fa2 OK')"; \
    else echo "SMOKE disabled"; fi

# ── 清理 ──
RUN rm -rf /tmp/FlashRT

# ── 入口: WS 策略服务器 ──
EXPOSE 3101
ENV PYTHONPATH=/opt/flashrt-robodojo-policy/vendor/FlashRT:/opt/flashrt-robodojo-policy/vendor/xp_lib:/opt/flashrt-robodojo-policy
ENV FLASHRT_PI05_STATE_PROMPT_MODE=fixed
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
CMD ["python3", "-u", "server/run_server.py", \
     "--framework", "jax", "--quantization", "fp8", "--hardware", "auto", \
     "--num-views", "3", "--action-dim", "14", "--port", "3101", "--host", "0.0.0.0"]
