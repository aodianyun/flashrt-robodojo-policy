# GOAI 2026 — flashrt-robodojo-policy 策略服务器镜像
#
# 完整自动化构建流程:
#   1. 基于 nvcr.io/nvidia/pytorch:25.10-py3
#   2. 拷贝本项目代码 (已含 FlashRT 源码 + 补丁 + tokenizer)
#   3. 从魔塔下载参赛模型
#   4. 获取并编译 FlashRT CUDA 内核 (含 CUTLASS)
#   5. 安装运行时依赖
#   6. 运行 WS 策略服务器
#
# 构建 (需 CUDA 工具链主机):
#   docker build --build-arg GPU_ARCH=120 -t flashrt-goai-robodojo-wsserver:v1.2 .
#
# 运行:
#   docker run --gpus all --shm-size=8g -p 3101:3101 flashrt-goai-robodojo-wsserver:v1.2

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE_IMAGE}

# 目标 GPU 架构 (120=RTX5090/5060Ti Blackwell, 89=RTX4090 Ada)
ARG GPU_ARCH=""
# FlashRT 上游 commit + CUTLASS 版本 (与项目 vendor/FlashRT 一致)
ARG FLASHRT_COMMIT=7fd75d20c0528f4c3ba70e23b9fbf579dbc68211
ARG CUTLASS_REF=v4.4.2
# 魔塔模型仓库
ARG MODEL_REPO=cpadyun/RoboDojo-goai2026-arx_x5-joint-0-pi05-flashrt-30000

# ── 基础工具 ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        ccache git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── 本项目代码 (含 vendor/FlashRT 源码 + 补丁 + tokenizer) ──
WORKDIR /opt/flashrt-robodojo-policy
COPY . .

# ── FlashRT 完整源码 (含 C++ csrc) + CUTLASS ──
RUN git clone https://github.com/flashrt-project/FlashRT.git /tmp/FlashRT \
    && cd /tmp/FlashRT \
    && git checkout ${FLASHRT_COMMIT} \
    && git clone --depth 1 --branch ${CUTLASS_REF} \
        https://github.com/NVIDIA/cutlass.git third_party/cutlass \
    && rm -rf third_party/cutlass/.git \
    && git apply /opt/flashrt-robodojo-policy/vendor/FlashRT/flashrt_goai_dualarm.patch \
    && mkdir -p third_party

# ── 编译 FlashRT CUDA 内核 ──
WORKDIR /tmp/FlashRT
RUN cmake -B build -S . \
        $( [ -n "${GPU_ARCH}" ] && echo "-DGPU_ARCH=${GPU_ARCH}" ) \
    && cmake --build build -j"$(nproc)" \
    && rm -rf build/CMakeFiles \
    # 编译产物 .so 合并回项目目录
    && cp flash_rt/*.so* /opt/flashrt-robodojo-policy/vendor/FlashRT/flash_rt/ 2>/dev/null || true

WORKDIR /opt/flashrt-robodojo-policy

# ── 运行时依赖 (JAX/Orbax 前端 + WS server) ──
RUN python3 -m pip install --no-cache-dir \
        modelscope \
        "jax==0.11.0" \
        "ml-dtypes==0.5.4" \
        "orbax-checkpoint==0.12.2" \
        "flax==0.12.8" \
        "numpy==2.5.1" \
        websockets msgpack msgpack-numpy pyyaml sentencepiece pillow \
        pydantic opencv-python-headless h5py

# ── 从魔塔下载参赛模型 ──
RUN mkdir -p /models/model \
    && modelscope download \
        --model ${MODEL_REPO} \
        --local_dir /models/model

# ── 自检: 内核可加载 ──
RUN PYTHONPATH=/opt/flashrt-robodojo-policy/vendor/FlashRT:/opt/flashrt-robodojo-policy/vendor/xp_lib:/opt/flashrt-robodojo-policy \
    python3 -c "import flash_rt; from flash_rt import flash_rt_kernels; from flash_rt import flash_rt_fa2; print('flash_rt kernels+fa2 OK')"

# ── 清理 ──
RUN rm -rf /tmp/FlashRT

# ── 入口: WS 策略服务器 (加载 /models/model) ──
EXPOSE 3101
ENV PYTHONPATH=/opt/flashrt-robodojo-policy/vendor/FlashRT:/opt/flashrt-robodojo-policy/vendor/xp_lib:/opt/flashrt-robodojo-policy
ENV FLASHRT_PI05_STATE_PROMPT_MODE=fixed
ENV FLASH_RT_PALIGEMMA_TOKENIZER=/opt/flashrt-robodojo-policy/vendor/FlashRT/assets/paligemma_tokenizer.model
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
CMD ["python3", "-u", "server/run_server.py", \
     "--framework", "jax", "--quantization", "fp8", "--hardware", "auto", \
     "--checkpoint", "/models/model", \
     "--num-views", "3", "--action-dim", "14", "--port", "3101", "--host", "0.0.0.0"]
