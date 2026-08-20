# GOAI 2026 参赛专用镜像 — flashrt-robodojo-policy 策略服务器
#
# 快速部署: 镜像内置预编译 SM120/RTX 内核 (flash_rt_kernels + flash_rt_fa2),
#           拉取即用, 无需在容器内编译 CUDA 内核。
#
# 基础: NVIDIA NGC PyTorch 25.10 (CUDA 13.0, PyTorch 2.9, Python 3.12, SM120 支持)
#
# 构建 (需 32GB+ 显存的 CUDA 主机, 或用非 GPU 主机绕过构建期自检):
#   docker build -t flashrt-robodojo:latest .
#   docker build --build-arg SMOKE=0 -t flashrt-robodojo:latest .   # 无 GPU 主机跳过自检
#
# 运行 (策略服务器, WebSocket):
#   docker run --gpus all --shm-size=8g \
#     -p 3101:3101 \
#     -v /path/to/model:/models/model \
#     flashrt-robodojo:latest \
#     --checkpoint /models/model --port 3101
#
# 模型 zip 解压到 /models/model 后即被 --checkpoint 加载。

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE_IMAGE}

ARG SMOKE=1

WORKDIR /opt/flashrt-robodojo-policy

# ── 项目代码 (含 vendor/FlashRT 源码 + 预编译 .so) ──
COPY . .

# ── 运行时依赖 (JAX/Orbax 前端) ──
RUN python3 -m pip install --no-cache-dir \
        "jax==0.11.0" \
        "ml-dtypes==0.5.4" \
        "orbax-checkpoint==0.12.2" \
        "flax==0.12.8" \
        "numpy==2.5.1" \
        websockets msgpack msgpack-numpy pyyaml sentencepiece pillow \
        pydantic opencv-python-headless h5py \
    && python3 -m pip install --no-cache-dir --no-build-isolation -e "/opt/flashrt-robodojo-policy/vendor/FlashRT[torch]"

# ── 自检 (可选, 无 GPU 构建时 SMOKE=0) ──
RUN if [ "${SMOKE}" = "1" ]; then \
      PYTHONPATH=/opt/flashrt-robodojo-policy/vendor/FlashRT:/opt/flashrt-robodojo-policy/vendor/xp_lib:/opt/flashrt-robodojo-policy \
      python3 -c "import flash_rt; from flash_rt import flash_rt_kernels; print('flash_rt kernels OK')"; \
    else echo "SMOKE disabled"; fi

# ── 入口: WS 策略服务器 ──
EXPOSE 3101
ENV PYTHONPATH=/opt/flashrt-robodojo-policy/vendor/FlashRT:/opt/flashrt-robodojo-policy/vendor/xp_lib:/opt/flashrt-robodojo-policy
ENV FLASHRT_PI05_STATE_PROMPT_MODE=fixed
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
CMD ["python3", "-u", "server/run_server.py", \
     "--framework", "jax", "--quantization", "fp8", "--hardware", "auto", \
     "--num-views", "3", "--action-dim", "14", "--port", "3101", "--host", "0.0.0.0"]
