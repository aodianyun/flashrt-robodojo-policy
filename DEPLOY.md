# 部署文档 — GOAI 2026 策略服务器

完整的部署流程。有两种方式: **裸机部署** 和 **Docker 自动化构建**。

## 环境要求

- GPU: NVIDIA RTX 50 系 (已验证 RTX 5060 Ti), 16GB+ 显存
- 驱动: NVIDIA 545+
- 基础镜像/环境: `nvcr.io/nvidia/pytorch:25.10-py3` (CUDA 13.0, Python 3.12)
- CMake 3.24+, GCC 11+

## 方式 A — 裸机部署

### 1. 获取项目代码

```bash
git clone https://github.com/aodianyun/flashrt-robodojo-policy.git
cd flashrt-robodojo-policy
```

### 2. 获取并编译 FlashRT

项目携带 FlashRT 源码 (commit `7fd75d2`) + 双臂补丁, 需编译 CUDA 内核:

```bash
cd vendor/FlashRT
# 拉取 FlashRT 完整源码 (含 C++ csrc) + CUTLASS
git clone https://github.com/flashrt-project/FlashRT.git /tmp/FlashRT
cd /tmp/FlashRT && git checkout 7fd75d20c0528f4c3ba70e23b9fbf579dbc68211
git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git third_party/cutlass
# 应用本项目双臂补丁
git apply /opt/flashrt-robodojo-policy/vendor/FlashRT/flashrt_goai_dualarm.patch
# 编译内核
cmake -B build -S . -DGPU_ARCH=120   # 120=RTX 50系 Blackwell
cmake --build build -j$(nproc)
# 将产物 .so 放回项目目录
cp flash_rt/*.so* <项目>/vendor/FlashRT/flash_rt/
```

### 3. 安装依赖

```bash
pip install "jax==0.11.0" "ml-dtypes==0.5.4" "orbax-checkpoint==0.12.2" \
            "flax==0.12.8" "numpy==2.5.1" \
            websockets msgpack msgpack-numpy pyyaml sentencepiece pillow \
            pydantic opencv-python-headless h5py
```

### 4. 下载模型 (魔塔)

```bash
pip install modelscope
modelscope download \
  --model cpadyun/RoboDojo-goai2026-arx_x5-joint-0-pi05-flashrt-30000 \
  --local_dir /models/model
```

### 5. 启动策略服务器

```bash
export PYTHONPATH="$(pwd)/vendor/FlashRT:$(pwd)/vendor/xp_lib:$(pwd)"
export FLASHRT_PI05_STATE_PROMPT_MODE=fixed
export FLASH_RT_PALIGEMMA_TOKENIZER="$(pwd)/vendor/FlashRT/assets/paligemma_tokenizer.model"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python3 -u server/run_server.py \
  --framework jax --quantization fp8 --hardware auto \
  --checkpoint /models/model --num-views 3 --action-dim 14 \
  --port 3101 --host 0.0.0.0
```

就绪标志:

```
INFO:client_server.ws.model_server:websocket policy server listening on ws://0.0.0.0:3101
```

## 方式 B — Docker 自动化构建

[`docker/Dockerfile`](./docker/Dockerfile) 自动完成全部流程:
获取代码 → 从魔塔下载模型 → 编译 FlashRT → 安装依赖 → 运行 server。

```bash
# 构建 (需 CUDA 工具链主机)
docker build --build-arg GPU_ARCH=120 -t flashrt-goai-robodojo-wsserver:v1.2 .

# 运行
docker run --gpus all --shm-size=8g -p 3101:3101 \
  flashrt-goai-robodojo-wsserver:v1.2
```

## 评测端连接 (RoboDojo 端)

批量评测 12 个任务 (每任务 5 次):

```bash
bash scripts/robodojo.sh client \
  --policy-name goai_flashrt \
  --policy-host <SERVER_IP> \
  --policy-port 3101 \
  --action-type joint \
  --ckpt goai_stack \
  --eval-num 5 \
  --only stack_bowls,push_T,pack_objects_into_box,fold_clothes,hang_mugs,sweep_blocks,pour_liquid_into_cup,make_toast,arrange_largest_number,sort_nesting_dolls_by_size,store_laptop_and_headphones,stack_blocks
```

> 省略 `--task` 即进入批量模式。单任务评测加 `--task stack_bowls` 即可。

评测端在容器内时, `--policy-host` 用 `host.docker.internal` (加 `--add-host=host.docker.internal:host-gateway`)。

## 故障排查

| 现象 | 处理 |
|---|---|
| 端口被占 | 换 `--port`, 并保持 client `--policy-port` 一致 |
| 显存不足 | 关闭其他 GPU 进程; 模型需 16GB+ |
| `paligemma_tokenizer.model not found` | 设置 `FLASH_RT_PALIGEMMA_TOKENIZER` 指向项目内 tokenizer |
