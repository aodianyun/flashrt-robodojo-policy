#!/usr/bin/env python3
"""GOAI FlashRT — XPolicyLab WS policy server.

Hosts the FlashRT Pi0.5 bimanual model and exposes the XPolicyLab
WebSocket protocol for the RoboDojo evaluator (robodojo.sh client).

Usage:
    python3 run_server.py --framework jax|torch \
        --quantization fp8|fp4|fp4-awq|bf16 \
        --checkpoint PATH [--port N] [--host IP] [--num-views N] [--action-dim N]

Normally launched via server/start_server.sh which sets system env and
daemonizes; this script is the foreground Python entry point.

Graceful shutdown: on SIGTERM/SIGINT the model server is stopped cleanly
(no kill -9 required).
"""
import os, sys, asyncio, argparse, pathlib, logging, signal

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("GOAI_PROJECT_ROOT", str(PROJECT_ROOT))
# Vendored FlashRT + XPolicyLab client lib. JAX stack is added by the
# model adapter when framework=jax (kept out of the torch venv which uses
# numpy 1.x and would break on jax's numpy-2 requirement).
for p in reversed((str(PROJECT_ROOT),
                   str(PROJECT_ROOT / "vendor" / "xp_lib"),
                   str(PROJECT_ROOT / "vendor" / "FlashRT"))):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import importlib
from client_server.ws.model_server import PolicyServer, PolicyServerConfig


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="GOAI FlashRT XPolicyLab WS policy server")
    ap.add_argument("--checkpoint", default=None,
                    help="explicit checkpoint path (overrides --model/--ckpt-dir)")
    ap.add_argument("--model", default="pi05-arx-x5",
                    help="model identifier (see scripts/download_checkpoint.sh)")
    ap.add_argument("--ckpt-dir", default="/data/ckpts",
                    help="checkpoint root dir; model weights live at <dir>/<model>/")
    ap.add_argument("--download-missing", action="store_true",
                    help="auto-download the checkpoint if missing")
    ap.add_argument("--port", type=int, default=3001)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--framework", choices=["jax", "torch"], default="jax",
                    help="inference framework")
    ap.add_argument("--quantization", choices=["fp8", "fp4", "fp4-awq", "bf16"],
                    default="fp8",
                    help="quantization: fp8 (default), fp4 (NVFP4), fp4-awq, bf16")
    ap.add_argument("--num-views", type=int, default=3)
    ap.add_argument("--action-dim", type=int, default=14)
    return ap.parse_args(argv)


def _resolve_checkpoint(args) -> str:
    """Explicit --checkpoint wins; otherwise <ckpt-dir>/<model>/ (auto-download optional)."""
    if args.checkpoint:
        return args.checkpoint
    resolve = PROJECT_ROOT / "scripts" / "resolve_checkpoint.sh"
    cmd = ["bash", str(resolve), args.model, args.ckpt_dir]
    if args.download_missing:
        cmd.append("--download")
    cmd += ["--format", "safetensors" if args.framework == "torch" else "orbax"]
    import subprocess
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        sys.exit(1)
    return r.stdout.strip()


def _build_model_cfg(args) -> dict:
    quant = args.quantization
    return {
        "policy_name": "goai_flashrt",
        "protocol": "ws",
        "host": args.host,
        "port": args.port,
        "action_type": "joint",
        "checkpoint_path": _resolve_checkpoint(args),
        "num_views": args.num_views,
        "action_dim": args.action_dim,
        "framework": args.framework,
        "use_fp8": quant in ("fp8", "fp4", "fp4-awq"),
        "use_fp4": quant in ("fp4", "fp4-awq"),
        "use_awq": quant == "fp4-awq",
        "weight_cache": False,
        "use_cuda_graph": False,
    }


def build_model(cfg: dict):
    mod = importlib.import_module("XPolicyLab.policy.goai_flashrt.model")
    return mod.Model(cfg)


async def _serve(args):
    cfg = _build_model_cfg(args)
    print(f"[SERVER] checkpoint: {cfg['checkpoint_path']}", flush=True)
    logging.basicConfig(level=logging.INFO)
    model = build_model(cfg)
    print(f"[SERVER] framework={args.framework} quant={args.quantization} "
          f"model loaded, starting on {args.host}:{args.port}", flush=True)
    server = PolicyServer(model, PolicyServerConfig(host=args.host, port=args.port))

    stop = asyncio.Event()

    def _request_stop(signum, frame):
        print(f"[SERVER] received signal {signum}, shutting down gracefully...",
              flush=True)
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_stop, sig, None)
        except NotImplementedError:  # pragma: no cover
            signal.signal(sig, _request_stop)

    serve_task = asyncio.create_task(server.serve_forever())
    await stop.wait()
    print("[SERVER] closing...", flush=True)
    try:
        await server.stop()
    except Exception as e:  # pragma: no cover
        print(f"[SERVER] stop warning: {e}", flush=True)
    serve_task.cancel()
    try:
        await serve_task
    except asyncio.CancelledError:
        pass
    print("[SERVER] exited cleanly", flush=True)


def main(argv=None):
    args = _parse_args(argv)
    try:
        asyncio.run(_serve(args))
    except KeyboardInterrupt:
        print("[SERVER] keyboard interrupt", flush=True)


if __name__ == "__main__":
    main()
