#!/usr/bin/env python3
"""XPolicyLab policy adapter that serves a FlashRT Pi0.5 bimanual model.

Hosted by XPolicyLab's WebSocket policy server (client_server/ws/model_server.py).
Implements the ModelTemplate surface:

    update_obs(obs) / update_obs_batch(obs_list)  -> cache observation
    get_action() / get_action_batch(env_idx_list) -> XPolicyLab dual-arm joint dicts
    reset()                                        -> reset frontend state

The underlying FlashRT Pi05JaxFrontendThor loads a cjgogo/openpi Orbax
checkpoint and emits 14-dim bimanual action chunks which are unpacked into the
XPolicyLab dual-arm joint action schema:
    {left_arm_joint_state(6), left_ee_joint_state(1),
     right_arm_joint_state(6), right_ee_joint_state(1)}

Usage (server side):
    bash server/start_server.sh --quantization fp8
"""
from __future__ import annotations

import os
import pathlib

import numpy as np

from XPolicyLab.model_template import ModelTemplate

# FlashRT + JAX stack — prefer the project's vendored copies.
# GOAI_PROJECT_ROOT is exported by server/start_server.sh.
_PROJ_ROOT = os.environ.get("GOAI_PROJECT_ROOT")
if not _PROJ_ROOT:
    # Fallback: this module lives at <root>/vendor/xp_lib/XPolicyLab/policy/goai_flashrt/
    _PROJ_ROOT = str(pathlib.Path(__file__).resolve().parents[5])
_FLASHRT_PATHS = [
    os.path.join(_PROJ_ROOT, "vendor", "FlashRT"),
]
sys_path = __import__("sys")
# insert at FRONT in priority order (first item wins).
for _p in reversed(_FLASHRT_PATHS):
    if _p and _p not in sys_path.path:
        sys_path.path.insert(0, _p)

from flash_rt.core.utils.norm_stats import load_norm_stats  # noqa: E402

# JAX frontends imported lazily in __init__ (only when framework=jax), so
# the torch venv (numpy 1.x) is not forced to load jax (which needs numpy 2.x).
Pi05JaxFrontendThor = None
Pi05JaxFrontendThorFP4 = None

ARM_DIM = 6
EE_DIM = 1
ACTION_DIM = 2 * ARM_DIM + 2 * EE_DIM  # 14
IMG_SIZE = 224


def _resize_with_pad(img, height=224, width=224):
    """openpi's resize_with_pad: keep aspect ratio, pad with black.

    Mirrors openpi/shared/image_tools.resize_with_pad (jax) used by the
    model's preprocess_observation. Without this the stretched image changes
    the spatial layout the policy was trained on (especially bimanual layout),
    which degrades/breaks dual-arm predictions.
    """
    from PIL import Image
    h, w = img.shape[:2]
    ratio = max(w / width, h / height)
    rh, rw = int(h / ratio), int(w / ratio)
    resized = np.asarray(Image.fromarray(img.astype(np.uint8)).resize(
        (rw, rh), Image.BILINEAR))
    out = np.zeros((height, width, 3), dtype=np.uint8)
    y0 = max(0, (height - rh) // 2)
    x0 = max(0, (width - rw) // 2)
    out[y0:y0 + rh, x0:x0 + rw] = resized
    return out


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.action_type = model_cfg.get("action_type", "joint")

        ckpt = model_cfg.get("checkpoint_path") or model_cfg.get(
            "model_path") or model_cfg.get("ckpt_name")
        if not ckpt:
            raise ValueError("model_cfg needs checkpoint_path/model_path/ckpt_name")
        ckpt = str(ckpt)

        num_views = int(model_cfg.get("num_views", 3))
        action_dim = int(model_cfg.get("action_dim", ACTION_DIM))

        # Quantization / framework selection — all via model_cfg (CLI params).
        framework = str(model_cfg.get("framework", "jax")).lower()
        self._framework = framework
        use_fp8 = bool(model_cfg.get("use_fp8", True))
        use_fp4 = bool(model_cfg.get("use_fp4", False))
        use_awq = bool(model_cfg.get("use_awq", True))
        print(f"[GOAI] framework={framework} fp8={use_fp8} fp4={use_fp4} "
              f"awq={use_awq}", flush=True)

        if framework == "jax":
            global Pi05JaxFrontendThor, Pi05JaxFrontendThorFP4
            from flash_rt.frontends.jax.pi05_thor import Pi05JaxFrontendThor  # noqa: E402
            try:  # NVFP4 path (optional, requires flash_rt_fp4 extension)
                from flash_rt.frontends.jax.pi05_thor_fp4 import Pi05JaxFrontendThorFP4  # noqa: E402
            except Exception:  # pragma: no cover
                Pi05JaxFrontendThorFP4 = None

        if framework == "torch":
            try:
                import torch  # noqa: F401
            except ImportError:
                raise RuntimeError(
                    "framework=torch requires torch installed (use the torch venv, "
                    "see vendor/venv-torch/install_torch_venv.sh)")
            from flash_rt.frontends.torch.pi05_thor import Pi05TorchFrontendThor
            self._pipe = Pi05TorchFrontendThor(
                ckpt,
                num_views=num_views,
                autotune=0,
                use_cuda_graph=bool(model_cfg.get("use_cuda_graph", False)),
                use_fp8=use_fp8,
                state_prompt_mode="fixed",
                action_dim=action_dim,
            )
        elif use_fp4 and Pi05JaxFrontendThorFP4 is not None:
            print(f"[GOAI] NVFP4 encoder-FFN path (Pi05JaxFrontendThorFP4, awq={use_awq})",
                  flush=True)
            self._pipe = Pi05JaxFrontendThorFP4(
                ckpt,
                num_views=num_views,
                autotune=0,
                use_cuda_graph=bool(model_cfg.get("use_cuda_graph", False)),
                use_fp8=use_fp8,
                weight_cache=bool(model_cfg.get("weight_cache", False)),
                action_dim=action_dim,
                use_fp4_encoder_ffn=True,
                fp4_layers=tuple(range(17)),   # all 17 live encoder FFN layers
                use_awq=use_awq,
                awq_alpha=float(model_cfg.get("awq_alpha", 0.5)),
                awq_calib_iters=int(model_cfg.get("awq_calib_iters", 8)),
                use_p1_split_gu=False,
            )
        else:
            self._pipe = Pi05JaxFrontendThor(
                ckpt,
                num_views=num_views,
                autotune=0,
                use_cuda_graph=bool(model_cfg.get("use_cuda_graph", False)),
                use_fp8=use_fp8,
                weight_cache=bool(model_cfg.get("weight_cache", False)),
                action_dim=action_dim,
            )

        # Inject the RoboDojo ARX-X5 bimanual norm stats.
        # - JAX frontend (orbax checkpoint): norm_stats.json is a separate
        #   asset; inject it here.
        # - Torch frontend (LeRobot safetensors): FlashRT already loads the
        #   LeRobot normalizer/unnormalizer processors; skip injection and
        #   reuse the frontend's stats.
        if framework == "torch":
            ns = getattr(self._pipe, "norm_stats", None)
            if ns is None:
                raise RuntimeError(
                    "torch frontend did not load norm stats (expected LeRobot "
                    "preprocessor/postprocessor safetensors)")
        else:
            ns_candidates = [
                pathlib.Path(ckpt) / "assets/arx_x5_sim/norm_stats.json",
                pathlib.Path(ckpt) / "assets" / "norm_stats.json",
                pathlib.Path(ckpt) / "norm_stats.json",
            ]
            ns = load_norm_stats(ns_candidates, strict=False)
            if ns is None:
                raise RuntimeError(
                    f"norm_stats not found; tried {[str(p) for p in ns_candidates]}")
            self._pipe.norm_stats = ns
        # openpi pi0.5 unnormalizes actions with q01/q99 (use_quantile_norm).
        if hasattr(self._pipe, "_use_quantile_unnorm"):
            self._pipe._use_quantile_unnorm = True
        # State normalization stats (openpi schema: state.q01/q99 quantiles).
        self._state_stats = (ns.get("state")
                             if isinstance(ns.get("state"), dict)
                             and "q01" in ns.get("state", {}) else None)
        # openpi's Policy.infer already applies the full output transform chain
        # (Unnormalize + AbsoluteActions), so the model output is already an
        # absolute target pose. FlashRT's unnormalize produces the same
        # absolute output; we must NOT add the state again (that double-adds
        # and drives the arm off to infinity).
        # Delta vs absolute actions: default True (RoboDojo official model uses
        # DeltaActions). LeRobot checkpoints carry config.json with
        # use_relative_actions — False means the model outputs absolute targets,
        # so we must NOT add the current state again (double-add drives the arm
        # off to infinity). CLI --use-delta-actions overrides auto-detection.
        use_delta_actions = model_cfg.get("use_delta_actions")
        if use_delta_actions is None:
            use_delta_actions = True  # default (RoboDojo official)
            cfg_json = pathlib.Path(ckpt) / "config.json"
            if cfg_json.exists():
                try:
                    import json as _json
                    c = _json.loads(cfg_json.read_text())
                    if "use_relative_actions" in c:
                        use_delta_actions = bool(c["use_relative_actions"])
                except Exception:
                    pass
        self._use_delta_actions = bool(use_delta_actions)
        print(f"[GOAI] delta_actions={self._use_delta_actions} (from config.json)",
              flush=True)
        self._last_raw_state = None
        self._num_views = num_views
        self._prompt = None
        self._obs_window = None

    # ── observation ingestion ──
    def update_obs(self, obs):
        if not getattr(self, "_obs_dumped", False):
            self._obs_dumped = True
            try:
                import json as _json
                vision = obs.get("vision", {})
                cam_desc = {}
                for k, v in (vision or {}).items():
                    if isinstance(v, dict):
                        for ik, iv in v.items():
                            if hasattr(iv, "shape"):
                                cam_desc[f"{k}.{ik}"] = list(iv.shape)
                            else:
                                cam_desc[f"{k}.{ik}"] = str(type(iv).__name__)
                    else:
                        cam_desc[k] = str(type(v).__name__)
                st = obs.get("state")
                st_desc = {}
                if isinstance(st, dict):
                    for k, v in st.items():
                        st_desc[k] = (list(np.asarray(v).shape)
                                      if hasattr(v, "shape") or isinstance(v, (list, np.ndarray))
                                      else str(type(v).__name__))
                print(f"[OBS-DUMP] keys={list(obs.keys())} "
                      f"vision={_json.dumps(cam_desc)} state={_json.dumps(st_desc)} "
                      f"instruction={str(obs.get('instruction'))[:80]!r}", flush=True)
            except Exception as e:
                print(f"[OBS-DUMP-ERR] {e}", flush=True)
        self._obs_window = self._encode(obs)

    def update_obs_batch(self, obs_list):
        if not getattr(self, "_obs_batch_dumped", False) and obs_list:
            self._obs_batch_dumped = True
            try:
                import json as _json
                o = obs_list[0]
                vision = o.get("vision", {})
                cam_desc = {}
                for k, v in (vision or {}).items():
                    if isinstance(v, dict):
                        for ik, iv in v.items():
                            if hasattr(iv, "shape"):
                                cam_desc[f"{k}.{ik}"] = list(iv.shape)
                            else:
                                cam_desc[f"{k}.{ik}"] = str(type(iv).__name__)
                    else:
                        cam_desc[k] = str(type(v).__name__)
                st = o.get("state")
                st_desc = {}
                if isinstance(st, dict):
                    for k, v in st.items():
                        st_desc[k] = (list(np.asarray(v).shape)
                                      if isinstance(v, (list, np.ndarray))
                                      else str(type(v).__name__))
                print(f"[OBS-BATCH-DUMP] n={len(obs_list)} "
                      f"keys={list(o.keys())} vision={_json.dumps(cam_desc)} "
                      f"state={_json.dumps(st_desc)} "
                      f"instruction={str(o.get('instruction'))[:80]!r}", flush=True)
            except Exception as e:
                print(f"[OBS-BATCH-DUMP-ERR] {e}", flush=True)
        self._obs_window = [self._encode(o) for o in obs_list]

    def _encode(self, obs: dict) -> dict:
        vision = obs.get("vision", {})
        base = self._extract(vision, ["cam_high", "cam_head", "head_camera", "top_camera"])
        left = self._extract(vision, ["cam_left_wrist", "left_wrist", "left_camera"])
        right = self._extract(vision, ["cam_right_wrist", "right_wrist", "right_camera"])
        state = self._extract_state(obs)
        # Keep the raw (physical-unit) state for delta->absolute action recovery.
        if state is not None:
            self._last_raw_state = self._extract_raw_state(obs)
        else:
            self._last_raw_state = None
        if not getattr(self, "_state_dumped", False):
            self._state_dumped = True
            try:
                st = obs.get("state")
                print(f"[STATE-DUMP] raw_state={np.round(self._last_raw_state,4).tolist() if self._last_raw_state is not None else None} "
                      f"normalized={np.round(state,4).tolist() if state is not None else None} "
                      f"st_keys={list(st.keys()) if isinstance(st,dict) else type(st).__name__}", flush=True)
            except Exception as e:
                print(f"[STATE-DUMP-ERR] {e}", flush=True)
        return {
            "image": base,
            "wrist_image": left if left is not None else base,
            "wrist_image_right": right if right is not None else base,
            "instruction": obs.get("instruction"),
            "state": state,
        }

    def _extract_raw_state(self, obs: dict):
        """Raw (unnormalized) 14-dim state [arm0(6), ee0(1), arm1(6), ee1(1)]."""
        st = obs.get("state")
        if not isinstance(st, dict):
            return None
        try:
            raw = np.concatenate([
                np.asarray(st["left_arm_joint_state"], dtype=np.float32).reshape(-1),
                np.asarray(st["left_ee_joint_state"], dtype=np.float32).reshape(-1),
                np.asarray(st["right_arm_joint_state"], dtype=np.float32).reshape(-1),
                np.asarray(st["right_ee_joint_state"], dtype=np.float32).reshape(-1),
            ])
        except (KeyError, ValueError):
            return None
        return raw if raw.shape[0] == ACTION_DIM else None

    def _extract_state(self, obs: dict):
        """Build the 14-dim bimanual state vector and normalize it.

        RoboDojo eval obs['state'] uses keys:
          left_arm_joint_state(6), left_ee_joint_state(1),
          right_arm_joint_state(6), right_ee_joint_state(1)
        Packed order must match training: [arm0, ee0, arm1, ee1].
        The openpi convention feeds a *normalized* state to the model
        (state prompt tokens are 256 bins over [-1, 1]).
        """
        st = obs.get("state")
        if not isinstance(st, dict):
            return None
        try:
            raw = np.concatenate([
                np.asarray(st["left_arm_joint_state"], dtype=np.float32).reshape(-1),
                np.asarray(st["left_ee_joint_state"], dtype=np.float32).reshape(-1),
                np.asarray(st["right_arm_joint_state"], dtype=np.float32).reshape(-1),
                np.asarray(st["right_ee_joint_state"], dtype=np.float32).reshape(-1),
            ])
        except (KeyError, ValueError) as exc:
            # Some eval clients emit unprefixed single-arm keys; try them.
            try:
                raw = np.concatenate([
                    np.asarray(st["arm_joint_state"], dtype=np.float32).reshape(-1),
                    np.asarray(st["ee_joint_state"], dtype=np.float32).reshape(-1),
                ])
            except (KeyError, ValueError):
                return None
        if raw.shape[0] != ACTION_DIM:
            return None
        # Normalize state per framework/format:
        # - JAX (openpi/RoboDojo orbax): q01/q99 quantile mapping.
        # - Torch (LeRobot safetensors): mean/std (LeRobot Normalize default).
        if self._state_stats is not None:
            if getattr(self, "_framework", "jax") == "torch" and "std" in self._state_stats:
                s_mean = np.asarray(self._state_stats["mean"], dtype=np.float32)
                s_std = np.asarray(self._state_stats["std"], dtype=np.float32)
                raw = (raw - s_mean) / (s_std + 1e-6)
            elif "q01" in self._state_stats and "q99" in self._state_stats:
                q01 = np.asarray(self._state_stats["q01"], dtype=np.float32)
                q99 = np.asarray(self._state_stats["q99"], dtype=np.float32)
                raw = 2.0 * (raw - q01) / (q99 - q01 + 1e-6) - 1.0
        return raw.astype(np.float32)

    @staticmethod
    def _extract(vision: dict, candidates):
        for name in candidates:
            cam = vision.get(name)
            if cam is None:
                continue
            img = cam["color"] if isinstance(cam, dict) and "color" in cam else (
                cam["rgb"] if isinstance(cam, dict) and "rgb" in cam else cam)
            a = np.asarray(img)
            if a.ndim == 3 and a.shape[-1] not in (3, 4) and a.shape[0] == 3:
                a = np.transpose(a, (1, 2, 0))
            if a.ndim == 3 and (a.shape[0], a.shape[1]) != (IMG_SIZE, IMG_SIZE):
                a = _resize_with_pad(a, IMG_SIZE, IMG_SIZE)
            return a
        return None
    # ── action generation ──
    def get_action(self):
        if self._obs_window is None:
            raise RuntimeError("update_obs must be called before get_action")
        fe = self._obs_window
        self._pipe.set_prompt(fe.get("instruction") or self._prompt or "",
                              state=fe.get("state"))
        out = self._pipe.infer(fe)
        acts = self._to_xp_actions(out["actions"])
        self._apply_absolute_actions(acts)
        self._log_actions("get_action", acts, fe)
        return acts

    def get_action_batch(self, env_idx_list=None):
        if self._obs_window is None:
            raise RuntimeError("update_obs_batch must be called first")
        windows = self._obs_window if isinstance(self._obs_window, list) else [self._obs_window]
        envs = env_idx_list if env_idx_list is not None else list(range(len(windows)))
        results = []
        for idx in envs:
            fe = windows[idx] if idx < len(windows) else windows[0]
            self._pipe.set_prompt(fe.get("instruction") or self._prompt or "",
                                  state=fe.get("state"))
            out = self._pipe.infer(fe)
            acts = self._to_xp_actions(out["actions"])
            self._apply_absolute_actions(acts)
            self._log_actions("get_action_batch", acts, fe, env=idx)
            results.append(acts)
        return results

    def _apply_absolute_actions(self, acts):
        """Convert model delta joint actions to absolute target positions.

        openpi trains with DeltaActions(mask=(6,-1,6,-1)): joint dims
        0:6 and 7:13 are deltas relative to the current joint state, gripper
        dims 6 and 13 are already absolute. At inference the deltas must be
        added back to the current (raw, physical-unit) joint state.
        """
        if not getattr(self, "_use_delta_actions", True):
            return
        cur = getattr(self, "_last_raw_state", None)
        if cur is None or cur.shape[0] != ACTION_DIM:
            return
        for a in acts:
            a["left_arm_joint_state"] = (np.asarray(a["left_arm_joint_state"]) + cur[0:6]).astype(np.float32)
            a["right_arm_joint_state"] = (np.asarray(a["right_arm_joint_state"]) + cur[7:13]).astype(np.float32)

    def _log_actions(self, tag, acts, fe, env=None):
        try:
            a0 = np.array(acts[0]["left_arm_joint_state"], dtype=np.float32)
            r0 = np.array(acts[0]["right_arm_joint_state"], dtype=np.float32)
            cur = getattr(self, "_last_raw_state", None)
            if cur is not None:
                dL = np.abs(a0 - cur[0:6]).mean()
            else:
                dL = float("nan")
            cur_s = "none" if cur is None else np.round(cur[:6], 3).tolist()
            print(f"[ACT-{tag}] env={env} left0={np.round(a0,3).tolist()} "
                  f"right0={np.round(r0,3).tolist()} |delta_left|={dL:.3f} "
                  f"state_left={cur_s} st={'none' if fe.get('state') is None else '14'}", flush=True)
        except Exception:
            pass

    def _to_xp_actions(self, actions: np.ndarray):
        actions = np.asarray(actions)
        result = []
        for row in actions:
            result.append({
                "left_arm_joint_state": row[0:ARM_DIM].astype(np.float32),
                "left_ee_joint_state": row[ARM_DIM:ARM_DIM + EE_DIM].astype(np.float32),
                "right_arm_joint_state": row[ARM_DIM + EE_DIM:2 * ARM_DIM + EE_DIM].astype(np.float32),
                "right_ee_joint_state": row[2 * ARM_DIM + EE_DIM:].astype(np.float32),
            })
        return result

    def reset(self):
        self._obs_window = None
        self._prompt = None
