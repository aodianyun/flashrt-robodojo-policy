"""FlashRT — Action post-processing utilities."""

import numpy as np

LIBERO_ACTION_DIM = 7


def unnormalize_actions(actions, norm_stats, use_quantile=True):
    """Unnormalize actions using norm_stats.

    Defaults to the openpi q01/q99 quantile mapping. When ``use_quantile`` is
    False (and the norm_stats carry ``mean``/``std``), uses the linear
    ``x * std + mean`` mapping that openpi's ``Unnormalize`` applies at
    inference time. For policies whose actions were normalized as *deltas*
    against a quantile range that does not match the action distribution
    (e.g. RoboDojo bimanual pi0.5 with DeltaActions), use_quantile=False is
    required to reproduce openpi bit-for-bit.
    """
    s = norm_stats["actions"]
    dim = min(actions.shape[-1], len(s["q01"]))
    if not use_quantile and "std" in s and "mean" in s:
        mean = np.array(s["mean"], dtype=np.float32)
        std = np.array(s["std"], dtype=np.float32)
        unnorm = actions.copy().astype(np.float32)
        unnorm[..., :dim] = unnorm[..., :dim] * (std[:dim] + 1e-6) + mean[:dim]
        return unnorm
    q01 = np.array(s["q01"], dtype=np.float32)
    q99 = np.array(s["q99"], dtype=np.float32)
    # IMPORTANT: no clip to [-1,1] — openpi's _unnormalize_quantile does not
    # clip, and flow-matching intermediate values can exceed [-1,1]. Clipping
    # truncates large-magnitude action dims (e.g. one arm) and desynchronizes
    # bimanual outputs vs openpi.
    unnorm = actions.copy().astype(np.float32)
    unnorm[..., :dim] = (
        (actions[..., :dim] + 1.0) / 2.0 * (q99[:dim] - q01[:dim] + 1e-6)
        + q01[:dim]
    )
    return unnorm
