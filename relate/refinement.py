"""
Training-free prediction refinement (paper Sec. 3.4).

When the joint prediction's confidence falls below a threshold, this
replaces low-confidence frames with the most reliable unimodal prediction
from the previous timestep, or falls back to the previous joint prediction
otherwise -- resolving ambiguous segments without any extra training.
"""

import torch


def refine_joint_prediction(
    rgb_pred: torch.Tensor,
    flow_pred: torch.Tensor,
    joint_pred: torch.Tensor,
    threshold: float = 0.7,
) -> torch.Tensor:
    """
    Args:
        rgb_pred, flow_pred, joint_pred: (T, C) per-frame class probabilities
            for the RGB branch, flow branch, and joint prediction.
        threshold: confidence threshold delta (Sec. 3.4; default matches
            the paper's delta=0.7, Sec. 4.1.2).
    Returns:
        (T, C) refined joint prediction.
    """
    modalities = torch.stack([rgb_pred, flow_pred, joint_pred])  # (3, T, C)
    mod_min = modalities.amin(dim=(1, 2), keepdim=True)
    mod_max = modalities.amax(dim=(1, 2), keepdim=True)
    modalities = (modalities - mod_min) / (mod_max - mod_min + 1e-8)

    T = modalities.shape[1]
    refined_joint = joint_pred.clone()
    prev_mod_conf = torch.max(modalities, dim=2).values  # (3, T)

    JOINT_IDX = 2
    for t in range(T):
        mod_conf = torch.max(modalities[:, t, :], dim=1).values
        joint_conf = torch.max(refined_joint[t, :], dim=0).values
        lowest_mod_idx = torch.argmin(mod_conf)

        if joint_conf < threshold:
            if lowest_mod_idx == JOINT_IDX:
                # The joint prediction is itself the least confident modality:
                # fall back to whichever unimodal prediction was most
                # confident at the previous timestep.
                best_prev_mod_idx = torch.argmax(prev_mod_conf[:, t - 1]) if t > 0 else 0
                refined_joint[t, :] = modalities[best_prev_mod_idx, t - 1, :]
            elif t > 0:
                # No unimodal alternative is clearly superior: retain the
                # previous joint prediction.
                refined_joint[t, :] = refined_joint[t - 1, :]

    return refined_joint
