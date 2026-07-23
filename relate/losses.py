"""
Training objective (paper Sec. 3.3):

    L = L_RGB + L_OF + L_J + gamma * L_CMKD

``L_RGB``, ``L_OF``, ``L_J`` are per-stage cross-entropy + temporal
smoothness losses (``segmentation_loss``), and ``L_CMKD`` is one of the
knowledge-distillation losses below. RELATE's main results use
instance-level logit distillation; Table 4 additionally reports the
relational-level alternatives (RKD, GCR) implemented here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def segmentation_loss(stage_logits: torch.Tensor, num_classes: int, batch_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Per-stage cross-entropy + truncated temporal-smoothness loss, summed
    over stages (used for L_RGB, L_OF, and L_J).

    Args:
        stage_logits: (num_stages, B, C, L) logits.
        batch_target: (B, L) integer class targets (``-100`` = ignore).
        mask: (B, C, L) padding mask.
    """
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    mse = nn.MSELoss(reduction="none")

    total_loss = 0.0
    for pred in stage_logits:
        ce_loss = ce(pred.transpose(2, 1).contiguous().view(-1, num_classes), batch_target.view(-1))
        smooth_loss = 0.15 * torch.mean(
            torch.clamp(
                mse(F.log_softmax(pred[:, :, 1:], dim=1), F.log_softmax(pred.detach()[:, :, :-1], dim=1)),
                min=0,
                max=16,
            )
            * mask[:, :, 1:]
        )
        total_loss = total_loss + ce_loss + smooth_loss
    return total_loss


def logit_distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """
    Instance-level KD loss (Gupta et al., 2016), RELATE's main CMKD
    objective (Table 4, "KD" / "RELATE + KD" rows).
    """
    p_student = F.log_softmax(student_logits / temperature, dim=1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(p_student, p_teacher, reduction="batchmean")


def rkd_distance_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Relational KD via pairwise L2 distance between frame features
    (Park et al., 2019), Table 4 "RKD" row.

    Args:
        student_feat, teacher_feat: (1, C, T) feature tensors.
    """
    s = student_feat.squeeze(0).transpose(0, 1)  # (T, C)
    t = teacher_feat.squeeze(0).transpose(0, 1)

    with torch.no_grad():
        t_dist = torch.cdist(t, t, p=2)
        t_dist = t_dist / (t_dist[t_dist > 0].mean() + eps)

    s_dist = torch.cdist(s, s, p=2)
    s_dist = s_dist / (s_dist[s_dist > 0].mean() + eps)
    return F.mse_loss(s_dist, t_dist)


def _covariance_upper_triangular(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Channel-wise covariance of a (1, C, T) feature, upper-triangular part."""
    feat = x.squeeze(0)  # (C, T)
    feat = feat - feat.mean(dim=1, keepdim=True)
    cov = (feat @ feat.transpose(0, 1)) / (feat.shape[1] - 1 + eps)
    iu = torch.triu_indices(cov.shape[0], cov.shape[1], offset=1)
    return cov[iu[0], iu[1]]


def global_covariance_relation_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """
    Global contextual relation distillation via channel covariance
    (Dai et al., 2021), Table 4 "GCR" row.
    """
    s_cov = _covariance_upper_triangular(student_feat)
    with torch.no_grad():
        t_cov = _covariance_upper_triangular(teacher_feat)
    return F.mse_loss(s_cov, t_cov)
