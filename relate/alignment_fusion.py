"""
Alignment Fusion Transformer (AFT), paper Sec. 3.2 and Fig. 2 (right).

The AFT is RELATE's core fusion module. It:
  1. Derives *action anchors* for each modality by average-pooling the
     query embeddings over each predicted segment (Sec. 3.2, paragraph
     defining ``a_i``).
  2. Uses Action-Gated Attention to compute a per-channel, per-frame
     alignment score between frame features and the action anchors, and
     gates the value embeddings with it (Eq. 1):

         A = sigmoid( sum_i (K ⊙ a_i) / sqrt(d) )
         Z~ = A ⊙ V

  3. Concatenates the gated RGB/OF features, normalizes, and runs a final
     self-attention + FFN layer to learn cross-modal importance before
     handing the fused feature to the joint segmenter.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import AttLayer


def action_anchor_pooling(stage_pred: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """
    Average-pool ``query`` over each predicted action segment to obtain the
    action anchors a_i (Sec. 3.2). Anchors are derived from *predicted*
    segments (not ground truth) so that training aligns with inference.

    Args:
        stage_pred: (1, num_classes, L) logits/probabilities used only to
            derive segment boundaries via argmax.
        query: (1, D, L) query embedding to pool.
    Returns:
        (1, D, S) pooled anchors, one per predicted segment.
    """
    device = query.device
    _, _, length = stage_pred.shape
    pred_classes = stage_pred.argmax(dim=1).squeeze(0)
    changes = pred_classes[1:] != pred_classes[:-1]
    change_indices = torch.nonzero(changes).squeeze(1) + 1
    change_indices = torch.cat(
        [torch.tensor([0], device=device), change_indices.to(device), torch.tensor([length], device=device)]
    )

    pooled = []
    for i in range(len(change_indices) - 1):
        start, end = change_indices[i].item(), change_indices[i + 1].item()
        pooled.append(query[:, :, start:end].mean(dim=2))
    return torch.stack(pooled, dim=2)  # (1, D, S)


class AlignmentFusionTransformer(nn.Module):
    """
    Fuses modality-specific RGB/OF features using action-gated attention
    followed by cross-modal self-attention (Sec. 3.2, Fig. 2 right).
    """

    def __init__(self, num_f_maps: int, device, window_size: int = 32):
        super().__init__()
        self.q_rgb = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.k_rgb = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.v_rgb = nn.Conv1d(num_f_maps, num_f_maps, 1)

        self.q_flow = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.k_flow = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.v_flow = nn.Conv1d(num_f_maps, num_f_maps, 1)

        self.instance_norm_rgb = nn.InstanceNorm1d(num_f_maps, track_running_stats=False)
        self.instance_norm_flow = nn.InstanceNorm1d(num_f_maps, track_running_stats=False)
        self.instance_norm_joint = nn.InstanceNorm1d(num_f_maps * 2, track_running_stats=False)

        # Self-attention + FFN over the concatenated, gated features
        # (Fig. 2 right: "Self-att & FFN" after "Concat.").
        self.self_attention = AttLayer(
            q_dim=num_f_maps * 2, k_dim=num_f_maps * 2, v_dim=num_f_maps * 2,
            r1=2, r2=2, r3=2, bl=window_size, stage="encoder", att_type="sliding_att", device=device,
        )
        self.conv_out = nn.Conv1d(num_f_maps * 2, num_f_maps, 1)
        self.sigmoid = torch.sigmoid

    def _action_gated_attention(self, feat, q_proj, k_proj, v_proj, stage_pred):
        """Eq. 1: A = sigmoid(sum_i K⊙a_i / sqrt(d)), Z~ = A ⊙ V."""
        anchors = action_anchor_pooling(stage_pred, q_proj(feat))  # (1, D, S)
        key = k_proj(feat)  # (1, D, L)

        gate = torch.zeros_like(key)
        for i in range(anchors.shape[2]):
            gate = gate + (anchors[:, :, i].unsqueeze(2) * key) / math.sqrt(anchors.shape[1])
        gate = self.sigmoid(gate)

        value = v_proj(feat)
        return value * gate  # Z~ (1, D, L)

    def forward(self, rgb_feat, flow_feat, p_rgb, p_flow, mask):
        """
        Args:
            rgb_feat, flow_feat: (1, D, L) modality-specific features
                (Z_RGB, Z_OF) from the last layer of each segmenter.
            p_rgb, p_flow: (1, num_classes, L) modality-specific stage
                predictions used to derive action anchors.
            mask: (1, C, L) padding mask.
        Returns:
            (1, D, L) fused feature, ready for the joint segmenter.
        """
        rgb_feat = self.instance_norm_rgb(rgb_feat)
        flow_feat = self.instance_norm_flow(flow_feat)

        rgb_mod = self._action_gated_attention(rgb_feat, self.q_rgb, self.k_rgb, self.v_rgb, p_rgb)
        flow_mod = self._action_gated_attention(flow_feat, self.q_flow, self.k_flow, self.v_flow, p_flow)

        fused = torch.cat([rgb_mod, flow_mod], dim=1)
        fused = self.instance_norm_joint(fused)
        return self.conv_out(self.self_attention(fused, None, mask))


class DotProductAlignmentFusion(nn.Module):
    """
    Dot-product variant of the AFT: anchors attend over frame keys with
    softmax instead of the Hadamard-product gate. Used as the
    "cross-attention" row of the fusion-mechanism ablation (Table 3).
    """

    def __init__(self, num_f_maps: int, device, window_size: int = 32):
        super().__init__()
        self.q_rgb = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.k_rgb = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.v_rgb = nn.Conv1d(num_f_maps, num_f_maps, 1)

        self.q_flow = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.k_flow = nn.Conv1d(num_f_maps, num_f_maps, 1)
        self.v_flow = nn.Conv1d(num_f_maps, num_f_maps, 1)

        self.instance_norm_rgb = nn.InstanceNorm1d(num_f_maps, track_running_stats=False)
        self.instance_norm_flow = nn.InstanceNorm1d(num_f_maps, track_running_stats=False)
        self.instance_norm_joint = nn.InstanceNorm1d(num_f_maps * 2, track_running_stats=False)

        self.self_attention = AttLayer(
            q_dim=num_f_maps * 2, k_dim=num_f_maps * 2, v_dim=num_f_maps * 2,
            r1=2, r2=2, r3=2, bl=window_size, stage="encoder", att_type="sliding_att", device=device,
        )
        self.conv_out = nn.Conv1d(num_f_maps * 2, num_f_maps, 1)

    def _dot_product_modulation(self, feat, q_proj, k_proj, v_proj, stage_pred):
        anchors = action_anchor_pooling(stage_pred, q_proj(feat))  # (1, D, S) treated as keys
        query = k_proj(feat)  # (1, D, L)
        attn = torch.bmm(query.transpose(1, 2), anchors) / math.sqrt(query.shape[1])  # (1, L, S)
        weights = F.softmax(attn, dim=-1)
        value = action_anchor_pooling(stage_pred, v_proj(feat))  # (1, D, S)
        return torch.bmm(value, weights.transpose(1, 2))  # (1, D, L)

    def forward(self, rgb_feat, flow_feat, p_rgb, p_flow, mask):
        rgb_feat = self.instance_norm_rgb(rgb_feat)
        flow_feat = self.instance_norm_flow(flow_feat)

        rgb_mod = self._dot_product_modulation(rgb_feat, self.q_rgb, self.k_rgb, self.v_rgb, p_rgb)
        flow_mod = self._dot_product_modulation(flow_feat, self.q_flow, self.k_flow, self.v_flow, p_flow)

        fused = torch.cat([rgb_mod, flow_mod], dim=1)
        fused = self.instance_norm_joint(fused)
        return self.conv_out(self.self_attention(fused, None, mask))


def naive_concat_fusion(rgb_feat: torch.Tensor, flow_feat: torch.Tensor) -> torch.Tensor:
    """Naive concatenation fusion baseline (Table 3, "Naive" row)."""
    return torch.cat([rgb_feat, flow_feat], dim=1)
