"""
Shared building blocks used by the MS-TCN and ASFormer backbones.

These are the standard temporal-convolution and windowed-self-attention
primitives from Farha & Gall (MS-TCN, CVPR 2019) and Yi et al. (ASFormer,
BMVC 2021), the two backbones RELATE is evaluated on (Sec. 4.1.2). They are
kept close to the original public implementations so that RELATE's
components (Sections 3.1-3.4) can be swapped in without changing the
underlying segmenter behaviour.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHelper(nn.Module):
    """Masked scaled dot-product attention over (B, C, L) tensors."""

    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        """
        Args:
            proj_query, proj_key, proj_val: (B, C, L) tensors.
            padding_mask: (B, 1, L) tensor with 1 for valid frames.
        Returns:
            out: (B, C, L) attended values.
            attention: (B, L, L) attention weights.
        """
        _, c1, _ = proj_query.shape
        _, c2, _ = proj_key.shape
        assert c1 == c2

        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # (B, L1, L2)
        attention = energy / np.sqrt(c1)
        # Mask padded frames by pushing their logits to -inf before softmax.
        attention = attention + torch.log(padding_mask + 1e-6)

        attention = self.softmax(attention)
        attention = attention * padding_mask
        attention = attention.permute(0, 2, 1)
        out = torch.bmm(proj_val, attention)
        return out, attention


class AttLayer(nn.Module):
    """
    Single-head windowed self/cross-attention layer.

    Supports three attention patterns (``att_type``):
      * ``normal_att``  — full self-attention over the whole sequence.
      * ``block_att``   — attention restricted to non-overlapping blocks.
      * ``sliding_att`` — attention over a local sliding window (used by
        ASFormer, and by RELATE's Alignment Fusion Transformer).
    """

    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, device):
        super().__init__()
        self.query_conv = nn.Conv1d(q_dim, q_dim // r1, 1)
        self.key_conv = nn.Conv1d(k_dim, k_dim // r2, 1)
        self.value_conv = nn.Conv1d(v_dim, v_dim // r3, 1)
        self.conv_out = nn.Conv1d(v_dim // r3, v_dim, 1)

        self.device = device
        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        assert self.att_type in ["normal_att", "block_att", "sliding_att"]
        assert self.stage in ["encoder", "decoder"]

        self.att_helper = AttentionHelper()
        self.window_mask = self._construct_window_mask()

    def _construct_window_mask(self):
        """Window mask of shape (1, bl, bl + bl) for sliding-window attention."""
        window_mask = torch.zeros((1, self.bl, self.bl + 2 * (self.bl // 2)), device=self.device)
        for i in range(self.bl):
            window_mask[:, :, i : i + self.bl] = 1
        return window_mask

    def forward(self, x1, x2, mask):
        # x1: features to attend from (query/key source).
        # x2: features to attend to for values, when stage == "decoder".
        query = self.query_conv(x1)
        key = self.key_conv(x1)

        if self.stage == "decoder":
            assert x2 is not None
            value = self.value_conv(x2)
        else:
            value = self.value_conv(x1)

        if self.att_type == "normal_att":
            return self._normal_self_att(query, key, value, mask)
        elif self.att_type == "block_att":
            return self._block_wise_self_att(query, key, value, mask)
        else:
            return self._sliding_window_self_att(query, key, value, mask)

    def _normal_self_att(self, q, k, v, mask):
        m, _, L = q.size()
        padding_mask = torch.ones((m, 1, L), device=self.device) * mask[:, 0:1, :]
        output, _ = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))[:, :, :L]
        return output * mask[:, 0:1, :]

    def _block_wise_self_att(self, q, k, v, mask):
        m, c1, L = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()

        nb = L // self.bl
        if L % self.bl != 0:
            pad = self.bl - L % self.bl
            q = torch.cat([q, torch.zeros((m, c1, pad), device=self.device)], dim=-1)
            k = torch.cat([k, torch.zeros((m, c2, pad), device=self.device)], dim=-1)
            v = torch.cat([v, torch.zeros((m, c3, pad), device=self.device)], dim=-1)
            nb += 1

        padding_mask = torch.cat(
            [
                torch.ones((m, 1, L), device=self.device) * mask[:, 0:1, :],
                torch.zeros((m, 1, self.bl * nb - L), device=self.device),
            ],
            dim=-1,
        )

        q = q.reshape(m, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m * nb, c1, self.bl)
        padding_mask = padding_mask.reshape(m, 1, nb, self.bl).permute(0, 2, 1, 3).reshape(m * nb, 1, self.bl)
        k = k.reshape(m, c2, nb, self.bl).permute(0, 2, 1, 3).reshape(m * nb, c2, self.bl)
        v = v.reshape(m, c3, nb, self.bl).permute(0, 2, 1, 3).reshape(m * nb, c3, self.bl)

        output, _ = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        output = output.reshape(m, nb, c3, self.bl).permute(0, 2, 1, 3).reshape(m, c3, nb * self.bl)[:, :, :L]
        return output * mask[:, 0:1, :]

    def _sliding_window_self_att(self, q, k, v, mask):
        m, c1, L = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()
        assert m == 1, "sliding-window attention currently only supports batch size 1"

        nb = L // self.bl
        if L % self.bl != 0:
            pad = self.bl - L % self.bl
            q = torch.cat([q, torch.zeros((m, c1, pad), device=self.device)], dim=-1)
            k = torch.cat([k, torch.zeros((m, c2, pad), device=self.device)], dim=-1)
            v = torch.cat([v, torch.zeros((m, c3, pad), device=self.device)], dim=-1)
            nb += 1

        padding_mask = torch.cat(
            [
                torch.ones((m, 1, L), device=self.device) * mask[:, 0:1, :],
                torch.zeros((m, 1, self.bl * nb - L), device=self.device),
            ],
            dim=-1,
        )

        half = self.bl // 2
        q = q.reshape(m, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m * nb, c1, self.bl)

        k = torch.cat([torch.zeros(m, c2, half, device=self.device), k, torch.zeros(m, c2, half, device=self.device)], dim=-1)
        v = torch.cat([torch.zeros(m, c3, half, device=self.device), v, torch.zeros(m, c3, half, device=self.device)], dim=-1)
        padding_mask = torch.cat(
            [torch.zeros(m, 1, half, device=self.device), padding_mask, torch.zeros(m, 1, half, device=self.device)], dim=-1
        )

        k = torch.cat([k[:, :, i * self.bl : (i + 1) * self.bl + half * 2] for i in range(nb)], dim=0)
        v = torch.cat([v[:, :, i * self.bl : (i + 1) * self.bl + half * 2] for i in range(nb)], dim=0)
        padding_mask = torch.cat(
            [padding_mask[:, :, i * self.bl : (i + 1) * self.bl + half * 2] for i in range(nb)], dim=0
        )
        final_mask = self.window_mask.repeat(m * nb, 1, 1) * padding_mask

        output, _ = self.att_helper.scalar_dot_att(q, k, v, final_mask)
        output = self.conv_out(F.relu(output))
        output = output.reshape(m, nb, -1, self.bl).permute(0, 2, 1, 3).reshape(m, -1, nb * self.bl)[:, :, :L]
        return output * mask[:, 0:1, :]


class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class FCFeedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(out_channels, out_channels, 1),
        )

    def forward(self, x):
        return self.layer(x)


class AttModule(nn.Module):
    """Dilated conv feed-forward + windowed attention, residual-connected."""

    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha, device):
        super().__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(
            in_channels, in_channels, out_channels, r1, r1, r2, dilation, stage=stage, att_type=att_type, device=device
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha

    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class PositionalEncoding(nn.Module):
    """Learnable sinusoidal-initialised positional encoding."""

    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0, 2, 1)  # (1, d_model, L)
        self.pe = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return x + self.pe[:, :, : x.shape[2]]


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super().__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]
