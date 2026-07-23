"""
Backbone stages for the two temporal-modeling segmenters RELATE is built on:
MS-TCN (Farha & Gall, 2019) and ASFormer (Yi et al., 2021). FACT
(Lu & Elhamifar, 2024), the third segmenter reported in Table 1, is a
frame-action matching model with its own repository; see
``docs/external_dependencies.md`` for how to plug it in.
"""

import copy

import torch
import torch.nn as nn

from .layers import AttModule, DilatedResidualLayer


class SingleStageModel(nn.Module):
    """One MS-TCN stage: 1x1 projection + a stack of dilated residual layers."""

    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps)) for i in range(num_layers)]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        feat = out
        logit = self.conv_out(out) * mask[:, 0:1, :]
        return logit, feat


class Encoder(nn.Module):
    """ASFormer encoder: channel-dropout input + stack of windowed-attention blocks."""

    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha, device):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2**i, num_f_maps, num_f_maps, r1, r2, att_type, "encoder", alpha, device) for i in range(num_layers)]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        if self.channel_masking_rate > 0:
            x = self.dropout(x.unsqueeze(2)).squeeze(2)

        feature = self.conv_1x1(x)
        feature_stack = feature.unsqueeze(0)  # (1, B, D, L)

        for layer in self.layers:
            feature = layer(feature, None, mask)
            feature_stack = torch.cat((feature_stack, feature.unsqueeze(0)), dim=0)

        logit = self.conv_out(feature) * mask[:, 0:1, :]
        return logit, feature_stack  # (B, num_classes, L), (num_layers+1, B, D, L)


class Decoder(nn.Module):
    """ASFormer decoder: cross-attends to the encoder's last-layer feature."""

    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha, device):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2**i, num_f_maps, num_f_maps, r1, r2, att_type, "decoder", alpha, device) for i in range(num_layers)]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, f_encoder, mask):
        feature = self.conv_1x1(x)
        feature_stack = feature.unsqueeze(0)

        for layer in self.layers:
            feature = layer(feature, f_encoder, mask)
            feature_stack = torch.cat((feature_stack, feature.unsqueeze(0)), dim=0)

        logit = self.conv_out(feature) * mask[:, 0:1, :]
        return logit, feature_stack


def exponential_decrease(idx_decoder, p=3):
    """Decoder-depth-dependent attenuation used for ASFormer's decoder stack."""
    import math

    return math.exp(-p * idx_decoder)
