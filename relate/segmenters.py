"""
Segmenters used in RELATE's dual-branch pipeline (paper Sec. 3.1, Fig. 2 left).

Four segmenters share the same backbone (MS-TCN or ASFormer), each playing a
different role:

  * ``RGBSegmenter``   — the RGB branch, encodes appearance features Z_RGB.
  * ``FlowSegmenter``  — the flow branch, learns motion-aware features Z_OF
    via cross-modal knowledge distillation (CMKD) from the teacher.
  * ``RefSegmenter``   — the frozen optical-flow teacher used only during
    training to supervise the flow branch (Fig. 2 left, "Teacher Segmenter").
  * ``JointSegmenter`` — consumes the fused RGB/OF features from the
    Alignment Fusion Transformer and produces the final joint prediction.

All four expose the same ``forward(x, mask) -> (stage_logits, stage_features)``
interface so they can be swapped freely by the training/inference scripts.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import Decoder, Encoder, SingleStageModel, exponential_decrease


class _MSTCNStack(nn.Module):
    """Shared MS-TCN multi-stage forward pass used by every segmenter role."""

    def __init__(self, num_layers, num_f_maps, input_dim, num_classes, num_extra_stages):
        super().__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, input_dim, num_classes)
        self.stages = nn.ModuleList(
            [SingleStageModel(num_layers, num_f_maps, num_classes, num_classes) for _ in range(num_extra_stages)]
        )

    def forward(self, x, mask):
        logits, features = self.stage1(x, mask)
        outputs = logits.unsqueeze(0)
        feature_stack = features.unsqueeze(0)

        for stage in self.stages:
            logits, features = stage(F.softmax(logits, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, logits.unsqueeze(0)), dim=0)
            feature_stack = torch.cat((feature_stack, features.unsqueeze(0)), dim=0)

        return outputs, feature_stack  # (num_stages, B, C, L), (num_stages, B, D, L)


class _ASFormerStack(nn.Module):
    """Shared ASFormer encoder-decoder forward pass used by every segmenter role."""

    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, num_decoders, device):
        super().__init__()
        self.encoder = Encoder(
            num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type="sliding_att", alpha=1, device=device
        )
        self.decoders = nn.ModuleList(
            [
                copy.deepcopy(
                    Decoder(
                        num_layers, r1, r2, num_f_maps, num_classes, num_classes,
                        att_type="sliding_att", alpha=exponential_decrease(s), device=device,
                    )
                )
                for s in range(num_decoders)
            ]
        )

    def forward(self, x, mask):
        # NOTE: ``feature`` from Encoder/Decoder is (num_layers+1, B, D, L)
        # (one entry per attention layer). We keep only each stage's
        # *last*-layer feature so ``feature_stack`` is (num_stages, B, D, L)
        # -- consistent with the MS-TCN stack and with how downstream code
        # (AFT, action anchors, CMKD losses) indexes ``feature_stack[-1]``.
        out, feature = self.encoder(x, mask)
        last_layer_feature = feature[-1]  # (B, D, L)
        outputs = out.unsqueeze(0)
        feature_stack = last_layer_feature.unsqueeze(0)

        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], last_layer_feature * mask[:, 0:1, :], mask)
            last_layer_feature = feature[-1]
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
            feature_stack = torch.cat((feature_stack, last_layer_feature.unsqueeze(0)), dim=0)

        return outputs, feature_stack


class _BaseSegmenter(nn.Module):
    """Common backbone-selection logic shared by every segmenter role."""

    def __init__(self, backbone, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, device):
        super().__init__()
        assert backbone in ("MS-TCN", "ASFormer"), f"Unsupported backbone: {backbone}"
        self.backbone = backbone
        if backbone == "MS-TCN":
            self.stack = _MSTCNStack(num_layers, num_f_maps, input_dim, num_classes, num_decoders)
        else:
            self.stack = _ASFormerStack(num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, num_decoders, device)

    def forward(self, x, mask):
        return self.stack(x, mask)


class RGBSegmenter(_BaseSegmenter):
    """RGB branch: encodes appearance features Z_RGB directly from RGB input."""


class FlowSegmenter(_BaseSegmenter):
    """
    Flow branch: learns motion-aware features Z_OF from an RGB(-derived)
    input via CMKD supervision from ``RefSegmenter`` (Sec. 3.1, Sec. 3.3).
    """


class RefSegmenter(_BaseSegmenter):
    """Optical-flow teacher segmenter. Frozen and only used at training time."""


class JointSegmenter(_BaseSegmenter):
    """Consumes fused RGB+OF features and predicts the final action labels."""
