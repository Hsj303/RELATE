"""
Lightweight shape/forward-pass smoke tests that need no dataset and run on
CPU, so they're suitable for CI. They check that the paper's architecture
(Sec. 3.1-3.4) wires together correctly, not full training convergence.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from relate.alignment_fusion import AlignmentFusionTransformer, action_anchor_pooling
from relate.losses import global_covariance_relation_loss, logit_distillation_loss, rkd_distance_loss, segmentation_loss
from relate.metrics import edit_score, f_score
from relate.modality_bridge import ModalityBridge
from relate.refinement import refine_joint_prediction
from relate.segmenters import FlowSegmenter, JointSegmenter, RGBSegmenter

DEVICE = torch.device("cpu")
B, L, C, D, NUM_CLASSES = 1, 64, 1024, 64, 5


def _dummy_batch():
    x = torch.randn(B, C, L)
    target = torch.randint(0, NUM_CLASSES, (B, L))
    mask = torch.ones(B, NUM_CLASSES, L)
    return x, target, mask


def test_mstcn_dual_branch_and_fusion():
    x, target, mask = _dummy_batch()
    rgb_model = RGBSegmenter("MS-TCN", 2, 4, 2, 2, D, C, NUM_CLASSES, 0.3, DEVICE)
    flow_model = FlowSegmenter("MS-TCN", 2, 4, 2, 2, D, C, NUM_CLASSES, 0.3, DEVICE)
    joint_model = JointSegmenter("MS-TCN", 2, 4, 2, 2, D, D, NUM_CLASSES, 0.3, DEVICE)
    aft = AlignmentFusionTransformer(D, DEVICE)

    p_rgb, rgb_feat = rgb_model(x, mask)
    p_flow, flow_feat = flow_model(x, mask)
    assert p_rgb.shape[-2:] == (NUM_CLASSES, L)

    fused = aft(rgb_feat[-1], flow_feat[-1], p_rgb[1], p_flow[1], mask)
    assert fused.shape == (B, D, L)

    p_joint, _ = joint_model(fused, mask)
    assert p_joint.shape[-2:] == (NUM_CLASSES, L)

    loss = segmentation_loss(p_joint, NUM_CLASSES, target, mask)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_asformer_dual_branch_and_fusion():
    x, target, mask = _dummy_batch()
    rgb_model = RGBSegmenter("ASFormer", 2, 4, 2, 2, D, C, NUM_CLASSES, 0.3, DEVICE)
    flow_model = FlowSegmenter("ASFormer", 2, 4, 2, 2, D, C, NUM_CLASSES, 0.3, DEVICE)
    joint_model = JointSegmenter("ASFormer", 2, 4, 2, 2, D, D, NUM_CLASSES, 0.3, DEVICE)
    aft = AlignmentFusionTransformer(D, DEVICE, window_size=16)

    p_rgb, rgb_feat = rgb_model(x, mask)
    p_flow, flow_feat = flow_model(x, mask)
    assert rgb_feat.dim() == 4  # (num_stages, B, D, L), consistent with MS-TCN

    fused = aft(rgb_feat[-1], flow_feat[-1], p_rgb[1], p_flow[1], mask)
    p_joint, _ = joint_model(fused, mask)
    assert p_joint.shape[-2:] == (NUM_CLASSES, L)


def test_action_anchor_pooling_matches_segment_count():
    pred = torch.zeros(1, NUM_CLASSES, L)
    pred[:, 0, :32] = 1
    pred[:, 1, 32:] = 1
    query = torch.randn(1, D, L)
    anchors = action_anchor_pooling(pred, query)
    assert anchors.shape == (1, D, 2)


def test_distillation_losses_run():
    student_logits = torch.randn(B, NUM_CLASSES, L)
    teacher_logits = torch.randn(B, NUM_CLASSES, L)
    student_feat = torch.randn(1, D, L)
    teacher_feat = torch.randn(1, D, L)

    assert torch.isfinite(logit_distillation_loss(student_logits, teacher_logits))
    assert torch.isfinite(rkd_distance_loss(student_feat, teacher_feat))
    assert torch.isfinite(global_covariance_relation_loss(student_feat, teacher_feat))


def test_prediction_refinement_shape():
    rgb_pred = torch.softmax(torch.randn(L, NUM_CLASSES), dim=-1)
    flow_pred = torch.softmax(torch.randn(L, NUM_CLASSES), dim=-1)
    joint_pred = torch.softmax(torch.randn(L, NUM_CLASSES), dim=-1)
    refined = refine_joint_prediction(rgb_pred, flow_pred, joint_pred, threshold=0.7)
    assert refined.shape == (L, NUM_CLASSES)


def test_modality_bridge_shape():
    bridge = ModalityBridge(dim=C, num_layers=2, hidden_dim=256, kernel_size=1)
    x = torch.randn(B, C, L)
    out = bridge(x)
    assert out.shape == x.shape


def test_metrics_perfect_prediction():
    gt = torch.randint(0, NUM_CLASSES, (L,))
    assert abs(f_score(gt, gt, overlap=0.5, background=-1) - 1.0) < 1e-4
    assert abs(edit_score(gt, gt, background=-1) - 100.0) < 1e-4
