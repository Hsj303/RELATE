#!/usr/bin/env python
"""
Train RELATE: RGB branch, flow branch (CMKD student), and the joint
segmenter fused via the Alignment Fusion Transformer (Sec. 3.1-3.3).

Assumes a pretrained optical-flow teacher segmenter (``RefSegmenter``)
checkpoint already exists -- train the teacher on optical-flow I3D features
with a plain segmenter first (e.g. by running this script with
``--teacher-only`` on flow features), or reuse an existing two-stream
model's flow branch as the teacher.

Example:
    python scripts/train.py \\
        --data-root /path/to/data --dataset gtea --split 1 \\
        --backbone MS-TCN --output-dir ./runs/gtea/split_1 \\
        --teacher-ckpt ./runs/gtea/split_1/teacher/epoch-120.pt
"""

import argparse
import os
import random
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from relate.alignment_fusion import AlignmentFusionTransformer, naive_concat_fusion
from relate.data import BatchGenerator
from relate.losses import global_covariance_relation_loss, logit_distillation_loss, rkd_distance_loss, segmentation_loss
from relate.metrics import segmental_f1
from relate.modality_bridge import ModalityBridge
from relate.segmenters import FlowSegmenter, JointSegmenter, RGBSegmenter, RefSegmenter


def parse_args():
    p = argparse.ArgumentParser(description="Train RELATE")
    p.add_argument("--data-root", required=True, help="Dataset root, e.g. /path/to/data/")
    p.add_argument("--dataset", required=True, choices=["gtea", "50salads", "breakfast"])
    p.add_argument("--split", default="1")
    p.add_argument("--output-dir", required=True, help="Where to save checkpoints")
    p.add_argument("--teacher-ckpt", required=True, help="Pretrained optical-flow teacher checkpoint")

    p.add_argument("--backbone", default="MS-TCN", choices=["MS-TCN", "ASFormer"])
    p.add_argument("--fusion", default="afti", choices=["afti", "concat"], help="'afti' = Alignment Fusion Transformer, 'concat' = naive concat baseline")
    p.add_argument("--distill", default="kd", choices=["kd", "rkd", "gcr"], help="CMKD loss (Table 4)")
    p.add_argument("--use-modality-bridge", action="store_true", help="Use a pretrained RGB->OF modality bridge instead of feeding raw RGB to the flow branch")
    p.add_argument("--modality-bridge-ckpt", default=None)

    p.add_argument("--num-stages", type=int, default=3)
    p.add_argument("--num-layers", type=int, default=10)
    p.add_argument("--num-f-maps", type=int, default=64)
    p.add_argument("--features-dim", type=int, default=1024, help="Per-modality (RGB or OF) feature dim; I3D=1024")
    p.add_argument("--channel-mask-rate", type=float, default=0.3)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=0.5, help="Weight of L_CMKD (paper default: 0.5)")
    p.add_argument("--seed", type=int, default=2222212)
    p.add_argument("--device", type=int, default=0)
    return p.parse_args()


def bg_class_for(dataset: str):
    return {"gtea": [10], "50salads": [], "breakfast": [0]}[dataset]


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    sample_rate = 2 if args.dataset == "50salads" else 1

    mapping_file = os.path.join(args.data_root, args.dataset, "mapping.txt")
    with open(mapping_file, "r") as f:
        actions = f.read().split("\n")[:-1]
    actions_dict = {a.split()[1]: int(a.split()[0]) for a in actions}
    num_classes = len(actions_dict)

    features_path = os.path.join(args.data_root, args.dataset, "features/")
    gt_path = os.path.join(args.data_root, args.dataset, "groundTruth/")
    train_list = os.path.join(args.data_root, args.dataset, "splits", f"train.split{args.split}.bundle")
    test_list = os.path.join(args.data_root, args.dataset, "splits", f"test.split{args.split}.bundle")

    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(train_list)
    batch_gen_test = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_test.read_data(test_list)

    r1 = r2 = 2  # ASFormer projection reduction factors
    rgb_model = RGBSegmenter(args.backbone, args.num_stages - 1, args.num_layers, r1, r2, args.num_f_maps, args.features_dim, num_classes, args.channel_mask_rate, device).to(device)
    flow_model = FlowSegmenter(args.backbone, args.num_stages - 1, args.num_layers, r1, r2, args.num_f_maps, args.features_dim, num_classes, args.channel_mask_rate, device).to(device)
    ref_model = RefSegmenter(args.backbone, args.num_stages - 1, args.num_layers, r1, r2, args.num_f_maps, args.features_dim, num_classes, args.channel_mask_rate, device).to(device)

    ref_model.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    joint_input_dim = args.num_f_maps if args.fusion == "afti" else args.num_f_maps * 2
    joint_model = JointSegmenter(args.backbone, args.num_stages - 1, args.num_layers, r1, r2, args.num_f_maps, joint_input_dim, num_classes, args.channel_mask_rate, device).to(device)

    aft = None
    trainable_params = list(rgb_model.parameters()) + list(flow_model.parameters()) + list(joint_model.parameters())
    if args.fusion == "afti":
        aft = AlignmentFusionTransformer(args.num_f_maps, device).to(device)
        trainable_params += list(aft.parameters())

    bridge = None
    if args.use_modality_bridge:
        bridge = ModalityBridge(args.features_dim).to(device)
        bridge.load_state_dict(torch.load(args.modality_bridge_ckpt, map_location=device))
        bridge.eval()
        for p in bridge.parameters():
            p.requires_grad_(False)

    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-5)
    os.makedirs(args.output_dir, exist_ok=True)

    best_f1 = 0.0
    for epoch in range(args.epochs):
        rgb_model.train()
        flow_model.train()
        joint_model.train()
        if aft is not None:
            aft.train()

        epoch_loss = 0.0
        while batch_gen.has_next():
            batch_input, batch_target, mask, _ = batch_gen.next_batch(args.batch_size, False)
            batch_rgb = batch_input[:, : args.features_dim, :].to(device)
            batch_flow = batch_input[:, args.features_dim :, :].to(device)
            batch_target = batch_target.to(device)
            mask = mask.to(device)

            flow_input = bridge(batch_rgb) if bridge is not None else batch_rgb

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                p_ref, ref_feat = ref_model(batch_flow, mask)

            p_rgb, rgb_feat = rgb_model(batch_rgb, mask)
            p_flow, flow_feat = flow_model(flow_input, mask)

            if args.fusion == "afti":
                fused = aft(rgb_feat[-1], flow_feat[-1], p_rgb[1], p_flow[1], mask)
            else:
                fused = naive_concat_fusion(rgb_feat[-1], flow_feat[-1])

            p_joint, _ = joint_model(fused, mask)

            loss = segmentation_loss(p_rgb, num_classes, batch_target, mask)
            loss = loss + segmentation_loss(p_flow, num_classes, batch_target, mask)
            loss = loss + segmentation_loss(p_joint, num_classes, batch_target, mask)

            if args.distill == "kd":
                cmkd = sum(logit_distillation_loss(p_flow[i], p_ref[i], args.temperature) for i in range(len(p_flow)))
            elif args.distill == "rkd":
                cmkd = rkd_distance_loss(flow_feat[-1], ref_feat[-1])
            else:
                cmkd = global_covariance_relation_loss(flow_feat[-1], ref_feat[-1])

            loss = loss + args.gamma * cmkd
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"epoch {epoch + 1}/{args.epochs} - loss {epoch_loss / max(len(batch_gen.list_of_examples), 1):.4f}")
        batch_gen.reset()

        if (epoch + 1) % args.eval_every == 0:
            best_f1 = _evaluate_and_checkpoint(
                rgb_model, flow_model, joint_model, aft, bridge, batch_gen_test, device, args, epoch, best_f1,
            )

    print("training complete")


@torch.no_grad()
def _evaluate_and_checkpoint(rgb_model, flow_model, joint_model, aft, bridge, batch_gen_test, device, args, epoch, best_f1):
    rgb_model.eval()
    flow_model.eval()
    joint_model.eval()
    if aft is not None:
        aft.eval()

    f1_sums = defaultdict(float)
    n = 0
    batch_gen_test.reset()
    while batch_gen_test.has_next():
        batch_input, batch_target, mask, _ = batch_gen_test.next_batch(1)
        batch_rgb = batch_input[:, : args.features_dim, :].to(device)
        ones = torch.ones_like(batch_rgb).to(device)

        flow_input = bridge(batch_rgb) if bridge is not None else batch_rgb
        rgb_pred, rgb_feat = rgb_model(batch_rgb, ones)
        flow_pred, flow_feat = flow_model(flow_input, ones)

        if aft is not None:
            fused = aft(rgb_feat[-1], flow_feat[-1], rgb_pred[1], flow_pred[1], ones)
        else:
            fused = naive_concat_fusion(rgb_feat[-1], flow_feat[-1])
        fused_ones = torch.ones_like(fused)
        joint_pred, _ = joint_model(fused, fused_ones)

        pred = torch.max(F.softmax(joint_pred[-1], dim=1).data, 1)[1]
        scores = segmental_f1(pred.squeeze(0), batch_target.squeeze(0))
        for k, v in scores.items():
            f1_sums[k] += v
        n += 1

    f1_10, f1_25, f1_50 = f1_sums[0.10] / max(n, 1), f1_sums[0.25] / max(n, 1), f1_sums[0.50] / max(n, 1)
    mean_f1 = (f1_10 + f1_25 + f1_50) / 3
    print(f"  eval @ epoch {epoch + 1}: F1@10={f1_10:.1f} F1@25={f1_25:.1f} F1@50={f1_50:.1f}")

    ckpt_dir = args.output_dir
    torch.save(rgb_model.state_dict(), os.path.join(ckpt_dir, f"rgb-epoch-{epoch + 1}.pt"))
    torch.save(flow_model.state_dict(), os.path.join(ckpt_dir, f"flow-epoch-{epoch + 1}.pt"))
    torch.save(joint_model.state_dict(), os.path.join(ckpt_dir, f"joint-epoch-{epoch + 1}.pt"))
    if aft is not None:
        torch.save(aft.state_dict(), os.path.join(ckpt_dir, f"aft-epoch-{epoch + 1}.pt"))

    return max(best_f1, mean_f1)


if __name__ == "__main__":
    main()
