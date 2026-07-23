#!/usr/bin/env python
"""
Run RELATE inference on the test split of a dataset, using only RGB frames
at inference time (no optical-flow extraction), and write one predicted
label file per video under ``--output-dir``.

Applies the training-free prediction refinement step (Sec. 3.4) by default;
disable with ``--no-refine``.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from relate.alignment_fusion import AlignmentFusionTransformer, naive_concat_fusion
from relate.data import BatchGenerator
from relate.modality_bridge import ModalityBridge
from relate.refinement import refine_joint_prediction
from relate.segmenters import FlowSegmenter, JointSegmenter, RGBSegmenter


def parse_args():
    p = argparse.ArgumentParser(description="RELATE inference")
    p.add_argument("--data-root", required=True)
    p.add_argument("--dataset", required=True, choices=["gtea", "50salads", "breakfast"])
    p.add_argument("--split", default="1")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--epoch", required=True, help="Checkpoint epoch suffix, e.g. '100'")
    p.add_argument("--output-dir", required=True)

    p.add_argument("--backbone", default="MS-TCN", choices=["MS-TCN", "ASFormer"])
    p.add_argument("--fusion", default="afti", choices=["afti", "concat"])
    p.add_argument("--use-modality-bridge", action="store_true")
    p.add_argument("--modality-bridge-ckpt", default=None)

    p.add_argument("--num-stages", type=int, default=3)
    p.add_argument("--num-layers", type=int, default=10)
    p.add_argument("--num-f-maps", type=int, default=64)
    p.add_argument("--features-dim", type=int, default=1024)
    p.add_argument("--refine", dest="refine", action="store_true", default=True)
    p.add_argument("--no-refine", dest="refine", action="store_false")
    p.add_argument("--refine-threshold", type=float, default=0.7)
    p.add_argument("--device", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    sample_rate = 2 if args.dataset == "50salads" else 1

    mapping_file = os.path.join(args.data_root, args.dataset, "mapping.txt")
    with open(mapping_file, "r") as f:
        actions = f.read().split("\n")[:-1]
    actions_dict = {a.split()[1]: int(a.split()[0]) for a in actions}
    index2label = {v: k for k, v in actions_dict.items()}
    num_classes = len(actions_dict)

    features_path = os.path.join(args.data_root, args.dataset, "features/")
    gt_path = os.path.join(args.data_root, args.dataset, "groundTruth/")
    test_list = os.path.join(args.data_root, args.dataset, "splits", f"test.split{args.split}.bundle")

    batch_gen_test = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_test.read_data(test_list)

    r1 = r2 = 2
    rgb_model = RGBSegmenter(args.backbone, args.num_stages - 1, args.num_layers, r1, r2, args.num_f_maps, args.features_dim, num_classes, 0.0, device).to(device)
    flow_model = FlowSegmenter(args.backbone, args.num_stages - 1, args.num_layers, r1, r2, args.num_f_maps, args.features_dim, num_classes, 0.0, device).to(device)
    joint_input_dim = args.num_f_maps if args.fusion == "afti" else args.num_f_maps * 2
    joint_model = JointSegmenter(args.backbone, args.num_stages - 1, args.num_layers, r1, r2, args.num_f_maps, joint_input_dim, num_classes, 0.0, device).to(device)

    rgb_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, f"rgb-epoch-{args.epoch}.pt"), map_location=device))
    flow_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, f"flow-epoch-{args.epoch}.pt"), map_location=device))
    joint_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, f"joint-epoch-{args.epoch}.pt"), map_location=device))
    rgb_model.eval()
    flow_model.eval()
    joint_model.eval()

    aft = None
    if args.fusion == "afti":
        aft = AlignmentFusionTransformer(args.num_f_maps, device).to(device)
        aft.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, f"aft-epoch-{args.epoch}.pt"), map_location=device))
        aft.eval()

    bridge = None
    if args.use_modality_bridge:
        bridge = ModalityBridge(args.features_dim).to(device)
        bridge.load_state_dict(torch.load(args.modality_bridge_ckpt, map_location=device))
        bridge.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        while batch_gen_test.has_next():
            _, _, _, vids = batch_gen_test.next_batch(1)
            vid = vids[0]
            features = np.load(features_path + vid.split(".")[0] + ".npy")
            features = features[:, ::sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            rgb_input = input_x[: args.features_dim, :].unsqueeze(0).to(device)
            ones = torch.ones_like(rgb_input).to(device)

            flow_input = bridge(rgb_input) if bridge is not None else rgb_input
            rgb_pred, rgb_feat = rgb_model(rgb_input, ones)
            flow_pred, flow_feat = flow_model(flow_input, ones)

            if aft is not None:
                fused = aft(rgb_feat[-1], flow_feat[-1], rgb_pred[1], flow_pred[1], ones)
            else:
                fused = naive_concat_fusion(rgb_feat[-1], flow_feat[-1])
            fused_ones = torch.ones_like(fused)
            joint_pred, _ = joint_model(fused, fused_ones)

            rgb_prob = F.softmax(rgb_pred[-1], dim=1).squeeze(0).transpose(0, 1)
            flow_prob = F.softmax(flow_pred[-1], dim=1).squeeze(0).transpose(0, 1)
            joint_prob = F.softmax(joint_pred[-1], dim=1).squeeze(0).transpose(0, 1)

            if args.refine:
                joint_prob = refine_joint_prediction(rgb_prob, flow_prob, joint_prob, threshold=args.refine_threshold)

            predicted = joint_prob.argmax(dim=1)
            recognition = [index2label[int(c)] for c in predicted for _ in range(sample_rate)]

            out_path = os.path.join(args.output_dir, vid.split(".")[0])
            with open(out_path, "w") as f:
                f.write("### Frame level recognition: ###\n")
                f.write(" ".join(recognition))

    batch_gen_test.reset()
    print(f"predictions written to {args.output_dir}")


if __name__ == "__main__":
    main()
