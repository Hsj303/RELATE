#!/usr/bin/env python
"""
Compute F1@{10,25,50}, Edit score, and frame-wise accuracy (paper Sec. 4.1.1,
Table 1) by comparing prediction files written by ``predict.py`` against
ground truth.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from relate.metrics import edit_score, f_score, frame_accuracy


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RELATE predictions")
    p.add_argument("--data-root", required=True)
    p.add_argument("--dataset", required=True, choices=["gtea", "50salads", "breakfast"])
    p.add_argument("--split", default="1")
    p.add_argument("--pred-dir", required=True, help="Directory of prediction files written by predict.py")
    return p.parse_args()


def load_label_sequence(path, actions_dict):
    with open(path, "r") as f:
        content = f.read().split("\n")[:-1]
    return torch.tensor([actions_dict[c] for c in content], dtype=torch.long).unsqueeze(0)


def load_prediction(path, actions_dict):
    with open(path, "r") as f:
        lines = f.read().split("\n")
    labels = lines[1].split()
    return torch.tensor([actions_dict[c] for c in labels], dtype=torch.long).unsqueeze(0)


def main():
    args = parse_args()
    mapping_file = os.path.join(args.data_root, args.dataset, "mapping.txt")
    with open(mapping_file, "r") as f:
        actions = f.read().split("\n")[:-1]
    actions_dict = {a.split()[1]: int(a.split()[0]) for a in actions}
    background = {"gtea": actions_dict.get("background", -1), "50salads": -1, "breakfast": actions_dict.get("SIL", -1)}[args.dataset]

    gt_path = os.path.join(args.data_root, args.dataset, "groundTruth/")
    test_list = os.path.join(args.data_root, args.dataset, "splits", f"test.split{args.split}.bundle")
    with open(test_list, "r") as f:
        vids = [line.strip() for line in f if line.strip()]

    f1_sums = {0.10: 0.0, 0.25: 0.0, 0.50: 0.0}
    edit_sum, acc_sum, n = 0.0, 0.0, 0

    for vid in vids:
        gt = load_label_sequence(os.path.join(gt_path, vid), actions_dict)
        pred_path = os.path.join(args.pred_dir, vid.split(".")[0])
        if not os.path.exists(pred_path):
            print(f"[warn] missing prediction for {vid}, skipping")
            continue
        pred = load_prediction(pred_path, actions_dict)

        length = min(gt.shape[1], pred.shape[1])
        gt, pred = gt[:, :length], pred[:, :length]

        for overlap in f1_sums:
            f1_sums[overlap] += f_score(gt.squeeze(0), pred.squeeze(0), overlap, background)
        edit_sum += edit_score(gt.squeeze(0), pred.squeeze(0), background)
        acc_sum += frame_accuracy(pred.squeeze(0), gt.squeeze(0), ignore_index=-999)
        n += 1

    if n == 0:
        print("No predictions found to evaluate.")
        return

    print(f"Videos evaluated: {n}")
    print(f"F1@10: {100 * f1_sums[0.10] / n:.1f}")
    print(f"F1@25: {100 * f1_sums[0.25] / n:.1f}")
    print(f"F1@50: {100 * f1_sums[0.50] / n:.1f}")
    print(f"Edit:  {edit_sum / n:.1f}")
    print(f"Acc:   {acc_sum / n:.1f}")


if __name__ == "__main__":
    main()
