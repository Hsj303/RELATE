"""
Evaluation metrics for temporal action segmentation: segmental F1@{10,25,50},
edit distance, and frame-wise accuracy (paper Sec. 4.1.1).
"""

from typing import Dict, Tuple

import numpy as np
import torch


def get_labels_start_end_time(framewise_label: torch.Tensor, background: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts a framewise label sequence into (labels, starts, ends) for each
    contiguous non-background segment.
    """
    labels, starts, ends = [], [], []
    framewise_label = framewise_label.squeeze(0)
    last_label = framewise_label[0].item()

    if last_label != background:
        labels.append(last_label)
        starts.append(0)

    diff = torch.diff(framewise_label)
    non_index = torch.nonzero(diff).squeeze()

    if non_index.numel() == 0:
        if last_label != background:
            ends.append(len(framewise_label))
        return torch.tensor(labels), torch.tensor(starts), torch.tensor(ends)

    non_index = non_index + 1
    if non_index.dim() == 0:
        non_index = non_index.unsqueeze(0)

    for i in non_index:
        current_label = framewise_label[i].item()
        if current_label != background:
            labels.append(current_label)
            starts.append(i.item())
        if last_label != background:
            ends.append(i.item())
        last_label = current_label

    if last_label != background:
        ends.append(len(framewise_label))

    return torch.tensor(labels), torch.tensor(starts), torch.tensor(ends)


def f_score(ground_truth: torch.Tensor, predicted: torch.Tensor, overlap: float, background: int) -> float:
    """Segmental F1 score at a given IoU ``overlap`` threshold."""
    p_label, p_start, p_end = get_labels_start_end_time(predicted, background)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, background)

    if p_start.numel() == 0 or p_end.numel() == 0:
        return 0.0

    tp, fp = 0, 0
    hits = torch.zeros(y_label.shape[0])

    for j in range(p_label.shape[0]):
        intersection = torch.min(p_end[j], y_end) - torch.max(p_start[j], y_start)
        union = torch.max(p_end[j], y_end) - torch.min(p_start[j], y_start)
        intersection = torch.clamp(intersection, min=0)
        iou = intersection / (union + 1e-6)

        if iou.numel() == 0:
            continue

        matching_labels = p_label[j] == y_label
        if matching_labels.any():
            idx = torch.argmax(iou * matching_labels.float())
            if iou[idx] >= overlap and not hits[idx]:
                tp += 1
                hits[idx] = 1
            else:
                fp += 1

    fn = len(y_label) - hits.sum().item()
    precision = tp / float(tp + fp + 1e-6)
    recall = tp / float(tp + fn + 1e-6)
    f1 = 2.0 * (precision * recall) / (precision + recall + 1e-6)
    return float(np.nan_to_num(f1))


def edit_score(ground_truth: torch.Tensor, predicted: torch.Tensor, background: int) -> float:
    """Levenshtein edit distance between predicted and ground-truth segment labels."""
    p_label, _, _ = get_labels_start_end_time(predicted, background)
    y_label, _, _ = get_labels_start_end_time(ground_truth, background)
    p_label, y_label = p_label.tolist(), y_label.tolist()

    m, n = len(p_label), len(y_label)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if p_label[i - 1] == y_label[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    if max(m, n) == 0:
        return 100.0
    return (1 - dp[m][n] / max(m, n)) * 100.0


def segmental_f1(predicted: torch.Tensor, ground_truth: torch.Tensor, background: int = -100) -> Dict[float, float]:
    """Convenience wrapper returning F1@{0.10, 0.25, 0.50}."""
    return {overlap: f_score(ground_truth, predicted, overlap, background) for overlap in (0.10, 0.25, 0.50)}


def frame_accuracy(predicted: torch.Tensor, ground_truth: torch.Tensor, ignore_index: int = -100) -> float:
    valid = ground_truth != ignore_index
    correct = (predicted[valid] == ground_truth[valid]).sum().item()
    total = valid.sum().item()
    return 100.0 * correct / max(total, 1)
