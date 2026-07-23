"""
Batch generator for I3D-feature temporal action segmentation datasets
(GTEA / 50Salads / Breakfast layout: ``features/*.npy`` + ``groundTruth/*.txt``
+ ``mapping.txt``).

NOTE: ``batch_gen.py`` was referenced by the uploaded scripts
(``from batch_gen import BatchGenerator``) but was not included in the
uploaded archive. This is a clean reimplementation of the standard
single-clip-per-step generator interface used by MS-TCN-style action
segmentation code (``has_next`` / ``next_batch`` / ``reset`` /
``list_of_examples``), following the public dataset layout from
Farha & Gall (MS-TCN, CVPR 2019). If your original ``batch_gen.py`` differs,
drop it in here instead -- the rest of the codebase only depends on this
interface.
"""

import random
from typing import Dict, List, Tuple

import numpy as np
import torch


class BatchGenerator:
    def __init__(self, num_classes: int, actions_dict: Dict[str, int], gt_path: str, features_path: str, sample_rate: int = 1):
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

        self.list_of_examples: List[str] = []
        self.index = 0

    def has_next(self) -> bool:
        return self.index < len(self.list_of_examples)

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def read_data(self, vid_list_file: str):
        with open(vid_list_file, "r") as f:
            self.list_of_examples = [line.strip() for line in f if line.strip()]
        random.shuffle(self.list_of_examples)

    def _load_example(self, vid: str) -> Tuple[np.ndarray, np.ndarray]:
        features = np.load(self.features_path + vid.split(".")[0] + ".npy")
        with open(self.gt_path + vid, "r") as f:
            content = f.read().split("\n")[:-1]

        classes = np.zeros(min(np.shape(features)[1], len(content)))
        for i in range(len(classes)):
            classes[i] = self.actions_dict[content[i]]
        return features[:, :: self.sample_rate], classes[:: self.sample_rate]

    def next_batch(self, batch_size: int, shuffle: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        batch = self.list_of_examples[self.index : self.index + batch_size]
        self.index += batch_size

        batch_input, batch_target = [], []
        for vid in batch:
            features, classes = self._load_example(vid)
            batch_input.append(features)
            batch_target.append(classes)

        length_of_sequences = [x.shape[0] for x in batch_target]
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * -100
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)

        for i in range(len(batch_input)):
            L = np.shape(batch_input[i])[1]
            batch_input_tensor[i, :, :L] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :L] = torch.from_numpy(batch_target[i])
            mask[i, :, :L] = torch.ones(self.num_classes, L)

        return batch_input_tensor, batch_target_tensor, mask, batch
