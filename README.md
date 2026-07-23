# RELATE: Distill, Suppress, and Fuse

Official code for **"Distill, Suppress, and Fuse: Cross-Modal Knowledge
Integration for Optical Flow-Free Temporal Action Segmentation"**
(ICML 2026).

RELATE (RGB-based action s**E**gmentation with a**L**ignment g**ATE**d
fusion) does temporal action segmentation from **RGB frames only** at
inference time. During training, it distills motion knowledge from an
optical-flow teacher, but — unlike prior cross-modal distillation methods —
it selectively **suppresses** transferred cues that are misaligned with the
video's action structure instead of blindly absorbing all of them. The
result matches two-stream (RGB + optical flow) accuracy while being
**~175x faster** at inference, since no optical flow is ever computed.

<p align="center">
  <img src="https://img.shields.io/badge/ICML-2026-blue" alt="ICML 2026">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
</p>


## Installation

```bash
git clone <this-repo-url>
cd RELATE
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.0+. A GPU is recommended for training
but not required to run the tests.

## Data preparation

RELATE is evaluated on **GTEA**, **50Salads**, and **Breakfast**, using
2048-d I3D features (1024-d RGB + 1024-d optical flow) laid out as:

```
data/<dataset>/features/<video_id>.npy
data/<dataset>/groundTruth/<video_id>
data/<dataset>/mapping.txt
data/<dataset>/splits/train.split<K>.bundle
data/<dataset>/splits/test.split<K>.bundle
```

This is the same layout used by the public MS-TCN / ASFormer releases. See
[`docs/external_dependencies.md`](docs/external_dependencies.md) for how to
extract I3D features and optical flow (these are external, published tools
and are not part of this repository).

## Usage

**1. Pretrain the optical-flow teacher** (`RefSegmenter`) on optical-flow
I3D features only, with a plain segmentation loss — any standard MS-TCN /
ASFormer training loop works, since the teacher has no RELATE-specific
components. Save its checkpoint as `<ckpt>.pt`.

**2. (Optional) Pretrain the modality bridge**, if you want the flow branch
to receive bridged RGB->OF features instead of raw RGB (Sec. 3.1):

```python
from relate.modality_bridge import ModalityBridge, train_modality_bridge
from relate.data import BatchGenerator

bridge = ModalityBridge(dim=1024)
batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
batch_gen.read_data(train_list_file)
train_modality_bridge(bridge, batch_gen, device="cuda:0", save_dir="./runs/gtea/bridge")
```

**3. Train RELATE:**

```bash
python scripts/train.py \
    --data-root ./data --dataset gtea --split 1 \
    --backbone MS-TCN --fusion afti --distill kd \
    --teacher-ckpt ./runs/gtea/teacher/epoch-120.pt \
    --output-dir ./runs/gtea/split_1 \
    --epochs 100 --lr 0.001 --gamma 0.5
```

Reference hyperparameters per dataset (from Sec. 4.1.2) are in
[`configs/`](configs/). `--fusion concat` and `--distill {rkd,gcr}` reproduce
the Table 3 / Table 4 ablations.

**4. Run RGB-only inference** (no optical flow needed):

```bash
python scripts/predict.py \
    --data-root ./data --dataset gtea --split 1 \
    --checkpoint-dir ./runs/gtea/split_1 --epoch 100 \
    --output-dir ./runs/gtea/split_1/predictions
```

Add `--no-refine` to disable the Sec. 3.4 prediction refinement step.

**5. Evaluate:**

```bash
python scripts/evaluate.py \
    --data-root ./data --dataset gtea --split 1 \
    --pred-dir ./runs/gtea/split_1/predictions
```

## Testing

```bash
pytest tests/ -v
```

The test suite is CPU-only and needs no dataset — it checks that the
dual-branch pipeline, Alignment Fusion Transformer, distillation losses,
prediction refinement, and metrics all wire together and produce the
expected tensor shapes.

## Citation

```bibtex
@inproceedings{han2026relate,
  title     = {Distill, Suppress, and Fuse: Cross-Modal Knowledge Integration
               for Optical Flow-Free Temporal Action Segmentation},
  author    = {Han, Seungjin and Kim, Gyeong-hyeon and Kim, Eunwoo},
  booktitle = {Proceedings of the 43rd International Conference on Machine
               Learning (ICML)},
  volume    = {306},
  publisher = {PMLR},
  year      = {2026}
}
```

## Acknowledgements

This research was supported in part by the Chung-Ang University Graduate
Research Scholarship in 2023 and in part by the Institute of Information &
Communications Technology Planning & Evaluation (IITP) grant funded by the
Korea government (MSIT) (2021-0-01341, Artificial Intelligence Graduate
School Program, Chung-Ang University).

## License

MIT — see [LICENSE](LICENSE). The MS-TCN- and ASFormer-derived backbone
components (`relate/backbones.py`, `relate/layers.py`) are adapted from the
respective public, MIT-licensed reference implementations of
Farha & Gall (2019) and Yi et al. (2021).
