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

## Method overview

| Paper section | What it does | Code |
|---|---|---|
| 3.1 Dual-Branch Pipeline | Independent RGB/flow-student/teacher segmenters | [`relate/segmenters.py`](relate/segmenters.py), [`relate/modality_bridge.py`](relate/modality_bridge.py) |
| 3.2 Alignment Fusion Transformer (Eq. 1) | Action-gated attention that suppresses misaligned cues before fusing modalities | [`relate/alignment_fusion.py`](relate/alignment_fusion.py) |
| 3.3 Training Objective | `L = L_RGB + L_OF + L_J + gamma * L_CMKD` | [`relate/losses.py`](relate/losses.py) |
| 3.4 Prediction Refinement | Training-free confidence-based refinement of the joint prediction | [`relate/refinement.py`](relate/refinement.py) |
| 4.1.1 Evaluation metrics | Segmental F1@{10,25,50}, Edit, frame accuracy | [`relate/metrics.py`](relate/metrics.py) |

```
                Flow Branch (student, CMKD-supervised)
RGB frames ─┬─▶ RGBSegmenter ──────────────┐
            │                              ▼
            └─▶ ModalityBridge ─▶ FlowSegmenter ─▶ AlignmentFusionTransformer ─▶ JointSegmenter ─▶ PredictionRefinement ─▶ labels
                                        ▲
Optical flow (train only) ─▶ RefSegmenter (teacher, frozen)
```

### Action-Gated Attention (Eq. 1)

For each modality, action anchors `a_i` are obtained by average-pooling
query embeddings over each *predicted* segment. Keys are gated by their
similarity to the anchors before being used to modulate the values:

```
A = sigmoid( sum_i (K ⊙ a_i) / sqrt(d) )
Z~ = A ⊙ V
```

This is what lets RELATE keep informative transferred motion cues while
suppressing the ones that don't align with the action structure — see
`relate/alignment_fusion.py::AlignmentFusionTransformer._action_gated_attention`.

## Repository layout

```
relate/
  layers.py             # AttLayer, AttModule, positional encoding, feed-forwards
  backbones.py           # MS-TCN and ASFormer stage implementations
  segmenters.py           # RGBSegmenter / FlowSegmenter / RefSegmenter / JointSegmenter
  alignment_fusion.py     # Alignment Fusion Transformer + Table 3 fusion ablations
  modality_bridge.py      # RGB -> optical-flow feature bridge (Sec. 3.1)
  losses.py                # segmentation loss + KD / RKD / GCR (Table 4)
  metrics.py                # F1@{10,25,50}, Edit, frame accuracy
  refinement.py              # training-free prediction refinement (Sec. 3.4)
  data.py                    # I3D-feature batch generator
scripts/
  train.py    # trains RGB + Flow (student) + Joint segmenters and the AFT
  predict.py  # RGB-only inference, with optional prediction refinement
  evaluate.py # computes F1 / Edit / Acc from saved predictions
configs/      # reference hyperparameters per dataset (Sec. 4.1.2)
tests/        # CPU-only smoke tests (shapes, forward passes, no data needed)
docs/external_dependencies.md  # I3D extraction, TV-L1 flow, FACT backbone
```

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

## Notes on this release

This repository is a cleaned-up version of the original research code:

- Hardcoded absolute paths were replaced with CLI arguments / config files.
- `batch_gen.py` and `eval.py` were referenced by the original scripts but
  not included in the source archive; `relate/data.py` and
  `relate/metrics.py` are clean reimplementations of the standard
  MS-TCN-style interfaces they exposed. If your original files differ, they
  can be dropped in as long as they expose the same interface.
- The prediction refinement step (Sec. 3.4) was implemented (`utils.py`)
  but not wired into the original inference loop; it is enabled by default
  in `scripts/predict.py` here.
- A shape inconsistency between the MS-TCN and ASFormer feature outputs
  (which broke `AlignmentFusionTransformer` for the ASFormer backbone) was
  fixed in `relate/backbones.py` / `relate/segmenters.py`.
- Experimental code not described in the paper (an additional C2KD-based
  distillation variant, a Gram-Schmidt feature booster, a few unused
  visualization/analysis utilities) was left out for clarity. The three
  distillation methods reported in Table 4 (KD, RKD, GCR) are included in
  `relate/losses.py`.
- The FACT backbone (third segmenter in Table 1) is an external, separately
  licensed repository and is not vendored here — see
  [`docs/external_dependencies.md`](docs/external_dependencies.md) for how
  to plug it in. MS-TCN and ASFormer are fully implemented.

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
