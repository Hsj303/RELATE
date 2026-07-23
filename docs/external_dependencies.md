# External dependencies

RELATE's dual-branch pipeline, Alignment Fusion Transformer, modality
bridge, CMKD losses, and prediction refinement are all self-contained in
this repository (`relate/`). Three pieces described in the paper are
external, published components that this repo does not re-implement:

## 1. I3D feature extraction (RGB and optical flow)

The paper uses I3D features (Carreira & Zisserman, 2017), 1024-d for RGB
and 1024-d for optical flow, concatenated to 2048-d per frame. Extract
these with a public I3D feature extractor, e.g.
[piergiaj/pytorch-i3d](https://github.com/piergiaj/pytorch-i3d), and lay
features out as:

```
data/<dataset>/features/<video_id>.npy   # shape (2048, T): [0:1024]=RGB, [1024:2048]=OF
data/<dataset>/groundTruth/<video_id>    # one action label per line
data/<dataset>/mapping.txt               # "<id> <action_name>" per line
data/<dataset>/splits/train.split<K>.bundle
data/<dataset>/splits/test.split<K>.bundle
```

This is the same layout used by MS-TCN and ASFormer's public releases.

## 2. Optical flow (TV-L1)

The optical-flow teacher (`RefSegmenter`) is trained on optical flow I3D
features, extracted with the TV-L1 algorithm (Wedel et al., 2009) at
224x224 with a 21-frame window (paper, Sec. 4.4.2, footnote 1). Any TV-L1
implementation (e.g. OpenCV's `cv2.optflow.DualTVL1OpticalFlow`) works; flow
is only needed to prepare training data for the teacher and is never used
at inference time — that's the entire point of RELATE.

## 3. FACT backbone (optional third segmenter, Table 1)

Table 1 additionally reports RELATE with FACT (Lu & Elhamifar, CVPR 2024)
as the backbone segmenter, instead of MS-TCN or ASFormer. FACT is a
frame-action cross-attention matching model with its own repository
([ZijiaLewisLu/CVPR2024-FACT](https://github.com/ZijiaLewisLu/CVPR2024-FACT)).
It is not vendored here since it's a separate cited work with its own
license and config system. To reproduce the FACT rows of Table 1:

1. Clone FACT alongside this repo and install it per its own instructions.
2. Implement a `FACTSegmenter` wrapper matching the
   `forward(x, mask) -> (stage_logits, stage_features)` interface used by
   `relate/segmenters.py`, calling into FACT's `FACT` / `FACT_joint`
   modules.
3. Pass `--backbone FACT` (not yet wired into `scripts/train.py` — the
   `_BaseSegmenter` dispatch in `relate/segmenters.py` is the place to add
   it) once the wrapper is in place.

MS-TCN and ASFormer are fully implemented in this repo (`relate/backbones.py`,
`relate/segmenters.py`) and are the recommended starting point for
reproducing or extending RELATE.
