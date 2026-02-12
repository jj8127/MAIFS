# Subproject Conda Environment Snapshots

**Generated**: 2026-01-22
**Scope**: TruFor, CAT-Net, MVSS-Net, OmniGuard, HiNet

---

## Overview

This document records conda environment snapshots created from the official subproject specs.
Each snapshot is exported with `conda env export --no-builds` and has the `prefix:` entry removed.

---

## Snapshot Files

| Subproject | Conda Env | Snapshot File |
|---|---|---|
| TruFor | `maifs-trufor` | `envs/conda-maifs-trufor.yml` |
| CAT-Net | `maifs-catnet` | `envs/conda-maifs-catnet.yml` |
| MVSS-Net | `maifs-mvss` | `envs/conda-maifs-mvss.yml` |
| OmniGuard | `maifs-omniguard` | `envs/conda-maifs-omniguard.yml` |
| HiNet | `maifs-hinet` | `envs/conda-maifs-hinet.yml` |

---

## Recreate Command

```bash
conda env create -f envs/conda-maifs-trufor.yml
conda env create -f envs/conda-maifs-catnet.yml
conda env create -f envs/conda-maifs-mvss.yml
conda env create -f envs/conda-maifs-omniguard.yml
conda env create -f envs/conda-maifs-hinet.yml
```

---

## Smoke Test Summary

### TruFor
- **Status**: OK (CPU inference smoke test)
- **Command**:
  ```bash
  conda run -n maifs-trufor python test.py -g -1 -in /tmp/maifs_trufor_smoke.png -out /tmp/maifs_trufor_out -exp trufor_ph3 TEST.MODEL_FILE "pretrained_models/trufor.pth.tar"
  ```
- **Output**: `/tmp/maifs_trufor_out/maifs_trufor_smoke.png.npz`
- **Notes**:
  - `jpegio` is installed as `0.2.4` (the git build that succeeded locally).
  - `tqdm` and `yacs` were added for runtime imports.

### CAT-Net
- **Status**: OK (inference smoke test)
- **Command**:
  ```bash
  conda run -n maifs-catnet python tools/infer.py
  ```
- **Notes**:
  - WISE checkpoint placed at `output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar`.
  - Output heatmap: `output_pred/smoke.png`.
  - `torch==1.6.0+cu101` and `torchvision==0.7.0+cu101` are used to load the zip-format checkpoint.
  - `jpegio` is installed as `0.2.4` (git build).

### MVSS-Net
- **Status**: OK (model load + forward pass)
- **Notes**:
  - First run downloads ResNet50 weights to `~/.cache/torch/checkpoints/`.
  - Single-image smoke output: `MVSS-Net-master/output_smoke/smoke_pred.png`.

### HiNet
- **Status**: OK (checkpoint load + forward pass)
- **Notes**:
  - Smoke test uses `checkpoint/model.pt` with a synthetic input tensor.
  - Test outputs: `HiNet-main/image/cover/00000.png`, `HiNet-main/image/steg/00000.png`, `HiNet-main/image/secret/00000.png`, `HiNet-main/image/secret-rev/00000.png`.
  - Dummy inputs created under `HiNet-main/data/DIV2K_train_HR/` and `HiNet-main/data/DIV2K_valid_HR/`.

### OmniGuard
- **Status**: OK (script smoke test)
- **Notes**:
  - Script output: `OmniGuard-main/output_smoke/stego.png` (with `cover.png`, `secret.png`).
  - `checkpoint/iml_vit.pth` uses zip-based `torch.save` format and does not load on PyTorch 1.0.1.
  - `maifs-omniguard` now uses `torch==1.13.1+cu117`, so `torch.load` succeeds and CUDA ops work.

---

## Real-Data Spot Checks

### MVSS-Net
- **Input**: `datasets/CASIA2_subset/Tp/Tp_D_CND_M_N_ani00018_sec00096_00138.tif`
- **Output**: `MVSS-Net-master/output_real/Tp_D_CND_M_N_ani00018_sec00096_00138_pred.png`

### HiNet
- **Input**: `datasets/DIV2K_subset` (train/valid copied into `HiNet-main/data/`)
- **Outputs**: `HiNet-main/image/cover/*.png`, `HiNet-main/image/steg/*.png`, `HiNet-main/image/secret/*.png`, `HiNet-main/image/secret-rev/*.png`

### OmniGuard
- **Input**: `datasets/DIV2K_subset/DIV2K_train_HR/0001.png` (cover), `0002.png` (secret)
- **Output**: `OmniGuard-main/output_real/stego.png` (with `cover.png`, `secret.png`)

---

## Known Deviations From Upstream Specs

- TruFor/CAT-Net use `jpegio==0.2.4` due to build compatibility.
- CAT-Net uses `torch==1.6.0+cu101` and `torchvision==0.7.0+cu101` to load the WISE checkpoint.
- OmniGuard uses `torch==1.13.1+cu117` for `checkpoint/iml_vit.pth` and CUDA-only ops; HiNet still uses `pytorch==1.0.1`.
- MVSS `torchvision` is a conda build labeled `0.6.0a0+35d732a`, which corresponds to the 0.6.1 series.
