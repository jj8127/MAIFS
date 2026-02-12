# Datasets and Subproject Runbook

**Updated**: 2026-01-23
**Scope**: CASIA v2.0, DIV2K, IMD2020, GenImage (BigGAN subset)

---

## 1) Shared layout

- All datasets live under `datasets/` (not tracked by git).
- Use `*_subset` directories for quick runs and smoke checks.
- Prefer symlinks into subprojects to avoid duplicate copies.

### Recommended symlinks (run from repo root)

```bash
mkdir -p CAT-Net-main/datasets MVSS-Net-master/data OmniGuard-main/data TruFor-main/TruFor_train_test/datasets HiNet-main/data

ln -s ../../datasets/CASIA2_subset CAT-Net-main/datasets/CASIA
ln -s ../../datasets/IMD2020_subset CAT-Net-main/datasets/IMD2020

ln -s ../../datasets/CASIA2_subset MVSS-Net-master/data/CASIA2_subset
ln -s ../../datasets/IMD2020_subset MVSS-Net-master/data/IMD2020_subset
ln -s ../../datasets/GenImage_subset/BigGAN/val MVSS-Net-master/data/GenImage

ln -s ../../datasets/DIV2K_subset OmniGuard-main/data/DIV2K_subset
ln -s ../../datasets/CASIA2_subset OmniGuard-main/data/CASIA2_subset

ln -s ../../datasets/DIV2K_subset/DIV2K_train_HR HiNet-main/data/DIV2K_train_HR
ln -s ../../datasets/DIV2K_subset/DIV2K_valid_HR HiNet-main/data/DIV2K_valid_HR

ln -s ../../../datasets/IMD2020_subset TruFor-main/TruFor_train_test/datasets/IMD2020
```

---

## 2) Dataset sources and staging

### CASIA v2.0 (Kaggle: divg07/casia-20-image-tampering-detection-dataset)

- Download (requires Kaggle token in `~/.config/kaggle/kaggle.json`):
  ```python
  import kagglehub
  path = kagglehub.dataset_download("divg07/casia-20-image-tampering-detection-dataset")
  print(path)
  ```
- Staged subset: `datasets/CASIA2_subset/`
  - `Au`: 50 authentic images
  - `Tp`: 50 tampered images
  - `GT`: 38 ground-truth masks

### DIV2K (Kaggle: soumikrakshit/div2k-high-resolution-images)

- Download:
  ```python
  import kagglehub
  path = kagglehub.dataset_download("soumikrakshit/div2k-high-resolution-images")
  print(path)
  ```
- Staged subset: `datasets/DIV2K_subset/`
  - `DIV2K_train_HR`: 50 images
  - `DIV2K_valid_HR`: 20 images

### IMD2020 (Generative Inpainting, Yu et al.)

- Source: `https://staff.utia.cas.cz/novozada/db/`
- Download (example):
  ```bash
  mkdir -p datasets/IMD2020_raw
  curl -L -o datasets/IMD2020_raw/IMD2020_Generative_Image_Inpainting_yu2018_01.zip \
    https://staff.utia.cas.cz/novozada/db/IMD2020_Generative_Image_Inpainting_yu2018_01.zip
  curl -L -o datasets/IMD2020_raw/IMD2020_Generative_Image_Inpainting_yu2018_mask.zip \
    https://staff.utia.cas.cz/novozada/db/IMD2020_Generative_Image_Inpainting_yu2018_mask.zip
  ```
- Staged subset:
  - `datasets/IMD2020_subset/IMD2020_Generative_Image_Inpainting_yu2018_01/images` (200)
  - `datasets/IMD2020_subset/IMD2020_Generative_Image_Inpainting_yu2018_01/masks` (200)

### GenImage (BigGAN subset)

- Repo: `datasets/GenImage` (clone from `https://github.com/GenImage-Dataset/GenImage.git`)
- Dataset host: Google Drive link in `datasets/GenImage/Readme.md`
  - Download only the `BigGAN` folder
  - Files used:
    - `imagenet_ai_0419_biggan.zip`
    - `imagenet_ai_0419_biggan.z01` to `imagenet_ai_0419_biggan.z07`
- Extract (requires `7z`):
  ```bash
  7z x datasets/GenImage_data/BigGAN/imagenet_ai_0419_biggan.zip -odatasets/GenImage_data/BigGAN
  ```
- Staged subset:
  - `datasets/GenImage_subset/BigGAN/val/ai` (50 images)
  - `datasets/GenImage_subset/BigGAN/val/nature` (50 images)

---

## 3) Subproject runbook (real data)

### MVSS-Net (localization)

- **Requires**: `MVSS-Net-master/ckpt/mvssnet_casia.pt`
- **Dataset**: CASIA v2.0 tampered images (`datasets/CASIA2_subset/Tp`)
- **Run**:
  ```bash
  conda run -n maifs-mvss python scripts/run_mvss_real.py \
    --image datasets/CASIA2_subset/Tp/<file>.tif
  ```
- **Output**: `MVSS-Net-master/output_real/<image>_pred.png`

### HiNet (watermark embed/extract)

- **Requires**: `HiNet-main/checkpoint/model.pt`
- **Dataset**: DIV2K subset
- **Run**:
  ```bash
  conda run -n maifs-hinet python HiNet-main/test.py
  ```
- **Outputs**:
  - `HiNet-main/image/cover/*.png`
  - `HiNet-main/image/steg/*.png`
  - `HiNet-main/image/secret/*.png`
  - `HiNet-main/image/secret-rev/*.png`

### OmniGuard (watermark embed demo)

- **Requires**: `OmniGuard-main/checkpoint/model_checkpoint_01500.pt`
- **Dataset**: DIV2K subset (cover/secret)
- **Run**:
  ```bash
  conda run -n maifs-omniguard python scripts/run_omniguard_real.py \
    --cover datasets/DIV2K_subset/DIV2K_train_HR/<cover>.png \
    --secret datasets/DIV2K_subset/DIV2K_train_HR/<secret>.png
  ```
- **Outputs**:
  - `OmniGuard-main/output_real/cover.png`
  - `OmniGuard-main/output_real/secret.png`
  - `OmniGuard-main/output_real/stego.png`

### CAT-Net and TruFor (optional)

- CAT-Net uses `CAT-Net-main/datasets/` for dataset roots (see `CAT-Net-main/project_config.py`).
- TruFor uses `TruFor-main/TruFor_train_test/datasets/` for dataset roots (see `TruFor-main/TruFor_train_test/project_config.py`).

---

## 4) Output locations (summary)

| Subproject | Output path |
| --- | --- |
| MVSS-Net | `MVSS-Net-master/output_real/*.png` |
| HiNet | `HiNet-main/image/{cover,steg,secret,secret-rev}/*.png` |
| OmniGuard | `OmniGuard-main/output_real/{cover,secret,stego}.png` |
