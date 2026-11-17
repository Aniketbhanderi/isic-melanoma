# Lightweight CNN v1 and v2 — ISIC 2020

Two convolutional networks trained **from scratch** using the team's shared
`isic_utils.py` preprocessing and patient-level splits.

- CNN v1: 19,457 trainable parameters (two convolutional blocks).
- CNN v2: 93,825 trainable parameters (three blocks with BatchNorm).
- Class-weighted BCEWithLogitsLoss, AdamW, warmup/cosine scheduling,
  early stopping/checkpoint selection on **validation AUPRC**.
- F1-maximizing decision thresholds selected **on validation only**.

## Results

See `validation_summary.csv` for **new-run validation results**.
See `test_summary.csv` for one-time held-out internal test results.

## Reproduction

Open `train_lightweight_cnn.ipynb` in Google Colab. Supply matching
`metadata.csv`, `images_uint8_new.npy`, and team `isic_utils.py`; edit the
input paths in the setup cell. Best `.pth` files are saved in temporary Colab storage at
`/content/checkpoints` and browser downloads are triggered on each improvement.
**Verify the final best checkpoints were downloaded to your computer.**

The 2020 competition test set is unlabeled; test results (if present) are
from the team's labeled internal patient-level holdout, not the competition test.

## Attribution

Shared preprocessing and dataset splitting are the team's work. The
CNN v1/v2 from-scratch architectures, training and evaluation are the
contribution of Muhammad Mubinur Rahman. AI assistance was used in code
preparation and debugging; results require actual execution.
