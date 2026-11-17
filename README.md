# Lightweight CNN for ISIC 2020 Melanoma Classification

This branch contains two compact convolutional neural networks (CNN v1 and CNN v2) trained **from scratch** on the SIIM-ISIC 2020 melanoma classification dataset. The experiments use the team's shared preprocessing and patient-level split and are separate from the team's pretrained ResNet experiments.

**Two experimental versions are retained:**

- **Version 1 — Original experiment:** initial validation-only CNN results, documented in the original notebook named `train_lightweight_cnn.ipynb` here.
- **Version 2 — Retrained experiment:** a subsequent run with fresh validation results and **one held-out test evaluation**, documented in `train_lightweight_cnn_colab_test.ipynb`. The retraining run also adds local checkpoint saving and download prompts after Google Drive mounting failed in Colab.

The results from the two versions are reported separately rather than overwriting the earlier experiment.

## Model architectures

| Model | Architecture | Trainable parameters |
|---|---|---:|
| CNN v1 | 2 convolutional layers | 19,457 |
| CNN v2 | 3 convolutional layers with BatchNorm | 93,825 |

Both models are lightweight, from-scratch baselines. CNN v2 has approximately 4.82 times as many trainable parameters as CNN v1.

## Version 1 — Original experiment (validation only)

| Metric | CNN v1 | CNN v2 |
|---|---:|---:|
| Best validation AUPRC | 0.0498 | 0.0778 |
| Validation AUROC at selected checkpoint | 0.7482 | 0.7978 |
| Precision at validation-F1 threshold | 0.0701 | 0.1236 |
| Recall (sensitivity) at validation-F1 threshold | 0.1765 | 0.1294 |
| F1 at validation-F1 threshold | 0.1003 | 0.1264 |

Additional original CNN v2 validation metrics: **accuracy 0.9674**, **specificity 0.9830**, **MCC 0.1099**.

In this original run, CNN v2 improved validation AUPRC by approximately **56.2%** relative to CNN v1. Its best validation checkpoint was at epoch 20; the recorded training time was **76.16 minutes**. **No held-out test results were reported for Version 1.**

## Version 2 — Retrained experiment (validation)

The second run used the same model designs and patient-level data partition, but produced a new set of training outcomes. The following numbers come from the uploaded final notebook's validation outputs.

| Metric | CNN v1 | CNN v2 |
|---|---:|---:|
| Best validation AUPRC | 0.0520 | 0.0584 |
| Validation AUROC | 0.7492 | 0.7903 |
| Accuracy | 0.9292 | 0.9003 |
| Precision | 0.0732 | 0.0642 |
| Recall (sensitivity) | 0.2471 | 0.3294 |
| F1 | 0.1129 | 0.1075 |
| Specificity | 0.9419 | 0.9109 |
| Balanced accuracy | 0.5945 | 0.6201 |
| MCC | 0.1052 | 0.1104 |
| Selected validation-F1 threshold | 0.2226 | 0.1996 |

In Version 2, CNN v2's best validation checkpoint occurred at **epoch 18**, and its recorded training time was **81.64 minutes**. CNN v2 had higher validation AUPRC and sensitivity, while CNN v1 had higher validation precision, F1, accuracy, and specificity at the selected thresholds.

## Version 2 — Held-out test results

The retrained models were evaluated on the **4,674-image patient-level held-out test split** using decision thresholds selected from validation data. The test set was not used to select thresholds.

| Metric | CNN v1 | CNN v2 |
|---|---:|---:|
| AUPRC | 0.0485 | **0.0850** |
| AUROC | 0.7424 | **0.8307** |
| Accuracy | **0.9204** | 0.8962 |
| Precision | 0.0469 | **0.0685** |
| Recall (sensitivity) | 0.1829 | **0.3902** |
| F1 | 0.0746 | **0.1166** |
| Specificity | **0.9336** | 0.9053 |
| Balanced accuracy | 0.5583 | **0.6478** |
| MCC | 0.0606 | **0.1294** |
| True positives (TP) | 15 | **32** |
| False positives (FP) | 305 | 435 |
| False negatives (FN) | 67 | **50** |
| True negatives (TN) | 4,287 | 4,157 |
| Threshold (selected on validation) | 0.2226 | 0.1996 |

**Interpretation:** CNN v2 detected **32 of 82** melanoma cases in the test split, compared with **15 of 82** for CNN v1, but generated more false positives (435 vs. 305). CNN v2's test AUPRC was approximately **75.4% higher** than CNN v1's. Accuracy alone is misleading because melanoma cases are rare. Both models still miss many melanoma cases and **are not suitable for clinical deployment**.

**Do not compare Version 1 validation metrics directly with Version 2 test metrics as if they came from the same evaluation split.** The two versions are separate training runs, and their results reflect training variability as well as different evaluation partitions when comparing validation with test.

## Dataset and preprocessing

The shared ISIC 2020 preprocessing pipeline was developed by **Monirul Islam and Aniket Bhanderi** and reused in these lightweight CNN experiments. The lightweight CNN contributions by **Muhammad Mubinur Rahman** include both model architectures, training, evaluation, and comparison.

- Cached RGB images: **32,701** images, shape `(32701, 256, 256, 3)`, stored as `uint8` in a roughly 6.4 GB NumPy array
- Input resolution: **224 × 224**
- Patient-level split: **23,364 train / 4,663 validation / 4,674 held-out test**
- Malignant examples: **414 train / 85 validation / 82 test**
- Shared augmentations and ImageNet normalization through `isic_utils.build_transforms`

## Training configuration

- Optimizer: **AdamW**, `lr=2e-4`, `weight_decay=1e-4`
- Loss: class-weighted `BCEWithLogitsLoss` with `pos_weight=sqrt(negative/positive)`
- Up to **20 epochs**, **2-epoch warmup** followed by cosine learning-rate decay
- Early-stopping patience: **5**
- Model selection based on **validation AUPRC**
- Decision threshold selected by **validation F1**; reused unchanged for held-out test evaluation
- Patient-level splitting to reduce leakage between training, validation, and test images

## Files and reproducibility

- `train_lightweight_cnn.ipynb` — original shareable notebook for Version 1
- `train_lightweight_cnn_colab_test.ipynb` — Version 2 notebook containing retraining, validation, held-out test evaluation, plots, and exports
- `requirements.txt` — dependencies, where present
- `.gitignore` — excludes large datasets and checkpoints, where configured

For the retrained notebook, provide the team's `isic_utils.py`, `metadata.csv`, and the cached image array. The Colab paths used in the recorded run were:

```text
/content/isic/isic_utils.py
/content/isic/metadata.csv
/content/images_uint8_new.npy
```

Version 2 saves best model checkpoints under `/content/checkpoints/` and triggers browser downloads. **Colab local storage is temporary; users must verify that both `.pth` checkpoint files and the results ZIP were actually downloaded to permanent storage.** The notebook records the paths and download attempts, but cannot establish that the files reached the user's computer. Large datasets and trained weights are not included in this README.

The experiments can vary across runs due to random initialization, stochastic augmentation, training order, and environment differences. The original validation-only results are retained as a separate historical experiment rather than being presented as test results.

## AI usage disclosure

AI assistance was used for implementation guidance, debugging, checkpoint-handling workflow, and documentation. The reported model training and evaluation results were generated by the actual experiments.
