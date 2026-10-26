# Lightweight CNN for ISIC 2020 melanoma classification

Two compact convolutional neural networks trained **from scratch** using the team's patient-level split and shared `isic_utils.py` preprocessing. This branch contributes the lightweight CNN experiments, separate from the ResNet work.

## Results (validation only)

| Metric | CNN v1 | CNN v2 |
|---|---:|---:|
| Convolutional blocks | 2 | 3 (with BatchNorm) |
| Trainable parameters | 19,457 | 93,825 |
| Best validation AUPRC | 0.0498 | 0.0778 |
| Validation AUROC at selected checkpoint | 0.7482 | 0.7978 |
| Precision at validation-F1 threshold | 0.0701 | 0.1236 |
| Recall at validation-F1 threshold | 0.1765 | 0.1294 |
| F1 at validation-F1 threshold | 0.1003 | 0.1264 |
| Metric | CNN v2 |
|---|---|
| Accuracy | 0.9674 (96.74%) |
| Specificity | 0.9830 (98.30%) |
| MCC | 0.1099 |

CNN v2 improved AUPRC by approximately 56.2% relative to v1, at 4.8 times the parameter count. CNN v2's best validation checkpoint occurred at epoch 20, with 76.16 minutes total training time in the recorded Colab run. Despite improvement, both models have limited sensitivity for melanoma and are **not suitable for clinical use**.

## Training configuration

- ISIC 2020 cached RGB images; 224 × 224 model input
- Patient-level split: 23,364 train / 4,663 validation / 4,674 held-out test
- Shared augmentations and ImageNet normalization via `isic_utils.build_transforms`
- AdamW (`lr=2e-4`, `weight_decay=1e-4`), class-weighted `BCEWithLogitsLoss` with `pos_weight=sqrt(negative/positive)`
- 20-epoch maximum, 2-epoch warmup and cosine decay, early stopping patience 5
- Select checkpoint using **validation AUPRC**; F1 threshold selected on validation
- **No test-set evaluation** in this contribution

## Files and reproducibility

- `train_lightweight_cnn.ipynb`: clean Colab notebook for both models, training, evaluation, plots, and exports
- `validation_summary.csv`: metrics from the original experiment
- `requirements.txt`: Python dependencies (use compatible versions for your runtime)
- `.gitignore`: excludes large datasets, checkpoints, and locally generated results

Before running, provide the team's `isic_utils.py` and corresponding `metadata.csv`, plus the 6.4 GB cached image array, and update the three paths near the start of the notebook. In the original Colab run they were `/content/isic/isic_utils.py`, `/content/isic/metadata.csv`, and `/content/images_uint8_new.npy`. The notebook has cleared execution outputs and does not bundle the dataset or trained checkpoints.

**Important:** The original notebook contained transient OneDrive download code and verbose multiprocessing warnings; those were removed from this shareable version. Original experimental result values are documented, but reproducibility depends on the dataset, software environment, and stochastic training.

**AI usage disclosure:** AI assistance was used for implementation guidance, debugging, and documentation. Model training and evaluation results were produced by the experiments.
