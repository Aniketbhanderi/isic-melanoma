# isic-melanoma

## EfficientNet training model

The patient-grouped training pipeline in [train.py](train.py) uses the
deduplicated RGB image cache created by [preprocess.ipynb](preprocess.ipynb).
Its architecture follows the linked SIIM-ISIC solution at a high level:
EfficientNet image features, an optional metadata embedding (age, sex, and
anatomical site), and five dropout passes whose classification logits are
averaged. Stratified folds keep patient groups intact, and the best validation
ROC-AUC checkpoint is saved. This is an experimental model, not a diagnostic
tool.

Install the project dependencies, then run `python train.py` to train the
default EfficientNet-B0 model for 15 epochs on fold 0. Checkpoints and epoch
history are written under `artifacts/`. Use `--no-pretrained` to avoid
downloading ImageNet weights, `--no-use-metadata` for image-only training, or
`--fold 1` to train a different patient fold.