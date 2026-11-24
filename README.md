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

## ISIC-DICM-17k training

Run `python preprocess_dicm.py` to build a separate cache from the JPEG images.
It keeps only rows with a definitive `diagnosis_2` of melanoma or benign
melanocytic proliferation, mapping them to targets 1 and 0 respectively. Then
run `python train.py --cache-dir artifacts/preprocessed_dicm17k --output
artifacts/efficientnet_dicm17k_model.pt` to train the same EfficientNet-B0
pipeline with metadata. The DICM manifest and model outputs remain separate
from the ISIC 2020 artifacts. Patient IDs are missing for many DICM records;
the cache groups those by lesion ID where available and otherwise by image, so
patient leakage cannot be ruled out for records without patient identifiers.