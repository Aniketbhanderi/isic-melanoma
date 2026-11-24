"""Build the RGB image cache for the ISIC-DICM-17k melanoma task.

Only records labeled as melanoma or benign melanocytic proliferations in
``diagnosis_2`` are retained. The output follows the cache contract consumed by
train.py and is kept separate from the ISIC 2020 cache.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "isic-dicm-17k.csv"
DEFAULT_IMAGES = ROOT / "images_isic-dicm-17k"
DEFAULT_CACHE = ROOT / "artifacts" / "preprocessed_dicm17k"
IMAGE_SIZE = 256
MELANOMA = "Malignant melanocytic proliferations (Melanoma)"
BENIGN = "Benign melanocytic proliferations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CACHE)
    return parser.parse_args()


def build_cache(csv_path: Path, images_dir: Path, output_dir: Path) -> None:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset metadata CSV not found: {csv_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")

    frame = pd.read_csv(csv_path)
    required = {"isic_id", "diagnosis_2", "patient_id", "lesion_id", "age_approx", "sex", "anatom_site_general"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset CSV is missing required columns: {sorted(missing)}")
    frame = frame.loc[frame["diagnosis_2"].isin((MELANOMA, BENIGN))].copy()
    if frame.empty:
        raise ValueError("No melanoma or benign melanocytic rows found in diagnosis_2.")
    if frame["isic_id"].isna().any() or frame["isic_id"].duplicated().any():
        raise ValueError("Selected image IDs must be non-empty and unique.")

    frame["target"] = frame["diagnosis_2"].eq(MELANOMA).astype(np.int64)
    patients = frame["patient_id"].astype("string").str.strip()
    lesions = frame["lesion_id"].astype("string").str.strip()
    ids = frame["isic_id"].astype(str)
    frame["patient_group"] = np.where(
        patients.notna() & patients.ne(""),
        "patient:" + patients.fillna(""),
        np.where(lesions.notna() & lesions.ne(""), "lesion:" + lesions.fillna(""), "image:" + ids),
    )
    frame["anatom_site_general_challenge"] = frame["anatom_site_general"].fillna("unknown")
    frame["cache_index"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "images_uint8.npy"
    metadata_path = output_dir / "clean_metadata.csv"
    manifest_path = output_dir / "cache_manifest.json"
    manifest = {
        "complete": False,
        "source_csv": str(csv_path.resolve()),
        "source_images": str(images_dir.resolve()),
        "n_images": int(len(frame)),
        "image_shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
        "dtype": "uint8",
        "label_column": "diagnosis_2",
        "target_mapping": {BENIGN: 0, MELANOMA: 1},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    images = np.lib.format.open_memmap(
        image_path, mode="w+", dtype=np.uint8,
        shape=(len(frame), IMAGE_SIZE, IMAGE_SIZE, 3),
    )
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc="Preprocessing DICM images"):
            image_file = images_dir / f"{row.isic_id}.jpg"
            if not image_file.is_file():
                raise FileNotFoundError(f"Image file is missing: {image_file}")
            with Image.open(image_file) as image:
                image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
                images[row.cache_index] = np.asarray(image, dtype=np.uint8)
        images.flush()
    except Exception:
        del images
        raise
    del images

    metadata_columns = [
        "cache_index", "image_name", "target", "patient_group", "age_approx",
        "sex", "anatom_site_general_challenge", "diagnosis_2", "patient_id", "lesion_id",
    ]
    metadata = frame.rename(columns={"isic_id": "image_name"})[metadata_columns]
    metadata.to_csv(metadata_path, index=False)
    manifest["complete"] = True
    manifest["class_counts"] = {
        "benign": int((frame["target"] == 0).sum()),
        "malignant": int((frame["target"] == 1).sum()),
    }
    manifest["unique_patient_groups"] = int(frame["patient_group"].nunique())
    manifest["rows_missing_patient_id"] = int(frame["patient_id"].isna().sum())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Saved cache: {output_dir}")


def main() -> None:
    args = parse_args()
    build_cache(args.csv, args.images_dir, args.output_dir)


if __name__ == "__main__":
    main()
