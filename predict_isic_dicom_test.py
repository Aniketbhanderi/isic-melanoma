"""Run inference on the unlabeled ISIC 2020 DICOM test directory.

The official ISIC test metadata contains no target labels, so this script writes
per-image probabilities/predictions but cannot calculate accuracy or F1.
"""

from __future__ import annotations

import argparse
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from pydicom.encaps import generate_frames
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from test import encode_test_metadata, resolve_device
from train import EfficientNetMelanoma, IMAGENET_MEAN, IMAGENET_STD


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "efficientnet_dicm17k_model.pt"
DEFAULT_METADATA = ROOT / "ISIC_2020_Test_Metadata.csv"
DEFAULT_DICOM_DIR = ROOT / "test"
DEFAULT_OUTPUT = ROOT / "artifacts" / "dicm_model_isic2020_official_test_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--dicom-dir", type=Path, default=DEFAULT_DICOM_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def dicom_to_rgb(path: Path, image_size: int) -> Image.Image:
    """Decode the ISIC baseline-JPEG DICOM without expanding to full resolution."""
    dataset = pydicom.dcmread(path)
    transfer_syntax = dataset.file_meta.TransferSyntaxUID
    if str(transfer_syntax) == "1.2.840.10008.1.2.4.50":
        frame = next(generate_frames(dataset.PixelData, number_of_frames=1))
        with Image.open(BytesIO(frame)) as image:
            image.draft("RGB", (image_size, image_size))
            return image.convert("RGB").resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            )

    pixels = dataset.pixel_array
    if pixels.ndim == 2:
        image = Image.fromarray(pixels).convert("RGB")
    elif pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
        image = Image.fromarray(pixels).convert("RGB")
    else:
        raise ValueError(f"Unsupported pixel array shape {pixels.shape} in {path}")
    return image.resize((image_size, image_size), Image.Resampling.BILINEAR)


class DicomTestDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        rows: pd.DataFrame,
        dicom_dir: Path,
        metadata: np.ndarray,
        image_size: int,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.dicom_dir = dicom_dir
        self.metadata = metadata
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows.iloc[index]
        image_id = str(row["image"])
        image = dicom_to_rgb(self.dicom_dir / f"{image_id}.dcm", self.image_size)
        return self.transform(image), torch.from_numpy(self.metadata[index]), image_id


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size must be positive and num-workers non-negative.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.metadata_csv.is_file():
        raise FileNotFoundError(f"ISIC test metadata not found: {args.metadata_csv}")
    if not args.dicom_dir.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {args.dicom_dir}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint is missing model_config.")
    if not config.get("use_metadata") or not checkpoint.get("metadata_schema"):
        raise ValueError("This inference script expects a checkpoint trained with metadata.")

    rows = pd.read_csv(args.metadata_csv)
    required = {"image", "age_approx", "sex", "anatom_site_general"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"Metadata CSV is missing columns: {sorted(missing)}")
    if rows.empty or rows["image"].isna().any() or rows["image"].duplicated().any():
        raise ValueError("Test metadata must contain unique, non-empty image IDs.")
    rows["anatom_site_general_challenge"] = rows["anatom_site_general"]
    missing_images = [
        image_id for image_id in rows["image"].astype(str)
        if not (args.dicom_dir / f"{image_id}.dcm").is_file()
    ]
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} metadata image(s) are missing; examples: {missing_images[:5]}"
        )

    metadata = encode_test_metadata(rows, checkpoint["metadata_schema"])
    expected_features = int(config["metadata_features"])
    if metadata.shape[1] != expected_features:
        raise ValueError(
            f"Checkpoint expects {expected_features} metadata features, got {metadata.shape[1]}"
        )
    image_size = int(config["image_size"])
    dataset = DicomTestDataset(rows, args.dicom_dir, metadata, image_size)
    device = resolve_device(args.device)
    amp_enabled = args.amp and device.type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = EfficientNetMelanoma(
        backbone_name=str(config["backbone"]),
        pretrained=False,
        metadata_features=expected_features,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    image_ids: list[str] = []
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_index, (images, batch_metadata, batch_ids) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            batch_metadata = batch_metadata.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images, batch_metadata)
            probabilities.append(logits.softmax(dim=1).float().cpu().numpy())
            image_ids.extend(batch_ids)
            if batch_index == 1 or batch_index % 50 == 0 or batch_index == len(loader):
                print(f"Inference batch {batch_index}/{len(loader)} | {len(image_ids):,}/{len(dataset):,}", flush=True)

    scores = np.concatenate(probabilities)
    predictions = pd.DataFrame({
        "image": image_ids,
        "predicted_target": scores.argmax(axis=1),
        "benign_probability": scores[:, 0],
        "malignant_probability": scores[:, 1],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "metadata_csv": str(args.metadata_csv.resolve()),
        "dicom_dir": str(args.dicom_dir.resolve()),
        "n_images": len(predictions),
        "ground_truth_available": False,
        "metrics": None,
        "predicted_counts_argmax": {
            "benign": int((predictions["predicted_target"] == 0).sum()),
            "malignant": int((predictions["predicted_target"] == 1).sum()),
        },
        "device": str(device),
        "amp_enabled": amp_enabled,
        "predictions_csv": str(args.output.resolve()),
    }
    summary_path = args.output.with_name(f"{args.output.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
