"""Evaluate a trained EfficientNet melanoma checkpoint on the labeled test split.

The evaluator uses the project's preprocessed image cache and the metadata
schema stored in the checkpoint, then writes test metrics and per-image scores.

python test.py --checkpoint artifacts/efficientnet_fold0.pt --cache-dir artifacts/jpeg_256 --test-csv test_cleaned.csv --device cuda --amp --batch-size 32 --output-predictions artifacts/efficientnet_b7_fold0_test.csv --output-metrics artifacts/efficientnet_b7_fold0_test_metrics.json 


"""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	confusion_matrix,
	f1_score,
	log_loss,
	precision_score,
	recall_score,
	roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from train import EfficientNetMelanoma, MelanomaDataset, load_training_data


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "artifacts" / "preprocessed_train"
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "efficientnet_melanoma_model.pt"
DEFAULT_TEST_CSV = ROOT / "test_cleaned.csv"
DEFAULT_PREDICTIONS = ROOT / "artifacts" / "efficientnet_test_predictions.csv"
DEFAULT_METRICS = ROOT / "artifacts" / "efficientnet_test_metrics.json"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Evaluate the trained EfficientNet melanoma model on labeled test data."
	)
	parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
	parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
	parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
	parser.add_argument("--log-interval", type=int, default=50)
	parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
	parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--output-predictions", type=Path, default=DEFAULT_PREDICTIONS)
	parser.add_argument("--output-metrics", type=Path, default=DEFAULT_METRICS)
	return parser.parse_args()


def encode_test_metadata(frame: pd.DataFrame, schema: dict[str, Any]) -> np.ndarray:
	"""Encode test metadata with training-only means and vocabularies."""
	age_mean = float(schema["age_mean"])
	age_std = max(float(schema["age_std"]), 1.0)
	ages = pd.to_numeric(frame["age_approx"], errors="coerce").to_numpy(dtype=np.float32)
	ages = np.where(np.isfinite(ages), ages, age_mean)
	features = [((ages - age_mean) / age_std).astype(np.float32).reshape(-1, 1)]

	categories_by_column = schema["categories"]
	for column in schema["categorical_columns"]:
		if column not in frame:
			raise ValueError(f"Test CSV is missing metadata column {column!r}.")
		categories = list(categories_by_column[column])
		if "unknown" not in categories:
			raise ValueError(f"Checkpoint metadata schema has no 'unknown' category for {column!r}.")
		category_index = {value: index for index, value in enumerate(categories)}
		values = frame[column].fillna("unknown").astype(str).str.strip()
		values = values.mask(values.eq(""), "unknown")
		encoded = np.zeros((len(frame), len(categories)), dtype=np.float32)
		unknown_index = category_index["unknown"]
		for row, value in enumerate(values):
			encoded[row, category_index.get(value, unknown_index)] = 1.0
		features.append(encoded)

	return np.concatenate(features, axis=1)


def load_test_data(
	test_csv: Path,
	cache_dir: Path,
	use_metadata: bool,
	metadata_schema: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, pd.DataFrame]:
	"""Match labeled test rows to cached RGB images and their cache indices."""
	if not test_csv.is_file():
		raise FileNotFoundError(f"Labeled test CSV not found: {test_csv}")
	test_frame = pd.read_csv(test_csv)
	required = {"image_name", "target"}
	missing_columns = required - set(test_frame.columns)
	if missing_columns:
		raise ValueError(f"Test CSV is missing required columns: {sorted(missing_columns)}")
	if test_frame.empty:
		raise ValueError(f"Test CSV contains no rows: {test_csv}")
	if test_frame["image_name"].isna().any() or test_frame["image_name"].duplicated().any():
		raise ValueError("Test CSV image_name values must be non-empty and unique.")
	test_frame["target"] = pd.to_numeric(test_frame["target"], errors="raise").astype(np.int64)
	if not test_frame["target"].isin((0, 1)).all():
		raise ValueError("Test labels must be binary (0=benign, 1=malignant).")
	if test_frame["target"].nunique() != 2:
		raise ValueError("Test metrics require both benign and malignant examples.")

	images, cache_frame = load_training_data(cache_dir)
	lookup = cache_frame[["image_name", "cache_index", "target"]]
	test_frame = test_frame.copy()
	test_frame["test_order"] = np.arange(len(test_frame))
	matched = test_frame.merge(
		lookup,
		on="image_name",
		how="left",
		suffixes=("_test", "_cache"),
		validate="one_to_one",
		indicator=True,
	).sort_values("test_order").reset_index(drop=True)
	missing_images = matched.loc[matched["_merge"] != "both", "image_name"].tolist()
	if missing_images:
		raise ValueError(
			f"{len(missing_images)} test image(s) are missing from the preprocessed cache; "
			f"examples: {missing_images[:5]}"
		)
	if not np.array_equal(
		matched["target_test"].to_numpy(dtype=np.int64),
		matched["target_cache"].to_numpy(dtype=np.int64),
	):
		raise ValueError("Test CSV labels disagree with labels in the preprocessed cache.")

	indices = matched["cache_index"].to_numpy(dtype=np.int64)
	labels = matched["target_test"].to_numpy(dtype=np.int64)
	metadata = None
	if use_metadata:
		if metadata_schema is None:
			raise ValueError("Checkpoint is configured for metadata but has no metadata schema.")
		metadata = encode_test_metadata(matched, metadata_schema)
	return images, indices, metadata, matched


def resolve_device(device_name: str) -> torch.device:
	if device_name == "cuda":
		if not torch.cuda.is_available():
			raise RuntimeError("CUDA was requested but is not available.")
		return torch.device("cuda")
	if device_name == "cpu":
		return torch.device("cpu")
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	use_metadata: bool,
	amp_enabled: bool,
	log_interval: int,
) -> tuple[np.ndarray, np.ndarray, float]:
	model.eval()
	criterion = nn.CrossEntropyLoss()
	all_labels: list[np.ndarray] = []
	all_probabilities: list[np.ndarray] = []
	total_loss = 0.0
	samples_seen = 0
	started_at = time.perf_counter()
	with torch.inference_mode():
		for batch_index, (images, metadata, labels) in enumerate(loader, start=1):
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)
			metadata = metadata.to(device, non_blocking=True) if use_metadata else None
			with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
				logits = model(images, metadata)
				loss = criterion(logits, labels)
			batch_size = len(labels)
			total_loss += float(loss.item()) * batch_size
			samples_seen += batch_size
			all_labels.append(labels.cpu().numpy())
			all_probabilities.append(logits.softmax(dim=1).float().cpu().numpy())
			if batch_index == 1 or batch_index % log_interval == 0 or batch_index == len(loader):
				elapsed = time.perf_counter() - started_at
				print(
					f"Test batch {batch_index}/{len(loader)} | "
					f"samples {samples_seen:,}/{len(loader.dataset):,} | "
					f"loss {total_loss / samples_seen:.4f} | {elapsed:.1f}s",
					flush=True,
				)
	return (
		np.concatenate(all_labels),
		np.concatenate(all_probabilities),
		total_loss / samples_seen,
	)


def main() -> None:
	args = parse_args()
	if args.batch_size < 1 or args.num_workers < 0 or args.log_interval < 1:
		raise ValueError("batch-size/log-interval must be positive and num-workers non-negative.")
	if not args.checkpoint.is_file():
		raise FileNotFoundError(f"Model checkpoint not found: {args.checkpoint}")
	checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
	model_config = checkpoint.get("model_config")
	if not isinstance(model_config, dict):
		raise ValueError("Checkpoint does not contain a valid model_config.")

	backbone = str(model_config["backbone"])
	image_size = int(model_config["image_size"])
	use_metadata = bool(model_config["use_metadata"])
	metadata_features = int(model_config["metadata_features"])
	if use_metadata != (metadata_features > 0):
		raise ValueError("Checkpoint metadata configuration is inconsistent.")

	images, indices, metadata, test_frame = load_test_data(
		args.test_csv,
		args.cache_dir,
		use_metadata,
		checkpoint.get("metadata_schema"),
	)
	labels = test_frame["target_test"].to_numpy(dtype=np.int64)
	if metadata is not None and metadata.shape[1] != metadata_features:
		raise ValueError(
			f"Checkpoint expects {metadata_features} metadata features, got {metadata.shape[1]}."
		)
	# MelanomaDataset indexes labels and metadata by the cache row, while these
	# test arrays are ordered only over the selected test rows.
	dataset_labels = np.zeros(len(images), dtype=np.int64)
	dataset_labels[indices] = labels
	dataset_metadata = None
	if metadata is not None:
		dataset_metadata = np.zeros((len(images), metadata_features), dtype=np.float32)
		dataset_metadata[indices] = metadata
	device = resolve_device(args.device)
	amp_enabled = args.amp and device.type == "cuda"
	dataset = MelanomaDataset(
		images=images,
		labels=dataset_labels,
		indices=indices,
		image_size=image_size,
		training=False,
		metadata=dataset_metadata,
	)
	loader = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=device.type == "cuda",
		persistent_workers=args.num_workers > 0,
	)

	model = EfficientNetMelanoma(
		backbone_name=backbone,
		pretrained=False,
		metadata_features=metadata_features,
	).to(device)
	model.load_state_dict(checkpoint["model_state_dict"], strict=True)
	actual_labels, probabilities, test_loss = evaluate(
		model, loader, device, use_metadata, amp_enabled, args.log_interval
	)
	malignant_probabilities = probabilities[:, 1]
	predicted_labels = probabilities.argmax(axis=1)
	metrics = {
		"checkpoint": str(args.checkpoint.resolve()),
		"test_csv": str(args.test_csv.resolve()),
		"n_test": int(len(actual_labels)),
		"class_counts": {
			"benign": int((actual_labels == 0).sum()),
			"malignant": int((actual_labels == 1).sum()),
		},
		"loss": test_loss,
		"accuracy": float(accuracy_score(actual_labels, predicted_labels)),
		"roc_auc": float(roc_auc_score(actual_labels, malignant_probabilities)),
		"average_precision": float(average_precision_score(actual_labels, malignant_probabilities)),
		"precision_malignant": float(precision_score(actual_labels, predicted_labels, zero_division=0)),
		"recall_malignant": float(recall_score(actual_labels, predicted_labels, zero_division=0)),
		"f1_malignant": float(f1_score(actual_labels, predicted_labels, zero_division=0)),
		"confusion_matrix_labels_0_1": confusion_matrix(
			actual_labels, predicted_labels, labels=[0, 1]
		).tolist(),
		"log_loss": float(log_loss(actual_labels, probabilities, labels=[0, 1])),
		"checkpoint_validation_roc_auc": checkpoint.get("validation_roc_auc"),
		"device": str(device),
		"amp_enabled": amp_enabled,
	}
	predictions = pd.DataFrame({
		"image_name": test_frame["image_name"].to_numpy(),
		"target": actual_labels,
		"predicted_target": predicted_labels,
		"benign_probability": probabilities[:, 0],
		"malignant_probability": malignant_probabilities,
	})
	args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
	args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
	predictions.to_csv(args.output_predictions, index=False)
	args.output_metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
	print(json.dumps(metrics, indent=2))
	print(f"Saved predictions: {args.output_predictions}")
	print(f"Saved metrics: {args.output_metrics}")


if __name__ == "__main__":
	main()