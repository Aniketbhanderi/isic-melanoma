"""Train a ResNet-50 melanoma classifier with patient metadata.

By default, images are read from the JPEG cache and training is restricted to
train_cleaned.csv. EfficientNet backbones remain available via --backbone.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / "artifacts" / "jpeg_256"
DEFAULT_TRAIN_CSV = ROOT / "train_cleaned.csv"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
METADATA_CATEGORICAL_COLUMNS = ("sex", "anatom_site_general_challenge")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Train a patient-split melanoma classifier with image and metadata inputs."
	)
	parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
	parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "efficientnet_b0_melanoma_model.pt")
	parser.add_argument("--backbone", default="efficientnet_b0", choices=(
		"resnet50",
		"efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
		"efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
	))
	parser.add_argument("--image-size", type=int, default=256)
	parser.add_argument("--epochs", type=int, default=15)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--log-interval", type=int, default=25,
						 help="Print training/validation progress every N batches.")
	parser.add_argument("--learning-rate", type=float, default=3e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--n-folds", type=int, default=5)
	parser.add_argument("--fold", type=int, default=0)
	parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--use-metadata", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV,
						help="Training CSV containing image names, labels, and metadata.")
	return parser.parse_args()


class MelanomaDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
	"""Memory-mapped RGB images paired with binary labels and optional metadata."""

	def __init__(
		self,
		images: np.ndarray,
		labels: np.ndarray,
		indices: np.ndarray,
		image_size: int,
		training: bool,
		metadata: np.ndarray | None = None,
		image_indices: np.ndarray | None = None,
	) -> None:
		self.images = images
		self.labels = labels
		self.indices = indices
		self.metadata = metadata
		self.image_indices = image_indices
		image_transforms: list[Any] = [transforms.Resize((image_size, image_size))]
		if training:
			image_transforms.extend([
				transforms.RandomHorizontalFlip(),
				transforms.RandomVerticalFlip(),
				transforms.RandomRotation(20),
				transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
			])
		image_transforms.extend([
			transforms.ToTensor(),
			transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
		])
		self.transform = transforms.Compose(image_transforms)

	def __len__(self) -> int:
		return len(self.indices)

	def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		row = int(self.indices[item])
		image_row = int(self.image_indices[row]) if self.image_indices is not None else row
		image = Image.fromarray(self.images[image_row], mode="RGB")
		image_tensor = self.transform(image)
		label = torch.tensor(self.labels[row], dtype=torch.long)
		if self.metadata is None:
			metadata_tensor = torch.empty(0, dtype=torch.float32)
		else:
			metadata_tensor = torch.from_numpy(self.metadata[row])
		return image_tensor, metadata_tensor, label


class MelanomaClassifier(nn.Module):
	"""Image encoder with optional tabular fusion and a multi-dropout head."""

	def __init__(
		self,
		backbone_name: str,
		pretrained: bool,
		metadata_features: int = 0,
		output_features: int = 2,
	) -> None:
		super().__init__()
		weights = models.get_model_weights(backbone_name).DEFAULT if pretrained else None
		self.backbone = models.get_model(backbone_name, weights=weights)
		if backbone_name == "resnet50":
			image_features = self.backbone.fc.in_features
			self.backbone.fc = nn.Identity()
		else:
			image_features = self.backbone.classifier[-1].in_features
			self.backbone.classifier = nn.Identity()

		self.metadata_encoder: nn.Module | None = None
		metadata_embedding_size = 0
		if metadata_features:
			metadata_embedding_size = 128
			self.metadata_encoder = nn.Sequential(
				nn.Linear(metadata_features, 512),
				nn.LayerNorm(512),
				nn.SiLU(),
				nn.Dropout(0.3),
				nn.Linear(512, metadata_embedding_size),
				nn.LayerNorm(metadata_embedding_size),
				nn.SiLU(),
			)

		self.dropouts = nn.ModuleList(nn.Dropout(0.5) for _ in range(5))
		self.classifier = nn.Linear(image_features + metadata_embedding_size, output_features)

	def forward(
		self, images: torch.Tensor, metadata: torch.Tensor | None = None
	) -> torch.Tensor:
		features = self.backbone(images)
		if self.metadata_encoder is not None:
			if metadata is None:
				raise ValueError("Metadata is required by this model configuration.")
			features = torch.cat((features, self.metadata_encoder(metadata)), dim=1)
		logits = [self.classifier(dropout(features)) for dropout in self.dropouts]
		return torch.stack(logits, dim=0).mean(dim=0)


# Preserve the public name used by existing EfficientNet inference scripts.
EfficientNetMelanoma = MelanomaClassifier


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = True


def load_training_data(cache_dir: Path, train_csv: Path | None = None) -> tuple[np.ndarray, pd.DataFrame]:
	"""Load a memory-mapped cache and optionally select rows from a training CSV."""

	manifest_path = cache_dir / "manifest.json"
	metadata_path = cache_dir / "metadata.csv"
	legacy_manifest_path = cache_dir / "cache_manifest.json"
	legacy_metadata_path = cache_dir / "clean_metadata.csv"
	image_path = cache_dir / "images_uint8.npy"

	if manifest_path.is_file():
		with manifest_path.open(encoding="utf-8") as file:
			manifest = json.load(file)
	elif legacy_manifest_path.is_file():
		with legacy_manifest_path.open(encoding="utf-8") as file:
			manifest = json.load(file)
	else:
		raise FileNotFoundError(f"No cache manifest found in {cache_dir}")
	if not manifest.get("complete"):
		raise ValueError(f"Preprocessed cache is incomplete: {cache_dir}")

	if metadata_path.is_file():
		frame = pd.read_csv(metadata_path).sort_values("cache_index").reset_index(drop=True)
	elif legacy_metadata_path.is_file():
		frame = pd.read_csv(legacy_metadata_path).sort_values("cache_index").reset_index(drop=True)
	else:
		raise FileNotFoundError(f"No metadata CSV found in {cache_dir}")

	if "patient_group" in frame.columns:
		groups = frame["patient_group"].astype("string").str.strip()
	else:
		groups = pd.Series(pd.NA, index=frame.index, dtype="string")
	if "patient_id" in frame.columns:
		patient_ids = frame["patient_id"].astype("string").str.strip()
		groups = groups.fillna(patient_ids)
	groups = groups.fillna(
		"unknown:image:" + frame["image_name"].astype("string").str.strip()
	)
	frame["patient_group"] = groups.astype(str)
	images = np.load(image_path, mmap_mode="r")
	if images.shape[:1] != (len(frame),) or images.dtype != np.uint8:
		raise ValueError(f"Unexpected image cache shape/dtype: {images.shape}, {images.dtype}")
	cache_indices = frame["cache_index"].to_numpy(dtype=np.int64)
	if len(np.unique(cache_indices)) != len(frame) or (cache_indices < 0).any() or (cache_indices >= len(images)).any():
		raise ValueError("Cache indices must be unique and within the image array.")

	if train_csv is not None and not train_csv.is_file():
		raise FileNotFoundError(f"Required training CSV not found: {train_csv}")
	if train_csv is not None:
		train_frame = pd.read_csv(train_csv)
		required = {"image_name", "target", "patient_id", "age_approx", *METADATA_CATEGORICAL_COLUMNS}
		missing = required - set(train_frame.columns)
		if missing:
			raise ValueError(f"train_cleaned.csv is missing columns: {sorted(missing)}")
		if train_frame.empty or train_frame["image_name"].isna().any() or train_frame["image_name"].astype(str).str.strip().eq("").any() or train_frame["image_name"].duplicated().any():
			raise ValueError("Training CSV image names must be non-empty and unique.")
		train_frame["target"] = pd.to_numeric(train_frame["target"], errors="raise")
		if not train_frame["target"].isin((0, 1)).all():
			raise ValueError("train_cleaned.csv must contain binary melanoma targets (0/1).")
		train_frame["target"] = train_frame["target"].astype(np.int64)

		matched = frame.merge(
			train_frame[["image_name", "target", "patient_id", "age_approx", *METADATA_CATEGORICAL_COLUMNS]],
			on="image_name",
			how="inner",
			suffixes=("_cache", "_train"),
			validate="one_to_one",
		)
		if len(matched) != len(train_frame):
			missing_names = sorted(set(train_frame["image_name"]) - set(frame["image_name"]))
			raise ValueError(
				f"Only {len(matched)}/{len(train_frame)} train CSV rows exist in the cache; "
				f"missing examples: {missing_names[:5]}"
			)
		if not np.array_equal(
			matched["target_cache"].to_numpy(dtype=np.int64),
			matched["target_train"].to_numpy(dtype=np.int64),
		):
			raise ValueError("train_cleaned.csv labels disagree with cache labels.")

		matched = matched.sort_values("cache_index").reset_index(drop=True)
		frame = matched.rename(columns={"target_cache": "target"}).copy()
		for column in ("patient_id", "age_approx", *METADATA_CATEGORICAL_COLUMNS):
			train_column = f"{column}_train" if f"{column}_train" in frame else column
			frame[column] = frame[train_column]
		patient_ids = frame["patient_id"].astype("string").str.strip().replace("", pd.NA)
		frame["patient_group"] = patient_ids.fillna(frame["patient_group"]).astype(str)
		print(
			f"Training set: {train_csv} | images={len(frame):,} | "
			f"patients={frame['patient_group'].nunique():,}",
			flush=True,
		)
		return images, frame

	print(
		f"Full cache loaded: images={len(frame):,} | "
		f"patients={frame['patient_group'].nunique():,}",
		flush=True,
	)
	return images, frame


def build_metadata_features(
	frame: pd.DataFrame, train_indices: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
	"""Fit feature vocabularies/scaling on training rows only."""
	ages = pd.to_numeric(frame["age_approx"], errors="coerce").to_numpy(dtype=np.float32)
	train_ages = ages[train_indices]
	age_mean = float(np.nanmean(train_ages)) if np.isfinite(train_ages).any() else 0.0
	age_std = float(np.nanstd(train_ages)) if np.isfinite(train_ages).any() else 1.0
	age_std = max(age_std, 1.0)
	ages = np.where(np.isfinite(ages), ages, age_mean)
	columns = [((ages - age_mean) / age_std).astype(np.float32).reshape(-1, 1)]

	categories_by_column: dict[str, list[str]] = {}
	for column in METADATA_CATEGORICAL_COLUMNS:
		values = frame[column].fillna("unknown").astype(str).str.strip()
		values = values.mask(values.eq(""), "unknown")
		categories = sorted(set(values.iloc[train_indices].tolist()) | {"unknown"})
		categories_by_column[column] = categories
		category_index = {value: index for index, value in enumerate(categories)}
		encoded = np.zeros((len(frame), len(categories)), dtype=np.float32)
		for row, value in enumerate(values):
			encoded[row, category_index.get(value, category_index["unknown"])] = 1.0
		columns.append(encoded)

	return np.concatenate(columns, axis=1), {
		"age_mean": age_mean,
		"age_std": age_std,
		"categories": categories_by_column,
		"categorical_columns": list(METADATA_CATEGORICAL_COLUMNS),
	}


def evaluate(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	use_metadata: bool,
	amp_enabled: bool,
	epoch: int,
	total_epochs: int,
	log_interval: int,
) -> tuple[float, float, float]:
	model.eval()
	total_loss = 0.0
	samples_seen = 0
	started_at = time.perf_counter()
	all_labels: list[np.ndarray] = []
	all_scores: list[np.ndarray] = []
	criterion = nn.CrossEntropyLoss()
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
			all_scores.append(logits.softmax(dim=1)[:, 1].float().cpu().numpy())
			if batch_index == 1 or batch_index % log_interval == 0 or batch_index == len(loader):
				elapsed = time.perf_counter() - started_at
				print(
					f"Epoch {epoch:02d}/{total_epochs} | validation batch "
					f"{batch_index}/{len(loader)} | samples {samples_seen:,}/{len(loader.dataset):,} "
					f"| loss {total_loss / samples_seen:.4f} | "
					f"{samples_seen / max(elapsed, 1e-9):.1f} img/s | {elapsed:.1f}s",
					flush=True,
				)

	labels_array = np.concatenate(all_labels)
	scores_array = np.concatenate(all_scores)
	auc = float(roc_auc_score(labels_array, scores_array))
	average_precision = float(average_precision_score(labels_array, scores_array))
	return total_loss / len(loader.dataset), auc, average_precision


def train_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	scaler: torch.amp.GradScaler,
	device: torch.device,
	use_metadata: bool,
	amp_enabled: bool,
	criterion: nn.Module,
	epoch: int,
	total_epochs: int,
	log_interval: int,
) -> float:
	model.train()
	total_loss = 0.0
	samples_seen = 0
	started_at = time.perf_counter()
	for batch_index, (images, metadata, labels) in enumerate(loader, start=1):
		images = images.to(device, non_blocking=True)
		metadata = metadata.to(device, non_blocking=True) if use_metadata else None
		labels = labels.to(device, non_blocking=True)
		optimizer.zero_grad(set_to_none=True)
		with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
			logits = model(images, metadata)
			loss = criterion(logits, labels)
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()
		batch_size = len(labels)
		total_loss += float(loss.item()) * batch_size
		samples_seen += batch_size
		if batch_index == 1 or batch_index % log_interval == 0 or batch_index == len(loader):
			elapsed = time.perf_counter() - started_at
			print(
				f"Epoch {epoch:02d}/{total_epochs} | train batch "
				f"{batch_index}/{len(loader)} | samples {samples_seen:,}/{len(loader.dataset):,} "
				f"| loss {total_loss / samples_seen:.4f} | "
				f"{samples_seen / max(elapsed, 1e-9):.1f} img/s | {elapsed:.1f}s",
				flush=True,
			)
	return total_loss / len(loader.dataset)


def main() -> None:
	args = parse_args()
	if args.image_size < 32 or args.epochs < 1 or args.batch_size < 1 or args.log_interval < 1:
		raise ValueError("image-size must be >=32, and epochs/batch-size/log-interval must be positive.")
	if not 0 <= args.fold < args.n_folds:
		raise ValueError("fold must be in [0, n-folds).")
	print(
		"Run parameters: "
		f"backbone={args.backbone}, image_size={args.image_size}, "
		f"epochs={args.epochs}, batch_size={args.batch_size}, "
		f"learning_rate={args.learning_rate:g}, weight_decay={args.weight_decay:g}, "
		f"optimizer=Adam, scheduler=CosineAnnealingLR, fold={args.fold}/{args.n_folds - 1}, "
		f"seed={args.seed}, pretrained={args.pretrained}, use_metadata={args.use_metadata}, "
		f"amp_requested={args.amp}, workers={args.num_workers}, "
		f"log_interval={args.log_interval}, cache={args.cache_dir}, output={args.output}",
		flush=True,
	)
	set_seed(args.seed)

	images, frame = load_training_data(args.cache_dir, args.train_csv)
	labels = frame["target"].to_numpy(dtype=np.int64)
	groups = frame["patient_group"].astype(str).to_numpy()
	splitter = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
	train_indices, validation_indices = list(splitter.split(np.zeros(len(labels)), labels, groups))[args.fold]
	if set(labels[train_indices]) != {0, 1} or set(labels[validation_indices]) != {0, 1}:
		raise ValueError("Both train and validation folds must contain benign and malignant images.")
	if not set(groups[train_indices]).isdisjoint(groups[validation_indices]):
		raise RuntimeError("Patient leakage detected between the train and validation folds.")

	metadata_features = None
	metadata_schema = None
	if args.use_metadata:
		metadata_features, metadata_schema = build_metadata_features(frame, train_indices)

	train_dataset = MelanomaDataset(
		images, labels, train_indices, args.image_size, training=True, metadata=metadata_features,
		image_indices=frame["cache_index"].to_numpy(dtype=np.int64),
	)
	validation_dataset = MelanomaDataset(
		images, labels, validation_indices, args.image_size, training=False, metadata=metadata_features,
		image_indices=frame["cache_index"].to_numpy(dtype=np.int64),
	)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	amp_enabled = args.amp and device.type == "cuda"
	loader_options: dict[str, Any] = {
		"batch_size": args.batch_size,
		"num_workers": args.num_workers,
		"pin_memory": device.type == "cuda",
		"persistent_workers": args.num_workers > 0,
	}
	train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
	validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

	model = MelanomaClassifier(
		backbone_name=args.backbone,
		pretrained=args.pretrained,
		metadata_features=metadata_features.shape[1] if metadata_features is not None else 0,
	).to(device)
	class_counts = np.bincount(labels[train_indices], minlength=2)
	class_weights = len(train_indices) / (2.0 * class_counts)
	criterion = nn.CrossEntropyLoss(
		weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
	)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
	scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	best_auc = -1.0
	history: list[dict[str, float | int]] = []
	print(
		f"Device: {device}; train images: {len(train_indices):,}; "
		f"validation images: {len(validation_indices):,}; malignant train: {class_counts[1]:,}"
	)
	for epoch in range(1, args.epochs + 1):
		train_loss = train_one_epoch(
			model, train_loader, optimizer, scaler, device, args.use_metadata, amp_enabled,
			criterion, epoch, args.epochs, args.log_interval,
		)
		val_loss, val_auc, val_ap = evaluate(
			model, validation_loader, device, args.use_metadata, amp_enabled,
			epoch, args.epochs, args.log_interval,
		)
		learning_rate = float(optimizer.param_groups[0]["lr"])
		history.append({
			"epoch": epoch,
			"learning_rate": learning_rate,
			"train_loss": train_loss,
			"validation_loss": val_loss,
			"validation_roc_auc": val_auc,
			"validation_average_precision": val_ap,
		})
		print(
			f"Epoch {epoch:02d}/{args.epochs} | lr={learning_rate:.2e} "
			f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
			f"val_auc={val_auc:.4f} val_ap={val_ap:.4f}"
		)
		if val_auc > best_auc:
			best_auc = val_auc
			torch.save({
				"model_state_dict": model.state_dict(),
				"model_config": {
					"backbone": args.backbone,
					"image_size": args.image_size,
					"use_metadata": args.use_metadata,
					"metadata_features": metadata_features.shape[1] if metadata_features is not None else 0,
				},
				"metadata_schema": metadata_schema,
				"class_names": ["benign", "malignant"],
				"validation_roc_auc": val_auc,
				"fold": args.fold,
				"seed": args.seed,
			}, args.output)
		scheduler.step()

	history_path = args.output.with_name(f"{args.output.stem}_history.csv")
	pd.DataFrame(history).to_csv(history_path, index=False)
	print(f"Best validation ROC-AUC: {best_auc:.4f}")
	print(f"Saved best checkpoint: {args.output}")
	print(f"Saved training history: {history_path}")


if __name__ == "__main__":
	main()
