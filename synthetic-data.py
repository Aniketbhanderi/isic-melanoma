"""Build a holdout-safe ISIC 2020 + ISIC-DICM-17k training cache.

Despite this filename, no synthetic images are generated. The output contains
only train_cleaned.csv rows and DICM melanomas with known, non-holdout patient
IDs. Original data and caches are never modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
MELANOMA = "Malignant melanocytic proliferations (Melanoma)"


def read_metadata(path: Path, required: set[str]) -> pd.DataFrame:
	frame = pd.read_csv(path, low_memory=False)
	missing = required - set(frame.columns)
	if missing:
		raise ValueError(f"{path} is missing columns: {sorted(missing)}")
	return frame


def load_base(cache_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
	manifest = json.loads((cache_dir / "cache_manifest.json").read_text())
	if not manifest.get("complete"):
		raise ValueError(f"ISIC 2020 cache is incomplete: {cache_dir}")
	frame = read_metadata(cache_dir / "clean_metadata.csv", {
		"cache_index", "image_name", "target", "patient_group", "patient_id",
		"age_approx", "sex", "anatom_site_general_challenge",
	}).sort_values("cache_index").reset_index(drop=True)
	images = np.load(cache_dir / "images_uint8.npy", mmap_mode="r", allow_pickle=False)
	if images.shape != (len(frame), 256, 256, 3) or images.dtype != np.uint8:
		raise ValueError("Expected an ISIC 2020 uint8 cache shaped (N, 256, 256, 3).")
	if not np.array_equal(frame["cache_index"], np.arange(len(frame))):
		raise ValueError("Base cache indices must be contiguous, starting at zero.")
	if frame.empty or not frame["target"].isin([0, 1]).all():
		raise ValueError("Base cache must contain rows with binary targets.")
	for column in ("image_name", "patient_group"):
		if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
			raise ValueError(f"Base cache has missing {column} values.")
	if frame["image_name"].duplicated().any():
		raise ValueError("Base cache image IDs must be unique.")
	return images, frame


def prepare_metadata(
	base: pd.DataFrame,
	dicm_csv: Path,
	images_dir: Path,
	train_split_csv: Path,
	holdout_csv: Path,
	all_training_csv: Path,
	official_test_csv: Path,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
	"""Select train-split ISIC rows and DICM cases outside the patient holdout."""
	train_split = read_metadata(train_split_csv, {"image_name", "target", "patient_id"})
	holdout = read_metadata(holdout_csv, {"image_name", "patient_id"})
	for frame, label in ((train_split, "training"), (holdout, "holdout")):
		ids = frame["image_name"].astype("string").str.strip()
		if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
			raise ValueError(f"{label} image IDs must be non-empty and unique.")
		frame["image_name"] = ids
	train_split["target"] = pd.to_numeric(train_split["target"], errors="raise").astype(np.int64)
	if train_split.empty or not train_split["target"].isin((0, 1)).all():
		raise ValueError("Training split must contain rows with binary targets.")
	holdout_patients = holdout["patient_id"].astype("string").str.strip()
	if holdout.empty or holdout_patients.isna().any() or holdout_patients.eq("").any():
		raise ValueError("Every holdout row needs a patient_id to enforce patient-level isolation.")
	train_ids = set(train_split["image_name"])
	holdout_ids = set(holdout["image_name"])
	test_patient_ids = set(holdout_patients)
	base_ids = base["image_name"].astype("string").str.strip()
	base_id_set = set(base_ids)
	missing_train_ids = train_ids - base_id_set
	missing_holdout_ids = holdout_ids - base_id_set
	if missing_train_ids:
		raise ValueError(f"{len(missing_train_ids)} train_split image IDs are missing from the base cache.")
	if missing_holdout_ids:
		raise ValueError(f"{len(missing_holdout_ids)} holdout image IDs are missing from the base cache.")
	base_by_id = base.assign(_normalized_id=base_ids).set_index("_normalized_id")
	train_labels = train_split.set_index("image_name")["target"]
	base_train = base.loc[base_ids.isin(train_ids)].copy()
	base_train_labels = base_train.set_index("image_name")["target"].reindex(train_labels.index)
	if not np.array_equal(base_train_labels.to_numpy(), train_labels.to_numpy()):
		raise ValueError("train_cleaned.csv labels disagree with labels in the base cache.")
	holdout_rows = base_by_id.loc[sorted(holdout_ids)]
	holdout_groups = set(holdout_rows["patient_group"].astype(str))
	base_patient_ids = base_train["patient_id"].astype("string").str.strip()
	base_in_holdout = base_train["patient_group"].astype(str).isin(holdout_groups)
	base_in_holdout |= base_patient_ids.isin(test_patient_ids)
	excluded_base_holdout_patients = int(base_in_holdout.sum())
	base_train = base_train.loc[~base_in_holdout].copy()
	base_cache_indices = base_train["cache_index"].to_numpy(dtype=np.int64)
	if set(base_train["image_name"]) & holdout_ids:
		raise RuntimeError("Holdout image IDs remain in the base training rows.")
	if set(base_train["patient_group"].astype(str)) & holdout_groups:
		raise RuntimeError("Holdout patient groups remain in the base training rows.")
	if set(base_train["patient_id"].dropna().astype(str).str.strip()) & test_patient_ids:
		raise RuntimeError("Holdout patient IDs remain in the base training rows.")
	if base_train.empty:
		raise ValueError("No ISIC training rows remain after holdout-patient exclusion.")

	external = read_metadata(dicm_csv, {
		"isic_id", "diagnosis_2", "patient_id", "lesion_id", "age_approx",
		"sex", "anatom_site_general",
	})
	external = external.loc[external["diagnosis_2"].eq(MELANOMA)].copy()
	ids = external["isic_id"].astype("string").str.strip()
	if ids.isna().any() or not ids.str.fullmatch(r"ISIC_\d+").all() or ids.duplicated().any():
		raise ValueError("DICM melanoma image IDs must be valid, non-empty and unique.")
	external["isic_id"] = ids

	# Keep DICM IDs out if they appear anywhere in the original ISIC training
	# set, including removed duplicates and the separate local holdout split.
	all_training = read_metadata(all_training_csv, {"image_name"})
	official_test = read_metadata(official_test_csv, {"image"})
	isic_ids = set(all_training["image_name"].dropna().astype(str))
	isic_ids |= set(official_test["image"].dropna().astype(str))
	isic_ids |= holdout_ids
	id_overlap = external["isic_id"].isin(isic_ids)
	external = external.loc[~id_overlap].copy()
	patient_ids = external["patient_id"].astype("string").str.strip()
	missing_patient = patient_ids.isna() | patient_ids.eq("")
	excluded_missing_patient = int(missing_patient.sum())
	external = external.loc[~missing_patient].copy()
	patient_ids = external["patient_id"].astype("string").str.strip()
	in_holdout_patient = patient_ids.isin(test_patient_ids)
	excluded_holdout_patient = int(in_holdout_patient.sum())
	external = external.loc[~in_holdout_patient].copy().reset_index(drop=True)
	report: dict[str, Any] = {
		"base_cache_images": len(base),
		"isic_train_split_rows": len(train_split),
		"excluded_base_rows_not_in_train_split": len(base) - int(base_ids.isin(train_ids).sum()),
		"excluded_isic_train_rows_matching_holdout_patient_or_group": excluded_base_holdout_patients,
		"holdout_images_excluded": len(holdout_ids),
		"holdout_patients_excluded": len(test_patient_ids),
		"dicm_melanomas": int(external.shape[0] + int(id_overlap.sum()) + excluded_missing_patient + excluded_holdout_patient),
		"excluded_dicm_isic_image_id_overlap": int(id_overlap.sum()),
		"excluded_dicm_missing_patient_id": excluded_missing_patient,
		"excluded_dicm_holdout_patient": excluded_holdout_patient,
	}
	paths = [images_dir / f"{image_id}.jpg" for image_id in external["isic_id"]]
	missing = [str(path) for path in paths if not path.is_file()]
	if missing:
		raise FileNotFoundError(f"{len(missing)} melanoma JPEGs are missing; first: {missing[0]}")

	# Share a known training patient group instead of splitting that patient by
	# dataset. Unknown-patient DICM rows were excluded above for holdout safety.
	patient_groups = base_train.dropna(subset=["patient_id"]).copy()
	patient_groups["patient_id"] = patient_groups["patient_id"].astype(str).str.strip()
	if patient_groups.groupby("patient_id")["patient_group"].nunique().gt(1).any():
		raise ValueError("A training patient belongs to multiple patient groups.")
	known_groups = patient_groups.set_index("patient_id")["patient_group"].to_dict()
	groups = []
	for row in external.itertuples(index=False):
		patient = str(row.patient_id).strip()
		if patient:
			group = known_groups.get(patient, f"dicm17k:patient:{patient}")
		else:
			raise RuntimeError("An unidentified DICM patient reached the training cache.")
		groups.append(group)

	external = external.rename(columns={
		"isic_id": "image_name",
		"anatom_site_general": "anatom_site_general_challenge",
	})
	external["target"] = 1
	external["diagnosis"] = "melanoma"
	external["benign_malignant"] = "malignant"
	external["patient_group"] = groups
	external["source_dataset"] = "isic-dicm-17k"
	external["source_path"] = [str(path.resolve()) for path in paths]
	external["anatom_site_general_challenge"] = external[
		"anatom_site_general_challenge"
	].replace({"anterior torso": "torso", "posterior torso": "torso", "lateral torso": "torso"})
	base_train["source_dataset"] = "isic-2020-train-split"
	combined = pd.concat([base_train, external], ignore_index=True, sort=False)
	combined["cache_index"] = np.arange(len(combined), dtype=np.int64)
	report.update({
		"isic_train_images": len(base_train),
		"added_melanomas": len(external),
		"total_images": len(combined),
		"class_counts": {
			"benign": int(combined["target"].eq(0).sum()),
			"malignant": int(combined["target"].eq(1).sum()),
		},
		"patient_grouping_note": (
			"The local test_cleaned image IDs, patient IDs, and base-cache patient groups "
			"are excluded. DICM records without patient IDs and DICM patient IDs found "
			"in test_cleaned are excluded. Remaining unknown cross-source patient aliases "
			"and visual duplicates with different IDs cannot be detected."
		),
	})
	return combined, base_cache_indices, report


def write_cache(
	base_images: np.ndarray,
	base_cache_indices: np.ndarray,
	combined: pd.DataFrame,
	report: dict[str, Any],
	output_dir: Path,
	provenance: dict[str, str],
) -> None:
	"""Stream selected base rows and permitted DICM JPEGs into a new cache."""
	if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
		raise FileExistsError(f"Refusing to overwrite a non-empty output: {output_dir}")
	output_dir.mkdir(parents=True, exist_ok=True)
	base_count = len(base_cache_indices)
	if base_count > len(combined) or np.any(base_cache_indices < 0) or np.any(base_cache_indices >= len(base_images)):
		raise ValueError("Selected base cache indices are inconsistent with the combined metadata.")
	required_bytes = len(combined) * 256 * 256 * 3
	if shutil.disk_usage(output_dir).free < required_bytes + 64 * 1024**2:
		raise OSError(f"Not enough free disk space; cache needs about {required_bytes / 1024**3:.2f} GiB.")
	manifest = {
		"complete": False,
		"shape": [len(combined), 256, 256, 3],
		"dtype": "uint8",
		"provenance": provenance,
		**report,
	}
	manifest_path = output_dir / "cache_manifest.json"
	manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
	images = np.lib.format.open_memmap(
		output_dir / "images_uint8.npy", mode="w+", dtype=np.uint8,
		shape=(len(combined), 256, 256, 3),
	)
	try:
		for start in tqdm(range(0, base_count, 128), desc="Copying ISIC train split"):
			stop = min(start + 128, base_count)
			images[start:stop] = base_images[base_cache_indices[start:stop]]
		added = combined.iloc[base_count:]
		for row in tqdm(added.itertuples(index=False), total=len(added), desc="Adding melanomas"):
			with Image.open(row.source_path) as image:
				image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
				images[row.cache_index] = np.asarray(image, dtype=np.uint8)
		images.flush()
	finally:
		del images
	combined.to_csv(output_dir / "clean_metadata.csv", index=False)
	manifest["complete"] = True
	manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--base-cache", type=Path, default=ROOT / "artifacts/preprocessed_train")
	parser.add_argument("--dicm-csv", type=Path, default=ROOT / "isic-dicm-17k.csv")
	parser.add_argument("--images-dir", type=Path, default=ROOT / "images_isic-dicm-17k")
	parser.add_argument("--train-split-csv", type=Path, default=ROOT / "train_cleaned.csv")
	parser.add_argument("--holdout-csv", type=Path, default=ROOT / "test_cleaned.csv")
	parser.add_argument("--all-training-csv", type=Path, default=ROOT / "ISIC_2020_Training_GroundTruth.csv")
	parser.add_argument("--official-test-csv", type=Path, default=ROOT / "ISIC_2020_Test_Metadata.csv")
	parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/preprocessed_train_plus_dicm_melanoma")
	parser.add_argument("--dry-run", action="store_true", help="Validate inputs and report counts without writing.")
	args = parser.parse_args()
	images, base = load_base(args.base_cache)
	combined, base_cache_indices, report = prepare_metadata(
		base, args.dicm_csv, args.images_dir, args.train_split_csv, args.holdout_csv,
		args.all_training_csv, args.official_test_csv,
	)
	print(json.dumps(report, indent=2))
	if not args.dry_run:
		write_cache(images, base_cache_indices, combined, report, args.output_dir, {
			"base_cache": str(args.base_cache.resolve()),
			"dicm_csv": str(args.dicm_csv.resolve()),
			"dicm_images": str(args.images_dir.resolve()),
			"train_split_csv": str(args.train_split_csv.resolve()),
			"holdout_csv": str(args.holdout_csv.resolve()),
			"all_training_csv": str(args.all_training_csv.resolve()),
			"official_test_csv": str(args.official_test_csv.resolve()),
		})
		print(f"Saved holdout-safe training cache: {args.output_dir}")


if __name__ == "__main__":
	main()
