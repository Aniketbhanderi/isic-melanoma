"""Small, isolated checks for the combined ISIC cache builder."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "synthetic-data.py"
SPEC = importlib.util.spec_from_file_location("dataset_merge", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
merge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge)


class DatasetMergeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = pd.DataFrame({
            "cache_index": [0, 1, 2], "image_name": ["ISIC_1", "ISIC_2", "ISIC_3"],
            "target": [1, 0, 1], "patient_id": ["P_TEST", "P2", "P3"],
            "patient_group": ["test_group", "P2", "P3"],
            "age_approx": [40, 50, 60], "sex": ["female", "male", "female"],
            "anatom_site_general_challenge": ["torso", "torso", "leg"],
        })
        self.train_split = self.root / "train_split.csv"
        pd.DataFrame({
            "image_name": ["ISIC_2", "ISIC_3"], "patient_id": ["P2", "P3"], "target": [0, 1],
        }).to_csv(self.train_split, index=False)
        self.holdout = self.root / "holdout.csv"
        pd.DataFrame({"image_name": ["ISIC_1"], "patient_id": ["P_TEST"]}).to_csv(self.holdout, index=False)
        self.all_training = self.root / "all_training.csv"
        pd.DataFrame({"image_name": ["ISIC_1", "ISIC_2", "ISIC_3"]}).to_csv(self.all_training, index=False)
        self.official_test = self.root / "official_test.csv"
        pd.DataFrame({"image": ["ISIC_4"]}).to_csv(self.official_test, index=False)
        self.csv = self.root / "dicm.csv"
        rows = []
        for number in range(1, 10):
            rows.append({
                "isic_id": f"ISIC_{number}",
                "diagnosis_2": "Benign melanocytic proliferations" if number == 9 else merge.MELANOMA,
                "patient_id": {5: "P_TEST", 6: "external", 8: "P2", 9: "benign"}.get(number),
                "lesion_id": "lesion7" if number == 7 else None,
                "age_approx": 50, "sex": "male", "anatom_site_general": "anterior torso",
                "copyright_license": "CC-BY",
            })
            Image.new("RGB", (12, 8), (number, 0, 0)).save(self.root / f"ISIC_{number}.jpg")
        pd.DataFrame(rows).to_csv(self.csv, index=False)

    def prepare(self):
        return merge.prepare_metadata(
            self.base, self.csv, self.root, self.train_split, self.holdout,
            self.all_training, self.official_test,
        )

    def test_selection_grouping_and_provenance(self):
        frame, base_indices, report = self.prepare()
        self.assertEqual(frame.image_name.tolist(), ["ISIC_2", "ISIC_3", "ISIC_6", "ISIC_8"])
        self.assertEqual(frame.target.tolist(), [0, 1, 1, 1])
        self.assertEqual(base_indices.tolist(), [1, 2])
        self.assertEqual(frame.patient_group.tolist(), [
            "P2", "P3", "dicm17k:patient:external", "P2",
        ])
        self.assertEqual(report["excluded_base_rows_not_in_train_split"], 1)
        self.assertEqual(report["excluded_isic_train_rows_matching_holdout_patient_or_group"], 0)
        self.assertEqual(report["excluded_dicm_isic_image_id_overlap"], 4)
        self.assertEqual(report["excluded_dicm_missing_patient_id"], 1)
        self.assertEqual(report["excluded_dicm_holdout_patient"], 1)
        self.assertEqual(report["added_melanomas"], 2)
        self.assertTrue(frame.iloc[2:].anatom_site_general_challenge.eq("torso").all())
        self.assertTrue(frame.iloc[2:].copyright_license.eq("CC-BY").all())
        self.assertFalse(set(frame.patient_id.astype(str)) & {"P_TEST"})
        self.assertEqual(self.base.columns.tolist().count("source_dataset"), 0)

    def test_cache_round_trip_and_overwrite_protection(self):
        frame, base_indices, report = self.prepare()
        original = np.stack([
            np.full((256, 256, 3), value, dtype=np.uint8) for value in (11, 22, 33)
        ])
        output = self.root / "combined"
        merge.write_cache(original, base_indices, frame, report, output, {})
        images, metadata = merge.load_base(output)
        np.testing.assert_array_equal(images[:2], original[1:])
        with Image.open(self.root / "ISIC_6.jpg") as image:
            expected = np.asarray(image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR))
        np.testing.assert_array_equal(images[2], expected)
        self.assertEqual(len(metadata), 4)
        self.assertTrue(json.loads((output / "cache_manifest.json").read_text())["complete"])
        with self.assertRaises(FileExistsError):
            merge.write_cache(original, base_indices, frame, report, output, {})

    def test_missing_image_rejected(self):
        (self.root / "ISIC_6.jpg").unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()

    def test_bad_image_leaves_cache_incomplete(self):
        frame, base_indices, report = self.prepare()
        (self.root / "ISIC_6.jpg").write_bytes(b"not an image")
        output = self.root / "failed"
        with self.assertRaises(OSError):
            merge.write_cache(
                np.zeros((3, 256, 256, 3), dtype=np.uint8), base_indices,
                frame, report, output, {},
            )
        self.assertFalse(json.loads((output / "cache_manifest.json").read_text())["complete"])
        with self.assertRaises(ValueError):
            merge.load_base(output)

    def test_empty_external_selection(self):
        data = pd.read_csv(self.csv)
        data["diagnosis_2"] = "Benign melanocytic proliferations"
        data.to_csv(self.csv, index=False)
        frame, base_indices, report = self.prepare()
        self.assertEqual(len(frame), 2)
        self.assertEqual(base_indices.tolist(), [1, 2])
        self.assertEqual(report["added_melanomas"], 0)

    def test_duplicate_external_id_rejected(self):
        data = pd.read_csv(self.csv)
        pd.concat([data, data.iloc[[4]]]).to_csv(self.csv, index=False)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_missing_holdout_patient_id_rejected(self):
        pd.DataFrame({"image_name": ["ISIC_1"], "patient_id": [None]}).to_csv(
            self.holdout, index=False,
        )
        with self.assertRaises(ValueError):
            self.prepare()


if __name__ == "__main__":
    unittest.main()