from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from eval.make_synthetic import make_split
from scripts.check_dataset import inspect_split


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        make_split(self.root, "train", 6, np.random.default_rng(0), 0, True, True)
        self.part = self.root / "train/impressions/part-00000.parquet"

    def test_valid_data(self):
        result = inspect_split(self.root, "train", 2)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["feeds"], 6)
        self.assertEqual(result["sampled_feeds"], 2)

    def test_duplicate_feed_ids_outside_sample(self):
        frame = pl.read_parquet(self.part)
        frame.tail(1).write_parquet(self.part.with_name("part-00001.parquet"))
        result = inspect_split(self.root, "train", 1)
        self.assertTrue(any("duplicate feed_id" in error for error in result["errors"]))

    def test_missing_columns(self):
        pl.read_parquet(self.part).drop("target").write_parquet(self.part)
        self.assertTrue(inspect_split(self.root, "train", 2)["errors"])

    def test_repeated_train_item_is_reported_without_rejecting_data(self):
        frame = pl.read_parquet(self.part)
        rows = frame.to_dicts()
        rows[0]['item_id'][1] = rows[0]['item_id'][0]
        pl.DataFrame(rows, schema=frame.schema).write_parquet(self.part)
        result = inspect_split(self.root, 'train', 2)
        self.assertEqual(result['errors'], [])
        self.assertTrue(result['warnings'])

    def test_repeated_test_item_is_rejected_for_submission_keys(self):
        make_split(self.root, 'test', 2, np.random.default_rng(1), 100, False, False)
        part = self.root / 'test/impressions/part-00000.parquet'
        frame = pl.read_parquet(part)
        rows = frame.to_dicts()
        rows[0]['item_id'][1] = rows[0]['item_id'][0]
        pl.DataFrame(rows, schema=frame.schema).write_parquet(part)
        self.assertTrue(inspect_split(self.root, 'test', 2)['errors'])

    def test_truncated_embeddings(self):
        (self.root / "train/item_embeddings/data.memmap").write_bytes(b"broken")
        self.assertTrue(any("memmap size" in error for error in inspect_split(self.root, "train", 2)["errors"]))

    def test_embedding_size_over_32_bits(self):
        # Match the real train shape without allocating a 10 GB test file.
        embedding_dir = self.root / "train/item_embeddings"
        meta = {"dtype": "int8", "shape": [81_557_944, 128], "order": "C"}
        (embedding_dir / "data.meta").write_text(json.dumps(meta), encoding="utf-8")
        original_stat = Path.stat

        def stat(path, *args, **kwargs):
            if path == embedding_dir / "data.memmap":
                return SimpleNamespace(st_size=10_439_416_832)
            return original_stat(path, *args, **kwargs)

        with patch.object(Path, "stat", autospec=True, side_effect=stat):
            self.assertEqual(inspect_split(self.root, "train", 2)["errors"], [])

    def test_missing_split(self):
        self.assertTrue(inspect_split(self.root, "val", 2)["errors"])


if __name__ == "__main__":
    unittest.main()
