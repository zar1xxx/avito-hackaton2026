from pathlib import Path
import tempfile
import unittest

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

    def test_truncated_embeddings(self):
        (self.root / "train/item_embeddings/data.memmap").write_bytes(b"broken")
        self.assertTrue(any("memmap size" in error for error in inspect_split(self.root, "train", 2)["errors"]))

    def test_missing_split(self):
        self.assertTrue(inspect_split(self.root, "val", 2)["errors"])


if __name__ == "__main__":
    unittest.main()
