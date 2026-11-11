import tempfile
from pathlib import Path
import unittest

import polars as pl

from scripts.validate_submission import validate


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.template = Path(self.tmp.name) / "template.parquet"
        self.submission = Path(self.tmp.name) / "submission.parquet"
        self.frame = pl.DataFrame({"feed_id": [1, 1, 2], "item_id": [10, 20, 30],
                                   "score": [0.1, 0.2, 0.3]},
                                  schema={"feed_id": pl.Int32, "item_id": pl.Int32, "score": pl.Float32})
        self.frame.write_parquet(self.template)

    def check_frame(self, frame):
        frame.write_parquet(self.submission)
        return validate(str(self.template), str(self.submission))

    def test_valid_and_reordered(self):
        self.assertEqual(self.check_frame(self.frame.reverse()), [])

    def test_missing_extra_duplicate(self):
        for frame, message in (
            (self.frame.head(2), "не хватает"),
            (self.frame.with_columns((pl.col("item_id") + 100).alias("item_id")), "лишних"),
            (pl.concat([self.frame, self.frame.head(1)]), "дубликаты"),
        ):
            with self.subTest(message=message):
                self.assertTrue(any(message in err for err in self.check_frame(frame)))

    def test_invalid_scores(self):
        for value in (float("nan"), float("inf"), float("-inf"), None):
            with self.subTest(value=value):
                self.assertTrue(self.check_frame(self.frame.with_columns(
                    pl.lit(value, dtype=pl.Float32).alias("score"))))

    def test_null_keys(self):
        for column in ("feed_id", "item_id"):
            with self.subTest(column=column):
                self.assertTrue(self.check_frame(self.frame.with_columns(
                    pl.lit(None, dtype=pl.Int32).alias(column))))

    def test_wrong_schema(self):
        for frame in (self.frame.drop("score"), self.frame.with_columns(pl.col("score").cast(pl.String)),
                      self.frame.with_columns(pl.col("feed_id").cast(pl.Int64))):
            self.assertTrue(self.check_frame(frame))

    def test_empty_template(self):
        self.frame.head(0).write_parquet(self.template)
        self.assertTrue(self.check_frame(self.frame.head(0)))


if __name__ == "__main__":
    unittest.main()
