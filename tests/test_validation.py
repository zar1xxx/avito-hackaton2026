import unittest

import polars as pl

from src.common.validation import make_validation_split


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.feeds = pl.DataFrame({"feed_id": list(range(20)), "timestamp": list(range(20))})

    def test_time_split_and_group_integrity(self):
        candidates = pl.concat([self.feeds, self.feeds.head(10)])
        result = make_validation_split(candidates)
        self.assertEqual(result.height, 20)
        public = result.filter(pl.col("leaderboard_split") == "public")
        private = result.filter(pl.col("leaderboard_split") == "private")
        self.assertEqual(public.height, 6)
        self.assertLess(public["timestamp"].max(), private["timestamp"].min())
        self.assertEqual(result["feed_id"].n_unique(), 20)
        self.assertEqual(sorted(result["fold"].value_counts()["count"].to_list()), [4] * 5)

    def test_determinism(self):
        first = make_validation_split(self.feeds)
        shuffled = self.feeds.sample(fraction=1.0, shuffle=True, seed=17)
        self.assertTrue(first.equals(make_validation_split(shuffled)))
        self.assertFalse(first.equals(make_validation_split(self.feeds, seed=1)))

    def test_timestamp_ties_never_cross_holdout(self):
        feeds = self.feeds.with_columns((pl.col("timestamp") // 4).alias("timestamp"))
        result = make_validation_split(feeds)
        self.assertEqual(result.group_by("timestamp").agg(pl.col("leaderboard_split").n_unique())
                         ["leaderboard_split"].max(), 1)

    def test_invalid_inputs(self):
        bad_frames = [self.feeds.head(0), self.feeds.head(2),
                      self.feeds.with_columns(pl.lit(0).alias("timestamp")),
                      self.feeds.with_columns(pl.lit(None).alias("timestamp")),
                      pl.concat([self.feeds, pl.DataFrame({"feed_id": [0], "timestamp": [999]})])]
        for frame in bad_frames:
            with self.subTest(rows=frame.height), self.assertRaises(ValueError):
                make_validation_split(frame)


if __name__ == "__main__":
    unittest.main()
