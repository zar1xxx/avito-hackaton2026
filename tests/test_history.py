import unittest

import numpy as np
import polars as pl

from src.common.history import _feed_block, build_feature_part, feature_names


class HistoryTests(unittest.TestCase):
    def test_hand_computed_block(self):
        # история от новых к старым: товары 7,5,7,9; категории 2,3,2,4; типы 1,2,2,1
        out = _feed_block(np.array([7, 5, 1]), np.array([2, 3, 8]),
                          np.array([7, 5, 7, 9]), np.array([2, 3, 2, 4]), np.array([1, 2, 2, 1]))
        np.testing.assert_allclose(out['h_item_cnt'], [2, 1, 0])
        np.testing.assert_allclose(out['h_item_last'], [0, 1 / 256, 1])
        np.testing.assert_allclose(out['h_item_last_evt'], [1, 2, 0])
        np.testing.assert_allclose(out['h_item_evt1'], [1, 0, 0])
        np.testing.assert_allclose(out['h_item_evt2'], [1, 1, 0])
        np.testing.assert_allclose(out['h_item_evt_nunique'], [2, 1, 0])
        np.testing.assert_allclose(out['h_cat_share'], [0.5, 0.25, 0])
        np.testing.assert_allclose(out['h_cat_is_last'], [1, 0, 0])
        np.testing.assert_allclose(out['h_cat_rank'], [0, 1, 3])
        np.testing.assert_allclose(out['h_cat_evt1'], [0.5, 0, 0])      # тип 1: категории 2 и 4
        np.testing.assert_allclose(out['h_ncats'], [3, 3, 3])
        np.testing.assert_allclose(out['h_user_evt1'], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(out['h_item_cnt_feedrank'], [2.5 / 3, 1.5 / 3, 0.5 / 3])

    def test_empty_history_and_alignment(self):
        cand = pl.DataFrame({'split': ['val'] * 5, 'feed_id': [2, 2, 1, 1, 1], 'candidate_index': [1, 0, 2, 0, 1],
                             'item_id': [10, 11, 5, 6, 7], 'category_id': [1, 1, 2, 3, 2]})
        feeds = pl.DataFrame({'feed_id': [1, 2], 'history__item_id': [[7, 7], []],
                              'history__category_id': [[2, 2], []], 'history__event_type': [[3, 1], []]},
                             schema_overrides={'history__item_id': pl.List(pl.Int64),
                                               'history__category_id': pl.List(pl.Int64),
                                               'history__event_type': pl.List(pl.Int64)})
        out = build_feature_part(cand, feeds)
        self.assertEqual(out.columns[:4], ['split', 'feed_id', 'candidate_index', 'item_id'])
        self.assertEqual(out.columns[4:], feature_names())
        row = out.filter((pl.col('feed_id') == 1) & (pl.col('item_id') == 7))
        self.assertEqual(row['h_item_cnt'][0], 2.0)
        self.assertEqual(row['h_item_last_evt'][0], 3.0)
        empty = out.filter(pl.col('feed_id') == 2)
        self.assertTrue((empty['h_item_cnt'] == 0).all() and (empty['h_item_last'] == 1).all())

    def test_candidate_order_does_not_change_values(self):
        rng = np.random.default_rng(0)
        ci, cc = rng.integers(1, 20, 12), rng.integers(1, 5, 12)
        h = (rng.integers(1, 20, 40), rng.integers(1, 5, 40), rng.integers(1, 11, 40))
        base = _feed_block(ci, cc, *h)
        perm = rng.permutation(12)
        shuffled = _feed_block(ci[perm], cc[perm], *h)
        for name in feature_names():
            np.testing.assert_allclose(shuffled[name], base[name][perm], rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
