from pathlib import Path
import tempfile
import unittest

import numpy as np

from eval.make_synthetic import make_split
from src.common.embedding_catalogue import EmbeddingCatalogue, build_catalogue, update_index


class CatalogueTests(unittest.TestCase):
    def test_conflicting_vector_inside_split_rejected(self):
        index = np.full(5, -1, dtype=np.int32)
        vectors = np.zeros((2, 128), dtype=np.int8)
        vectors[1, 0] = 1
        with self.assertRaisesRegex(ValueError, 'within one split'):
            update_index(index, np.array([2, 2]), np.array([0, 1]), vectors)

    def test_identical_aliases_and_missing_indices(self):
        index = np.full(5, -1, dtype=np.int32)
        vectors = np.ones((2, 128), dtype=np.int8)
        aliases = update_index(index, np.array([0, 1, 2, 2, 3]), np.array([0, -1, 0, 1, -1]), vectors)
        self.assertEqual(aliases, 1)
        self.assertEqual(index.tolist(), [-1, -1, 0, -1, -1])

    def test_all_split_versions_roundtrip_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, output = root / 'raw', root / 'catalogue'
            rng = np.random.default_rng(9)
            for number, split in enumerate(('train', 'val', 'test')):
                make_split(raw, split, 10, rng, number * 100, split == 'train', split != 'test')
            manifest = build_catalogue(raw, output)
            self.assertTrue(manifest['complete'])
            self.assertGreater(manifest['audit']['train/val']['different'], 0)
            store = EmbeddingCatalogue(output)
            valid = np.flatnonzero((store.indices['train'] >= 0) & (store.indices['val'] >= 0))[:10]
            self.assertGreater(len(valid), 0)
            train = store.gather('train', valid)
            val = store.gather('val', valid)
            self.assertFalse(np.array_equal(train, val))
            np.testing.assert_array_equal(train, store.vectors['train'][store.lookup('train', valid)].astype(np.float32))
            self.assertEqual(store.gather('train', [0, -1, manifest['max_item_id'] + 1]).sum(), 0.)
            self.assertEqual(store.gather('train', valid, np.ones(len(valid), dtype=bool)).sum(), 0.)
            # Release Windows memmaps before rerunning or TemporaryDirectory cleanup.
            del store
            again = build_catalogue(raw, output, resume=True)
            self.assertEqual(manifest['audit'], again['audit'])


if __name__ == '__main__':
    unittest.main()
