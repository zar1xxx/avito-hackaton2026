"""PyTorch Dataset и collate_fn для ранжирования лент рекомендаций Avito.

Поддерживает:
- ShardedFeedRankingDataset (IterableDataset): быстрое чтение 512+ партов через Polars
  (каждый парт читается ровно 1 раз за эпоху, ленты перемешиваются внутри парта, zero .as_py()).
- FeedRankingDataset (Map-style Dataset): случайный доступ по глобальному индексу (для отладки и Subset).
- Строгие per-split таблицы эмбеддингов (np.memmap int8 -> float32, -1 = нулевой вектор + маска).
- Динамический паддинг и 15 инженерных признаков кандидатов.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import polars as pl
import pyarrow.parquet as pq

try:
    import torch
    from torch.utils.data import Dataset, IterableDataset, get_worker_info
except ImportError:
    Dataset = object
    IterableDataset = object
    get_worker_info = lambda: None
    torch = None

EMB_DIM = 128
FLOAT_FEATS = ["f1", "f2", "f3", "f4", "f5"]
NUM_CAND_FEATS = 15  # f1..f5 (5) + f1_z..f5_z (5) + item_in_hist (1) + cat_share_hist (1) + feed_size (1) + hist_len (1) + cand_rel_pos (1)

EXPECTED_COLUMNS = [
    "feed_id",
    "user_id",
    "timestamp",
    "user_embedding",
    "item_id",
    "category_id",
    "item_id__embedding_idx",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "history__item_id",
    "history__category_id",
    "history__event_type",
    "history__event_source",
    "history__item_id__embedding_idx",
    "target",
    "pos",
]


def find_parquet_parts(data_root: str | Path, split: str) -> list[Path]:
    """Находит parquet-файлы для сплита, поддерживая разные структуры каталогов."""
    root = Path(data_root)
    imp_dir = root / split / "impressions"
    parts = sorted(imp_dir.glob("part-*.parquet"))
    if not parts:
        parts = sorted(imp_dir.glob("*.parquet"))
    if not parts:
        parts = sorted((root / split).glob("*.parquet"))
    if not parts:
        parts = sorted(root.glob(f"*{split}*.parquet"))

    if not parts:
        raise FileNotFoundError(f"Не найдены parquet файлы для сплита '{split}' в {root}")
    return parts


def load_item_embeddings_memmap(data_root: str | Path, split: str) -> np.memmap:
    """Открывает таблицу эмбеддингов именно этого сплита в режиме memmap r.

    У каждого сплита (train, val, test) своя таблица эмбеддингов и свои индексы строк,
    поэтому подставлять таблицу другого сплита категорически нельзя.
    """
    emb_dir = Path(data_root) / split / "item_embeddings"
    meta_path = emb_dir / "data.meta"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Не найден {meta_path}: у каждого сплита должна быть своя таблица эмбеддингов "
            "(подстановка таблицы train для val/test дает неверные векторы)."
        )

    meta = json.loads(meta_path.read_text())
    return np.memmap(
        emb_dir / "data.memmap",
        dtype=meta["dtype"],
        mode="r",
        shape=tuple(meta["shape"]),
        order=meta.get("order", "C"),
    )


def gather_embeddings(emb_memmap: np.memmap, indices: np.ndarray, dim: int = EMB_DIM) -> tuple[np.ndarray, np.ndarray]:
    """Извлекает эмбеддинги по индексам. Индекс -1 = вектор нулей."""
    idx = np.asarray(indices, dtype=np.int64)
    n = len(idx)
    out = np.zeros((n, dim), dtype=np.float32)
    has_emb = idx >= 0
    if has_emb.any():
        out[has_emb] = emb_memmap[idx[has_emb]].astype(np.float32) / 128.0
    return out, has_emb


def compute_candidate_features(
    row: dict[str, Any],
    cand_items: np.ndarray,
    cand_cats: np.ndarray,
    h_items: np.ndarray,
    h_cats: np.ndarray,
) -> np.ndarray:
    """Вычисляет 15 информативных признаков кандидата на лету через быстрый NumPy."""
    m = len(cand_items)
    h = len(h_items)
    if m == 0:
        return np.zeros((0, NUM_CAND_FEATS), dtype=np.float32)

    base_feats = np.zeros((m, len(FLOAT_FEATS)), dtype=np.float32)
    for col_idx, col in enumerate(FLOAT_FEATS):
        val = row.get(col)
        if val is not None and len(val) == m:
            base_feats[:, col_idx] = np.asarray(val, dtype=np.float32)

    mean = np.mean(base_feats, axis=0, keepdims=True)
    std = np.std(base_feats, axis=0, keepdims=True)
    z_feats = (base_feats - mean) / (std + 1e-5)

    if h > 0:
        eq_item = cand_items[:, None] == h_items[None, :]
        item_in_hist = eq_item.any(axis=1).astype(np.float32)

        eq_cat = cand_cats[:, None] == h_cats[None, :]
        cat_share_hist = eq_cat.mean(axis=1).astype(np.float32)
    else:
        item_in_hist = np.zeros(m, dtype=np.float32)
        cat_share_hist = np.zeros(m, dtype=np.float32)

    feed_size = np.full(m, m / 40.0, dtype=np.float32)
    hist_len = np.full(m, h / 256.0, dtype=np.float32)
    cand_rel_pos = np.arange(m, dtype=np.float32) / max(m, 1)

    return np.column_stack([
        base_feats,       # 5
        z_feats,          # 5
        item_in_hist,     # 1
        cat_share_hist,   # 1
        feed_size,        # 1
        hist_len,         # 1
        cand_rel_pos,     # 1
    ]).astype(np.float32)


def _as_int64_array(val: Any, fallback: np.ndarray, max_len: int | None = None) -> np.ndarray:
    if val is None:
        return fallback
    arr = np.asarray(val, dtype=np.int64)
    if max_len is not None:
        arr = arr[:max_len]
    return arr


def row_to_sample(row: dict[str, Any], emb: np.memmap, max_hist_len: int = 256) -> dict[str, Any]:
    """Преобразует словарь одной ленты в готовый сэмпл с эмбеддингами и фичами."""
    feed_id = int(row["feed_id"])
    user_id = int(row["user_id"]) if row.get("user_id") is not None else 0
    timestamp = int(row["timestamp"]) if row.get("timestamp") is not None else 0

    raw_user_emb = row.get("user_embedding")
    if raw_user_emb is not None and len(raw_user_emb) > 0:
        user_emb = (np.asarray(raw_user_emb, dtype=np.float32) / 128.0).astype(np.float32)
    else:
        user_emb = np.zeros(EMB_DIM, dtype=np.float32)

    cand_items = np.asarray(row["item_id"], dtype=np.int64)
    m = len(cand_items)
    zeros_m = np.zeros(m, dtype=np.int64)
    minus_ones_m = np.full(m, -1, dtype=np.int64)

    cand_cats = _as_int64_array(row.get("category_id"), zeros_m)
    cand_emb_idx = _as_int64_array(row.get("item_id__embedding_idx"), minus_ones_m)

    raw_h_items = row.get("history__item_id")
    if raw_h_items is None or len(raw_h_items) == 0:
        h_items = np.zeros(0, dtype=np.int64)
        h_cats = np.zeros(0, dtype=np.int64)
        h_ev_types = np.zeros(0, dtype=np.int64)
        h_ev_sources = np.zeros(0, dtype=np.int64)
        h_emb_idx = np.zeros(0, dtype=np.int64)
    else:
        h_items = np.asarray(raw_h_items, dtype=np.int64)[:max_hist_len]
        h = len(h_items)
        zeros_h = np.zeros(h, dtype=np.int64)
        minus_ones_h = np.full(h, -1, dtype=np.int64)
        h_cats = _as_int64_array(row.get("history__category_id"), zeros_h, max_hist_len)
        h_ev_types = _as_int64_array(row.get("history__event_type"), zeros_h, max_hist_len)
        h_ev_sources = _as_int64_array(row.get("history__event_source"), zeros_h, max_hist_len)
        h_emb_idx = _as_int64_array(row.get("history__item_id__embedding_idx"), minus_ones_h, max_hist_len)

    cand_embs, cand_has_emb = gather_embeddings(emb, cand_emb_idx, dim=EMB_DIM)
    hist_embs, hist_has_emb = gather_embeddings(emb, h_emb_idx, dim=EMB_DIM)
    cand_features = compute_candidate_features(row, cand_items, cand_cats, h_items, h_cats)

    sample = {
        "feed_id": feed_id,
        "user_id": user_id,
        "timestamp": timestamp,
        "user_emb": user_emb,
        "cand_len": m,
        "cand_item_id": cand_items,
        "cand_category_id": cand_cats,
        "cand_features": cand_features,
        "cand_emb": cand_embs,
        "cand_has_emb": cand_has_emb,
        "hist_len": len(h_items),
        "hist_item_id": h_items,
        "hist_category_id": h_cats,
        "hist_event_type": h_ev_types,
        "hist_event_source": h_ev_sources,
        "hist_emb": hist_embs,
        "hist_has_emb": hist_has_emb,
    }

    raw_target = row.get("target")
    if raw_target is not None:
        sample["target"] = np.asarray(raw_target, dtype=np.float32)

    raw_pos = row.get("pos")
    if raw_pos is not None:
        # Поддержка возможного null внутри списка pos
        try:
            sample["pos"] = np.asarray(raw_pos, dtype=np.int64)
        except (TypeError, ValueError):
            sample["pos"] = np.array([-1 if p is None else int(p) for p in raw_pos], dtype=np.int64)

    return sample


class ShardedFeedRankingDataset(IterableDataset):
    """Быстрый шардированный загрузчик для 512+ партов и миллионов лент.

    Оптимизации:
    1. Каждый parquet-парт читается через Polars ровно 1 раз за эпоху (только нужные колонки).
    2. Колонки парта конвертируются в списки за один Rust-вызов (без .as_py() по строкам).
    3. При shuffle=True перемешивается порядок партов и порядок лент внутри парта.
    4. Корректно распределяет парты между воркерами DataLoader при num_workers > 0.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        parts: Sequence[Path] | None = None,
        max_parts: int | None = None,
        max_feeds: int | None = None,
        max_hist_len: int = 256,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.max_hist_len = max_hist_len
        self.max_feeds = max_feeds
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        if parts is None:
            self.parts = find_parquet_parts(self.data_root, self.split)
        else:
            self.parts = list(parts)

        if max_parts is not None:
            self.parts = self.parts[:max_parts]

        self.part_sizes: list[int] = [pq.ParquetFile(p).metadata.num_rows for p in self.parts]
        self.total_rows = int(sum(self.part_sizes))
        if self.max_feeds is not None and self.max_feeds > 0:
            self.total_rows = min(self.total_rows, self.max_feeds)

        schema_cols = set(pq.ParquetFile(self.parts[0]).schema_arrow.names)
        self.cols_to_read = [c for c in EXPECTED_COLUMNS if c in schema_cols]
        self._emb_memmap: np.memmap | None = None

    def set_epoch(self, epoch: int) -> None:
        """Обновляет эпоху для детерминированного перемешивания партов."""
        self.epoch = epoch

    @property
    def emb_memmap(self) -> np.memmap:
        if self._emb_memmap is None:
            self._emb_memmap = load_item_embeddings_memmap(self.data_root, self.split)
        return self._emb_memmap

    def __len__(self) -> int:
        return self.total_rows

    def __iter__(self) -> Iterator[dict[str, Any]]:
        emb = self.emb_memmap
        rng = np.random.default_rng(self.seed + self.epoch)

        n_parts = len(self.parts)
        if self.shuffle:
            part_order = rng.permutation(n_parts)
        else:
            part_order = np.arange(n_parts)

        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            worker_part_indices = part_order[worker_id::num_workers]
            # Пропорциональная квота max_feeds на воркер
            if self.max_feeds is not None and self.max_feeds > 0:
                base_q = self.max_feeds // num_workers
                rem_q = self.max_feeds % num_workers
                quota = base_q + (1 if worker_id < rem_q else 0)
            else:
                quota = None
        else:
            worker_part_indices = part_order
            quota = self.max_feeds if (self.max_feeds is not None and self.max_feeds > 0) else None

        yielded = 0
        for p_idx in worker_part_indices:
            if quota is not None and yielded >= quota:
                break

            part_path = self.parts[int(p_idx)]
            df = pl.read_parquet(part_path, columns=self.cols_to_read)
            n_rows = df.height
            if n_rows == 0:
                continue

            if self.shuffle:
                row_indices = rng.permutation(n_rows)
            else:
                row_indices = np.arange(n_rows)

            if quota is not None:
                remaining = quota - yielded
                if remaining < len(row_indices):
                    row_indices = row_indices[:remaining]

            # Быстрая пакетная распаковка колонок без построчного pyarrow.as_py()
            col_data = {col: df[col].to_list() for col in self.cols_to_read}
            del df

            for r_idx in row_indices:
                idx_int = int(r_idx)
                row = {col: col_data[col][idx_int] for col in self.cols_to_read}
                yield row_to_sample(row, emb, self.max_hist_len)
                yielded += 1


class FeedRankingDataset(Dataset):
    """Map-style датасет рекомендательных лент с кэшем последнего парта через Polars."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        parts: Sequence[Path] | None = None,
        max_parts: int | None = None,
        max_feeds: int | None = None,
        max_hist_len: int = 256,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.max_hist_len = max_hist_len
        self.max_feeds = max_feeds

        if parts is None:
            self.parts = find_parquet_parts(self.data_root, self.split)
        else:
            self.parts = list(parts)

        if max_parts is not None:
            self.parts = self.parts[:max_parts]

        self.part_sizes: list[int] = [pq.ParquetFile(p).metadata.num_rows for p in self.parts]
        self.cum_sizes = np.cumsum([0] + self.part_sizes)
        self.total_rows = int(self.cum_sizes[-1])

        if self.max_feeds is not None and self.max_feeds > 0:
            self.total_rows = min(self.total_rows, self.max_feeds)
            active_parts_count = int(np.searchsorted(self.cum_sizes, self.total_rows, side="right"))
            self.parts = self.parts[:max(active_parts_count, 1)]

        schema_cols = set(pq.ParquetFile(self.parts[0]).schema_arrow.names)
        self.cols_to_read = [c for c in EXPECTED_COLUMNS if c in schema_cols]

        self._emb_memmap: np.memmap | None = None
        self._cached_part_idx: int = -1
        self._cached_cols: dict[str, list[Any]] | None = None

    @property
    def emb_memmap(self) -> np.memmap:
        if self._emb_memmap is None:
            self._emb_memmap = load_item_embeddings_memmap(self.data_root, self.split)
        return self._emb_memmap

    def __len__(self) -> int:
        return self.total_rows

    def _locate_row(self, global_idx: int) -> tuple[int, int]:
        part_idx = int(np.searchsorted(self.cum_sizes, global_idx, side="right") - 1)
        local_idx = global_idx - int(self.cum_sizes[part_idx])
        return part_idx, local_idx

    def _get_row(self, global_idx: int) -> dict[str, Any]:
        part_idx, local_idx = self._locate_row(global_idx)
        if self._cached_part_idx != part_idx or self._cached_cols is None:
            df = pl.read_parquet(self.parts[part_idx], columns=self.cols_to_read)
            self._cached_cols = {col: df[col].to_list() for col in self.cols_to_read}
            self._cached_part_idx = part_idx

        return {col: self._cached_cols[col][local_idx] for col in self.cols_to_read}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._get_row(idx)
        return row_to_sample(row, self.emb_memmap, self.max_hist_len)


def collate_ranking_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Собирает батч лент с динамическим паддингом и булевыми масками."""
    if torch is None:
        raise RuntimeError("PyTorch не установлен!")

    b = len(batch)
    m_max = max(max(item["cand_len"] for item in batch), 1)
    h_max = max(max(item["hist_len"] for item in batch), 1)
    num_cand_feats = batch[0]["cand_features"].shape[1] if batch[0]["cand_features"].ndim == 2 else NUM_CAND_FEATS

    has_target = any("target" in item for item in batch)
    has_pos = any("pos" in item for item in batch)

    candidate_mask = torch.zeros((b, m_max), dtype=torch.bool)
    cand_has_emb_mask = torch.zeros((b, m_max), dtype=torch.bool)
    cand_emb = torch.zeros((b, m_max, EMB_DIM), dtype=torch.float32)
    cand_item_id = torch.zeros((b, m_max), dtype=torch.long)
    cand_category_id = torch.zeros((b, m_max), dtype=torch.long)
    cand_features = torch.zeros((b, m_max, num_cand_feats), dtype=torch.float32)
    target = torch.zeros((b, m_max), dtype=torch.float32) if has_target else None
    pos = torch.full((b, m_max), -1, dtype=torch.long) if has_pos else None

    history_mask = torch.zeros((b, h_max), dtype=torch.bool)
    hist_has_emb_mask = torch.zeros((b, h_max), dtype=torch.bool)
    hist_emb = torch.zeros((b, h_max, EMB_DIM), dtype=torch.float32)
    hist_item_id = torch.zeros((b, h_max), dtype=torch.long)
    hist_category_id = torch.zeros((b, h_max), dtype=torch.long)
    hist_event_type = torch.zeros((b, h_max), dtype=torch.long)
    hist_event_source = torch.zeros((b, h_max), dtype=torch.long)

    feed_ids = torch.zeros(b, dtype=torch.long)
    user_ids = torch.zeros(b, dtype=torch.long)
    timestamps = torch.zeros(b, dtype=torch.long)
    user_embs = torch.zeros((b, EMB_DIM), dtype=torch.float32)
    cand_lens = torch.zeros(b, dtype=torch.long)
    hist_lens = torch.zeros(b, dtype=torch.long)

    for i, item in enumerate(batch):
        m = item["cand_len"]
        h = item["hist_len"]

        feed_ids[i] = item["feed_id"]
        user_ids[i] = item["user_id"]
        timestamps[i] = item.get("timestamp", 0)
        user_embs[i] = torch.from_numpy(item["user_emb"])
        cand_lens[i] = m
        hist_lens[i] = h

        if m > 0:
            candidate_mask[i, :m] = True
            cand_has_emb_mask[i, :m] = torch.from_numpy(item["cand_has_emb"])
            cand_emb[i, :m] = torch.from_numpy(item["cand_emb"])
            cand_item_id[i, :m] = torch.from_numpy(item["cand_item_id"])
            cand_category_id[i, :m] = torch.from_numpy(item["cand_category_id"])
            if num_cand_feats > 0:
                cand_features[i, :m] = torch.from_numpy(item["cand_features"])

            if has_target and "target" in item:
                target[i, :m] = torch.from_numpy(item["target"])
            if has_pos and "pos" in item:
                pos[i, :m] = torch.from_numpy(item["pos"])

        if h > 0:
            history_mask[i, :h] = True
            hist_has_emb_mask[i, :h] = torch.from_numpy(item["hist_has_emb"])
            hist_emb[i, :h] = torch.from_numpy(item["hist_emb"])
            hist_item_id[i, :h] = torch.from_numpy(item["hist_item_id"])
            hist_category_id[i, :h] = torch.from_numpy(item["hist_category_id"])
            hist_event_type[i, :h] = torch.from_numpy(item["hist_event_type"])
            hist_event_source[i, :h] = torch.from_numpy(item["hist_event_source"])

    result = {
        "feed_id": feed_ids,
        "user_id": user_ids,
        "timestamp": timestamps,
        "user_emb": user_embs,
        "cand_len": cand_lens,
        "hist_len": hist_lens,
        "candidate_mask": candidate_mask,
        "cand_has_emb_mask": cand_has_emb_mask,
        "cand_emb": cand_emb,
        "cand_item_id": cand_item_id,
        "cand_category_id": cand_category_id,
        "cand_features": cand_features,
        "history_mask": history_mask,
        "hist_has_emb_mask": hist_has_emb_mask,
        "hist_emb": hist_emb,
        "hist_item_id": hist_item_id,
        "hist_category_id": hist_category_id,
        "hist_event_type": hist_event_type,
        "hist_event_source": hist_event_source,
    }

    if has_target:
        result["target"] = target
    if has_pos:
        result["pos"] = pos
    if "sample_weight" in batch[0]:
        result["sample_weight"] = torch.tensor([item.get("sample_weight", 1.0) for item in batch], dtype=torch.float32)

    return result
