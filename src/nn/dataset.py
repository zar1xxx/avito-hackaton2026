"""PyTorch Dataset и collate_fn для ранжирования лент рекомендаций Avito.

Поддерживает:
- Чтение реальных данных соревнования (гибкий поиск паркетов и memmap по разным путям)
- Ленивый np.memmap без блокировок памяти
- Быстрое извлечение признаков (f1..f5 + внутрилентовые z-score + пересечения с историей)
- Динамический паддинг и маски валидности/наличия эмбеддингов
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    Dataset = object
    torch = None

EMB_DIM = 128
FLOAT_FEATS = ["f1", "f2", "f3", "f4", "f5"]
NUM_CAND_FEATS = 15  # f1..f5 (5) + f1_z..f5_z (5) + item_in_hist (1) + cat_share_hist (1) + feed_size (1) + hist_len (1) + cand_rel_pos (1)


def find_parquet_parts(data_root: str | Path, split: str) -> list[Path]:
    """Находит parquet-файлы для сплита, поддерживая разные структуры каталогов."""
    root = Path(data_root)
    # 1. split/impressions/part-*.parquet
    imp_dir = root / split / "impressions"
    parts = sorted(imp_dir.glob("part-*.parquet"))
    if not parts:
        # 2. split/impressions/*.parquet
        parts = sorted(imp_dir.glob("*.parquet"))
    if not parts:
        # 3. split/*.parquet
        parts = sorted((root / split).glob("*.parquet"))
    if not parts:
        # 4. Резерв: files matching split in data_root
        parts = sorted(root.glob(f"*{split}*.parquet"))

    if not parts:
        raise FileNotFoundError(f"Не найдены parquet файлы для сплита '{split}' в {root}")
    return parts


def load_item_embeddings_memmap(data_root: str | Path, split: str) -> np.memmap:
    """Открывает бинарный файл эмбеддингов в режиме memmap r."""
    root = Path(data_root)
    candidates = [
        root / split / "item_embeddings",
        root / "item_embeddings",
        root / "train" / "item_embeddings",
    ]
    meta_path = None
    for d in candidates:
        if (d / "data.meta").exists():
            meta_path = d / "data.meta"
            emb_dir = d
            break

    if meta_path is None:
        raise FileNotFoundError(
            f"Не найден data.meta для эмбеддингов ни по одному из путей: {[str(c) for c in candidates]}"
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

    # 1. f1..f5 (5 признаков)
    base_feats = np.zeros((m, len(FLOAT_FEATS)), dtype=np.float32)
    for col_idx, col in enumerate(FLOAT_FEATS):
        val = row.get(col)
        if val is not None and len(val) == m:
            base_feats[:, col_idx] = np.asarray(val, dtype=np.float32)

    # 2. Внутрилентовый z-score для f1..f5 (5 признаков)
    mean = np.mean(base_feats, axis=0, keepdims=True)
    std = np.std(base_feats, axis=0, keepdims=True)
    z_feats = (base_feats - mean) / (std + 1e-5)

    # 3. Пересечение с историей
    if h > 0:
        eq_item = cand_items[:, None] == h_items[None, :]
        item_in_hist = eq_item.any(axis=1).astype(np.float32)

        eq_cat = cand_cats[:, None] == h_cats[None, :]
        cat_share_hist = eq_cat.mean(axis=1).astype(np.float32)
    else:
        item_in_hist = np.zeros(m, dtype=np.float32)
        cat_share_hist = np.zeros(m, dtype=np.float32)

    # 4. Контекстные фичи ленты и позиции
    feed_size = np.full(m, m / 40.0, dtype=np.float32)
    hist_len = np.full(m, h / 256.0, dtype=np.float32)
    cand_rel_pos = np.arange(m, dtype=np.float32) / max(m, 1)

    feats_all = np.column_stack([
        base_feats,       # 5
        z_feats,          # 5
        item_in_hist,     # 1
        cat_share_hist,   # 1
        feed_size,        # 1
        hist_len,         # 1
        cand_rel_pos,     # 1
    ]).astype(np.float32)

    return feats_all


class FeedRankingDataset(Dataset):
    """Датасет рекомендательных лент с чтением parquet порциями и memmap эмбеддингами."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        parts: Sequence[Path] | None = None,
        max_parts: int | None = None,
        max_hist_len: int = 256,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.max_hist_len = max_hist_len

        # Поиск parquet-партов
        if parts is None:
            self.parts = find_parquet_parts(self.data_root, self.split)
        else:
            self.parts = list(parts)

        if max_parts is not None:
            self.parts = self.parts[:max_parts]

        # Строим индекс смещений строк без полной загрузки данных в память
        self.part_sizes: list[int] = []
        for p in self.parts:
            meta = pq.ParquetFile(p).metadata
            self.part_sizes.append(meta.num_rows)

        self.cum_sizes = np.cumsum([0] + self.part_sizes)
        self.total_rows = int(self.cum_sizes[-1])

        self._emb_memmap: np.memmap | None = None
        self._cached_part_idx: int = -1
        self._cached_table: Any = None

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
        if self._cached_part_idx != part_idx:
            self._cached_table = pq.read_table(self.parts[part_idx])
            self._cached_part_idx = part_idx

        return {col: self._cached_table[col][local_idx].as_py() for col in self._cached_table.column_names}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._get_row(idx)
        emb = self.emb_memmap

        feed_id = int(row["feed_id"])
        user_id = int(row.get("user_id", 0))

        # Эмбеддинг пользователя
        raw_user_emb = row.get("user_embedding")
        if raw_user_emb is not None and len(raw_user_emb) > 0:
            user_emb = (np.asarray(raw_user_emb, dtype=np.float32) / 128.0).astype(np.float32)
        else:
            user_emb = np.zeros(EMB_DIM, dtype=np.float32)

        # Кандидаты ленты
        cand_items = np.asarray(row["item_id"], dtype=np.int64)
        cand_cats = np.asarray(row.get("category_id") or np.zeros_like(cand_items), dtype=np.int64)
        cand_emb_idx = np.asarray(row.get("item_id__embedding_idx") or np.full_like(cand_items, -1), dtype=np.int64)
        m = len(cand_items)

        # История
        h_items = np.asarray(row.get("history__item_id") or [], dtype=np.int64)[:self.max_hist_len]
        h_cats = np.asarray(row.get("history__category_id") or np.zeros_like(h_items), dtype=np.int64)[:self.max_hist_len]
        h_ev_types = np.asarray(row.get("history__event_type") or np.zeros_like(h_items), dtype=np.int64)[:self.max_hist_len]
        h_ev_sources = np.asarray(row.get("history__event_source") or np.zeros_like(h_items), dtype=np.int64)[:self.max_hist_len]
        h_emb_idx = np.asarray(row.get("history__item_id__embedding_idx") or np.full_like(h_items, -1), dtype=np.int64)[:self.max_hist_len]

        # Извлечение эмбеддингов
        cand_embs, cand_has_emb = gather_embeddings(emb, cand_emb_idx, dim=EMB_DIM)
        hist_embs, hist_has_emb = gather_embeddings(emb, h_emb_idx, dim=EMB_DIM)

        # Инженерные признаки кандидатов
        cand_features = compute_candidate_features(row, cand_items, cand_cats, h_items, h_cats)

        sample = {
            "feed_id": feed_id,
            "user_id": user_id,
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

        if "target" in row and row["target"] is not None:
            sample["target"] = np.asarray(row["target"], dtype=np.float32)
        if "pos" in row and row["pos"] is not None:
            sample["pos"] = np.asarray(row["pos"], dtype=np.int64)

        return sample


def collate_ranking_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Собирает батч лент с динамическим паддингом и булевыми масками."""
    if torch is None:
        raise RuntimeError("PyTorch не установлен!")

    b = len(batch)
    m_max = max(max(item["cand_len"] for item in batch), 1)
    h_max = max(max(item["hist_len"] for item in batch), 1)
    num_cand_feats = batch[0]["cand_features"].shape[1] if batch[0]["cand_features"].ndim == 2 else NUM_CAND_FEATS

    has_target = "target" in batch[0]
    has_pos = "pos" in batch[0]

    candidate_mask = torch.zeros((b, m_max), dtype=torch.bool)
    cand_has_emb_mask = torch.zeros((b, m_max), dtype=torch.bool)
    cand_emb = torch.zeros((b, m_max, EMB_DIM), dtype=torch.float32)
    cand_item_id = torch.zeros((b, m_max), dtype=torch.long)
    cand_category_id = torch.zeros((b, m_max), dtype=torch.long)
    cand_features = torch.zeros((b, m_max, num_cand_feats), dtype=torch.float32)
    target = torch.zeros((b, m_max), dtype=torch.float32) if has_target else None
    pos = torch.zeros((b, m_max), dtype=torch.long) if has_pos else None

    history_mask = torch.zeros((b, h_max), dtype=torch.bool)
    hist_has_emb_mask = torch.zeros((b, h_max), dtype=torch.bool)
    hist_emb = torch.zeros((b, h_max, EMB_DIM), dtype=torch.float32)
    hist_item_id = torch.zeros((b, h_max), dtype=torch.long)
    hist_category_id = torch.zeros((b, h_max), dtype=torch.long)
    hist_event_type = torch.zeros((b, h_max), dtype=torch.long)
    hist_event_source = torch.zeros((b, h_max), dtype=torch.long)

    feed_ids = torch.zeros(b, dtype=torch.long)
    user_ids = torch.zeros(b, dtype=torch.long)
    user_embs = torch.zeros((b, EMB_DIM), dtype=torch.float32)
    cand_lens = torch.zeros(b, dtype=torch.long)
    hist_lens = torch.zeros(b, dtype=torch.long)

    for i, item in enumerate(batch):
        m = item["cand_len"]
        h = item["hist_len"]

        feed_ids[i] = item["feed_id"]
        user_ids[i] = item["user_id"]
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

    return result
