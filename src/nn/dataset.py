"""PyTorch Dataset и collate_fn для ранжирования лент рекомендаций.

Каждый элемент выборки — это одна рекомендательная лента (feed):
- user_embedding (вектор D, float32)
- кандидаты длины M: item_id, category_id, f1..f5, эмбеддинги из memmap
- история длины H (до 256): item_id, category_id, event_type, event_source, эмбеддинги из memmap

В collate_fn выполняется динамический паддинг до M_max и H_max в батче,
формируются маски валидных элементов (candidate_mask, history_mask)
и маски наличия эмбеддингов (cand_has_emb_mask, hist_has_emb_mask).
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
    # Заглушка на случай, если torch еще доустанавливается
    Dataset = object
    torch = None

EMB_DIM = 128
FLOAT_FEATS = ["f1", "f2", "f3", "f4", "f5"]


def load_item_embeddings_memmap(data_root: str | Path, split: str) -> np.memmap:
    """Открывает бинарный файл эмбеддингов в режиме memmap r."""
    d = Path(data_root) / split / "item_embeddings"
    meta = json.loads((d / "data.meta").read_text())
    return np.memmap(
        d / "data.memmap",
        dtype=meta["dtype"],
        mode="r",
        shape=tuple(meta["shape"]),
        order=meta.get("order", "C"),
    )


def gather_embeddings(emb_memmap: np.memmap, indices: np.ndarray, dim: int = EMB_DIM) -> tuple[np.ndarray, np.ndarray]:
    """Извлекает эмбеддинги по индексам.
    
    Args:
        emb_memmap: np.memmap размерности (N_items, dim), int8.
        indices: массив индексов int32/int64 (может содержать -1).
        dim: размерность эмбеддинга.

    Returns:
        embs_float: np.ndarray (len(indices), dim) в float32, нулевой вектор для missing.
        has_emb_mask: np.ndarray (len(indices),) bool, True если индекс >= 0.
    """
    idx = np.asarray(indices, dtype=np.int64)
    n = len(idx)
    out = np.zeros((n, dim), dtype=np.float32)
    has_emb = idx >= 0
    if has_emb.any():
        # Чтение int8 из memmap и каст в float32 с нормализацией /128.0
        out[has_emb] = emb_memmap[idx[has_emb]].astype(np.float32) / 128.0
    return out, has_emb


class FeedRankingDataset(Dataset):
    """Датасет рекомендательных лент с чтением parquet порциями и memmap эмбеддингами.
    
    Не разворачивает кандидатов в плоские строки: одна запись = одна лента.
    """

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
        impressions_dir = self.data_root / split / "impressions"
        if parts is None:
            self.parts = sorted(impressions_dir.glob("part-*.parquet"))
            if not self.parts:
                raise FileNotFoundError(f"Не найдены parquet файлы в {impressions_dir}")
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

        # Lazy memmap: инициализируется при первом обращении в текущем процессе
        self._emb_memmap: np.memmap | None = None

        # Кэш последнего прочитанного парта (для ускорения последовательного чтения)
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
        """Определяет (индекс парта, локальный индекс строки в парте)."""
        part_idx = int(np.searchsorted(self.cum_sizes, global_idx, side="right") - 1)
        local_idx = global_idx - int(self.cum_sizes[part_idx])
        return part_idx, local_idx

    def _get_row(self, global_idx: int) -> dict[str, Any]:
        part_idx, local_idx = self._locate_row(global_idx)
        if self._cached_part_idx != part_idx:
            # Читаем парт в память как pyarrow таблицу (быстро и компактно)
            self._cached_table = pq.read_table(self.parts[part_idx])
            self._cached_part_idx = part_idx

        # Извлекаем словарь строки
        row_dict = {col: self._cached_table[col][local_idx].as_py() for col in self._cached_table.column_names}
        return row_dict

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._get_row(idx)
        emb = self.emb_memmap

        feed_id = int(row["feed_id"])
        user_id = int(row["user_id"])

        # Эмбеддинг пользователя (int8 -> float32)
        raw_user_emb = row.get("user_embedding")
        if raw_user_emb is not None and len(raw_user_emb) > 0:
            user_emb = (np.asarray(raw_user_emb, dtype=np.float32) / 128.0).astype(np.float32)
        else:
            user_emb = np.zeros(EMB_DIM, dtype=np.float32)

        # Кандидаты ленты
        cand_items = np.asarray(row["item_id"], dtype=np.int64)
        cand_cats = np.asarray(row["category_id"], dtype=np.int64)
        cand_emb_idx = np.asarray(row["item_id__embedding_idx"], dtype=np.int64)
        m = len(cand_items)

        # Числовые признаки кандидатов (f1..f5)
        cand_feats_list = [
            np.asarray(row[col], dtype=np.float32) if col in row else np.zeros(m, dtype=np.float32)
            for col in FLOAT_FEATS
        ]
        cand_features = np.column_stack(cand_feats_list).astype(np.float32) if cand_feats_list else np.zeros((m, 0), dtype=np.float32)

        # Извлечение эмбеддингов кандидатов из memmap
        cand_embs, cand_has_emb = gather_embeddings(emb, cand_emb_idx, dim=EMB_DIM)

        # История (обрезка до max_hist_len событий, самые свежие сначала)
        h_items = np.asarray(row.get("history__item_id") or [], dtype=np.int64)[:self.max_hist_len]
        h_cats = np.asarray(row.get("history__category_id") or [], dtype=np.int64)[:self.max_hist_len]
        h_ev_types = np.asarray(row.get("history__event_type") or [], dtype=np.int64)[:self.max_hist_len]
        h_ev_sources = np.asarray(row.get("history__event_source") or [], dtype=np.int64)[:self.max_hist_len]
        h_emb_idx = np.asarray(row.get("history__item_id__embedding_idx") or [], dtype=np.int64)[:self.max_hist_len]

        hist_embs, hist_has_emb = gather_embeddings(emb, h_emb_idx, dim=EMB_DIM)

        sample = {
            "feed_id": feed_id,
            "user_id": user_id,
            "user_emb": user_emb,
            # Кандидаты
            "cand_len": m,
            "cand_item_id": cand_items,
            "cand_category_id": cand_cats,
            "cand_features": cand_features,
            "cand_emb": cand_embs,
            "cand_has_emb": cand_has_emb,
            # История
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
    """Собирает батч лент с динамическим паддингом и булевыми масками.
    
    Маски:
    - candidate_mask: [B, M_max] — True для существующих кандидатов, False для паддингов.
    - cand_has_emb_mask: [B, M_max] — True если у кандидата есть валидный эмбеддинг (не -1 и не паддинг).
    - history_mask: [B, H_max] — True для реальных событий истории, False для паддингов.
    - hist_has_emb_mask: [B, H_max] — True если событие истории имеет валидный эмбеддинг.
    """
    if torch is None:
        raise RuntimeError("PyTorch не установлен!")

    b = len(batch)
    m_max = max(max(item["cand_len"] for item in batch), 1)
    h_max = max(max(item["hist_len"] for item in batch), 1)
    num_cand_feats = batch[0]["cand_features"].shape[1] if batch[0]["cand_features"].ndim == 2 else 0

    has_target = "target" in batch[0]
    has_pos = "pos" in batch[0]

    # Тензоры для кандидатов
    candidate_mask = torch.zeros((b, m_max), dtype=torch.bool)
    cand_has_emb_mask = torch.zeros((b, m_max), dtype=torch.bool)
    cand_emb = torch.zeros((b, m_max, EMB_DIM), dtype=torch.float32)
    cand_item_id = torch.zeros((b, m_max), dtype=torch.long)
    cand_category_id = torch.zeros((b, m_max), dtype=torch.long)
    cand_features = torch.zeros((b, m_max, num_cand_feats), dtype=torch.float32)
    target = torch.zeros((b, m_max), dtype=torch.float32) if has_target else None
    pos = torch.zeros((b, m_max), dtype=torch.long) if has_pos else None

    # Тензоры для истории
    history_mask = torch.zeros((b, h_max), dtype=torch.bool)
    hist_has_emb_mask = torch.zeros((b, h_max), dtype=torch.bool)
    hist_emb = torch.zeros((b, h_max, EMB_DIM), dtype=torch.float32)
    hist_item_id = torch.zeros((b, h_max), dtype=torch.long)
    hist_category_id = torch.zeros((b, h_max), dtype=torch.long)
    hist_event_type = torch.zeros((b, h_max), dtype=torch.long)
    hist_event_source = torch.zeros((b, h_max), dtype=torch.long)

    # Мета и пользователь
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

        # Заполнение кандидатов
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

        # Заполнение истории
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
        # Тензоры кандидатов
        "candidate_mask": candidate_mask,
        "cand_has_emb_mask": cand_has_emb_mask,
        "cand_emb": cand_emb,
        "cand_item_id": cand_item_id,
        "cand_category_id": cand_category_id,
        "cand_features": cand_features,
        # Тензоры истории
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
