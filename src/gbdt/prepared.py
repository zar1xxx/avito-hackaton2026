"""Чтение подготовленных данных Димы (data/prepared + кеши признаков) для раннера GBDT.

Вместо сборки признаков из сырых лент (src/common/data.py — медленно и весь split в памяти)
берём готовые таблицы кандидатов и группы признаков:
    data/prepared/<split>/candidates/part-*.parquet   (scripts.reproduce_data / prepare_data)
    data/features/cosine, data/features/infeed         (build_similarity_features / build_infeed_features)
Ключ строки — (feed_id, candidate_index); группы присоединяются строго 1:1 (attach_features Димы).

Train огромен (512 шардов, ~247 млн кандидатов), поэтому берём --train-parts N шардов,
равномерно по времени (как run_baseline_check). Кеш группы признаков должен покрывать эти шарды.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

BASE = ["f1", "f2", "f3", "f4", "f5", "category_id"]


def _manifest(root: Path) -> dict:
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not m.get("complete"):
        raise ValueError(f"{root}: подготовка данных не завершена (manifest complete=false)")
    return m


def select_train_parts(prepared: Path, n_parts: int) -> list[dict]:
    parts = _manifest(prepared)["splits"]["train"]["parts"]
    if not 1 <= n_parts <= len(parts):
        raise ValueError(f"--train-parts должен быть от 1 до {len(parts)}")
    return [parts[int(i)] for i in np.linspace(0, len(parts) - 1, n_parts, dtype=int)]


def load_prepared(prepared: str | Path, split: str, feature_roots: list[str | Path], train_parts: int = 1,
                  ) -> tuple[pl.DataFrame, list[str]]:
    """Таблица кандидатов split с признаками групп. Возвращает (frame, имена признаков модели)."""
    from src.common.similarity import attach_features  # код Димы

    prepared = Path(prepared)
    man = _manifest(prepared)
    parts = select_train_parts(prepared, train_parts) if split == "train" else man["splits"][split]["parts"]
    schema = pl.read_parquet_schema(prepared / parts[0]["candidate_path"])
    cols = ["feed_id", "timestamp", "candidate_index", "item_id", *BASE]
    cols += [c for c in ("target", "pos") if c in schema]
    df = pl.concat([pl.read_parquet(prepared / p["candidate_path"], columns=cols) for p in parts])
    df = df.sort("feed_id", "candidate_index")
    feats = list(BASE)
    sources = [p["source"] for p in parts]
    for root in feature_roots:
        df, names = attach_features(df, root, split, sources)
        clash = set(feats) & set(names)
        if clash:
            raise ValueError(f"повтор имён признаков между группами: {sorted(clash)}")
        feats += names
    return df.sort("feed_id", "candidate_index"), feats
