"""Подключение внешних признаков (Антон: история, Август: нейро-OOF/эмбеддинги).

КОНТРАКТ на файл признаков (по одному файлу на группу и на сплит):
    <dir>/<group>/<split>.parquet,  split in {train, val, test}
    колонки: feed_id:int32, item_id:int32 + любые числовые колонки.
    Имена колонок начинаются с "x_<group>_" (чтобы не пересекаться с базовыми признаками).
    Строк ровно столько же, сколько кандидатов в сплите, без дублей (feed_id, item_id).

Анти-утечка (главное правило): признак, посчитанный по target (CTR, нейро-скор), для train и val
должен быть OOF — посчитан моделью/статистикой, которая НЕ видела эту ленту. Для test —
на полной модели/статистике. Нельзя считать статистики по val/test target и нельзя брать
историю пользователя «из будущего» без согласования с организаторами (см. раздел «Риски» плана).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl


def attach(df: pl.DataFrame, feature_dir: str | Path | None, groups: list[str], split: str
           ) -> tuple[pl.DataFrame, list[str]]:
    if not groups:
        return df, []
    new_cols: list[str] = []
    for g in groups:
        path = Path(feature_dir) / g / f"{split}.parquet"
        ext = pl.read_parquet(path)
        keys = ["feed_id", "item_id"]
        cols = [c for c in ext.columns if c not in keys]
        bad = [c for c in cols if not c.startswith(f"x_{g}_")]
        if bad:
            raise ValueError(f"{path}: колонки {bad} должны начинаться с 'x_{g}_'")
        if ext.select(keys).is_duplicated().any():
            raise ValueError(f"{path}: дубли (feed_id, item_id)")
        ext = ext.cast({"feed_id": pl.Int32, "item_id": pl.Int32})
        before = df.height
        df = df.join(ext, on=keys, how="left")
        if df.height != before:
            raise ValueError(f"{path}: join изменил число строк ({before} -> {df.height})")
        new_cols += cols
    return df, new_cols
