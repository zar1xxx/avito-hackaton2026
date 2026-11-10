"""Проверка сабмита перед отправкой: те же пары, что в шаблоне, без дублей, NaN и inf, нужные типы.

    python scripts/validate_submission.py competition/sample_submission.parquet runs/x/submission.parquet
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl


def validate(template_path: str, sub_path: str) -> list[str]:
    errors: list[str] = []
    tpl = pl.read_parquet(template_path)
    sub = pl.read_parquet(sub_path)
    if sub.columns != ["feed_id", "item_id", "score"]:
        errors.append(f"колонки {sub.columns}, нужно ['feed_id','item_id','score']")
        return errors
    want = {"feed_id": pl.Int32, "item_id": pl.Int32, "score": pl.Float32}
    for c, t in want.items():
        if sub.schema[c] != t:
            errors.append(f"тип {c}: {sub.schema[c]}, нужен {t}")
    if sub.height != tpl.height:
        errors.append(f"строк {sub.height}, в шаблоне {tpl.height}")
    if sub.select("feed_id", "item_id").is_duplicated().any():
        errors.append("есть дубликаты (feed_id, item_id)")
    s = sub["score"].to_numpy()
    if not np.isfinite(s).all():
        errors.append("есть NaN или inf в score")
    miss = tpl.select("feed_id", "item_id").join(sub.select("feed_id", "item_id"), on=["feed_id", "item_id"], how="anti")
    if miss.height:
        errors.append(f"не хватает {miss.height} пар из шаблона")
    extra = sub.select("feed_id", "item_id").join(tpl.select("feed_id", "item_id"), on=["feed_id", "item_id"], how="anti")
    if extra.height:
        errors.append(f"лишних пар: {extra.height}")
    return errors


if __name__ == "__main__":
    errs = validate(sys.argv[1], sys.argv[2])
    print("OK" if not errs else "\n".join(errs))
    sys.exit(1 if errs else 0)
