"""Парное сравнение двух запусков по val_pred.parquet: разница NDCG@30 по лентам и бутстреп-интервал.

Отвечает на вопрос «прирост больше шума?» — обе модели оцениваются на одних и тех же лентах,
поэтому интервал заметно уже, чем у сравнения двух независимых средних.

    uv run --locked python -m scripts.paired_compare --base runs/cb8 --cand runs/cb8-hist
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.common.cv import pseudo_split
from src.common.metric import per_feed_ndcg


def val_timestamps(prepared: Path) -> pl.DataFrame:
    manifest = json.loads((prepared / 'manifest.json').read_text(encoding='utf-8'))
    parts = manifest['splits']['val']['parts']
    return pl.concat([pl.read_parquet(prepared / p['feed_path'], columns=['feed_id', 'timestamp']) for p in parts])


def per_feed(run: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pl.read_parquet(run / 'val_pred.parquet')
    return per_feed_ndcg(df['feed_id'].to_numpy(), df['target'].to_numpy(), df['score'].to_numpy(), cutoff=30)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--cand', type=Path, required=True)
    ap.add_argument('--prepared', type=Path, default=Path('data/prepared'))
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    fb, vb = per_feed(a.base)
    fc, vc = per_feed(a.cand)
    if not np.array_equal(fb, fc):
        raise SystemExit('runs are evaluated on different val feeds')
    ts = val_timestamps(a.prepared)
    pub, priv = pseudo_split(ts['feed_id'].to_numpy(), ts['timestamp'].to_numpy())
    rng = np.random.default_rng(a.seed)
    out = {'base': str(a.base), 'cand': str(a.cand)}
    for name, subset in (('pub', pub), ('priv', priv), ('all', fb)):
        mask = np.isin(fb, subset)
        d = vc[mask] - vb[mask]
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.n_boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[name] = {'feeds': int(mask.sum()), 'base': round(float(vb[mask].mean()), 5),
                     'cand': round(float(vc[mask].mean()), 5), 'diff': round(float(d.mean()), 5),
                     'ci95': [round(float(lo), 5), round(float(hi), 5)],
                     'p_cand_better': round(float((boot > 0).mean()), 3)}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
