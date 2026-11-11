"""Benchmark both readers on one real shard and verify exact cached values."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import polars as pl

from src.common.data import load_embeddings
from src.common.preparation import fingerprint, write_json
from src.common.resources import memory_snapshot
from src.common.similarity import (build_feature_part, feature_names, prefetch_vectors, required_vector_rows)


def benchmark(raw,prepared,cache,output):
    raw,prepared,cache,output=(Path(path) for path in (raw,prepared,cache,output))
    source=json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    meta=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    entry=source['splits']['train']['parts'][0]
    cached=meta['splits']['train']['parts'][entry['source']]
    path=raw/'train/impressions'/entry['source']
    if not meta['complete'] or cached['source']!=fingerprint(path):
        raise ValueError('cached reference and raw shard must match')
    frame=pl.read_parquet(path,columns=['feed_id','item_id','item_id__embedding_idx','history__item_id__embedding_idx','user_embedding'])
    start=time.monotonic(); chunks=[]
    for i in range(0,frame.height,128):
        vectors=load_embeddings(raw,'train')
        try:
            chunks.append(build_feature_part(frame.slice(i,128),'train',vectors))
        finally:
            vectors._mmap.close()
    mapped=pl.concat(chunks)
    mapped_seconds=time.monotonic()-start
    print(f'Mapped reader: {mapped_seconds:.2f}s',flush=True)
    start=time.monotonic()
    rows=required_vector_rows(frame)
    vectors=prefetch_vectors(raw,'train',rows)
    sequential=build_feature_part(frame,'train',vectors,vector_rows=rows)
    sequential_seconds=time.monotonic()-start
    reference=pl.read_parquet(cache/cached['path'])
    for candidate in (mapped,sequential):
        if not reference.select('split','feed_id','candidate_index','item_id').equals(candidate.select('split','feed_id','candidate_index','item_id')):
            raise ValueError('reader candidate keys differ from cached reference')
        for name in feature_names():
            if not np.array_equal(reference[name].to_numpy().view(np.uint32),candidate[name].to_numpy().view(np.uint32)):
                raise ValueError(f'reader changed Float32 bits: {name}')
    report={'complete':True,'source':entry['source'],'feeds':frame.height,'candidates':mapped.height,
            'mapped_seconds':mapped_seconds,'sequential_seconds':sequential_seconds,
            'observed_speed_ratio':mapped_seconds/sequential_seconds,
            'retained_vector_rows':len(rows),'retained_vector_bytes':vectors.nbytes,
            'float32_bits_match_mapped_and_cache':True,'memory':memory_snapshot(),
            'scope':'one shard, mapped then sequential, warm filesystem-cache effects may influence timings',
            'implementation_sha256':hashlib.sha256(Path('src/common/similarity.py').read_bytes()).hexdigest()}
    write_json(output,report)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,default=Path('competition'))
    parser.add_argument('--prepared',type=Path,default=Path('data/prepared'))
    parser.add_argument('--cache',type=Path,default=Path('data/features/cosine'))
    parser.add_argument('--out',type=Path,default=Path('runs/cosine-reader-benchmark.json'))
    args=parser.parse_args(); benchmark(args.raw,args.prepared,args.cache,args.out)
