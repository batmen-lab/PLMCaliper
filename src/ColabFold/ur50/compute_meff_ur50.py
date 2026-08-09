#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
_AP = argparse.ArgumentParser()
_AP.add_argument("--method", default="dhr_postprocess")
_AP.add_argument("--metrics-name", default="msa_metrics.tsv",
                 help="the per-iter table to add meff to (msa_metrics.tsv or metrics.tsv)")
_ARGS, _ = _AP.parse_known_args()
BASE = ROOT / f"results/ColabFold/ur50_db/{_ARGS.method}/iter1"
RUNS = BASE / "runs"
SRC = BASE / _ARGS.metrics_name
SEQID, SAMPLE = 0.8, 3000
RNG = np.random.default_rng(0)


def a3m_matrix(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    seqs, buf, in_seq = [], [], False
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if in_seq:
                seqs.append("".join(buf))
            buf, in_seq = [], True
        else:
            buf.append("".join(c for c in line.strip() if c == "-" or c.isupper()))
    if in_seq:
        seqs.append("".join(buf))
    seqs = [s for s in seqs if s]
    if not seqs:
        return None
    L = len(seqs[0])
    seqs = [s for s in seqs if len(s) == L]
    return np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), L)


def meff(path: Path) -> float:
    M = a3m_matrix(path)
    if M is None:
        return 0.0
    N, L = M.shape
    if N == 1:
        return 1.0
    m = N if N <= SAMPLE else SAMPLE
    idx = np.arange(N) if N <= SAMPLE else RNG.choice(N, m, replace=False)
    Ms = M[idx]
    thr = SEQID * L
    ic = np.zeros((m, N), dtype=np.float32)
    for s in np.unique(M):
        ic += (Ms == s).astype(np.float32) @ (M == s).astype(np.float32).T
    nsim = (ic >= thr).sum(axis=1)                 # includes self
    return float(N * np.mean(1.0 / nsim))


def main():
    d = pd.read_csv(SRC, sep="\t")
    vals = []
    for k, (q, t) in enumerate(zip(d["query_id"], d["tag"]), 1):
        vals.append(round(meff(RUNS / str(q) / str(t) / "msa.a3m"), 2))
        if k % 100 == 0:
            print(f"  {k}/{len(d)} done", flush=True)
    d["meff"] = vals
    d.to_csv(SRC, sep="\t", index=False)
    fdr = d[d["efdr_limit"].notna()]
    print(f"[DONE] meff added to {SRC}")
    print("by tag median meff:")
    print(d.groupby("tag")["meff"].median().round(1).to_string())


if __name__ == "__main__":
    main()
