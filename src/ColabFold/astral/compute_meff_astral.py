import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
METHODS = ["dhr_postprocess", "plm", "tmvec"]
SEQID = 0.8


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
    seqs = [s for s in seqs if len(s) == L]            # keep consistently-aligned rows
    return np.array([[ord(c) for c in s] for s in seqs], dtype=np.int16)


def meff(path: Path) -> float:
    M = a3m_matrix(path)
    if M is None:
        return 0.0
    n = M.shape[0]
    if n == 1:
        return 1.0
    L = M.shape[1]
    # pairwise fractional identity (N x N), then 80%-identity reweighting
    ident = (M[:, None, :] == M[None, :, :]).sum(axis=2) / L
    nsim = (ident >= SEQID).sum(axis=1)                # includes self
    return float((1.0 / nsim).sum())


def main():
    for mdir in METHODS:
        shards = sorted(glob.glob(str(ROOT / f"results/ColabFold/astral_db/{mdir}/iter1/parallel/out_gpu*.tsv")))
        for f in shards:
            f = Path(f)
            runs = f.parent / "runs"
            d = pd.read_csv(f, sep="\t")
            d["meff"] = [round(meff(runs / str(q) / str(t) / "msa.a3m"), 3)
                         for q, t in zip(d["query_id"], d["tag"])]
            d.to_csv(f, sep="\t", index=False)
        # report
        dd = pd.concat([pd.read_csv(x, sep="\t") for x in shards], ignore_index=True)
        fdr = dd[dd["efdr_limit"].notna()]
        print(f"[{mdir}] meff: median(FDR)={fdr['meff'].median():.2f} "
              f"max={dd['meff'].max():.1f}  vanilla median={dd[dd.tag=='vanilla']['meff'].median():.2f}")
    print("[DONE] meff column added to all astral_db shards")


if __name__ == "__main__":
    main()
