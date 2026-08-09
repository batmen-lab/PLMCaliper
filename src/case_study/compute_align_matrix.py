import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tmtools import tm_align
from tmtools.io import get_structure, get_residue_data

import config

PDBSTYLE = config.PDBSTYLE_DIR
DEFAULT_OUTDIR = config.PROJECT_ROOT / "results" / "case_study" / "align_matrix"

INDEX_COLUMNS = ["query", "target", "qlen", "tlen", "s_npy", "path_npy", "lddt_npy", "seqxA", "seqyA"]


def find_ent(sid: str) -> Path:
    hashed = PDBSTYLE / sid[2:4] / f"{sid}.ent"
    if hashed.exists():
        return hashed
    hits = list(PDBSTYLE.rglob(f"{sid}.ent"))
    if not hits:
        sys.exit(f"Could not find structure file for '{sid}' under {PDBSTYLE}")
    return hits[0]


def load(sid: str):
    chain = next(get_structure(str(find_ent(sid))).get_chains())
    coords, seq = get_residue_data(chain)
    return coords, seq


def aligned_path(seqxA: str, seqyA: str) -> np.ndarray:
    qi = ti = 0
    pairs = []
    for a, b in zip(seqxA, seqyA):
        if a != "-" and b != "-":
            pairs.append((qi, ti))
        if a != "-":
            qi += 1
        if b != "-":
            ti += 1
    if not pairs:
        return np.zeros((0, 2), dtype=np.int32)
    return np.array(pairs, dtype=np.int32)


def tmscore_d0(length: int) -> float:
    return max(0.5, 1.24 * (length - 15) ** (1.0 / 3.0) - 1.8)


def similarity_matrix(q_super: np.ndarray, t: np.ndarray, d0: float) -> np.ndarray:
    diff = q_super[:, None, :] - t[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))          # L_q x L_t Cα-Cα distances
    return 1.0 / (1.0 + (d / d0) ** 2)         # TM-score-style, in (0, 1]


def lddt_matrix(
    cq: np.ndarray,
    ct: np.ndarray,
    path: np.ndarray,
    R_cutoff: float = 15.0,
    thresholds: tuple = (0.5, 1.0, 2.0, 4.0),
) -> np.ndarray:
    S = np.full((len(cq), len(ct)), np.nan)
    if len(path) == 0:
        return S

    qi, ti = path[:, 0], path[:, 1]
    cq_aln = cq[qi]
    ct_aln = ct[ti]

    dq = np.sqrt(((cq_aln[:, None] - cq_aln[None]) ** 2).sum(-1))
    dt = np.sqrt(((ct_aln[:, None] - ct_aln[None]) ** 2).sum(-1))

    n = len(qi)
    in_sphere = (dt < R_cutoff)
    in_sphere[np.arange(n), np.arange(n)] = False

    diff = np.abs(dq - dt)

    for k in range(n):
        mask = in_sphere[k]
        if not mask.any():
            continue
        S[qi[k], ti[k]] = np.mean([(diff[k, mask] < thr).mean() for thr in thresholds])

    return S


def compute_one(query: str, target: str, outdir: Path) -> dict:
    cq, sq = load(query)
    ct, st = load(target)
    res = tm_align(cq, ct, sq, st)

    q_super = cq @ res.u.T + res.t
    d0 = tmscore_d0(len(cq))
    S = similarity_matrix(q_super, ct, d0)
    path = aligned_path(res.seqxA, res.seqyA)
    S_lddt = lddt_matrix(cq, ct, path)

    pair = f"{query}-{target}"
    s_npy = outdir / f"{pair}.S.npy"
    path_npy = outdir / f"{pair}.path.npy"
    lddt_npy = outdir / f"{pair}.lddt.npy"
    np.save(s_npy, S.astype(np.float32, copy=False))
    np.save(path_npy, path)
    np.save(lddt_npy, S_lddt.astype(np.float32, copy=False))

    return {
        "query": query, "target": target, "qlen": len(cq), "tlen": len(ct),
        "s_npy": s_npy.name, "path_npy": path_npy.name, "lddt_npy": lddt_npy.name,
        "seqxA": res.seqxA, "seqyA": res.seqyA,
    }


def run(pairs, outdir: Path = DEFAULT_OUTDIR) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for query, target in pairs:
        print(f"\n=== TM-align compute: {query} -> {target} ===")
        row = compute_one(query, target, outdir)
        rows.append(row)
        print(f"[OK] {query}-{target}: S {row['qlen']}x{row['tlen']}")

    index_path = outdir / "index.csv"
    pd.DataFrame(rows, columns=INDEX_COLUMNS).to_csv(index_path, index=False)
    print(f"\n[OK] wrote {len(rows)} pair(s) -> {outdir}")
    print(f"[OK] index -> {index_path}")
    print("     Next: python src/case_study/plot_align_matrix.py")


if __name__ == "__main__":
    # ---- edit this list of (query, target) pairs ----
    PAIRS = [
        ("d3tg7a1", "d3ku3a1"),
        ("d5awwy_", "d2a65a1"),
    ]
    run(PAIRS)
