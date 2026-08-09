#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")               # headless; must precede the plot-module imports
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent            # src/case_study
_CORE = _HERE.parent / "core"                      # src/core
for _p in (str(_HERE), str(_CORE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config                                        # noqa: E402  (src/case_study/config.py)

FIGS_ROOT = config.DATA_DIR / "plot_data" / "case_study" / "figs"
# reuse the matrices already computed under data/plot_data (esp. the GPU-expensive DeepBLAST ones)
ALNMAT_CACHE = config.DATA_DIR / "plot_data" / "case_study" / "align_matrix"
DEEPBLAST_CACHE = config.DATA_DIR / "plot_data" / "case_study" / "deepblast_heatmaps"


def _method_dir(method: str) -> Path:
    d = FIGS_ROOT / method
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# 1) score distribution  (needs the search / calibrated / target-noisy files)
# --------------------------------------------------------------------------- #
def make_score_dist(method, pairs, *, decoy_method, query_name, target_name, rep_id,
                    data_dir, out_dir):
    import lookup_pair_scores as lps           # src/core/lookup_pair_scores.py
    df = lps.lookup_pair_scores(
        pairs, search_method=method, query_name=query_name, target_name=target_name,
        decoy_method=decoy_method, rep_id=rep_id, data_dir=data_dir)
    lps.plot_score_distributions(
        df, search_method=method, query_name=query_name, target_name=target_name,
        data_dir=data_dir, plot_dir=out_dir)


# --------------------------------------------------------------------------- #
# 2) TM-align: alignment matrix (alnmat) AND sequence alignment (seqaln)
#    Both come from one TM-align compute; needs pdbstyle structures + tmtools.
# --------------------------------------------------------------------------- #
_ALIGN_COLS = ["query", "target", "qlen", "tlen", "s_npy", "path_npy", "lddt_npy",
               "seqxA", "seqyA"]


def _load_align_index(cache_dir):
    p = cache_dir / "index.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {(str(r.query), str(r.target)): r._asdict() for r in df.itertuples(index=False)}


def _upsert_align_index(cache_dir, row):
    p = cache_dir / "index.csv"
    df = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=_ALIGN_COLS)
    df = df[~((df["query"] == row["query"]) & (df["target"] == row["target"]))]
    df = pd.concat([df, pd.DataFrame([row], columns=_ALIGN_COLS)], ignore_index=True)
    df.to_csv(p, index=False)


def make_align(pairs, *, out_dir, which, cache_dir=ALNMAT_CACHE):
    cache_dir.mkdir(parents=True, exist_ok=True)
    import plot_align_matrix as pam            # matplotlib only
    idx = _load_align_index(cache_dir)
    cam = None                                 # lazy (needs tmtools) only when a compute is due
    for q, t in pairs:
        s_npy = cache_dir / f"{q}-{t}.S.npy"
        p_npy = cache_dir / f"{q}-{t}.path.npy"
        l_npy = cache_dir / f"{q}-{t}.lddt.npy"
        row = idx.get((q, t))
        cached = (row is not None and s_npy.exists() and p_npy.exists() and l_npy.exists()
                  and isinstance(row.get("seqxA"), str) and isinstance(row.get("seqyA"), str))
        if not cached:
            if cam is None:
                import compute_align_matrix as cam  # noqa: F811
            row = cam.compute_one(q, t, cache_dir)   # writes npys, returns seqxA/seqyA
            _upsert_align_index(cache_dir, row)
        if "alnmat" in which:
            pam.plot_alnmat(np.load(s_npy), np.load(p_npy), q, t, out_dir)
        if "seqaln" in which:
            pam.plot_seq_alignment(q, t, str(row["seqxA"]), str(row["seqyA"]),
                                   np.load(l_npy), out_dir)


# --------------------------------------------------------------------------- #
# 3) DeepBLAST residue-alignment heatmap  (needs the DeepBLAST/ProtT5 model, GPU)
# --------------------------------------------------------------------------- #
def make_deepblast(pairs, *, out_dir, fasta=None, cache_dir=DEEPBLAST_CACHE):
    cache_dir.mkdir(parents=True, exist_ok=True)
    import plot_deepblast_heatmap as pdh       # matplotlib only
    idx_path = cache_dir / "index.csv"

    # cache hit = a row in a prior index whose heat .npy is still on disk
    have = {}
    if idx_path.exists():
        for r in pd.read_csv(idx_path).itertuples(index=False):
            if (cache_dir / str(r.heat_npy)).exists():
                have[(str(r.query), str(r.target))] = r
    todo = [(q, t) for (q, t) in pairs if (q, t) not in have]

    rows = {k: have[k] for k in pairs if k in have}
    if todo:                                    # import (and load) the model ONLY for misses
        import compute_deepblast_heatmap as cdh
        kw = {"outdir": cache_dir}
        if fasta is not None:
            kw["fasta"] = fasta
        cdh.run(todo, **kw)                     # overwrites index.csv with just `todo`
        for r in pd.read_csv(idx_path).itertuples(index=False):
            rows[(str(r.query), str(r.target))] = r

    for (q, t) in pairs:
        r = rows.get((q, t))
        if r is None:
            print(f"[deepblast][WARN] no matrix for {q}-{t}, skipped")
            continue
        heat = np.load(cache_dir / str(r.heat_npy))
        pp = None
        if isinstance(r.path_npy, str) and r.path_npy:
            pp = np.load(cache_dir / str(r.path_npy))
        # draw_path=False matches plot_deepblast_heatmap.run()'s default (the original figures)
        pdh.plot_heatmap(heat, pp, q, t, out_dir, str(r.stem), draw_path=False)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_all(pairs_by_method, *, decoy_method, query_name="astral", target_name="astral",
            rep_id=0, data_dir=None, fasta=None,
            which=("score_dist", "alnmat", "seqaln", "deepblast")):
    data_dir = Path(data_dir) if data_dir is not None else config.DATA_DIR
    for method, pairs in pairs_by_method.items():
        pairs = [tuple(p) for p in pairs]
        if not pairs:
            continue
        out_dir = _method_dir(method)
        print(f"\n########## {method}: {len(pairs)} pair(s) -> {out_dir} ##########")
        if "score_dist" in which:
            try:
                make_score_dist(method, pairs, decoy_method=decoy_method,
                                query_name=query_name, target_name=target_name,
                                rep_id=rep_id, data_dir=data_dir, out_dir=out_dir)
            except Exception as e:
                print(f"[score_dist][SKIP] {method}: {type(e).__name__}: {e}")
        align_which = tuple(w for w in ("alnmat", "seqaln") if w in which)
        if align_which:
            try:
                make_align(pairs, out_dir=out_dir, which=align_which)
            except Exception as e:
                print(f"[align][SKIP] {method}: {type(e).__name__}: {e}")
        if "deepblast" in which:
            try:
                make_deepblast(pairs, out_dir=out_dir, fasta=fasta)
            except Exception as e:
                print(f"[deepblast][SKIP] {method}: {type(e).__name__}: {e}")
        pdfs = sorted(p.name for p in out_dir.glob("*.pdf"))
        print(f"[{method}] {len(pdfs)} PDF(s) in {out_dir}")


if __name__ == "__main__":
    # ============================================================= #
    #  EDIT HERE: put each (qid, tid) pair under its search method.  #
    #  The pair's 3 figures are saved into figs/<method>/.           #
    # ============================================================= #
    PAIRS_BY_METHOD = {
        "dhr_postprocess": [
            ("d1a12a_", "d6jx5a_"),
            ("d12asa_", "d1f7ua2"),
            ("d16vpa_", "d1ho8a_"),
            ("d1ry9a_", "d1jyoa_"),
            ("d4k8ga1", "d1r0ma2"),
            ("d6rh5a1", "d1foea2"),
        ],
        "plm": [
            ("d1a8qa_", "d4yv7a_"),
            ("d1brta_", "d5dkva1"),
            ("d1a7sa_", "d3su6a1"),
            ("d1aqea_", "d1ofwa_"),
            ("d1b77a1", "d1t6la2"),
        ],
        "tmvec": [
            ("d1cvra2", "d1t1ga_"),
            ("d16vpa_", "d1no7a_"),
            ("d16vpa_", "d5id6a5"),
            ("d1g8fa3", "d1viaa_"),
            ("d1i7na1", "d2w70a1"),
            ("d1ywfa1", "d6uzta2"),
        ],
    }

    # shared knobs (rarely need changing)
    DECOY_METHOD = "extended_mkv2"     # only affects score_dist (calibrated-decoy file name)
    QUERY_NAME = "astral"
    TARGET_NAME = "astral"
    REP_ID = 0
    DATA_DIR = os.environ.get("CASE_DATA_DIR") or None   # None -> config.DATA_DIR (repo data/)
    FASTA = None                       # None -> config.fasta_path("astral") for DeepBLAST
    # figure kinds to make; override per run with e.g. CASE_WHICH=deepblast (comma-separated).
    # NOTE envs: alnmat/seqaln need `tmtools` (dplm env); deepblast needs the DeepBLAST/ProtT5
    # backend (tmvec env). No single env has both, so run the tool once per env with CASE_WHICH.
    WHICH = tuple(os.environ.get(
        "CASE_WHICH", "score_dist,alnmat,seqaln,deepblast").split(","))

    run_all(PAIRS_BY_METHOD, decoy_method=DECOY_METHOD, query_name=QUERY_NAME,
            target_name=TARGET_NAME, rep_id=REP_ID, data_dir=DATA_DIR, fasta=FASTA,
            which=WHICH)
