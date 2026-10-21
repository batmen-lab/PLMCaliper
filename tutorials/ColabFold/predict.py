import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASETS = ("astral", "ur50")
# the run tree is derived data; only the figures go to results/
RESULTS_DIR = {"astral": PROJECT_ROOT / "data" / "ColabFold" / "astral_db",
               "ur50": PROJECT_ROOT / "data" / "ColabFold" / "ur50_exp"}

# ColabFold prediction settings (Methods)
COLABFOLD_NUM_MODELS = 1
COLABFOLD_NUM_RECYCLE = 3
# --------------------------------------------------------------------------
# helpers: run ColabFold, read back its pLDDT and model
# --------------------------------------------------------------------------
def resolve_colabfold() -> str | None:
    p = os.environ.get("COLABFOLD_BATCH")
    if p and Path(p).exists():
        return p
    r = subprocess.run(["bash", "-lc", "command -v colabfold_batch"],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def run_colabfold(msa_in: Path, out_dir: Path, single: bool) -> tuple[str, float]:
    cf = resolve_colabfold()
    if not cf:
        return "colabfold_missing", float("nan")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [cf, "--num-models", str(COLABFOLD_NUM_MODELS),
           "--num-recycle", str(COLABFOLD_NUM_RECYCLE), str(msa_in), str(out_dir)]
    if single:
        cmd += ["--msa-mode", "single_sequence"]
    t = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t
    (out_dir / "colabfold.log").write_text(r.stdout + r.stderr)
    return ("ok" if r.returncode == 0 else "failed"), dt


def extract_plddt(out_dir: Path) -> float | None:
    for pat in ("**/*scores_rank_001*.json", "**/*scores_rank_*.json"):
        hits = sorted(Path(out_dir).glob(pat))
        if hits:
            pl = json.loads(hits[0].read_text()).get("plddt")
            return float(np.mean(pl)) if pl else None
    return None


def find_pred_pdb(out_dir: Path) -> Path | None:
    for pat in ("**/*_unrelaxed_rank_001*.pdb", "**/*rank_001*.pdb", "**/*.pdb"):
        hits = sorted(Path(out_dir).glob(pat))
        if hits:
            return hits[0]
    return None


def write_metrics(rows: list[dict], out: Path, columns: list[str]) -> None:
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    ordered = columns + [c for c in df.columns if c not in columns]
    out.parent.mkdir(parents=True, exist_ok=True)
    df[ordered].to_csv(out, sep="\t", index=False)


def read_query_ids(path: Path) -> list[str]:
    ids, seen = [], set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        qid = line.split("\t")[0].strip()
        if qid and qid not in seen:
            ids.append(qid)
            seen.add(qid)
    return ids


PRED_COLUMNS = ["method", "query_id", "tag", "single_seq", "plddt",
                "predict_seconds", "pred_pdb", "af2_status"]


def fold_one(row, runs_dir: Path, force: bool) -> dict:
    qid, tag = str(row.query_id), str(row.tag)
    d = runs_dir / qid / tag
    a3m, qfa = d / "msa.a3m", d / "query.fa"
    out_dir = d / "af2_out"

    depth = int(row.msa_depth) if row.msa_depth == row.msa_depth else 0
    single = depth <= 1                       # nothing to align: fold the bare query
    status = str(row.msa_status)

    done = find_pred_pdb(out_dir)
    if done is not None and not force:
        return dict(method=row.method, query_id=qid, tag=tag, single_seq=single,
                    plddt=extract_plddt(out_dir), predict_seconds=float("nan"),
                    pred_pdb=str(done), af2_status="reused")

    if "missing" in status or not qfa.exists():
        return dict(method=row.method, query_id=qid, tag=tag, single_seq=single,
                    plddt=None, predict_seconds=float("nan"), pred_pdb=None,
                    af2_status="skip_no_msa")

    t0 = time.perf_counter()
    af2, predict_s = run_colabfold(qfa if single else a3m, out_dir, single)
    if predict_s != predict_s:
        predict_s = time.perf_counter() - t0
    plddt = extract_plddt(out_dir) if af2 == "ok" else None
    pred = find_pred_pdb(out_dir)
    return dict(method=row.method, query_id=qid, tag=tag, single_seq=single,
                plddt=plddt, predict_seconds=round(predict_s, 2),
                pred_pdb=str(pred) if pred else None, af2_status=af2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 4: fold every MSA built in step 3 with ColabFold.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--method", default="dhr_postprocess")
    p.add_argument("--msa-table", type=Path, default=None, help="default: step 3's")
    p.add_argument("--query-list", type=Path, default=None)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--jack-iters", type=int, default=1)
    p.add_argument("--force-refold", action="store_true")
    p.add_argument("--runs-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    base = RESULTS_DIR[a.dataset] / a.method / f"iter{a.jack_iters}"
    msa_table = a.msa_table or base / "msa.tsv"
    out = a.out or base / "pred.tsv"
    runs_dir = a.runs_dir or base / "runs"
    if not msa_table.exists():
        raise FileNotFoundError(f"{msa_table} -- run build_msa.py first")

    df = pd.read_csv(msa_table, sep="\t", dtype={"query_id": str, "tag": str})
    if a.query_list:
        keep = set(read_query_ids(a.query_list))
        df = df[df["query_id"].isin(keep)]
    if a.max_queries:
        keep = list(dict.fromkeys(df["query_id"]))[: a.max_queries]
        df = df[df["query_id"].isin(keep)]

    print(f"[step4] folding {len(df)} (query, condition) pairs", flush=True)
    rows = []
    for row in df.itertuples():
        r = fold_one(row, runs_dir, a.force_refold)
        rows.append(r)
        print(f"  [{r['query_id']}] {r['tag']}: plddt={r['plddt']} {r['af2_status']}",
              flush=True)
    write_metrics(rows, out, PRED_COLUMNS)
    print(f"[step4] -> {out}")


if __name__ == "__main__":
    main()
