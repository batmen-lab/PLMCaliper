import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASETS = ("astral", "ur50")
# the run tree is derived data; only the figures go to results/
RESULTS_DIR = {"astral": PROJECT_ROOT / "data" / "ColabFold" / "astral_db",
               "ur50": PROJECT_ROOT / "data" / "ColabFold" / "ur50_exp"}
DATA_DIR = PROJECT_ROOT / "data"
PLOT_DATA_DIR = {"astral": DATA_DIR / "ColabFold" / "plot_data" / "astral_db",
                 "ur50": DATA_DIR / "ColabFold" / "plot_data" / "ur50_db"}

LIBS = PROJECT_ROOT / "libs"
TMALIGN = LIBS / "dplm" / "analysis" / "TMalign"
NATIVE_PDB_DIR = LIBS / "dplm" / "gen.fasta" / "astral" / "folding" / "pdb"
PDBSTYLE_DIR = DATA_DIR / "pdbstyle-2.08"

METRIC_COLUMNS = [
    "method", "query_id", "tag", "efdr_limit", "qlen", "n_hits", "msa_depth",
    "meff", "plddt", "tm_score", "rmsd", "msa_seconds", "predict_seconds",
    "total_seconds", "msa_status", "af2_status", "has_native",
]


# --------------------------------------------------------------------------
# locate the experimental structure, run TM-align
# --------------------------------------------------------------------------
def native_pdb(qid: str) -> Path | None:
    p = NATIVE_PDB_DIR / f"{qid}.pdb"
    if p.exists():
        return p
    ent = PDBSTYLE_DIR / qid[2:4] / f"{qid}.ent"
    return ent if ent.exists() else None


def run_tmalign(model: Path, native: Path | None) -> dict:
    if native is None or not Path(native).exists() or not TMALIGN.exists():
        return {"tm_score": None, "rmsd": None}
    r = subprocess.run([str(TMALIGN), str(model), str(native)],
                       capture_output=True, text=True, check=False)
    tm = rmsd = None
    for line in (r.stdout + r.stderr).splitlines():
        if line.startswith("TM-score=") and "Chain_1" in line:
            m = re.search(r"TM-score=\s*([0-9.]+)", line)
            if m and tm is None:
                tm = float(m.group(1))
        if "RMSD=" in line:
            m = re.search(r"RMSD=\s*([0-9.]+)", line)
            if m:
                rmsd = float(m.group(1))
    return {"tm_score": tm, "rmsd": rmsd}


def evaluate(pred: pd.DataFrame) -> pd.DataFrame:
    """TM-score and r.m.s.d. per (query, condition); native looked up once per query."""
    natives = {q: native_pdb(q) for q in pred["query_id"].unique()}
    n_missing = sum(v is None for v in natives.values())
    if n_missing:
        print(f"[step5] {n_missing}/{len(natives)} queries have no native structure")

    rows = []
    for r in pred.itertuples():
        nat = natives.get(str(r.query_id))
        tm = rmsd = None
        if r.pred_pdb and isinstance(r.pred_pdb, str) and Path(r.pred_pdb).exists():
            m = run_tmalign(Path(r.pred_pdb), nat)
            tm, rmsd = m["tm_score"], m["rmsd"]
        rows.append({"query_id": str(r.query_id), "tag": str(r.tag),
                     "tm_score": tm, "rmsd": rmsd, "has_native": nat is not None})
    return pd.DataFrame(rows)


def merge_all(msa: pd.DataFrame, pred: pd.DataFrame, evald: pd.DataFrame) -> pd.DataFrame:
    df = (msa.merge(pred.drop(columns=["method"], errors="ignore"),
                    on=["query_id", "tag"], how="left")
             .merge(evald, on=["query_id", "tag"], how="left"))
    df["total_seconds"] = (df[["msa_seconds", "predict_seconds"]]
                           .apply(lambda r: float(np.nansum(r)), axis=1).round(2))
    for col in METRIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[METRIC_COLUMNS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 5: score each predicted structure against its experimental structure.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--method", default="dhr_postprocess")
    p.add_argument("--jack-iters", type=int, default=1)
    p.add_argument("--msa-table", type=Path, default=None)
    p.add_argument("--pred-table", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None, help="default: metrics.tsv")
    p.add_argument("--sync-plot-data", action="store_true",
                   help="also copy the table into data/plot_data for step 6")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    base = RESULTS_DIR[a.dataset] / a.method / f"iter{a.jack_iters}"
    msa_p = a.msa_table or base / "msa.tsv"
    pred_p = a.pred_table or base / "pred.tsv"
    out = a.out or base / "metrics.tsv"
    for p in (msa_p, pred_p):
        if not p.exists():
            raise FileNotFoundError(f"{p} -- run the earlier step first")

    msa = pd.read_csv(msa_p, sep="\t", dtype={"query_id": str, "tag": str})
    pred = pd.read_csv(pred_p, sep="\t", dtype={"query_id": str, "tag": str})
    print(f"[step5] {len(msa)} MSA rows, {len(pred)} prediction rows", flush=True)

    df = merge_all(msa, pred, evaluate(pred))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"[step5] -> {out}  ({len(df)} rows)")

    ok = df[df["af2_status"].isin(["ok", "reused"])]
    print(f"[step5] folded {len(ok)}/{len(df)}; "
          f"with native {int(df['has_native'].sum())}")

    if a.sync_plot_data:
        dst = PLOT_DATA_DIR[a.dataset] / f"metrics_{a.method}.tsv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dst, sep="\t", index=False)
        print(f"[step5] plot data -> {dst}")


if __name__ == "__main__":
    main()
