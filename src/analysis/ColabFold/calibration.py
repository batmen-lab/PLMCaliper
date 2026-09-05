import argparse
import importlib.util
import re
import subprocess
from io import StringIO
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

DATASETS = ("astral", "ur50")
RESULTS_DIR = {"astral": PROJECT_ROOT / "results" / "ColabFold" / "astral_db",
               "ur50": PROJECT_ROOT / "results" / "ur50_exp"}
# step 1 writes these; step 2 reads them. --real/--decoy override both.
RAW_REAL = {"astral": DATA_DIR / "result_{m}_astral_astral.txt",
            "ur50": DATA_DIR / "search_data" / "result_{m}_astral4f_ur50.txt"}
RAW_DECOY = {"astral": DATA_DIR / "_hf_decoy" / "raw" / "scores" / "astral_main"
                       / "result_{m}_astral_extended_mkv2_astral.txt",
             "ur50": DATA_DIR / "search_data" / "result_{m}_astral4f_extended_mkv2_ur50.txt"}
# an ASTRAL self-search returns the decoy queries too; UR50 targets carry no decoys
DROP_DECOY_HITS = {"astral": True, "ur50": False}
DECOY_SUFFIX = "_mkv"

EFDR_THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 10)]   # 0.1 .. 0.9


def efdr_tag(q: float) -> str:
    return f"efdr_{q:.2f}".replace(".", "p")

# --------------------------------------------------------------------------
# helpers: read one query's block, spot a decoy target
# --------------------------------------------------------------------------
def grep_rows(path: Path, prefix: str) -> pd.DataFrame:
    """Rows whose first column equals `prefix`, without reading the whole table."""
    proc = subprocess.run(["grep", "-E", f"^{re.escape(prefix)}\t", str(path)],
                          capture_output=True, text=True, check=False)
    if not proc.stdout.strip():
        return pd.DataFrame()
    with open(path) as fh:
        header = fh.readline().rstrip("\n")
    return pd.read_csv(StringIO(header + "\n" + proc.stdout), sep="\t")


def is_decoy_tid(tid) -> bool:
    t = str(tid)
    return any(s in t for s in (DECOY_SUFFIX, "_rev", "_mkv", "_dup"))


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


PLMCALIPER = PROJECT_ROOT / "src" / "PLMCaliper" / "PLMCaliper.py"


def load_plmcaliper():
    """Import the one PLM-Caliper calibration implementation used by src."""
    if not PLMCALIPER.exists():
        raise FileNotFoundError(f"{PLMCALIPER} is missing")
    spec = importlib.util.spec_from_file_location("PLMCaliper_core", PLMCALIPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Calibration settings of the published ColabFold runs.  scripts/run_calibration.sh
# passed no --tau, so those runs used the pipeline default 0.2 -- NOT the 0.25 the
# BAGEL and ASTRAL FDR figures use.  Keep 0.2 here to stay comparable with the
# existing metrics tables; pass --tau to override.
CALIB = dict(tau=0.2, tol=0.2, max_iter=2, noise_lambda=0.02, n_reps=1, seed=43,
             num_points=1000)


def cutoffs_path(dataset: str, method: str) -> Path:
    return RESULTS_DIR[dataset] / method / "cutoffs.tsv"


def hits_path(dataset: str, method: str) -> Path:
    return RESULTS_DIR[dataset] / method / "hits.tsv"


def run_plmcaliper(dataset: str, method: str, real: Path, decoy: Path, out: Path,
                   python_exe: str | None = None, *, tau: float = CALIB["tau"]) -> None:
    """Raw score tables -> per-(query, q) score cutoffs, via the released tool."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if python_exe:
        cmd = [python_exe, str(PLMCALIPER), "calibrate",
               "--real", str(real), "--decoy", str(decoy), "--out", str(out),
               "--decoy-suffix", DECOY_SUFFIX,
               "--target-fdr", *[f"{q:g}" for q in EFDR_THRESHOLDS],
               "--tau", str(tau), "--tol", str(CALIB["tol"]),
               "--max-iter", str(CALIB["max_iter"]),
               "--noise-lambda", str(CALIB["noise_lambda"]),
               "--n-reps", str(CALIB["n_reps"]), "--seed", str(CALIB["seed"]),
               "--num-points", str(CALIB["num_points"])]
        print("[step2] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        return

    print(f"[step2] importing PLM-Caliper core: {PLMCALIPER}", flush=True)
    pc = load_plmcaliper()
    pc.calibrate(
        str(real), str(decoy), str(out),
        decoy_suffix=DECOY_SUFFIX, q_levels=EFDR_THRESHOLDS,
        tau=tau, tol=CALIB["tol"],
        max_iter=CALIB["max_iter"], noise_lambda=CALIB["noise_lambda"],
        n_reps=CALIB["n_reps"], seed=CALIB["seed"],
        num_points=CALIB["num_points"],
    )


def select_hits(dataset: str, method: str, cutoffs: Path, real: Path,
                qids: list[str] | None) -> pd.DataFrame:
    """Apply each (query, q) cutoff to the real scores; drop self-hits and decoys."""
    cut = pd.read_csv(cutoffs, sep="\t")
    if "target_fdr" not in cut.columns and "q" in cut.columns:
        cut = cut.rename(columns={"q": "target_fdr"})
    cut = cut[cut["rep_id"] == 0]
    wanted = set(qids) if qids else set(cut["qid"].astype(str))

    rows = []
    for qid in sorted(wanted):
        df_q = grep_rows(real, qid)
        if df_q.empty:
            print(f"[step2] {qid}: no rows in {real.name}")
            continue
        if "rep_id" in df_q.columns:
            df_q = df_q[df_q["rep_id"] == 0]
        s = df_q[df_q["tid"].astype(str) != qid]
        if DROP_DECOY_HITS[dataset]:
            s = s[~s["tid"].map(is_decoy_tid)]
        s = s.sort_values("score", ascending=False)

        for r in cut[cut["qid"].astype(str) == qid].itertuples():
            if pd.isna(r.cutoff_score):        # q unreachable for this query
                continue
            target_fdr = float(r.target_fdr)
            keep = s[s["score"] >= r.cutoff_score]
            rows.append(pd.DataFrame({"qid": qid, "q": target_fdr, "tag": efdr_tag(target_fdr),
                                      "tid": keep["tid"].astype(str),
                                      "score": keep["score"]}))
    if not rows:
        return pd.DataFrame(columns=["qid", "q", "tag", "tid", "score"])
    return pd.concat(rows, ignore_index=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 2: PLM-Caliper calibration -> the hit set to build each MSA from.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--method", default="dhr_postprocess")
    p.add_argument("--real", type=Path, default=None,
                   help="raw real score table (default: the dataset's)")
    p.add_argument("--decoy", type=Path, default=None,
                   help="raw decoy score table (default: the dataset's)")
    p.add_argument("--query-list", type=Path, default=None)
    p.add_argument("--tau", type=float, default=CALIB["tau"],
                   help="pinball tau; 0.2 reproduces the published ColabFold runs")
    p.add_argument("--skip-calibration", action="store_true",
                   help="reuse an existing cutoffs.tsv and only re-select hits")
    p.add_argument("--python", default=None,
                   help="optional interpreter for running the same PLMCaliper.py out of process")
    p.add_argument("--out-cutoffs", type=Path, default=None)
    p.add_argument("--out-hits", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    real = a.real or Path(str(RAW_REAL[a.dataset]).format(m=a.method))
    decoy = a.decoy or Path(str(RAW_DECOY[a.dataset]).format(m=a.method))
    cutoffs = a.out_cutoffs or cutoffs_path(a.dataset, a.method)
    hits_out = a.out_hits or hits_path(a.dataset, a.method)

    if not a.skip_calibration:
        for p in (real, decoy):
            if not p.exists():
                raise FileNotFoundError(f"{p} -- run build_data.py first")
        run_plmcaliper(a.dataset, a.method, real, decoy, cutoffs, a.python, tau=a.tau)
    elif not cutoffs.exists():
        raise FileNotFoundError(f"{cutoffs} -- drop --skip-calibration")

    qids = read_query_ids(a.query_list) if a.query_list else None
    hits = select_hits(a.dataset, a.method, cutoffs, real, qids)
    hits_out.parent.mkdir(parents=True, exist_ok=True)
    hits.to_csv(hits_out, sep="\t", index=False)

    n_per = hits.groupby(["q"])["tid"].count() / max(hits["qid"].nunique(), 1)
    print(f"\n[step2] {hits['qid'].nunique()} queries -> {hits_out}")
    print(f"{'q':>6} {'mean hits/query':>16}")
    for q, n in n_per.items():
        print(f"{q:>6.2f} {n:>16.1f}")


if __name__ == "__main__":
    main()
