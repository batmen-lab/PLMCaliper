import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
CORE = ROOT / "src" / "core"
sys.path.insert(0, str(CORE))
from fdr import compute_tda_fdr_per_query, make_pair_table  # noqa: E402

# ---- astral config ----
ASTRAL_FASTA = DATA / "astral.fa"
TARGET_NAME = "astral"
DECOY_METHOD = "extended_shuf"
DECOY_SUFFIX = "_shuf"
WEIGHT_METHOD = "AdaptiveBell"
EFDR_THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 10)]
VANILLA_TAG = "vanilla"
QJACKHMMER = ROOT / "libs/Dense-Homolog-Retrieval-main/bin/qjackhmmer"
TMALIGN = ROOT / "libs/dplm/analysis/TMalign"
NATIVE_PDB_DIRS = [ROOT / "libs/dplm/gen.fasta/astral/folding/pdb"]
PDBSTYLE_DIR = DATA / "pdbstyle-2.08"
RESULTS = ROOT / "results" / "ColabFold" / "astral_db"
PLOT_DATA = ROOT / "data" / "plot_data" / "ColabFold" / "astral_db"
JH = dict(cpu=8, F1="0.0005", F2="5e-05", F3="5e-07")
VANILLA_INCE = "1e-3"   # AF2-default-style E-value filter for the vanilla baseline
FDR_INCE = "1e9"        # effectively NO E-value filter for FDR-selected hits (avoid collapse)
JACK_ITERS = 1          # JackHMMER iterations (-N); set from --jack-iters, start at 1
MEFF_SEQID = 0.8
MEFF_SAMPLE = 3000
MEFF_RNG = np.random.default_rng(0)
METRIC_COLUMNS = [
    "method", "query_id", "tag", "efdr_limit", "qlen", "n_hits", "msa_depth",
    "meff", "plddt", "tm_score", "rmsd", "msa_seconds", "predict_seconds",
    "total_seconds", "msa_status", "af2_status", "has_native",
]


def path_noisy(method): return DATA / f"result_{method}_{TARGET_NAME}_target_noisy_{TARGET_NAME}.txt"
def path_calib(method): return DATA / f"result_{method}_{TARGET_NAME}_{DECOY_METHOD}_calibrated_gam_{WEIGHT_METHOD}_{TARGET_NAME}.txt"


def efdr_tag(q): return f"efdr_{q:.2f}".replace(".", "p")


def grep_rows(path: Path, prefix: str) -> pd.DataFrame:
    proc = subprocess.run(["grep", "-E", f"^{re.escape(prefix)}\t", str(path)],
                          capture_output=True, text=True, check=False)
    if not proc.stdout.strip():
        return pd.DataFrame()
    header = open(path).readline().rstrip("\n")
    return pd.read_csv(StringIO(header + "\n" + proc.stdout), sep="\t")


def native_pdb(qid: str) -> Path | None:
    for d in NATIVE_PDB_DIRS:
        p = d / f"{qid}.pdb"
        if p.exists():
            return p
    ent = PDBSTYLE_DIR / qid[2:4] / f"{qid}.ent"
    return ent if ent.exists() else None


def load_astral_seqs() -> dict:
    return {r.id: str(r.seq) for r in SeqIO.parse(str(ASTRAL_FASTA), "fasta")}


def select_hits_at_efdr(curve, q, df_real_q, qid) -> pd.DataFrame:
    valid = curve[curve["est_fdp"] <= q]
    if valid.empty:
        return pd.DataFrame(columns=df_real_q.columns)
    thr = float(valid.loc[valid["threshold"].idxmin(), "threshold"])
    hits = df_real_q[df_real_q["tid"] != qid].sort_values("score", ascending=False)
    return hits[hits["score"] >= thr].reset_index(drop=True)


def is_decoy_tid(tid: str) -> bool:
    return DECOY_SUFFIX in str(tid) or "_rev" in str(tid) or "_mkv" in str(tid) or "_dup" in str(tid)


def run_qjackhmmer(query_fa: Path, target_fa: Path, out_a3m: Path, incE: str) -> int:
    # vanilla uses the default filter; FDR arm uses a huge incE (no E-value filtering).
    cmd = [str(QJACKHMMER), "-B", str(out_a3m), "--noali",
           "--incE", incE, "--incdomE", incE, "-E", incE, "--domE", incE,
           "--cpu", str(JH["cpu"]), "-N", str(JACK_ITERS), "--F1", JH["F1"], "--F2", JH["F2"],
           "--F3", JH["F3"], "-o", str(out_a3m.with_suffix(".jh.log")), str(query_fa), str(target_fa)]
    subprocess.run(cmd, check=True)
    if not out_a3m.exists() or out_a3m.stat().st_size == 0:
        return 0
    return sum(1 for l in out_a3m.read_text().splitlines() if l.startswith(">"))


def resolve_colabfold():
    p = os.environ.get("COLABFOLD_BATCH")
    if p and Path(p).exists():
        return p
    r = subprocess.run(["bash", "-lc", "command -v colabfold_batch"], capture_output=True, text=True)
    return r.stdout.strip() or None


def run_colabfold(msa_in: Path, out_dir: Path, single: bool):
    cf = resolve_colabfold()
    if not cf:
        return "colabfold_missing", float("nan")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [cf, "--num-models", "1", "--num-recycle", "3", str(msa_in), str(out_dir)]
    if single:
        cmd += ["--msa-mode", "single_sequence"]
    t = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "colabfold.log").write_text(r.stdout + r.stderr)
    return ("ok" if r.returncode == 0 else "failed"), time.perf_counter() - t


def extract_plddt(out_dir: Path):
    for pat in ("**/*scores_rank_001*.json", "**/*scores_rank_*.json"):
        h = sorted(out_dir.glob(pat))
        if h:
            pl = json.loads(h[0].read_text()).get("plddt")
            return float(np.mean(pl)) if pl else None
    return None


def find_pred_pdb(out_dir: Path):
    for pat in ("**/*_unrelaxed_rank_001*.pdb", "**/*rank_001*.pdb", "**/*.pdb"):
        h = sorted(out_dir.glob(pat))
        if h:
            return h[0]
    return None


def run_tmalign(model: Path, native: Path) -> dict:
    if not TMALIGN.exists() or native is None or not Path(native).exists():
        return {"tm_score": None, "rmsd": None}
    r = subprocess.run([str(TMALIGN), str(model), str(native)], capture_output=True, text=True, check=False)
    txt = r.stdout + r.stderr
    tm = rmsd = None
    for line in txt.splitlines():
        if line.startswith("TM-score=") and "Chain_1" in line:
            m = re.search(r"TM-score=\s*([0-9.]+)", line)
            if m and tm is None:
                tm = float(m.group(1))
        if "RMSD=" in line:
            m = re.search(r"RMSD=\s*([0-9.]+)", line)
            if m:
                rmsd = float(m.group(1))
    return {"tm_score": tm, "rmsd": rmsd}


def a3m_depth(a3m: Path) -> int:
    return sum(1 for l in a3m.read_text().splitlines() if l.startswith(">")) if a3m.exists() else 0


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
    length = len(seqs[0])
    seqs = [s for s in seqs if len(s) == length]
    if not seqs:
        return None
    return np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), length)


def meff(path: Path) -> float:
    mat = a3m_matrix(path)
    if mat is None:
        return 0.0
    n, length = mat.shape
    if n == 1:
        return 1.0
    sample_n = n if n <= MEFF_SAMPLE else MEFF_SAMPLE
    idx = np.arange(n) if n <= MEFF_SAMPLE else MEFF_RNG.choice(n, sample_n, replace=False)
    sampled = mat[idx]
    threshold = MEFF_SEQID * length
    identical_cols = np.zeros((sample_n, n), dtype=np.float32)
    for symbol in np.unique(mat):
        identical_cols += (sampled == symbol).astype(np.float32) @ (mat == symbol).astype(np.float32).T
    n_similar = (identical_cols >= threshold).sum(axis=1)
    return float(n * np.mean(1.0 / n_similar))


def write_fa(path: Path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f">{i}\n{s}\n" for i, s in recs))


def default_out(method: str, jack_iters: int) -> Path:
    return RESULTS / method / f"iter{jack_iters}" / "metrics.tsv"


def default_runs_dir(out: Path) -> Path:
    direct = out.parent / "runs"
    parallel = out.parent / "parallel" / "runs"
    return parallel if parallel.is_dir() and not direct.is_dir() else direct


def write_metrics(rows: list[dict], out: Path) -> None:
    df = pd.DataFrame(rows)
    for col in METRIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    ordered = METRIC_COLUMNS + [c for c in df.columns if c not in METRIC_COLUMNS]
    out.parent.mkdir(parents=True, exist_ok=True)
    df[ordered].to_csv(out, sep="\t", index=False)


def sync_plot_data(src: Path, method: str, dst: Path | None = None) -> Path:
    dst = dst or PLOT_DATA / f"metrics_{method}.tsv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"[PLOT-DATA] {src} -> {dst}", flush=True)
    return dst


def _missing(value) -> bool:
    return value is None or pd.isna(value)


def fill_derived_metrics(rows: list[dict], qseqs: dict, runs_dir: Path) -> None:
    for row in rows:
        qid = str(row.get("query_id", ""))
        tag = str(row.get("tag", ""))
        if qid in qseqs and ("qlen" not in row or _missing(row.get("qlen"))):
            row["qlen"] = len(qseqs[qid])
        if tag and ("meff" not in row or _missing(row.get("meff"))):
            row["meff"] = round(meff(runs_dir / qid / tag / "msa.a3m"), 3)


def run_query(qid, method, qseqs, *, skip_af2, runs_dir, tmp_dir):
    dr = grep_rows(path_noisy(method), qid)
    dr = dr[dr["rep_id"] == 0] if "rep_id" in dr.columns else dr
    dd = grep_rows(path_calib(method), f"{qid}{DECOY_SUFFIX}")
    dd = dd[dd["rep_id"] == 0] if "rep_id" in dd.columns else dd
    if dr.empty or dd.empty:
        return [{"method": method, "query_id": qid, "tag": "ERR", "msa_status": "no_scores", "af2_status": "skip"}]
    # drop decoy targets from real hit pool
    dr = dr[~dr["tid"].astype(str).map(is_decoy_tid)].copy()

    tmp_q = tmp_dir / qid
    tmp_q.mkdir(parents=True, exist_ok=True)
    dr.to_csv(tmp_q / "r.tsv", sep="\t", index=False)
    dd.to_csv(tmp_q / "d.tsv", sep="\t", index=False)
    curve = compute_tda_fdr_per_query(make_pair_table(str(tmp_q / "r.tsv"), str(tmp_q / "d.tsv"), DECOY_SUFFIX))

    nat = native_pdb(qid)
    conditions = [(efdr_tag(q), q) for q in EFDR_THRESHOLDS] + [(VANILLA_TAG, None)]
    rows = []
    for tag, q in conditions:
        d = runs_dir / qid / tag
        d.mkdir(parents=True, exist_ok=True)
        qfa = d / "query.fa"
        write_fa(qfa, [(qid, qseqs[qid])])
        a3m = d / "msa.a3m"
        n_hits = 0
        single = False
        msa_status = "pending"
        t0 = time.perf_counter()
        try:
            if q is None:
                target = ASTRAL_FASTA
                n_hits = -1
            else:
                hits = select_hits_at_efdr(curve, q, dr, qid)
                recs = [(t, qseqs[t]) for t in hits["tid"].astype(str) if t in qseqs]
                n_hits = len(recs)
                if n_hits == 0:
                    msa_status, single = "query_only", True
                    target = None
                else:
                    target = d / "hits.fa"
                    write_fa(target, recs)
            if target is not None:
                incE = VANILLA_INCE if q is None else FDR_INCE
                n = run_qjackhmmer(qfa, target, a3m, incE)
                msa_status = "jackhmmer_ok" if n > 1 else "thin_msa"
                if n <= 1:
                    single = True
        except Exception as exc:
            msa_status = f"msa_failed:{exc}"
        msa_s = time.perf_counter() - t0
        depth = a3m_depth(a3m)
        meff_val = round(meff(a3m), 3)

        plddt = tm = rmsd = None
        predict_s = float("nan")
        af2 = "skipped" if skip_af2 else "pending"
        if not skip_af2 and msa_status in {"jackhmmer_ok", "query_only", "thin_msa"}:
            cf_in = qfa if single else a3m
            af2, predict_s = run_colabfold(cf_in, d / "af2_out", single)
            if af2 == "ok":
                plddt = extract_plddt(d / "af2_out")
                pred = find_pred_pdb(d / "af2_out")
                if pred is not None:
                    m = run_tmalign(pred, nat)
                    tm, rmsd = m["tm_score"], m["rmsd"]
        rows.append({"method": method, "query_id": qid, "tag": tag, "efdr_limit": q,
                     "qlen": len(qseqs[qid]), "n_hits": n_hits, "msa_depth": depth,
                     "meff": meff_val, "plddt": plddt,
                     "tm_score": tm, "rmsd": rmsd, "msa_seconds": round(msa_s, 2),
                     "predict_seconds": round(predict_s, 2) if predict_s == predict_s else None,
                     "total_seconds": round(float(np.nansum([msa_s, predict_s])), 2),
                     "msa_status": msa_status, "af2_status": af2, "has_native": nat is not None})
        print(f"  [{qid}] {tag}: hits={n_hits} depth={depth} plddt={plddt} "
              f"tm={tm} rmsd={rmsd} af2={af2}", flush=True)
    return rows


def load_query_list(path: Path) -> list[str]:
    out, seen = [], set()
    for l in path.read_text().splitlines():
        l = l.strip()
        if l and not l.startswith("#") and l not in seen:
            out.append(l); seen.add(l)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="dhr_postprocess")
    ap.add_argument("--query-id", default=None)
    ap.add_argument("--query-list", type=Path, default=None)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--skip-af2", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--runs-dir", type=Path, default=None,
                    help="Directory for query/tag MSA and ColabFold outputs. "
                         "Default: next to --out, under runs/")
    ap.add_argument("--tmp-dir", type=Path, default=None,
                    help="Directory for temporary per-query FDR inputs. "
                         "Default: next to --out, under _tmp/")
    ap.add_argument("--jack-iters", type=int, default=1,
                    help="JackHMMER iterations (-N) for both arms. Start at 1.")
    ap.add_argument("--sync-plot-data", action="store_true",
                    help="After writing metrics.tsv, copy it to "
                         "data/plot_data/ColabFold/astral_db/metrics_<method>.tsv")
    ap.add_argument("--plot-data-out", type=Path, default=None,
                    help="Override the --sync-plot-data destination.")
    args = ap.parse_args()
    global JACK_ITERS
    JACK_ITERS = args.jack_iters

    qseqs = load_astral_seqs()
    if args.query_id:
        qids = [args.query_id]
    elif args.query_list:
        qids = [q for q in load_query_list(args.query_list) if q in qseqs]
    else:
        raise SystemExit("provide --query-id or --query-list")
    if args.max_queries:
        qids = qids[:args.max_queries]

    out = args.out or default_out(args.method, args.jack_iters)
    # runs/tmp live next to the output table, so each results/ColabFold/<case>/iter<N>
    # is self-contained and keeps the MSA intermediates used for plotting backfills.
    runs_dir = args.runs_dir or default_runs_dir(out)
    tmp_dir = args.tmp_dir or out.parent / "_tmp"
    for p in (runs_dir, tmp_dir, out.parent):
        p.mkdir(parents=True, exist_ok=True)

    all_rows, done = [], set()
    if out.exists():
        prev = pd.read_csv(out, sep="\t")
        done = set(prev.loc[prev["tag"] == VANILLA_TAG, "query_id"].astype(str))
        all_rows = prev[prev["query_id"].astype(str).isin(done)].to_dict("records")
        print(f"[RESUME] {len(done)} queries done -> skip", flush=True)

    print(f"[INFO] {len(qids)} queries, method={args.method}, out={out}", flush=True)
    changed = False
    for qid in qids:
        if qid in done:
            continue
        all_rows += run_query(qid, args.method, qseqs, skip_af2=args.skip_af2,
                              runs_dir=runs_dir, tmp_dir=tmp_dir)
        fill_derived_metrics(all_rows, qseqs, runs_dir)
        write_metrics(all_rows, out)
        changed = True
    if changed or not out.exists():
        fill_derived_metrics(all_rows, qseqs, runs_dir)
        write_metrics(all_rows, out)
    if args.sync_plot_data:
        if out.exists():
            all_rows = pd.read_csv(out, sep="\t").to_dict("records")
            fill_derived_metrics(all_rows, qseqs, runs_dir)
            write_metrics(all_rows, out)
        sync_plot_data(out, args.method, args.plot_data_out)
    print(f"[DONE] -> {out}")


if __name__ == "__main__":
    main()
