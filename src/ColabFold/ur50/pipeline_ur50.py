#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

sys.path.insert(0, str(C.CORE_DIR))
from fdr import compute_tda_fdr_per_query, make_pair_table  # noqa: E402

JACKHMMER = dict(cpu=8, F1="0.0005", F2="5e-05", F3="5e-07")  # DHR-repo default prefilters
BASE_INCE = "1e-3"      # baselines: DHR-repo default inclusion (incE = -E = 1e-3)
FDR_INCE = "1e9"        # FDR arm: effectively NO E-value filter (as in the astral-subset exp)
# Three baselines, all with DHR-default filtering; they differ in target DB + iterations.
#   (tag, target-kind, jackhmmer iters)
BASELINES = [("base_alldb_iter1", "alldb", 1),
             ("base_alldb_iter2", "alldb", 2),
             ("base_top400k_iter2", "top400k", 2)]
TOP400K_N = 400000
TOP400K_DIR = C.DATA_DIR / "ur50_top400k"   # prebuilt per-method/per-query fastas: <method>/<qid>.fa


# ---------- IO ----------
def path_noisy(method: str) -> Path:
    return C.DATA_DIR / f"result_{method}_{C.QUERY_NAME}_target_noisy_{C.UR50_NAME}.txt"


def path_calib(method: str) -> Path:
    return (C.DATA_DIR / f"result_{method}_{C.QUERY_NAME}_{C.DECOY_METHOD}"
            f"_calibrated_gam_{C.WEIGHT_METHOD}_{C.UR50_NAME}.txt")


def grep_rows(path: Path, prefix: str) -> pd.DataFrame:
    proc = subprocess.run(["grep", "-E", f"^{re.escape(prefix)}\t", str(path)],
                          capture_output=True, text=True, check=False)
    if not proc.stdout.strip():
        return pd.DataFrame()
    header = path.read_text().split("\n", 1)[0] if path.stat().st_size < 4096 else \
        open(path).readline().rstrip("\n")
    return pd.read_csv(StringIO(header + "\n" + proc.stdout), sep="\t")


def load_query_seqs() -> dict:
    return {l.split("\t")[0]: l.rstrip("\n").split("\t")[1]
            for l in open(C.QUERY_TSV) if l.strip()}


def build_candidate_seqs(tids: set[str]) -> dict:
    out: dict[str, str] = {}
    with open(C.UR50_TSV) as fh:
        for line in fh:
            i = line.find("\t")
            if i > 0 and line[:i] in tids:
                out[line[:i]] = line[i + 1:].rstrip("\n")
                if len(out) == len(tids):
                    break
    return out


# ---------- FDR selection ----------
def select_hits_at_efdr(df_curve, q, df_real_q, qid) -> pd.DataFrame:
    valid = df_curve[df_curve["est_fdp"] <= q]
    if valid.empty:
        return pd.DataFrame(columns=df_real_q.columns)
    thr = float(valid.loc[valid["threshold"].idxmin(), "threshold"])
    hits = df_real_q[df_real_q["tid"] != qid].sort_values("score", ascending=False)
    return hits[hits["score"] >= thr].reset_index(drop=True)


# ---------- MSA + fold ----------
def run_qjackhmmer(query_fa: Path, target_fa: Path, out_a3m: Path, incE: str, iters: int) -> int:
    # incE controls inclusion: baselines use the DHR default (1e-3), the FDR arm uses a
    # huge value (no E-value filtering). iters = -N (JackHMMER iterations). --inc*E + -E/--domE
    # are all set to incE so reporting and inclusion agree.
    cmd = [str(C.QJACKHMMER_BIN), "-B", str(out_a3m), "--noali",
           "--incE", incE, "--incdomE", incE, "-E", incE, "--domE", incE,
           "--cpu", str(JACKHMMER["cpu"]), "-N", str(iters),
           "--F1", JACKHMMER["F1"], "--F2", JACKHMMER["F2"],
           "--F3", JACKHMMER["F3"], "-o", str(out_a3m.with_suffix(".jh.log")),
           str(query_fa), str(target_fa)]
    subprocess.run(cmd, check=True)
    if not out_a3m.exists() or out_a3m.stat().st_size == 0:
        return 0
    return sum(1 for l in out_a3m.read_text().splitlines() if l.startswith(">"))


def resolve_colabfold() -> str | None:
    p = os.environ.get("COLABFOLD_BATCH")
    if p and Path(p).exists():
        return p
    r = subprocess.run(["bash", "-lc", "command -v colabfold_batch"], capture_output=True, text=True)
    return r.stdout.strip() or None


def run_colabfold(msa_in: Path, out_dir: Path, single: bool) -> tuple[str, float]:
    cf = resolve_colabfold()
    if not cf:
        return "colabfold_missing", float("nan")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [cf, "--num-models", "1", "--num-recycle", "3", str(msa_in), str(out_dir)]
    if single:
        cmd += ["--msa-mode", "single_sequence"]
    t = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t
    (out_dir / "colabfold.log").write_text(r.stdout + r.stderr)
    return ("ok" if r.returncode == 0 else "failed"), dt


def extract_plddt(out_dir: Path) -> float | None:
    for pat in ("**/*scores_rank_001*.json", "**/*scores_rank_*.json"):
        hits = sorted(out_dir.glob(pat))
        if hits:
            pl = json.loads(hits[0].read_text()).get("plddt")
            return float(np.mean(pl)) if pl else None
    return None


def write_fa(path: Path, recs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f">{i}\n{s}\n" for i, s in recs))


def a3m_depth(a3m: Path) -> int:
    return sum(1 for l in a3m.read_text().splitlines() if l.startswith(">")) if a3m.exists() else 0


def native_pdb(qid: str) -> Path | None:
    # SCOP/ASTRAL native structures (queries are now astral 4-family domains)
    p = C.PROJECT_ROOT / "libs/dplm/gen.fasta/astral/folding/pdb" / f"{qid}.pdb"
    if p.exists():
        return p
    ent = C.DATA_DIR / "pdbstyle-2.08" / qid[2:4] / f"{qid}.ent"
    return ent if ent.exists() else None


def find_pred_pdb(out_dir: Path) -> Path | None:
    for pat in ("**/*_unrelaxed_rank_001*.pdb", "**/*rank_001*.pdb", "**/*.pdb"):
        h = sorted(out_dir.glob(pat))
        if h:
            return h[0]
    return None


TMALIGN = C.PROJECT_ROOT / "libs/dplm/analysis/TMalign"


def run_tmalign(model: Path, native: Path | None) -> dict:
    if native is None or not Path(native).exists() or not TMALIGN.exists():
        return {"tm_score": None, "rmsd": None}
    r = subprocess.run([str(TMALIGN), str(model), str(native)], capture_output=True, text=True, check=False)
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


# ---------- per query ----------
def _conditions():
    fdr = [(C.efdr_tag(q) if hasattr(C, "efdr_tag") else f"efdr_{q:.2f}".replace('.', 'p'),
            "fdr", q, FDR_INCE, 1) for q in C.EFDR_THRESHOLDS]
    base = [(tag, kind, None, BASE_INCE, iters) for tag, kind, iters in BASELINES]
    return fdr + base


def run_query(
    qid: str,
    method: str,
    qseqs: dict,
    cand: dict,
    *,
    skip_af2: bool,
    af2_only: bool,
    force_rebuild: bool,
    prior: dict,
    runs_dir: Path,
    tmp_dir: Path,
) -> list[dict]:
    df_real = grep_rows(path_noisy(method), qid)
    df_real = df_real[df_real["rep_id"] == 0] if "rep_id" in df_real.columns else df_real
    df_decoy = grep_rows(path_calib(method), f"{qid}{C.DECOY_SUFFIX}")
    df_decoy = df_decoy[df_decoy["rep_id"] == 0] if "rep_id" in df_decoy.columns else df_decoy
    if df_real.empty or df_decoy.empty:
        return [{"method": method, "query_id": qid, "tag": "ERR", "msa_status": "no_scores", "af2_status": "skip"}]

    tmp_q = tmp_dir / qid
    tmp_q.mkdir(parents=True, exist_ok=True)
    df_real.to_csv(tmp_q / "r.tsv", sep="\t", index=False)
    df_decoy.to_csv(tmp_q / "d.tsv", sep="\t", index=False)
    curve = compute_tda_fdr_per_query(make_pair_table(str(tmp_q / "r.tsv"), str(tmp_q / "d.tsv"), C.DECOY_SUFFIX))

    rows = []
    out_root = runs_dir / qid
    for tag, kind, q, incE, iters in _conditions():
        d = out_root / tag
        d.mkdir(parents=True, exist_ok=True)
        qfa = d / "query.fa"
        write_fa(qfa, [(qid, qseqs[qid])])
        a3m = d / "msa.a3m"
        built = a3m.exists() and a3m.stat().st_size > 0
        n_hits = 0
        msa_status = "pending"
        target = None
        t0 = time.perf_counter()
        try:
            if kind == "fdr":
                hits = select_hits_at_efdr(curve, q, df_real, qid)
                recs = [(t, cand[t]) for t in hits["tid"].astype(str) if t in cand]
                n_hits = len(recs)
                if n_hits == 0:
                    msa_status = "query_only"
                else:
                    target = d / "hits.fa"
                    write_fa(target, recs)
            elif kind == "alldb":
                target, n_hits = C.UR50_FASTA, -1
            elif kind == "top400k":
                target, n_hits = TOP400K_DIR / method / f"{qid}.fa", -2
                if not target.exists():
                    target, msa_status = None, "top400k_missing"
            # build the MSA unless it already exists (resume) or we're in af2-only mode
            if target is not None and not af2_only and (force_rebuild or not built):
                n = run_qjackhmmer(qfa, target, a3m, incE, iters)
                msa_status = "jackhmmer_ok" if n > 1 else "thin_msa"
            elif built:
                msa_status = "reused"
        except Exception as exc:
            msa_status = f"msa_failed:{exc}"
        msa_s = time.perf_counter() - t0

        depth = a3m_depth(a3m)
        single = depth <= 1                     # fold the bare query if the MSA is empty/thin
        # preserve the real build time from a prior (build) pass when we reuse the MSA
        prior_row = prior.get((qid, tag))
        if (af2_only or (built and not force_rebuild)) and prior_row is not None:
            msa_s = prior_row.get("msa_seconds") or msa_s
        if msa_status in {"reused", "pending"} and depth > 0:
            msa_status = "reused" if built else msa_status

        plddt = tm = rmsd = None
        predict_s = float("nan")
        af2 = "skipped" if skip_af2 else "pending"
        foldable = depth >= 1 or msa_status in {"query_only", "thin_msa"}
        if not skip_af2 and prior_row is not None and prior_row.get("af2_status") == "ok" \
                and not force_rebuild:                       # reuse a prior AF2 fold
            af2, plddt = "ok", prior_row.get("plddt")
            tm, rmsd = prior_row.get("tm_score"), prior_row.get("rmsd")
            predict_s = prior_row.get("predict_seconds") or float("nan")
        elif not skip_af2 and foldable and "missing" not in msa_status:
            cf_in = qfa if single else a3m
            af2, predict_s = run_colabfold(cf_in, d / "af2_out", single)
            if af2 == "ok":
                plddt = extract_plddt(d / "af2_out")
                pred = find_pred_pdb(d / "af2_out")
                if pred is not None:
                    tmr = run_tmalign(pred, native_pdb(qid))
                    tm, rmsd = tmr["tm_score"], tmr["rmsd"]
        msa_s_val = round(msa_s, 2) if msa_s == msa_s else None
        rows.append({"method": method, "query_id": qid, "tag": tag, "efdr_limit": q,
                     "qlen": len(qseqs[qid]), "n_hits": n_hits, "msa_depth": depth,
                     "plddt": plddt, "tm_score": tm, "rmsd": rmsd,
                     "msa_seconds": msa_s_val,
                     "predict_seconds": round(predict_s, 2) if predict_s == predict_s else None,
                     "total_seconds": round(np.nansum([msa_s, predict_s]), 2),
                     "msa_status": msa_status, "af2_status": af2})
        print(f"  [{qid}] {tag}: hits={n_hits} depth={depth} plddt={plddt} tm={tm} rmsd={rmsd} "
              f"msa={msa_status} af2={af2}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="dhr_postprocess")
    ap.add_argument("--query-id", default=None)
    ap.add_argument("--query-list", type=Path, default=None,
                    help="File of query ids (one per line) to process. Enables "
                         "per-query parallel sharding across processes/GPUs.")
    ap.add_argument("--cand-seqs", type=Path, default=None,
                    help="Prebuilt candidate-seq TSV (tid<TAB>seq) to load instead "
                         "of streaming the full UR50 TSV (avoids N-way 17GB scans).")
    ap.add_argument("--skip-af2", action="store_true", help="build MSAs only, no folding")
    ap.add_argument("--af2-only", action="store_true",
                    help="fold pre-built MSAs only; never (re)build a3m")
    ap.add_argument("--force-rebuild", action="store_true",
                    help="rebuild MSA / refold even if outputs already exist")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--runs-dir", type=Path, default=None,
                    help="Directory for query/tag MSA and ColabFold outputs. Default: results/ur50_exp/<method>/runs")
    ap.add_argument("--tmp-dir", type=Path, default=None,
                    help="Directory for temporary per-query pair tables. Default: results/ur50_exp/<method>/_tmp")
    args = ap.parse_args()

    qseqs = load_query_seqs()
    if args.query_id:
        qids = [args.query_id]
    elif args.query_list:
        wanted = [l.strip() for l in args.query_list.read_text().splitlines()
                  if l.strip() and not l.startswith("#")]
        qids = [q for q in wanted if q in qseqs]
    else:
        qids = list(qseqs.keys())

    # candidate tids needed for FDR hits.fa (real-query retrieved targets),
    # scoped to the queries THIS process handles (keeps the seq dict small when
    # sharding across GPUs).
    real = pd.read_csv(path_noisy(args.method), sep="\t", usecols=["qid", "tid"])
    real = real[real["qid"].astype(str).isin(set(qids))]
    cand_tids = set(real["tid"].astype(str))
    if args.cand_seqs and args.cand_seqs.exists():
        print(f"[INFO] loading prebuilt candidate seqs from {args.cand_seqs}", flush=True)
        cand = {}
        with open(args.cand_seqs) as fh:
            for line in fh:
                if not line.strip():
                    continue
                tid, seq = line.rstrip("\n").split("\t", 1)
                if tid in cand_tids:
                    cand[tid] = seq
    else:
        print(f"[INFO] building seq lookup for {len(cand_tids):,} candidate tids ...", flush=True)
        cand = build_candidate_seqs(cand_tids)
    print(f"[INFO] got {len(cand):,} sequences", flush=True)

    method_root = C.method_results_dir(args.method)
    runs_dir = args.runs_dir or C.method_runs_dir(args.method)
    tmp_dir = args.tmp_dir or C.method_tmp_dir(args.method)
    out = args.out or C.method_summary_path(args.method)
    runs_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] method output root: {method_root}", flush=True)
    print(f"[INFO] runs dir: {runs_dir}", flush=True)

    # resume: load prior rows keyed by (qid, tag). Rows for queries NOT in this shard are
    # carried through untouched; rows for our queries seed msa_seconds / AF2 reuse.
    n_cond = len(_conditions())
    prior = {}
    carry = []
    if out.exists():
        prev = pd.read_csv(out, sep="\t").to_dict("records")
        ours = set(qids)
        for r in prev:
            qt = (str(r["query_id"]), str(r["tag"]))
            if str(r["query_id"]) in ours:
                prior[qt] = r
            else:
                carry.append(r)
        print(f"[RESUME] {len(prior)} prior rows for this shard's queries", flush=True)

    def fully_done(qid):
        rs = [prior.get((qid, t)) for t, *_ in _conditions()]
        if any(r is None for r in rs):
            return False
        if args.skip_af2:
            return all(r.get("msa_status") not in (None, "pending") for r in rs)
        return all(r.get("af2_status") in ("ok", "skipped") for r in rs)

    all_rows = list(carry)
    for qid in qids:
        if not args.force_rebuild and fully_done(qid):
            all_rows += [prior[(qid, t)] for t, *_ in _conditions()]
            continue
        all_rows += run_query(
            qid,
            args.method,
            qseqs,
            cand,
            skip_af2=args.skip_af2,
            af2_only=args.af2_only,
            force_rebuild=args.force_rebuild,
            prior=prior,
            runs_dir=runs_dir,
            tmp_dir=tmp_dir,
        )
        pd.DataFrame(all_rows).to_csv(out, sep="\t", index=False)  # incremental save
    pd.DataFrame(all_rows).to_csv(out, sep="\t", index=False)
    print(f"[DONE] -> {out}")


if __name__ == "__main__":
    main()
