import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

EXPS = {
    "astral_db": dict(
        pipeline=ROOT / "src/ColabFold/astral/pipeline_astral.py",
        full_query_tsv=ROOT / "data/querylist_4families.txt",
        prebuild_cand=False,
        pass_jack_iters=True,   
    ),
    "ur50_db": dict(
        pipeline=ROOT / "src/ColabFold/ur50/pipeline_ur50.py",
        full_query_tsv=ROOT / "data/querylist_4families.txt",  # astral queries
        noisy=ROOT / "data/result_dhr_postprocess_astral4f_target_noisy_ur50.txt",
        ur50_tsv=ROOT / "data/uniref50/uniref50.tsv",
        prebuild_cand=True,
        pass_jack_iters=False, 
        pass_runs_dir=True,
    ),
}
METHOD = "dhr_postprocess"


def out_for(case: str, method: str, jack_iters: int) -> Path:
    return ROOT / f"results/ColabFold/{case}/{method}/iter{jack_iters}/metrics.tsv"


def read_query_ids(path: Path) -> list[str]:
    ids, seen = [], set()
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        qid = ln.split("\t")[0].strip()
        if qid and qid not in seen:
            ids.append(qid)
            seen.add(qid)
    return ids


def done_queries(out: Path, shard_dir: Path) -> set[str]:
    done: set[str] = set()
    files = [out] + sorted(shard_dir.glob("out_gpu*.tsv"))
    for f in files:
        if f.exists() and f.stat().st_size > 0:
            try:
                d = pd.read_csv(f, sep="\t")
                done |= set(d.loc[d["tag"] == "vanilla", "query_id"].astype(str))
            except Exception:
                pass
    return done


def prebuild_cand_seqs(cfg, remaining: list[str], dst: Path) -> Path | None:
    noisy, ur50_tsv = cfg["noisy"], cfg["ur50_tsv"]
    if not noisy.exists() or not ur50_tsv.exists():
        print(f"[prebuild] missing {noisy} or {ur50_tsv}; workers will self-scan")
        return None
    real = pd.read_csv(noisy, sep="\t", usecols=["qid", "tid"])
    cand_tids = set(real.loc[real["qid"].astype(str).isin(set(remaining)), "tid"].astype(str))
    print(f"[prebuild] {len(cand_tids):,} candidate tids; scanning UR50 TSV once ...", flush=True)
    n = 0
    with open(ur50_tsv) as fh, open(dst, "w") as out:
        for line in fh:
            tid = line.split("\t", 1)[0]
            if tid in cand_tids:
                out.write(line if line.endswith("\n") else line + "\n")
                n += 1
    print(f"[prebuild] wrote {n:,} candidate seqs -> {dst}", flush=True)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=list(EXPS), required=True)
    ap.add_argument("--gpus", required=True, help="comma-separated GPU ids, e.g. 2,3,4")
    ap.add_argument("--method", default=METHOD)
    ap.add_argument("--query-list", type=Path, default=None,
                    help="override the full query universe")
    ap.add_argument("--no-prebuild", action="store_true",
                    help="skip the shared candidate-seq cache (ur50)")
    ap.add_argument("--af2-only", action="store_true",
                    help="fold pre-built MSAs only; workers never (re)build a3m")
    ap.add_argument("--skip-af2", action="store_true",
                    help="build MSAs only; no folding (passed through to workers)")
    ap.add_argument("--jack-iters", type=int, default=1,
                    help="JackHMMER iterations; also names the output folder iter<N>.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the per-GPU split plan and exit (no folding)")
    args = ap.parse_args()

    cfg = dict(EXPS[args.exp])
    if "noisy" in cfg:
        cfg["noisy"] = ROOT / f"data/result_{args.method}_astral4f_target_noisy_ur50.txt"
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    pipeline = cfg["pipeline"]
    out = out_for(args.exp, args.method, args.jack_iters)
    full_list = args.query_list or cfg["full_query_tsv"]

    py = os.environ.get("EVALPLMS_PYTHON_PATH")
    if not py:
        sys.exit("EVALPLMS_PYTHON_PATH not set — run `source src/PLMs_cmds/.env` first")

    work = out.parent / "parallel"
    shard_dir = work
    work.mkdir(parents=True, exist_ok=True)

    full = read_query_ids(full_list)
    done = done_queries(out, shard_dir)
    remaining = [q for q in full if q not in done]
    print(f"[INFO] exp={args.exp} gpus={gpus}")
    print(f"[INFO] queries: {len(full)} total, {len(done)} done, {len(remaining)} remaining")
    if not remaining:
        print("[DONE] nothing to do — all queries already folded.")
        merge(out, shard_dir)
        return

    buckets = {g: [] for g in gpus}
    for i, q in enumerate(remaining):
        buckets[gpus[i % len(gpus)]].append(q)

    if args.dry_run:
        for g in gpus:
            print(f"  GPU {g}: {len(buckets[g])} queries  {buckets[g]}")
        print(f"[DRY-RUN] would launch {len([g for g in gpus if buckets[g]])} workers; "
              f"prebuild_cand={cfg.get('prebuild_cand') and not args.no_prebuild}")
        return

    cand_seqs = None
    if cfg.get("prebuild_cand") and not args.no_prebuild:
        existing = work / "cand_seqs.tsv"
        if existing.exists() and existing.stat().st_size > 0:
            print(f"[prebuild] reusing existing candidate-seq cache {existing}", flush=True)
            cand_seqs = existing
        else:
            cand_seqs = prebuild_cand_seqs(cfg, remaining, existing)

    procs = []
    for g in gpus:
        qids = buckets[g]
        if not qids:
            continue
        shard = work / f"queries_gpu{g}.txt"
        shard.write_text("\n".join(qids) + "\n")
        shard_out = work / f"out_gpu{g}.tsv"
        log = work / f"gpu{g}.log"
        cmd = [py, str(pipeline), "--method", args.method,
               "--query-list", str(shard), "--out", str(shard_out)]
        if cfg.get("pass_jack_iters", True):
            cmd += ["--jack-iters", str(args.jack_iters)]
        if cfg.get("pass_runs_dir", False):
            cmd += ["--runs-dir", str(out.parent / "runs"), "--tmp-dir", str(out.parent / "tmp")]
        if cand_seqs:
            cmd += ["--cand-seqs", str(cand_seqs)]
        if args.af2_only:
            cmd.append("--af2-only")
        if args.skip_af2:
            cmd.append("--skip-af2")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g)
        print(f"[LAUNCH] GPU {g}: {len(qids)} queries -> {shard_out} (log {log})")
        lf = open(log, "w")
        procs.append((g, subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                                          stdout=lf, stderr=subprocess.STDOUT), lf))

    rc = 0
    for g, p, lf in procs:
        r = p.wait()
        lf.close()
        print(f"[WORKER] GPU {g} exited rc={r}")
        rc |= (r != 0)
    merge(out, shard_dir)
    print(f"[DONE] merged -> {out}" + ("" if rc == 0 else "  (some workers nonzero; re-run to resume)"))


def merge(out: Path, shard_dir: Path) -> None:
    frames = []
    if out.exists() and out.stat().st_size > 0:
        frames.append(pd.read_csv(out, sep="\t"))
    for f in sorted(shard_dir.glob("out_gpu*.tsv")):
        if f.stat().st_size > 0:
            frames.append(pd.read_csv(f, sep="\t"))
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    key = ["method", "query_id", "tag"]
    if "efdr_limit" in df.columns:
        df["_e"] = df["efdr_limit"].fillna(-1)
        key = key + ["_e"]
    df = df.drop_duplicates(subset=key, keep="last").drop(columns=[c for c in ["_e"] if c in df.columns])
    df = df.sort_values(["query_id", "tag"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"[MERGE] {len(df)} rows, {df.loc[df['tag']=='vanilla','query_id'].nunique()} queries -> {out}")


if __name__ == "__main__":
    main()
