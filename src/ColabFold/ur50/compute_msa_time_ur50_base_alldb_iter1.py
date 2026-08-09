#!/usr/bin/env python3
import argparse
import hashlib
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
QJACKHMMER = ROOT / "libs/Dense-Homolog-Retrieval-main/bin/qjackhmmer"
UR50_FASTA = ROOT / "data/uniref50/uniref50.fasta"
QUERY_TSV = ROOT / "data/astral4f_query.tsv"
EXISTING_RUNS = ROOT / "results/ColabFold/ur50_db/dhr_postprocess/iter1/runs"
RETIME_DIR = ROOT / "results/ColabFold/ur50_db/dhr_postprocess/iter1/retime_base_alldb_iter1"
PLOT_TSV = ROOT / "data/plot_data/ColabFold/ur50_db/metrics_dhr_postprocess.tsv"

TAG = "base_alldb_iter1"
JH = dict(cpu=8, F1="0.0005", F2="5e-05", F3="5e-07")
INCE = "1e-3"
ITERS = 1


def load_query_seqs() -> dict[str, str]:
    seqs: dict[str, str] = {}
    with open(QUERY_TSV) as fh:
        for line in fh:
            if not line.strip():
                continue
            qid, seq = line.rstrip("\n").split("\t", 1)
            seqs[qid] = seq
    return seqs


def a3m_depth(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.startswith(">"))


def sha256(path: Path) -> str | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_fa(path: Path, qid: str, seq: str) -> None:
    path.write_text(f">{qid}\n{seq}\n")


def run_one(qid: str, seq: str, out_root: str) -> dict:
    out_dir = Path(out_root) / "runs" / qid / TAG
    out_dir.mkdir(parents=True, exist_ok=True)
    qfa = out_dir / "query.fa"
    out_a3m = out_dir / "msa.a3m"
    write_fa(qfa, qid, seq)

    cmd = [
        str(QJACKHMMER),
        "-B",
        str(out_a3m),
        "--noali",
        "--incE",
        INCE,
        "--incdomE",
        INCE,
        "-E",
        INCE,
        "--domE",
        INCE,
        "--cpu",
        str(JH["cpu"]),
        "-N",
        str(ITERS),
        "--F1",
        JH["F1"],
        "--F2",
        JH["F2"],
        "--F3",
        JH["F3"],
        "-o",
        str(out_a3m.with_suffix(".jh.log")),
        str(qfa),
        str(UR50_FASTA),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    seconds = time.perf_counter() - t0

    old_a3m = EXISTING_RUNS / qid / TAG / "msa.a3m"
    old_depth = a3m_depth(old_a3m)
    new_depth = a3m_depth(out_a3m)
    old_sha = sha256(old_a3m)
    new_sha = sha256(out_a3m)
    return {
        "query_id": qid,
        "tag": TAG,
        "seconds": round(seconds, 2),
        "old_depth": old_depth,
        "new_depth": new_depth,
        "depth_match": old_depth == new_depth,
        "old_sha256": old_sha,
        "new_sha256": new_sha,
        "sha256_match": old_sha is not None and old_sha == new_sha,
        "returncode": proc.returncode,
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-8:]),
    }


def patch_plot_data(summary: pd.DataFrame, path: Path) -> None:
    ok = summary[(summary["returncode"] == 0) & summary["depth_match"]].copy()
    if ok.empty:
        raise RuntimeError("No successful depth-matched retime rows to patch.")

    df = pd.read_csv(path, sep="\t")
    lookup = dict(zip(ok["query_id"].astype(str), ok["seconds"].astype(float)))
    mask = (df["tag"] == TAG) & df["query_id"].astype(str).isin(lookup)
    df.loc[mask, "msa_seconds"] = df.loc[mask, "query_id"].astype(str).map(lookup)
    df.to_csv(path, sep="\t", index=False)
    print(f"[PATCH] {int(mask.sum())} rows updated in {path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--query-ids", nargs="+", default=None)
    p.add_argument("--all", action="store_true", help="process all queries in the plotting table")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=RETIME_DIR)
    p.add_argument("--plot-tsv", type=Path, default=PLOT_TSV)
    p.add_argument("--update-plot-data", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    qseqs = load_query_seqs()
    if args.all:
        df = pd.read_csv(args.plot_tsv, sep="\t")
        qids = sorted(df.loc[df["tag"] == TAG, "query_id"].astype(str).unique())
    elif args.query_ids:
        qids = args.query_ids
    else:
        raise SystemExit("Pass --query-ids ... or --all")

    missing = [q for q in qids if q not in qseqs]
    if missing:
        raise SystemExit(f"Unknown query ids: {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "retime_base_alldb_iter1.tsv"
    print(f"[INFO] qjackhmmer={QJACKHMMER}", flush=True)
    print(f"[INFO] ur50={UR50_FASTA}", flush=True)
    print(f"[INFO] queries={len(qids)} workers={args.workers} out={args.out_dir}", flush=True)

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, qid, qseqs[qid], str(args.out_dir)): qid for qid in qids}
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            pd.DataFrame(rows).sort_values("query_id").to_csv(summary_path, sep="\t", index=False)
            status = "OK" if row["returncode"] == 0 and row["depth_match"] else "BAD"
            print(
                f"[{status}] {row['query_id']} {row['seconds']:.2f}s "
                f"depth {row['new_depth']} old {row['old_depth']} "
                f"sha_match={row['sha256_match']}",
                flush=True,
            )

    summary = pd.DataFrame(rows).sort_values("query_id")
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"[DONE] summary -> {summary_path}", flush=True)

    bad = summary[(summary["returncode"] != 0) | (~summary["depth_match"])]
    if not bad.empty:
        print(bad[["query_id", "returncode", "old_depth", "new_depth", "stderr_tail"]].to_string(index=False))
        raise SystemExit(2)
    if args.update_plot_data:
        patch_plot_data(summary, args.plot_tsv)


if __name__ == "__main__":
    main()
