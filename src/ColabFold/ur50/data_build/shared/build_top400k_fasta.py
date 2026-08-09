#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
UR50_TSV = ROOT / "data/uniref50/uniref50.tsv"


def retrieval_path(method: str) -> Path:
    return ROOT / f"data/result_{method}_astral4f_ur50.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="dhr_postprocess")
    ap.add_argument("--n", type=int, default=400000)
    args = ap.parse_args()

    outdir = ROOT / "data/ur50_top400k" / args.method
    outdir.mkdir(parents=True, exist_ok=True)
    rpath = retrieval_path(args.method)
    if not rpath.exists():
        raise SystemExit(f"[abort] retrieval file missing: {rpath}")

    # 1. per query -> set of its top-N tids; tid -> list of queries needing it
    print(f"[1/3] reading top-{args.n} tids/query from {rpath.name} ...", flush=True)
    want = defaultdict(list)            # tid -> [qid, ...]
    per_q_count = defaultdict(int)
    with open(rpath) as fh:
        header = fh.readline()          # qid tid score homo_type rank
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            qid, tid, rank = p[0], p[1], int(p[4])
            if rank <= args.n:
                want[tid].append(qid)
                per_q_count[qid] += 1
    print(f"      {len(per_q_count)} queries, {len(want):,} unique tids "
          f"(median per query ~{sorted(per_q_count.values())[len(per_q_count)//2]:,})", flush=True)

    # 2. open one fasta handle per query (skip queries already fully built)
    handles, written = {}, defaultdict(int)
    for qid in per_q_count:
        fp = outdir / f"{qid}.fa"
        if fp.exists() and fp.stat().st_size > 0:   # resume: skip done queries
            continue
        handles[qid] = open(fp, "w")
    print(f"[2/3] writing {len(handles)} fastas (resume: {len(per_q_count)-len(handles)} already done)", flush=True)
    if not handles:
        print("[DONE] all per-query fastas already present.")
        return

    # 3. ONE streaming pass over UR50 TSV, routing each needed seq to its queries
    print(f"[3/3] streaming {UR50_TSV.name} ...", flush=True)
    n_lines = 0
    with open(UR50_TSV) as fh:
        for line in fh:
            n_lines += 1
            i = line.find("\t")
            if i <= 0:
                continue
            tid = line[:i]
            qs = want.get(tid)
            if not qs:
                continue
            seq = line[i + 1:].rstrip("\n")
            rec = f">{tid}\n{seq}\n"
            for qid in qs:
                h = handles.get(qid)
                if h is not None:
                    h.write(rec)
                    written[qid] += 1
            if n_lines % 5_000_000 == 0:
                print(f"      scanned {n_lines:,} UR50 lines ...", flush=True)
    for h in handles.values():
        h.close()
    miss = {q: per_q_count[q] - written[q] for q in handles if written[q] < per_q_count[q]}
    print(f"[DONE] wrote {len(handles)} fastas -> {outdir}")
    print(f"       median seqs/query = {sorted(written.values())[len(written)//2]:,}" if written else "")
    if miss:
        print(f"[WARN] {len(miss)} queries had tids not found in UR50 TSV (e.g. {list(miss.items())[:3]})")


if __name__ == "__main__":
    main()
