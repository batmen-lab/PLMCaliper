#!/usr/bin/env python3
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
UR50_TSV = ROOT / "data/uniref50/uniref50.tsv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="dhr_postprocess",
                    help="retriever token in the result filenames (dhr_postprocess/tmvec/plm)")
    args = ap.parse_args()
    NOISY = ROOT / f"data/result_{args.method}_astral4f_target_noisy_ur50.txt"
    OUT = ROOT / f"results/ColabFold/ur50_db/{args.method}/iter1/parallel/cand_seqs.tsv"

    print(f"[1/2] reading candidate tids from {NOISY.name} ...", flush=True)
    tids = set()
    with open(NOISY) as fh:
        fh.readline()
        for line in fh:
            i = line.find("\t")
            j = line.find("\t", i + 1)
            if i > 0 and j > i:
                tids.add(line[i + 1:j])
    print(f"      {len(tids):,} unique candidate tids", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"[2/2] streaming {UR50_TSV.name} -> {OUT} ...", flush=True)
    n = found = 0
    with open(UR50_TSV) as fh, open(OUT, "w") as out:
        for line in fh:
            n += 1
            i = line.find("\t")
            if i > 0 and line[:i] in tids:
                out.write(line if line.endswith("\n") else line + "\n")
                found += 1
            if n % 5_000_000 == 0:
                print(f"      scanned {n:,} lines, found {found:,}", flush=True)
    print(f"[DONE] wrote {found:,}/{len(tids):,} candidate seqs -> {OUT}")


if __name__ == "__main__":
    main()
