#!/usr/bin/env python3
"""Convert a FASTA to the two-column `id<TAB>sequence` TSV that DHR searches take."""
import argparse
import os


def fa2tsv(fa_path, tsv_path):
    n = 0
    with open(fa_path) as fin, open(tsv_path, "w") as fout:
        seq_id, chunks = None, []
        for line in fin:
            line = line.strip()
            if line.startswith(">"):
                if seq_id is not None:
                    fout.write(f"{seq_id}\t{''.join(chunks)}\n"); n += 1
                seq_id, chunks = line[1:].split(None, 1)[0], []
            elif line:
                chunks.append(line)
        if seq_id is not None:
            fout.write(f"{seq_id}\t{''.join(chunks)}\n"); n += 1
    print(f"[OK] {n} sequences -> {tsv_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", default=None, help="default: <fasta stem>.tsv")
    args = ap.parse_args()
    fa2tsv(args.fasta, args.out or f"{os.path.splitext(args.fasta)[0]}.tsv")


if __name__ == "__main__":
    main()
