#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np

EMB_DIM = 512


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-dir", type=Path, default=Path("data/db/db_ur50_tmvec/chunks"))
    ap.add_argument("--split-dir", type=Path, default=Path("data/uniref50/tmvec_chunks"),
                    help="dir of chunk_*.fasta (to check completeness)")
    ap.add_argument("--out-prefix", default="data/db/db_ur50_tmvec/ur50_embedding")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="merge whatever chunk npys exist (skip the completeness check)")
    ap.add_argument("--delete-chunks-after", action="store_true",
                    help="after a fully-verified merge, delete the per-chunk .npy/.ids.txt to "
                         "free disk (~38 GB). Only runs once the memmap+ids are complete.")
    args = ap.parse_args()

    expected = sorted(p.stem for p in args.split_dir.glob("chunk_*.fasta"))
    npys = {p.stem: p for p in args.chunk_dir.glob("chunk_*.npy")}
    if not expected:
        sys.exit(f"[fatal] no chunk_*.fasta in {args.split_dir}")
    missing = [s for s in expected if s not in npys]
    if missing and not args.allow_incomplete:
        sys.exit(f"[fatal] {len(missing)}/{len(expected)} chunks not encoded yet "
                 f"(e.g. {missing[:3]}). Wait for the encode to finish, or pass "
                 f"--allow-incomplete.")
    order = [s for s in expected if s in npys]

    # 1st pass: total N (and validate dims)
    N = 0
    for stem in order:
        a = np.load(npys[stem], mmap_mode="r")
        if a.ndim != 2 or a.shape[1] != EMB_DIM:
            sys.exit(f"[fatal] {stem}: bad shape {a.shape}")
        N += a.shape[0]
    print(f"[merge] {len(order)} chunks, N={N:,} x {EMB_DIM} (f16 -> "
          f"{N*EMB_DIM*2/1e9:.1f} GB)", flush=True)

    out_f16 = Path(f"{args.out_prefix}.f16")
    out_ids = Path(f"{args.out_prefix}.ids.txt")
    out_shape = Path(f"{args.out_prefix}.shape.txt")
    out_f16.parent.mkdir(parents=True, exist_ok=True)

    mm = np.memmap(out_f16, dtype=np.float16, mode="w+", shape=(N, EMB_DIM))
    row = 0
    with open(out_ids, "w") as fids:
        for i, stem in enumerate(order):
            a = np.load(npys[stem]).astype(np.float16, copy=False)
            n = a.shape[0]
            mm[row:row + n] = a
            row += n
            ids = (args.chunk_dir / f"{stem}.ids.txt").read_text().split("\n")
            ids = [x for x in ids if x]
            if len(ids) != n:
                sys.exit(f"[fatal] {stem}: {len(ids)} ids vs {n} rows")
            fids.write("\n".join(ids) + "\n")
            if i % 20 == 0:
                print(f"  merged {i+1}/{len(order)} chunks, {row:,} rows", flush=True)
    mm.flush()
    del mm
    out_shape.write_text(f"{N} {EMB_DIM}\n")
    print(f"[done] {out_f16} ({N:,} x {EMB_DIM})  +  {out_ids.name}  +  {out_shape.name}",
          flush=True)

    if args.delete_chunks_after:

        n_ids = sum(1 for _ in open(out_ids))
        exp_bytes = N * EMB_DIM * 2
        if out_f16.stat().st_size != exp_bytes or n_ids != N:
            sys.exit(f"[fatal] refusing to delete chunks: merge looks incomplete "
                     f"(f16 {out_f16.stat().st_size} vs {exp_bytes} bytes, ids {n_ids} vs {N})")
        freed = 0
        for stem in order:
            for p in (npys[stem], args.chunk_dir / f"{stem}.ids.txt"):
                if p.exists():
                    freed += p.stat().st_size
                    p.unlink()
        print(f"[cleanup] deleted {len(order)} chunk .npy/.ids.txt, freed {freed/1e9:.1f} GB",
              flush=True)


if __name__ == "__main__":
    main()
