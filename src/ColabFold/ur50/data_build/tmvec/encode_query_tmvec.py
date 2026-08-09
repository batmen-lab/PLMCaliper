#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encode_ur50_tmvec_worker import DEFAULT_MAX_LEN, encode_chunk, load_models


def read_query_tsv(path: Path):
    ids, seqs = [], []
    for line in open(path):
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        ids.append(parts[0])
        seqs.append(parts[1].replace("*", ""))
    return ids, seqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-tsv", required=True, type=Path)
    ap.add_argument("--out-prefix", required=True,
                    help="writes <prefix>.npy [Q,512] f32 and <prefix>.ids.txt")
    ap.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("[fatal] CUDA not available")
    device = torch.device("cuda:0")  # pinned by CUDA_VISIBLE_DEVICES

    ids, seqs = read_query_tsv(args.query_tsv)
    print(f"[query] {len(ids)} sequences from {args.query_tsv}", flush=True)
    tokenizer, model, model_deep = load_models(device)
    emb, n_clip, n_fail = encode_chunk(seqs, model_deep, model, tokenizer, device,
                                       args.max_len)
    emb = np.ascontiguousarray(emb, dtype=np.float32)   # keep query at f32

    out_npy = Path(f"{args.out_prefix}.npy")
    out_ids = Path(f"{args.out_prefix}.ids.txt")
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, emb)
    out_ids.write_text("\n".join(ids) + "\n")
    print(f"[done] {emb.shape} -> {out_npy}  (clip>{args.max_len}={n_clip} fail={n_fail})",
          flush=True)


if __name__ == "__main__":
    main()
