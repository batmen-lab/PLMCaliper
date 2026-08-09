import argparse
import glob
import os
from pathlib import Path

import faiss
import pandas as pd
import torch
from pyarrow import csv as pacsv


def load_vec(pt_path: Path):
    obj = torch.load(str(pt_path), map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)):
        if len(obj) == 1 and isinstance(obj[0], torch.Tensor):
            return obj[0]
        return torch.cat(list(obj), dim=0)
    if isinstance(obj, dict):
        tensors = [v for v in obj.values() if isinstance(v, torch.Tensor)]
        if not tensors:
            raise TypeError(f"no tensors in {pt_path}")
        return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
    raise TypeError(f"unsupported type {type(obj)} in {pt_path}")


def shard_pt(ebd_base: Path, shard_name: str) -> Path:
    cands = sorted(glob.glob(str(ebd_base / shard_name / "ebd" / "*" / "*.pt")))
    if not cands:
        raise FileNotFoundError(f"no .pt under {ebd_base/shard_name}/ebd/")
    return Path(cands[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-tsv-glob", required=True)
    ap.add_argument("--ebd-base", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dim", type=int, default=480)
    args = ap.parse_args()

    tsvs = sorted(glob.glob(args.shard_tsv_glob))
    if not tsvs:
        raise SystemExit(f"no shard TSVs match {args.shard_tsv_glob}")
    print(f"[agg] {len(tsvs)} shards in order:")

    index = faiss.IndexFlatL2(args.dim)
    df_parts = []
    for tsv in tsvs:
        name = Path(tsv).stem  # shard_00
        pt = shard_pt(args.ebd_base, name)
        vec = load_vec(pt)
        df = pacsv.read_csv(tsv, read_options=pacsv.ReadOptions(column_names=["id", "sequence"]),
                            parse_options=pacsv.ParseOptions(delimiter="\t")).to_pandas()
        if vec.shape[0] != len(df):
            raise SystemExit(f"[FATAL] {name}: {vec.shape[0]} embeddings vs {len(df)} seqs — misaligned!")
        index.add(vec.cpu().numpy())
        df_parts.append(df)
        print(f"  {name}: {vec.shape[0]:,} vecs (dim {vec.shape[1]})")

    args.out.mkdir(parents=True, exist_ok=True)
    full = pd.concat(df_parts, ignore_index=True)
    full.to_pickle(str(args.out / "df-ebd.pkl"))
    faiss.write_index(index, str(args.out / "index-ebd.index"))
    print(f"[OK] index size {index.ntotal:,} -> {args.out}/index-ebd.index")


if __name__ == "__main__":
    main()
