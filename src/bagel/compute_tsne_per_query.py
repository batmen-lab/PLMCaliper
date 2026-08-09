import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
OUT_DIR = DATA_DIR / "plot_data" / "bagel" / "tsne_per_query"

DECOY_METHOD = "extended_shuf"
Q_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

# user-facing label -> on-disk method name (discovery table naming)
METHOD_DISK = {"plm": "plm", "tmvec": "tmvec", "dhr": "dhr_postprocess",
               "blastp": "blastp_postprocessed"}

COORDS_NPY = DATA_DIR / "X_2d_all.npy"
IDS_NPY = DATA_DIR / "seqs_all.npy"
CLASS_FILE = DATA_DIR / "bagel_class.txt"


def discovery_hits_file(disk_method: str, decoy_method: str) -> Path:
    return DATA_DIR / f"putative_discovery_hits_scan_{disk_method}_{decoy_method}.tsv"


def write_coords_and_ids() -> int:
    coords = np.load(COORDS_NPY)
    ids = np.load(IDS_NPY, allow_pickle=True).astype(str)
    if coords.shape[0] != ids.shape[0]:
        raise ValueError(f"coords/ids length mismatch: {coords.shape} vs {ids.shape}")
    np.save(OUT_DIR / "coords.npy", coords)
    (OUT_DIR / "ids.txt").write_text("\n".join(ids) + "\n")
    return len(ids)


def write_id2class() -> int:
    df = pd.read_csv(CLASS_FILE, sep="\t")
    class_map = {"ClassI": "1", "ClassII": "2", "ClassIII": "3"}
    df["class_code"] = df["class"].astype(str).map(class_map).fillna(df["class"].astype(str))
    df[["ids", "class_code"]].to_csv(OUT_DIR / "id2class.tsv", sep="\t", index=False,
                                     header=["id", "class"])
    return len(df)


def write_hits_for_method(label: str, decoy_method: str) -> int:
    disk = METHOD_DISK[label]
    path = discovery_hits_file(disk, decoy_method)
    if not path.exists():
        print(f"  [WARN] {label}: missing {path.name}; skip")
        return 0
    df = pd.read_csv(path, sep="\t")
    df["q"] = pd.to_numeric(df["q"], errors="coerce")
    df = df[df["q"].notna()]
    # keep only the q levels we plot; union tids over reps per (qid, q)
    df = df[df["q"].apply(lambda q: any(np.isclose(q, ql) for ql in Q_LEVELS))]
    hits = (
        df[["qid", "q", "tid"]]
        .astype({"qid": str, "tid": str})
        .drop_duplicates()
        .sort_values(["qid", "q", "tid"])
    )
    out = OUT_DIR / f"hits_{label}.tsv"
    hits.to_csv(out, sep="\t", index=False)
    return hits["qid"].nunique()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["plm", "tmvec", "dhr"],
                   choices=("plm", "tmvec", "dhr", "blastp"))
    p.add_argument("--decoy-method", default=DECOY_METHOD)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_pts = write_coords_and_ids()
    n_cls = write_id2class()
    print(f"[OK] coords + ids: {n_pts} points   id2class: {n_cls} targets")

    for label in args.methods:
        n_q = write_hits_for_method(label, args.decoy_method)
        print(f"[OK] hits_{label}.tsv: {n_q} queries")

    meta_path = OUT_DIR / "meta.json"
    existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    methods_all = sorted(set(existing.get("methods", [])) | set(args.methods))
    meta_path.write_text(json.dumps({
        "q_levels": Q_LEVELS,
        "decoy_method": args.decoy_method,
        "methods": methods_all,
    }, indent=2))
    print(f"\n[DONE] wrote plotting data -> {OUT_DIR}")


if __name__ == "__main__":
    main()
