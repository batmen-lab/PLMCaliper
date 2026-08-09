import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DB_DIR = DATA_DIR / "db"

TMVEC_SRC = BASE_DIR / "libs" / "tm-vec-master"
TMVEC_MODEL = TMVEC_SRC / "model" / "tm_vec_cath_model.ckpt"
TMVEC_CONFIG = TMVEC_SRC / "model" / "tm_vec_cath_model_params.json"
PROTRANS_MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"
BUILD_DB_SCRIPT = TMVEC_SRC / "scripts" / "tmvec-build-database"

CLASS_FILE = DATA_DIR / "bagel_class.txt"


def build_or_load_embeddings(fasta: Path, db_out: Path, device: str) -> tuple[np.ndarray, list[str]]:
    db_npy, meta_npy = db_out / "db.npy", db_out / "meta.npy"
    if not (db_npy.exists() and meta_npy.exists()):
        print(f"[build] TM-Vec embedding {fasta.name} -> {db_out}")
        cmd = [
            sys.executable, str(BUILD_DB_SCRIPT),
            "--input-fasta", str(fasta),
            "--tm-vec-model", str(TMVEC_MODEL),
            "--tm-vec-config-path", str(TMVEC_CONFIG),
            "--protrans-model", PROTRANS_MODEL,
            "--device", device,
            "--output", str(db_out),
        ]
        subprocess.run(cmd, check=True)
    else:
        print(f"[cache] reusing {db_out}")
    emb = np.load(db_npy)
    ids = [str(x) for x in np.load(meta_npy, allow_pickle=True)]
    print(f"        {fasta.name}: {emb.shape[0]} seqs, dim={emb.shape[1]}")
    return emb, ids


def load_id2class() -> dict[str, str]:
    import pandas as pd
    df = pd.read_csv(CLASS_FILE, sep="\t")
    m = {"ClassI": "1", "ClassII": "2", "ClassIII": "3"}
    code = df["class"].astype(str).map(m).fillna(df["class"].astype(str))
    return dict(zip(df["ids"].astype(str), code))


def run_tsne(X: np.ndarray, *, scale: bool, perplexity: float, seed: int,
             init: str, learning_rate) -> np.ndarray:
    from sklearn.manifold import TSNE
    if scale:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)
    print(f"[tsne] N={X.shape[0]} dim={X.shape[1]} perplexity={perplexity} "
          f"seed={seed} init={init} scale={scale}")
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate,
                init=init, random_state=seed)
    return tsne.fit_transform(X)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=DATA_DIR,
                   help="Where to write X_2d_all/seqs_all/labels_all.npy (default: data/; "
                        "point elsewhere first to avoid overwriting the working file).")
    p.add_argument("--putative-fa", type=Path, default=DATA_DIR / "putative.fa")
    p.add_argument("--class-all-fa", type=Path, default=DATA_DIR / "class_all.fa")
    p.add_argument("--seed", type=int, default=123, help="t-SNE random_state (default 123).")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--init", default="pca", choices=("pca", "random"))
    p.add_argument("--learning-rate", default="auto")
    p.add_argument("--no-scale", action="store_true", help="Skip StandardScaler before t-SNE.")
    p.add_argument("--device", default="gpu", choices=("gpu", "cpu"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) embeddings (putative first, then class_all — matches seqs_all ordering)
    emb_p, ids_p = build_or_load_embeddings(
        args.putative_fa, DB_DIR / "db_putative_tmvec", args.device)
    emb_c, ids_c = build_or_load_embeddings(
        args.class_all_fa, DB_DIR / "db_class_all_tmvec", args.device)

    X = np.vstack([emb_p, emb_c])
    ids = ids_p + ids_c

    # 2) labels: putative vs BAGEL class
    id2class = load_id2class()
    labels = (["Putative"] * len(ids_p)
              + [f"Class {id2class.get(i, '?')}" for i in ids_c])

    # 3) t-SNE
    X_2d = run_tsne(X, scale=not args.no_scale, perplexity=args.perplexity,
                    seed=args.seed, init=args.init, learning_rate=args.learning_rate)

    # 4) save
    np.save(args.out_dir / "X_2d_all.npy", X_2d.astype(np.float32))
    np.save(args.out_dir / "seqs_all.npy", np.asarray(ids, dtype=object))
    np.save(args.out_dir / "labels_all.npy", np.asarray(labels, dtype=object))
    print(f"[OK] wrote X_2d_all/seqs_all/labels_all.npy ({X_2d.shape[0]} points) -> {args.out_dir}")


if __name__ == "__main__":
    main()
