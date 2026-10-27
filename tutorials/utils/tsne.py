import subprocess
import sys
from pathlib import Path
import numpy as np
from sklearn.manifold import TSNE

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = DATA_DIR / "bagel"
TMVEC_DIR = REPO_ROOT / "libs" / "tm-vec-master"

def embed(name):
    # TM-Vec embeddings (ProtT5 -> TM-Vec)
    db = DATA_DIR / "db" / f"db_{name}_tmvec"
    if not db.exists():
        db.parent.mkdir(parents=True, exist_ok=True)
        # run libs/tm-vec-master/scripts/tmvec-build-database
        subprocess.run([sys.executable, TMVEC_DIR / "scripts" / "tmvec-build-database",
                        "--input-fasta", DATA_DIR / f"{name}.fa",
                        "--tm-vec-model", TMVEC_DIR / "model" / "tm_vec_cath_model.ckpt",
                        "--tm-vec-config-path", TMVEC_DIR / "model" / "tm_vec_cath_model_params.json",
                        "--protrans-model", "Rostlab/prot_t5_xl_half_uniref50-enc",
                        "--device", "gpu", "--output", db], check=True)
    emb = np.load(db / "db.npy")
    return emb / np.linalg.norm(emb, axis=1, keepdims=True), np.load(db / "meta.npy").astype(str)

def run_tsne():
    # t-SNE of the putative queries
    (emb_p, ids_p), (emb_c, ids_c) = embed("putative"), embed("class_all")
    xy = TSNE(n_components=2, perplexity=15, init="random", learning_rate=200, random_state=0).fit_transform(np.vstack([emb_p, emb_c]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "X_2d_all.npy", xy.astype(np.float32))
    np.save(OUT_DIR / "seqs_all.npy", np.concatenate([ids_p, ids_c]))
    np.save(OUT_DIR / "labels_all.npy", np.array(["Putative"] * len(ids_p) + [f"Class {i.split(';')[0].split('.')[1]}" for i in ids_c]))
    print(f"save X_2d_all/seqs_all/labels_all.npy -> {OUT_DIR}")


if __name__ == "__main__":
    if (OUT_DIR / "X_2d_all.npy").exists():
        sys.exit(f"{OUT_DIR / 'X_2d_all.npy'} exists.")
    run_tsne()
