import sys
from pathlib import Path
import numpy as np
import pandas as pd
import config

BAGEL_DIR = config.PROJECT_ROOT / "src" / "bagel"
sys.path.insert(0, str(BAGEL_DIR))
from compute_alignment_heatmap import (  # noqa: E402
    DeepBlastRunner,
    read_fasta,
    sanitize_filename,
)

DEFAULT_FASTA = config.fasta_path("astral")
DEFAULT_OUTDIR = config.PROJECT_ROOT / "results" / "case_study" / "deepblast_heatmaps"
DEEPBLAST_MODEL = (
    config.PROJECT_ROOT / "libs" / "tm-vec-master" / "model" / "deepblast-v3.ckpt"
)
PROTRANS_MODEL = "Rostlab/prot_t5_xl_uniref50"

INDEX_COLUMNS = ["query", "target", "qlen", "tlen", "heat_npy", "path_npy", "stem"]


def run(
    pairs,
    *,
    fasta: Path = DEFAULT_FASTA,
    outdir: Path = DEFAULT_OUTDIR,
    device: str = "auto",
) -> None:
    records = read_fasta(fasta)
    runner = DeepBlastRunner(
        model_path=DEEPBLAST_MODEL,
        protrans_model=PROTRANS_MODEL,
        alignment_mode="needleman-wunsch",
        device=device,
    )
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for query, target in pairs:
        if query not in records:
            raise KeyError(f"qid not found in {fasta}: {query}")
        if target not in records:
            raise KeyError(f"tid not found in {fasta}: {target}")
        print(f"\n=== DeepBLAST compute: {query} -> {target} ===")
        heat, aln = runner.align_pair(
            records[query], records[target], cache_key=(query, target)
        )

        stem = f"deepblast_{sanitize_filename(query)}_{sanitize_filename(target)}"
        heat_npy = outdir / f"{stem}.npy"
        np.save(heat_npy, heat.astype(np.float32, copy=False))

        path_name = ""
        if aln is not None and aln.pairs:
            path_npy = outdir / f"{stem}.path.npy"
            np.save(path_npy, np.asarray(aln.pairs, dtype=np.int32))
            path_name = path_npy.name

        rows.append({
            "query": query, "target": target,
            "qlen": len(records[query]), "tlen": len(records[target]),
            "heat_npy": heat_npy.name, "path_npy": path_name, "stem": stem,
        })
        print(f"[OK] {stem}.npy  ({heat.shape[0]}x{heat.shape[1]})")

    index_path = outdir / "index.csv"
    pd.DataFrame(rows, columns=INDEX_COLUMNS).to_csv(index_path, index=False)
    print(f"\n[OK] wrote {len(rows)} matrix(es) -> {outdir}")
    print(f"[OK] index -> {index_path}")
    print("     Next: python src/case_study/plot_deepblast_heatmap.py")


if __name__ == "__main__":
    # ---- edit this list of (query, target) pairs ----
    PAIRS = [
        ("d3tg7a1", "d3ku3a1"),
        ("d5awwy_", "d2a65a1"),
        ("d4zgyb_", "d1nsla1"),
        ("d4zgyb_", "d2g3aa1"),
        ("d4zgyb_", "d2ge3a1"),
        ("d4zgyb_", "d3igra_"),
    ]
    run(PAIRS)
