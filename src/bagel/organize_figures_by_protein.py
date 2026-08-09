import argparse
import glob
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BAGEL_DIR = Path(__file__).resolve().parent
BASE_DIR = BAGEL_DIR.parents[1]
sys.path.insert(0, str(BAGEL_DIR))
from plot_alignment_heatmap import (  # noqa: E402  (reuse the exact heatmap renderer)
    AlignmentPath,
    PairInfo,
    plot_heatmap,
    sanitize_filename,
)

DECOY_METHOD = "extended_shuf"
Q_LEVELS = (0.10, 0.20)

# user-facing label -> on-disk method name
METHOD_DISK = {"plm": "plm", "tmvec": "tmvec", "dhr": "dhr_postprocess"}

LIST_DIR = BAGEL_DIR / "protein_lists"
PUTATIVE_FA = BASE_DIR / "data" / "putative.fa"
OUT_ROOT = BASE_DIR / "results" / "bagel" / "figures_by_protein"


def heatmap_root(disk_method: str) -> Path:
    return BASE_DIR / "results" / "bagel" / f"alignment_heatmaps_all_hits_{disk_method}"


def tsne_root(disk_method: str) -> Path:
    return BASE_DIR / "results_final" / "bagel" / f"tsne_per_query_{DECOY_METHOD}_{disk_method}"


def read_prefixes(list_path: Path) -> list[str]:
    ids = []
    for line in list_path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def load_prefix_to_qid(fasta: Path) -> dict[str, str]:
    mapping = {}
    for line in fasta.read_text().splitlines():
        if line.startswith(">"):
            qid = line[1:].split()[0]
            mapping[qid.split(";")[0]] = qid
    return mapping


def find_tsne_pdf(disk_method: str, prefix: str) -> Path | None:
    root = tsne_root(disk_method)
    hits = sorted(glob.glob(str(root / "q_*" / f"tsne_{DECOY_METHOD}_{prefix}_*.pdf")))
    return Path(hits[0]) if hits else None


def render_heatmaps_for_qid(index_df: pd.DataFrame, hm_root: Path, qid: str,
                            out_dir: Path) -> int:
    rows = index_df[index_df["qid"].astype(str) == qid].copy()
    rows["q"] = rows["q"].astype(float)
    eps = 1e-6
    n = 0
    for q in Q_LEVELS:
        rows_q = rows[rows["q"] <= q + eps].sort_values(
            ["q", "rank"], na_position="last"
        )
        for seq, row in enumerate(rows_q.itertuples(index=False), start=1):
            heat = np.load(hm_root / row.heat_npy)
            aln = None
            path_npy = getattr(row, "path_npy", "")
            if isinstance(path_npy, str) and path_npy:
                arr = np.load(hm_root / path_npy)
                aln = AlignmentPath(pairs=[(int(i), int(j)) for i, j in arr])
            pair = PairInfo(
                qid=str(row.qid), tid=str(row.tid),
                rank=None if pd.isna(row.rank) else int(row.rank),
                q=None if pd.isna(row.q) else float(row.q),
            )
            # running index keeps names unique/ordered within the cumulative set
            stem = f"deepblast_q{q:.2f}_{seq:02d}__{sanitize_filename(str(row.tid))}"
            label = row.label if isinstance(row.label, str) else "Residue alignment probability"
            plot_heatmap(
                heat, pair=pair, out_prefix=out_dir / stem, formats=["pdf"], dpi=300,
                label=label, aln=aln,
                crop_bounds=(int(row.q0), int(row.q1), int(row.t0), int(row.t1)),
                draw_path=True,
            )
            n += 1
    return n


def organize_method(label: str, prefix2qid: dict[str, str]) -> None:
    disk = METHOD_DISK[label]
    prefixes = read_prefixes(LIST_DIR / f"{label}.txt")
    index_path = heatmap_root(disk) / "index.csv"
    index_df = pd.read_csv(index_path) if index_path.exists() else pd.DataFrame()
    print(f"\n=== {label} ({disk}): {len(prefixes)} proteins ===")

    for prefix in prefixes:
        qid = prefix2qid.get(prefix)
        if qid is None:
            print(f"  [WARN] {prefix}: not found in {PUTATIVE_FA.name}; skip")
            continue
        out_dir = OUT_ROOT / label / sanitize_filename(qid)
        out_dir.mkdir(parents=True, exist_ok=True)

        tsne_pdf = find_tsne_pdf(disk, prefix)
        if tsne_pdf is not None:
            shutil.copyfile(tsne_pdf, out_dir / "tsne.pdf")
            tsne_msg = f"tsne <- {tsne_pdf.parent.name}/"
        else:
            tsne_msg = "tsne MISSING"

        n_hm = 0
        if not index_df.empty:
            n_hm = render_heatmaps_for_qid(index_df, heatmap_root(disk), qid, out_dir)
        print(f"  {prefix:8s} {qid:45s} {tsne_msg:22s} heatmaps={n_hm}")


def emit_qidlists(label: str, prefix2qid: dict[str, str]) -> Path:
    prefixes = read_prefixes(LIST_DIR / f"{label}.txt")
    qids = [prefix2qid[p] for p in prefixes if p in prefix2qid]
    out = LIST_DIR / f"{label}_qids.txt"
    out.write_text("\n".join(qids) + "\n")
    print(f"[OK] {label}: {len(qids)} qids -> {out}")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["plm", "tmvec", "dhr"],
                   choices=("plm", "tmvec", "dhr"))
    p.add_argument("--emit-qidlists", action="store_true",
                   help="Only expand the protein lists to full-qid files "
                        "(<method>_qids.txt) for compute_alignment_heatmap, then exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prefix2qid = load_prefix_to_qid(PUTATIVE_FA)
    if args.emit_qidlists:
        for label in args.methods:
            emit_qidlists(label, prefix2qid)
        return
    for label in args.methods:
        organize_method(label, prefix2qid)
    print(f"\n[DONE] figures under {OUT_ROOT}")


if __name__ == "__main__":
    main()
