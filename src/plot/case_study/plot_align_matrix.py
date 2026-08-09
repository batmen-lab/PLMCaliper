#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style  # noqa: F401 - shared pdf.fonttype=42 rcParams
from paths import experiment_dir

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

DEFAULT_DATA_DIR = experiment_dir("case_study") / "align_matrix"


def plot_alnmat(S: np.ndarray, path: np.ndarray, query: str, target: str, outdir: Path,
                formats=("pdf",), dpi: int = 400, draw_path: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(S, origin="upper", aspect="auto", interpolation="nearest",
                   cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xlim(-0.5, S.shape[1] - 0.5)
    ax.set_ylim(S.shape[0] - 0.5, -0.5)
    if draw_path and len(path):
        ax.plot(path[:, 1], path[:, 0], color="white", lw=0.9, alpha=0.85)
    ax.set_xlabel(f"{target}  (Target residue)")
    ax.set_ylabel(f"{query}  (Query residue)")
    ax.set_title("TM-align alignment matrix")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Structural similarity", rotation=270, labelpad=16)
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = outdir / f"alnmat_{query}-{target}.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[OK] {out}")
    plt.close(fig)


def plot_seq_alignment(query: str, target: str, seqxA: str, seqyA: str, S_lddt: np.ndarray,
                       outdir: Path, formats=("pdf",), cols: int = 60, dpi: int = 150) -> None:
    n = len(seqxA)

    # per-alignment-column lDDT (NaN where either side is a gap)
    sims = np.full(n, np.nan)
    qi = ti = 0
    for k, (a, b) in enumerate(zip(seqxA, seqyA)):
        if a != "-" and b != "-":
            sims[k] = S_lddt[qi, ti]
        if a != "-":
            qi += 1
        if b != "-":
            ti += 1

    # lDDT is a 0-1 sequential quality score -> perceptually-uniform sequential
    # map, not a diverging one (P14; diverging implies a meaningful centre).
    cmap = matplotlib.cm.viridis
    norm = Normalize(vmin=0, vmax=1)
    n_blocks = (n + cols - 1) // cols

    char_h = 1.5
    block_gap = 1.2
    label_w = 9          # chars reserved for the sequence-ID label on the left

    total_w = label_w + cols
    total_h = n_blocks * (2 * char_h + block_gap)

    scale = 0.12         # inches per character unit
    fig, ax = plt.subplots(figsize=(total_w * scale + 0.8, total_h * scale + 0.5))
    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)   # top-down y axis
    ax.axis("off")

    fs = 6.5

    for blk in range(n_blocks):
        start = blk * cols
        end = min(start + cols, n)
        y0 = blk * (2 * char_h + block_gap)

        ax.text(label_w - 0.3, y0 + char_h * 0.5, query[:label_w - 1],
                va="center", ha="right", fontsize=fs, fontfamily="monospace", color="darkorange")
        ax.text(label_w - 0.3, y0 + char_h * 1.5, target[:label_w - 1],
                va="center", ha="right", fontsize=fs, fontfamily="monospace", color="steelblue")

        for c, k in enumerate(range(start, end)):
            a, b, s = seqxA[k], seqyA[k], sims[k]
            x = label_w + c
            if not np.isnan(s):
                ax.add_patch(Rectangle((x, y0), 1, 2 * char_h,
                                       facecolor=cmap(norm(s)), edgecolor="none", zorder=0))
            ax.text(x + 0.5, y0 + char_h * 0.5, a, ha="center", va="center", fontsize=fs,
                    fontfamily="monospace", zorder=1)
            ax.text(x + 0.5, y0 + char_h * 1.5, b, ha="center", va="center", fontsize=fs,
                    fontfamily="monospace", zorder=1)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.4, pad=0.01, aspect=20)
    cb.set_label("lDDT", fontsize=8)

    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = outdir / f"seqaln_{query}-{target}.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[OK] {out}")
    plt.close(fig)


def run(data_dir: Path, plot_dir: Path, formats=("pdf",), draw_path: bool = True) -> None:
    index_path = data_dir / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}")
    df = pd.read_csv(index_path)
    for row in df.itertuples(index=False):
        query, target = str(row.query), str(row.target)
        S = np.load(data_dir / row.s_npy)
        path = np.load(data_dir / row.path_npy)
        S_lddt = np.load(data_dir / row.lddt_npy)
        print(f"\n=== render: {query} -> {target} ===")
        plot_alnmat(S, path, query, target, plot_dir, formats=formats, draw_path=draw_path)
        plot_seq_alignment(query, target, str(row.seqxA), str(row.seqyA), S_lddt,
                           plot_dir, formats=formats)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--plot-dir", type=Path, default=DEFAULT_DATA_DIR / "figures")
    p.add_argument("--formats", nargs="+", default=["pdf"], choices=("png", "pdf", "svg"))
    p.add_argument("--no-path", action="store_true", help="Do not draw the alignment path line.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.data_dir, args.plot_dir, formats=tuple(args.formats), draw_path=not args.no_path)


if __name__ == "__main__":
    main()
