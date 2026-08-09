from pathlib import Path

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

DEFAULT_OUTDIR = config.PROJECT_ROOT / "results" / "case_study" / "deepblast_heatmaps"


def sequence_ticks(length: int, max_ticks: int = 9) -> list[int]:
    if length <= 0:
        return []
    if length == 1:
        return [0]
    ticks = np.linspace(0, length - 1, num=min(max_ticks, length))
    ticks = sorted({int(round(x)) for x in ticks})
    if ticks[0] != 0:
        ticks.insert(0, 0)
    if ticks[-1] != length - 1:
        ticks.append(length - 1)
    return ticks


def plot_heatmap(heat, path_pairs, query: str, target: str, outdir: Path, stem: str,
                 *, formats=("pdf",), dpi: int = 400, draw_path: bool = True) -> None:
    fig_w = max(4.8, min(8.2, 2.5 + 0.09 * heat.shape[1]))
    fig_h = max(4.6, min(8.2, 2.4 + 0.09 * heat.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(heat, origin="upper", aspect="auto", interpolation="nearest",
                   cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_xlim(-0.5, heat.shape[1] - 0.5)
    ax.set_ylim(heat.shape[0] - 0.5, -0.5)
    ax.set_xticks(sequence_ticks(heat.shape[1]))
    ax.set_yticks(sequence_ticks(heat.shape[0]))

    if draw_path and path_pairs is not None and len(path_pairs):
        ax.plot(path_pairs[:, 1], path_pairs[:, 0], color="white", linewidth=0.8, alpha=0.85)

    ax.set_xlabel(f"{target}  (Target residue)", fontsize=13)
    ax.set_ylabel(f"{query}  (Query residue)", fontsize=13)
    ax.set_title("DeepBLAST residue alignment", fontsize=14)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Residue alignment probability", rotation=270, labelpad=18, fontsize=13)

    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = outdir / f"{stem}.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[OK] heatmap -> {out}")
    plt.close(fig)


def run(outdir: Path = DEFAULT_OUTDIR, *, formats=("pdf",), draw_path: bool = False,
        dpi: int = 400) -> None:
    index_path = outdir / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing {index_path}. Run: python src/case_study/compute_deepblast_heatmap.py"
        )
    df = pd.read_csv(index_path)
    for row in df.itertuples(index=False):
        heat = np.load(outdir / row.heat_npy)
        path_pairs = None
        if isinstance(row.path_npy, str) and row.path_npy:
            path_pairs = np.load(outdir / row.path_npy)
        plot_heatmap(heat, path_pairs, str(row.query), str(row.target), outdir, str(row.stem),
                     formats=formats, dpi=dpi, draw_path=draw_path)


if __name__ == "__main__":
    run(formats=("pdf",))
