import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUT_PARENT = BASE_DIR / "results" / "bagel"
DEFAULT_SEARCH_METHOD = "plm"


@dataclass(frozen=True)
class PairInfo:
    qid: str
    tid: str
    rank: int | None = None
    q: float | None = None


@dataclass(frozen=True)
class AlignmentPath:
    pairs: list[tuple[int, int]] = field(default_factory=list)


def sanitize_filename(text: str, max_len: int = 90) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:max_len] if len(text) > max_len else text


def append_extension(path: Path, extension: str) -> Path:
    extension = extension if extension.startswith(".") else f".{extension}"
    return path.parent / f"{path.name}{extension}"


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


def parse_bagel_label(seq_id: str) -> tuple[str, str, str]:
    left, _, right = seq_id.partition(";")
    parts = left.split(".")
    class_code = parts[1] if len(parts) >= 2 else "?"
    class_label = f"Class {class_code}" if class_code in {"1", "2", "3"} else "Class ?"
    display_name = right if right else seq_id
    return class_code, class_label, display_name


def short_axis_label(seq_id: str, *, prefix: str = "") -> str:
    _, class_label, display_name = parse_bagel_label(seq_id)
    if prefix:
        return f"{prefix}\n{seq_id}"
    return f"{class_label}, {display_name}"


def plot_heatmap(
    heat: np.ndarray,
    *,
    pair: PairInfo,
    out_prefix: Path,
    formats: list[str],
    dpi: int,
    label: str,
    aln: AlignmentPath | None,
    crop_bounds: tuple[int, int, int, int],
    draw_path: bool,
) -> None:
    q0, q1, t0, t1 = crop_bounds
    fig_w = max(4.8, min(8.2, 2.5 + 0.09 * heat.shape[1]))
    fig_h = max(4.6, min(8.2, 2.4 + 0.09 * heat.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        heat,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xlim(-0.5, heat.shape[1] - 0.5)
    ax.set_ylim(heat.shape[0] - 0.5, -0.5)
    xticks = sequence_ticks(heat.shape[1])
    yticks = sequence_ticks(heat.shape[0])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([str(int(x + t0)) if x >= 0 else "" for x in xticks])
    ax.set_yticklabels([str(int(y + q0)) if y >= 0 else "" for y in yticks])

    if draw_path and aln is not None and aln.pairs:
        xs = [j - t0 for _, j in aln.pairs if t0 <= j < t1]
        ys = [i - q0 for i, _ in aln.pairs if q0 <= i < q1]
        if len(xs) == len(ys) and xs:
            ax.plot(xs, ys, color="white", linewidth=0.8, alpha=0.85)

    rank_label = f"Near neighbor {pair.rank}" if pair.rank is not None else "BAGEL pair"
    if pair.q is not None:
        rank_label = f"Target FDR threshold = {pair.q:.2f} | {rank_label}"
    ax.set_title(rank_label, fontsize=14)
    ax.set_ylabel(short_axis_label(pair.qid, prefix="Putative bacteriocin"), fontsize=13)
    ax.set_xlabel(short_axis_label(pair.tid), fontsize=13)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, rotation=270, labelpad=18, fontsize=13)

    fig.tight_layout()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = append_extension(out_prefix, fmt)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[OK] heatmap -> {out_path}")
    plt.close(fig)


def _opt_int(value) -> int | None:
    return None if pd.isna(value) else int(value)


def _opt_float(value) -> float | None:
    return None if pd.isna(value) else float(value)


def render_from_index(index_path: Path, *, formats: list[str], dpi: int, draw_path: bool) -> int:
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing heatmap index: {index_path}\n"
            f"Run: python compute_alignment_heatmap.py ..."
        )
    root = index_path.parent
    df = pd.read_csv(index_path)
    if df.empty:
        print(f"[INFO] {index_path} has no rows; nothing to plot")
        return 0

    n = 0
    for row in df.itertuples(index=False):
        heat = np.load(root / row.heat_npy)

        aln: AlignmentPath | None = None
        path_npy = getattr(row, "path_npy", "")
        if isinstance(path_npy, str) and path_npy:
            arr = np.load(root / path_npy)
            aln = AlignmentPath(pairs=[(int(i), int(j)) for i, j in arr])

        pair = PairInfo(
            qid=str(row.qid),
            tid=str(row.tid),
            rank=_opt_int(row.rank),
            q=_opt_float(row.q),
        )
        crop_bounds = (int(row.q0), int(row.q1), int(row.t0), int(row.t1))
        label = row.label if isinstance(row.label, str) else "Residue alignment probability"

        plot_heatmap(
            heat,
            pair=pair,
            out_prefix=root / row.out_prefix,
            formats=formats,
            dpi=dpi,
            label=label,
            aln=aln,
            crop_bounds=crop_bounds,
            draw_path=draw_path,
        )
        n += 1
    return n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render BAGEL residue-alignment heatmaps from a precomputed index.csv + npy matrices.",
    )
    parser.add_argument(
        "--search-method", default=DEFAULT_SEARCH_METHOD,
        help="Used only to resolve the default index directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory holding index.csv + matrices. "
             "Default results/bagel/alignment_heatmaps_all_hits_<search-method>.",
    )
    parser.add_argument(
        "--index", type=Path, default=None,
        help="Explicit index.csv path (overrides --output-dir).",
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=("png", "pdf", "svg"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--draw-path", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.index is not None:
        index_path = args.index
    else:
        out_dir = args.output_dir or (
            DEFAULT_OUT_PARENT / f"alignment_heatmaps_all_hits_{sanitize_filename(args.search_method)}"
        )
        index_path = out_dir / "index.csv"

    print(f"[INFO] index: {index_path}")
    n = render_from_index(index_path, formats=args.formats, dpi=args.dpi, draw_path=args.draw_path)
    print(f"\n[DONE] rendered {n} heatmap(s)")


if __name__ == "__main__":
    main()
