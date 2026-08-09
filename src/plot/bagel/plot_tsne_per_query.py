#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style  # noqa: F401 - shared Nature rcParams (font, sizes, linewidths)
from style import FONT_LABEL, FONT_LEGEND, FONT_TICK, FONT_TITLE
from paths import experiment_dir

DATA_DIR = experiment_dir("bagel") / "tsne_per_query"

# --- styling, ported verbatim from src/bagel/classify_putative.py ---
CLASS_MARKERS = {"1": "o", "2": "s", "3": "^"}
CLASS_COLORS = {"1": "#0173B2", "2": "#DE8F05", "3": "#029E73", "Unknown": "#BBBBBB"}
Q_BAND_CMAP = "plasma"
ACCEPTED_HIT_SIZE = 18
QUERY_STAR_SIZE = 90


def class_from_qid(qid: str) -> str:
    parts = qid.split(";", 1)[0].split(".")
    return parts[1] if len(parts) >= 2 and parts[1] in {"1", "2", "3"} else "?"


def safe_filename(text: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", str(text))


def _short_qid(qid: str) -> str:
    left = qid.split(";", 1)[0]
    rest = qid.split(";", 1)[1] if ";" in qid else qid
    if len(rest) > 36:
        rest = rest[:33] + "..."
    return f"{left};{rest}" if ";" in qid else left


def min_discoverable_q(hits_by_q, q_levels):
    for q in sorted(q_levels):
        if hits_by_q.get(q, set()):
            return q
    return None


def q_discovery_folder_name(q_min, q_levels):
    if q_min is not None:
        return f"q_{q_min:.2f}"
    return f"no_hit_above_q{max(q_levels):.2f}"


def partition_hits_by_first_q(hits_by_q, q_levels):
    q_levels = sorted(q_levels)
    tid2first_q, new_by_q, seen = {}, {q: set() for q in q_levels}, set()
    for q in q_levels:
        at_q = hits_by_q.get(q, set())
        new_tids = at_q - seen
        new_by_q[q] = new_tids
        for tid in new_tids:
            tid2first_q[tid] = q
        seen |= at_q
    return tid2first_q, new_by_q


def _q_band_colors(q_levels):
    cmap = plt.get_cmap(Q_BAND_CMAP)
    n = len(q_levels)
    return {q: cmap(i / max(n - 1, 1)) for i, q in enumerate(sorted(q_levels))}


def _plot_q_color_bars(ax_qbar, q_levels, q_colors, new_by_q, *,
                       label_fontsize=FONT_TICK, title_fontsize=FONT_TITLE):
    q_levels = sorted(q_levels)
    counts = [len(new_by_q.get(q, set())) for q in q_levels]
    y_pos = np.arange(len(q_levels))
    ax_qbar.barh(y_pos, counts, color=[q_colors[q] for q in q_levels],
                 edgecolor="black", linewidth=0.4, height=0.72)
    ax_qbar.set_yticks(y_pos)
    ax_qbar.set_yticklabels([f"q = {q:.2f}" for q in q_levels], fontsize=label_fontsize)
    ax_qbar.set_title("q band colors", fontsize=title_fontsize)
    ax_qbar.invert_yaxis()
    if max(counts, default=0) == 0:
        ax_qbar.set_xlim(0, 1)
        ax_qbar.text(0.5, 0.5, "no hits", transform=ax_qbar.transAxes,
                     ha="center", va="center", fontsize=label_fontsize, color="gray")


def _draw_query_tsne_q_bands(ax, ax_qbar, qid, hits_by_q, q_levels, X_2d, seq2idx,
                             id2class, *, bg_size=18, label_fontsize=FONT_TICK) -> bool:
    if qid not in seq2idx:
        return False
    q_levels = sorted(q_levels)
    q_colors = _q_band_colors(q_levels)
    tid2first_q, new_by_q = partition_hits_by_first_q(hits_by_q, q_levels)
    highlighted = set(tid2first_q.keys())

    for cls, marker in CLASS_MARKERS.items():
        idxs = [seq2idx[s] for s in seq2idx
                if s != qid and id2class.get(s, "Unknown") == cls and s not in highlighted]
        if idxs:
            idxs = np.array(idxs)
            ax.scatter(X_2d[idxs, 0], X_2d[idxs, 1], c=CLASS_COLORS[cls], marker=marker,
                       s=bg_size, alpha=0.18, linewidths=0, zorder=1)

    for q in q_levels:
        for cls, marker in CLASS_MARKERS.items():
            idxs = [seq2idx[tid] for tid in new_by_q.get(q, set())
                    if tid in seq2idx and id2class.get(tid, "Unknown") == cls]
            if idxs:
                idxs = np.array(idxs)
                ax.scatter(X_2d[idxs, 0], X_2d[idxs, 1], c=[q_colors[q]], marker=marker,
                           s=ACCEPTED_HIT_SIZE, alpha=0.90, edgecolors="black",
                           linewidths=0.35, zorder=10)

    qi = seq2idx[qid]
    ax.scatter(X_2d[qi, 0], X_2d[qi, 1], c="red", marker="*", s=QUERY_STAR_SIZE,
               edgecolors="black", linewidths=0.5, zorder=100)
    ax.set_title(_short_qid(qid), fontsize=FONT_TITLE, fontweight="bold")
    ax.set_xlabel("t-SNE 1", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("t-SNE 2", fontsize=FONT_LABEL, fontweight="bold")
    ax.tick_params(labelsize=FONT_TICK)
    ax.set_box_aspect(1)
    ax_qbar.set_box_aspect(1)
    _plot_q_color_bars(ax_qbar, q_levels, q_colors, new_by_q,
                       label_fontsize=FONT_TICK, title_fontsize=FONT_TITLE)
    ax_qbar.set_xlabel("# new hits", fontsize=FONT_TICK)
    ax_qbar.tick_params(labelsize=FONT_TICK)
    return True


def plot_query(qid, hits_by_q, q_levels, X_2d, ids, id2class, out_path: Path,
               dpi: int = 400) -> bool:
    seq2idx = {s: i for i, s in enumerate(ids)}
    if qid not in seq2idx:
        print(f"  [WARN] {qid} not in ids; skip")
        return False

    fig = plt.figure(figsize=(7.2, 5.0))  # Nature double column
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.0], wspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    ax_qbar = fig.add_subplot(gs[0, 1])
    _draw_query_tsne_q_bands(ax, ax_qbar, qid, hits_by_q, q_levels, X_2d, seq2idx, id2class)

    handles = [
        mlines.Line2D([], [], marker=CLASS_MARKERS[c], linestyle="None",
                      markerfacecolor=CLASS_COLORS[c], markeredgecolor=CLASS_COLORS[c],
                      markersize=5, alpha=0.7, label=f"Class {c}")
        for c in ("1", "2", "3")
    ] + [mlines.Line2D([], [], marker="*", linestyle="None", markerfacecolor="red",
                       markeredgecolor="black", markersize=8, label="query")]

    fig.subplots_adjust(bottom=0.20)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.06),
               ncol=4, fontsize=FONT_LEGEND, frameon=True)
    fig.text(0.5, 0.01, f"BAGEL class {class_from_qid(qid)}", ha="center", fontsize=FONT_TICK)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format="pdf")
    plt.close(fig)
    return True


def load_data(data_dir: Path):
    coords = np.load(data_dir / "coords.npy")
    ids = np.asarray((data_dir / "ids.txt").read_text().split("\n"))
    ids = np.array([s for s in ids if s]).astype(str)
    dfc = pd.read_csv(data_dir / "id2class.tsv", sep="\t", dtype=str)
    id2class = dict(zip(dfc["id"], dfc["class"]))
    meta = json.loads((data_dir / "meta.json").read_text())
    return coords, ids, id2class, meta


def hits_by_q_per_query(data_dir: Path, label: str, q_levels):
    df = pd.read_csv(data_dir / f"hits_{label}.tsv", sep="\t", dtype={"qid": str, "tid": str})
    out = {}
    for qid, df_q in df.groupby("qid"):
        by_q = {q: set() for q in q_levels}
        for q, df_qq in df_q.groupby("q"):
            for ql in q_levels:
                if np.isclose(float(q), ql):
                    by_q[ql] = set(df_qq["tid"])
        out[str(qid)] = by_q
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=None,
                   help="Subset of methods (default: all in meta.json).")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--plot-dir", type=Path, default=DATA_DIR / "figures")
    p.add_argument("--decoy-method", default="extended_shuf")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    coords, ids, id2class, meta = load_data(args.data_dir)
    q_levels = meta["q_levels"]
    methods = args.methods or meta["methods"]

    for label in methods:
        hits_file = args.data_dir / f"hits_{label}.tsv"
        if not hits_file.exists():
            print(f"[skip] {label}: no {hits_file.name}")
            continue
        hits = hits_by_q_per_query(args.data_dir, label, q_levels)
        print(f"\n=== {label}: {len(hits)} queries ===")
        n_ok = 0
        for qid, hits_by_q in hits.items():
            q_min = min_discoverable_q(hits_by_q, q_levels)
            subdir = q_discovery_folder_name(q_min, q_levels)
            out_path = (args.plot_dir / label / subdir
                        / f"tsne_{args.decoy_method}_{safe_filename(qid)}.pdf")
            n_ok += int(plot_query(qid, hits_by_q, q_levels, coords, ids, id2class, out_path))
        print(f"[OK] {label}: {n_ok} figures -> {args.plot_dir / label}")


if __name__ == "__main__":
    main()
