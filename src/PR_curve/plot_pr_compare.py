import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_efdr_recall import (
    DATA_SUBDIR,
    Q_LABEL_LEVELS,
    RESULTS_DIR,
    combo_id,
    combo_legend_text,
    format_q_label,
    pooled_recall_precision_functions,
    q_should_label,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
FIG_SIZE = (7.0, 7.0)
FIG_DPI = 300
LINE_WIDTH = 2.0
MARKER_SIZE = 5.5
CONTOUR_COLOR = "0.55"
CONTOUR_ALPHA = 0.55
CONTOUR_STYLE = ":"
LABEL_FONTSIZE = 8
TITLE_FONTSIZE = 14
AXIS_FONTSIZE = 13
LEGEND_FONTSIZE = 11

# Wong / colorblind-friendly
COMBO_STYLES = [
    {"color": "#E69F00", "marker": "o"},   # PLM-ish gold
    {"color": "#CC79A7", "marker": "s"},   # plum
    {"color": "#0072B2", "marker": "^"}, # steel blue
    {"color": "#009E73", "marker": "D"},
    {"color": "#D55E00", "marker": "v"},
    {"color": "#56B4E9", "marker": "P"},
]


@dataclass(frozen=True)
class ComboSpec:
    search_method: str
    decoy_method: str

    @property
    def combo_id(self) -> str:
        return combo_id(self.search_method, self.decoy_method)

    @property
    def label(self) -> str:
        return combo_legend_text(self.search_method, self.decoy_method)


@dataclass
class PrSeries:
    label: str
    q: np.ndarray
    recall: np.ndarray
    precision: np.ndarray


def _find_plot_data_tsv(out_dir: Path, spec: ComboSpec) -> Path | None:
    cid = spec.combo_id
    data_dir = out_dir / DATA_SUBDIR
    search_dirs = [data_dir, out_dir]  # out_dir fallback for legacy runs
    for base in search_dirs:
        if not base.exists():
            continue
        for pattern in (
            f"pooled_pr_plot_data_{cid}_*.tsv",
            f"fdr_metrics_{cid}_*.tsv",
            f"fdr_metrics_pooled_*{spec.search_method}*{spec.decoy_method}*.tsv",
        ):
            matches = sorted(base.glob(pattern))
            if matches:
                return matches[-1]
        combined = sorted(base.glob("pooled_pr_plot_data_*.tsv"))
        if combined:
            return combined[-1]
        combined = sorted(base.glob("fdr_metrics_pooled_*.tsv"))
        if combined:
            return combined[-1]
    return None


def load_pr_series(out_dir: Path, spec: ComboSpec) -> PrSeries:
    path = _find_plot_data_tsv(out_dir, spec)
    if path is None:
        raise FileNotFoundError(
            f"No plot data for {spec.label} under {out_dir}. "
            "Run compute_efdr_recall.py first."
        )
    print(f"[INFO] load {path.name} -> {spec.label}")
    df = pd.read_csv(path, sep="\t")
    if "combo_id" in df.columns:
        cid = spec.combo_id
        sub = df[df["combo_id"] == cid]
        if sub.empty and len(df["combo_id"].unique()) > 1:
            raise ValueError(f"combo_id={cid} not in {path.name}")
        df = sub if not sub.empty else df
    if "mean_recall" not in df.columns:
        df = pooled_recall_precision_functions(df)
    df = df.sort_values("q")
    return PrSeries(
        label=spec.label,
        q=df["q"].astype(float).values,
        recall=df["mean_recall"].astype(float).values,
        precision=df["mean_precision"].astype(float).values,
    )


def _align_q_levels(series_list: list[PrSeries]) -> np.ndarray:
    qs = [set(np.round(s.q, 6)) for s in series_list]
    common = sorted(qs[0].intersection(*qs[1:]) if len(qs) > 1 else qs[0])
    if common:
        return np.array(common)
    all_q = sorted({float(q) for s in series_list for q in s.q})
    return np.array(all_q)


def _resample_at_q(series: PrSeries, q_levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    recall = np.full(len(q_levels), np.nan)
    precision = np.full(len(q_levels), np.nan)
    qmap = {round(float(q), 6): i for i, q in enumerate(q_levels)}
    for q, r, p in zip(series.q, series.recall, series.precision):
        key = round(float(q), 6)
        if key in qmap:
            idx = qmap[key]
            recall[idx] = r
            precision[idx] = p
    return recall, precision


def plot_pr_with_q_contours(
    series_list: list[PrSeries],
    *,
    title: str,
    subtitle: str,
    out_path: Path,
    q_label_levels: np.ndarray | None = None,
    label_combo_for_q: int | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] = (-0.02, 1.05),
) -> None:
    if not series_list:
        raise ValueError("series_list is empty")

    q_levels = _align_q_levels(series_list)
    aligned: list[tuple[PrSeries, np.ndarray, np.ndarray]] = []
    for s in series_list:
        r, p = _resample_at_q(s, q_levels)
        aligned.append((s, r, p))

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)

    # q-linking contours (draw under curves)
    label_idx = label_combo_for_q
    if label_idx is None:
        label_idx = 0
    for qi, q in enumerate(q_levels):
        xs, ys = [], []
        for _, r, p in aligned:
            if not np.isnan(r[qi]) and not np.isnan(p[qi]):
                xs.append(r[qi])
                ys.append(p[qi])
        if len(xs) < 2:
            continue
        ax.plot(xs, ys, linestyle=CONTOUR_STYLE, color=CONTOUR_COLOR, alpha=CONTOUR_ALPHA, zorder=1)
        if q_should_label(float(q), q_label_levels) and label_idx < len(aligned):
            _, r, p = aligned[label_idx]
            if not np.isnan(r[qi]) and not np.isnan(p[qi]):
                ax.text(
                    r[qi] + 0.008,
                    p[qi] + 0.012,
                    f"q={format_q_label(q)}",
                    fontsize=LABEL_FONTSIZE,
                    color="dimgray",
                    alpha=0.85,
                    zorder=2,
                )

    for idx, (s, r, p) in enumerate(aligned):
        style = COMBO_STYLES[idx % len(COMBO_STYLES)]
        ax.plot(
            r,
            p,
            marker=style["marker"],
            markersize=MARKER_SIZE,
            color=style["color"],
            linewidth=LINE_WIDTH,
            label=s.label,
            zorder=3,
        )

    if xlim is None:
        all_r = np.concatenate([r[~np.isnan(r)] for _, r, _ in aligned])
        if all_r.size:
            xmin = max(0.0, float(np.nanmin(all_r)) - 0.05)
            xlim = (xmin, min(1.02, float(np.nanmax(all_r)) + 0.02))
        else:
            xlim = (0.0, 1.02)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Recall(q)", fontsize=AXIS_FONTSIZE, fontweight="bold")
    ax.set_ylabel("Precision(q)", fontsize=AXIS_FONTSIZE, fontweight="bold")
    # ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=12)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="lower left", frameon=True, edgecolor="0.35")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=AXIS_FONTSIZE - 1, color="0.35")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def run_comparison(
    out_dir: Path,
    specs: list[ComboSpec],
    *,
    title: str,
    out_name: str,
    seq_name: str,
    q_label_levels: np.ndarray | None = None,
    xlim: tuple[float, float] | None = None,
) -> None:
    series = [load_pr_series(out_dir, s) for s in specs]
    decoys = sorted({s.decoy_method for s in specs})
    subtitle = f"{seq_name}; decoy: {', '.join(decoys)}"
    plot_pr_with_q_contours(
        series,
        title=title,
        subtitle=subtitle,
        out_path=out_dir / out_name,
        q_label_levels=q_label_levels,
        xlim=xlim,
    )


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Parameters (edit here)
    # ------------------------------------------------------------------
    SEQ_NAME = "astral_s3000_seed123"
    OUT_DIR = RESULTS_DIR / SEQ_NAME

    # Compare search methods (same decoy)
    SEARCH_COMPARE_DECOY = "extended_shuf"
    SEARCH_COMPARE_SPECS = [
        ComboSpec("plm", SEARCH_COMPARE_DECOY),
        ComboSpec("tmvec", SEARCH_COMPARE_DECOY),
        ComboSpec("dhr_postprocess", SEARCH_COMPARE_DECOY),
    ]
    SEARCH_COMPARE_TITLE = "FDR-controlled PR (search methods)"
    SEARCH_COMPARE_PDF = f"pr_compare_search_{SEARCH_COMPARE_DECOY}_{SEQ_NAME}.pdf"

    # Compare decoy generation methods (same search)
    DECOY_COMPARE_SEARCH = "plm"
    DECOY_COMPARE_SPECS = [
        ComboSpec(DECOY_COMPARE_SEARCH, "extended_shuf"),
        ComboSpec(DECOY_COMPARE_SEARCH, "extended_rev"),
        ComboSpec(DECOY_COMPARE_SEARCH, "shuf"),
    ]
    DECOY_COMPARE_TITLE = f"FDR-controlled PR (decoy methods; {DECOY_COMPARE_SEARCH})"
    DECOY_COMPARE_PDF = f"pr_compare_decoy_{DECOY_COMPARE_SEARCH}_{SEQ_NAME}.pdf"

    # Text labels on q contours only at these FDR levels (contours still drawn for all q)
    Q_LABEL_LEVELS_OVERRIDE = None  # default: [0.1, 0.3, 0.5, 0.7, 0.9] from compute_efdr_recall
    PR_XLIM: tuple[float, float] | None = None  # e.g. (0.4, 1.0) to zoom

    # ------------------------------------------------------------------
    run_comparison(
        OUT_DIR,
        SEARCH_COMPARE_SPECS,
        title=SEARCH_COMPARE_TITLE,
        out_name=SEARCH_COMPARE_PDF,
        seq_name=SEQ_NAME,
        q_label_levels=Q_LABEL_LEVELS_OVERRIDE,
        xlim=PR_XLIM,
    )
    run_comparison(
        OUT_DIR,
        DECOY_COMPARE_SPECS,
        title=DECOY_COMPARE_TITLE,
        out_name=DECOY_COMPARE_PDF,
        seq_name=SEQ_NAME,
        q_label_levels=Q_LABEL_LEVELS_OVERRIDE,
        xlim=PR_XLIM,
    )
