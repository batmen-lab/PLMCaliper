import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

# Nature-style rcParams, kept local to this experiment.
FONT_LABEL = 7
FONT_TITLE = 7
FONT_TICK = 6
FONT_LEGEND = 6
LINEWIDTH = 0.9
AXES_LINEWIDTH = 0.6
GRID_LINEWIDTH = 0.4
MARKERSIZE = 3.0
_SANS_PREF = ["Helvetica", "Arial", "Nimbus Sans", "Liberation Sans",
              "TeX Gyre Heros", "DejaVu Sans"]


def _apply_rcparams() -> None:
    rc = matplotlib.rcParams
    rc["pdf.fonttype"] = 42
    rc["ps.fonttype"] = 42
    rc["svg.fonttype"] = "none"
    rc["font.family"] = "sans-serif"
    rc["font.sans-serif"] = _SANS_PREF + list(rc["font.sans-serif"])
    rc["axes.unicode_minus"] = False
    rc["font.size"] = FONT_TICK
    rc["axes.labelsize"] = FONT_LABEL
    rc["axes.titlesize"] = FONT_TITLE
    rc["xtick.labelsize"] = FONT_TICK
    rc["ytick.labelsize"] = FONT_TICK
    rc["legend.fontsize"] = FONT_LEGEND
    rc["axes.linewidth"] = AXES_LINEWIDTH
    rc["lines.linewidth"] = LINEWIDTH
    rc["lines.markersize"] = MARKERSIZE
    rc["grid.linewidth"] = GRID_LINEWIDTH
    rc["xtick.major.width"] = AXES_LINEWIDTH
    rc["ytick.major.width"] = AXES_LINEWIDTH
    rc["xtick.major.size"] = 2.5
    rc["ytick.major.size"] = 2.5
    rc["savefig.dpi"] = 300
    rc["figure.dpi"] = 150
    rc["legend.frameon"] = True
    rc["legend.framealpha"] = 1.0


def data_root() -> Path:
    override = os.environ.get("PLMCALIPER_PLOT_DATA")
    if override:
        return Path(override)
    return PROJECT_ROOT / "data"


def experiment_dir(name: str) -> Path:
    return data_root() / name


_apply_rcparams()

DEFAULT_DATA_DIR = experiment_dir("ColabFold")
PROJECT_ROOT_ = Path(__file__).resolve().parents[3]
DATASETS = ("astral", "ur50")
# the run tree is derived data; only the figures go to results/
RESULTS_DIR = {"astral": PROJECT_ROOT_ / "data" / "ColabFold" / "astral_db",
               "ur50": PROJECT_ROOT_ / "data" / "ColabFold" / "ur50_exp"}


def metrics_path(dataset: str, method: str, jack_iters: int) -> Path:
    return RESULTS_DIR[dataset] / method / f"iter{jack_iters}" / "metrics.tsv"


def infer_baseline(df: pd.DataFrame) -> str:
    """The baseline is whichever arm is not an eFDR arm; metrics.tsv already says so."""
    tags = [t for t in df["tag"].unique() if not str(t).startswith("efdr_")]
    if not tags:
        raise SystemExit("[step6] no baseline arm in this metrics table")
    return sorted(tags)[0]


# how many sequences the unfiltered database holds, for the n_hits panel
FULL_DB_SIZE = {"ur50": 38_794_121, "astral": 15_177}

EFDR = [round(0.1 * i, 1) for i in range(1, 10)]
EFDR_COLOR = "#029E73"
VAN_COLOR = "#E69F00"
EFDR_CMAP = plt.cm.Blues  # sequential blue for the eFDR bars (dark 0.1 -> light 0.9)
LEG = dict(bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0,
           frameon=True, edgecolor="0.7", fontsize=6)
STRIP_MAX_N = 15
LABELS = {"msa_depth": "MSA depth", "msa_seconds": "MSA build time (s)",
          "plddt": "Mean pLDDT", "rmsd": "r.m.s.d.", "tm_score": "TM-score vs native",
          "n_hits": "JackHMMER input sequences", "meff": "Meff (effective sequences)"}
LOGY = {"msa_depth", "msa_seconds", "n_hits", "meff"}
SEARCH_DISPLAY = {"plm": "PLMsearch", "tmvec": "TMvec", "dhr_postprocess": "DHR"}


def _sci10(v: float) -> str:
    if not np.isfinite(v) or v <= 0:
        return "0"
    exp = int(np.floor(np.log10(v)))
    return rf"${v / 10.0 ** exp:.1f}\times10^{{{exp}}}$"

VALUE_FMT = {"tm_score": lambda v: f"{v:.2f}", "plddt": lambda v: f"{v:.1f}",
             "rmsd": lambda v: f"{v:.2f}", "msa_depth": lambda v: f"{v:,.0f}",
             "msa_seconds": _sci10, "n_hits": lambda v: f"{v:,.0f}",
             "meff": lambda v: f"{v:,.0f}"}
VALUE_ROT = {"msa_seconds": 45}

VALUE_FONTSIZE = {"msa_seconds": 4.5}



def _sig_stars(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _mean_ci95(vals) -> tuple[float, float, int]:
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    n = len(v)
    if n == 0:
        return np.nan, 0.0, 0
    if n == 1:
        return float(v[0]), 0.0, 1
    return float(v.mean()), 1.96 * float(v.std(ddof=1)) / np.sqrt(n), n





# the six metrics the figures report, plus n_hits as the selection-size control
METRICS = ["n_hits", "msa_depth", "meff", "msa_seconds", "plddt", "tm_score", "rmsd"]

def _holm_adjust(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    finite = np.where(np.isfinite(p))[0]
    order = finite[np.argsort(p[finite])]
    m = len(order)
    adj = np.full(p.shape, np.nan)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))   # step-down, monotone
        adj[i] = running
    return adj


def _ttest_rel_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided paired t-test, same quantity as scipy.stats.ttest_rel."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(sps.ttest_rel(a, b).pvalue)


def _paired_p(df: pd.DataFrame, metric: str, q: float, vanilla_tag: str) -> float:
    cond = df[df["efdr_limit"].notna() & np.isclose(df["efdr_limit"], q)][["query_id", metric]]
    base = df[df["tag"] == vanilla_tag][["query_id", metric]]
    m = cond.merge(base, on="query_id", suffixes=("_q", "_b")).dropna()
    if len(m) < 2:
        return np.nan
    return _ttest_rel_p(m[f"{metric}_q"].to_numpy(float), m[f"{metric}_b"].to_numpy(float))


def _bracket(ax, x1: float, x2: float, y_frac: float, label: str) -> None:
    trans = ax.get_xaxis_transform()
    tick = 0.012
    ax.plot([x1, x1, x2, x2], [y_frac - tick, y_frac, y_frac, y_frac - tick],
            transform=trans, color="0.35", lw=0.6, clip_on=False)
    ax.text((x1 + x2) / 2.0, y_frac + 0.004, label, transform=trans, ha="center",
            va="bottom", fontsize=5.5, color="0.2", clip_on=False)


def _save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def barplot_metric(df: pd.DataFrame, metric: str, out: Path, vanilla_tag: str, method: str,
                   add_vanilla: bool = True, vanilla_value: float | None = None,
                   brackets: bool = True):
    if metric not in df.columns or not df[metric].notna().any():
        print(f"[skip] {metric}: no data")
        return
    efdr = df[df["efdr_limit"].notna()]
    van = df[df["tag"] == vanilla_tag]

    # per-condition mean + 95% CI (9 eFDR, then vanilla last if requested)
    means, cis, ns = [], [], []
    for q in EFDR:
        m, c, n = _mean_ci95(efdr[np.isclose(efdr["efdr_limit"], q)][metric].tolist())
        means.append(m); cis.append(c); ns.append(n)
    if add_vanilla:
        if vanilla_value is not None:
            vm, vc = float(vanilla_value), 0.0   # fixed count -> no CI
        else:
            vm, vc, _ = _mean_ci95(van[metric].tolist())
        means.append(vm); cis.append(vc)
    means = np.asarray(means); cis = np.asarray(cis)
    x = np.arange(len(means))                 # eFDR at 0..8 (+ vanilla at 9)
    log = metric in LOGY
    if not np.isfinite(means).any():
        print(f"[skip] {metric}: no finite means"); return

    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    ax.set_box_aspect(1)
    if log:
        ax.set_yscale("log")

    colors = [EFDR_CMAP(0.85 - 0.5 * i / (len(EFDR) - 1)) for i in range(len(EFDR))]
    if add_vanilla:
        colors.append(VAN_COLOR)
    ax.bar(x, means, width=0.8, color=colors, edgecolor="none", zorder=2)

    lo = means - cis
    hi = means + cis
    if log:
        lo = np.clip(lo, np.nanmin(means[means > 0]) * 0.5, None)

    yerr = np.clip(np.nan_to_num(np.vstack([means - lo, hi - means]), nan=0.0), 0.0, None)
    ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black", elinewidth=0.7,
                capsize=0, zorder=4)

    data_lo = float(np.nanmin(lo)); data_hi = float(np.nanmax(hi))
    bar_top_frac = 0.52 if (add_vanilla and brackets) else 0.82
    if log:
        bottom = max(np.nanmin(means[means > 0]) * 0.5, 1e-9)
        top = bottom * (data_hi / bottom) ** (1.0 / bar_top_frac)
    else:
        span = max(data_hi - data_lo, 1e-9)
        bottom = max(0.0, data_lo - 0.12 * span) if data_lo >= 0 else data_lo - 0.12 * span
        top = bottom + (data_hi - bottom) / bar_top_frac
    ax.set_ylim(bottom, top)

    fmt = VALUE_FMT.get(metric, lambda v: f"{v:.2f}")
    rot = VALUE_ROT.get(metric, 0)
    fsz = VALUE_FONTSIZE.get(metric, 5.5)
    lab_ha = "left" if rot else "center"
    for xi, m, h in zip(x, means, hi):
        if not np.isfinite(m):
            continue
        ax.annotate(fmt(m), (xi, h), textcoords="offset points", xytext=(0, 2),
                    ha=lab_ha, va="bottom", fontsize=fsz, fontweight="bold",
                    color="black", rotation=rot)

    if add_vanilla and brackets:
        adj_p = _holm_adjust([_paired_p(df, metric, q, vanilla_tag) for q in EFDR])
        vpos = x[-1]
        order = sorted(range(len(EFDR)), key=lambda i: -EFDR[i])   # 0.9, 0.8, ..., 0.1
        y0, y1 = 0.60, 0.97
        step = (y1 - y0) / max(len(order) - 1, 1)
        for k, i in enumerate(order):
            if not np.isfinite(means[i]):
                continue
            _bracket(ax, x[i], vpos, y0 + step * k, _sig_stars(adj_p[i]))

    ax.set_xticks(x)
    labels = [f"{q:.1f}" for q in EFDR] + (["vanilla"] if add_vanilla else [])
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_xlabel("Target FDR level", fontweight="bold")
    ax.set_ylabel(LABELS.get(metric, metric), fontweight="bold")
    ax.set_title(SEARCH_DISPLAY.get(method, method), fontsize=7, fontweight="bold",
                 color="black")
    ax.set_axisbelow(True)
    _save(fig, out)


def depth_vs_qlen(df: pd.DataFrame, out: Path, vanilla_tag: str):
    if "qlen" not in df.columns or "msa_depth" not in df.columns:
        print("[skip] depth_vs_qlen: missing qlen/msa_depth")
        return
    efdr = df[df["efdr_limit"].notna()]
    van = df[df["tag"] == vanilla_tag]
    fdr_best = efdr.groupby("query_id").agg(qlen=("qlen", "first"),
                                            depth=("msa_depth", "max")).reset_index()
    vanq = van.groupby("query_id").agg(qlen=("qlen", "first"),
                                       depth=("msa_depth", "max")).reset_index()
    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    ax.set_box_aspect(1)
    ax.scatter(vanq["qlen"], vanq["depth"].clip(lower=1), s=16, color=VAN_COLOR,
               edgecolors="k", linewidths=0.3, label=f"vanilla ({vanilla_tag})", zorder=3)
    ax.scatter(fdr_best["qlen"], fdr_best["depth"].clip(lower=1), s=16, color=EFDR_COLOR,
               edgecolors="k", linewidths=0.3, label="FDR-best", zorder=4)
    ax.axhline(1, color="#C44E52", ls=":", lw=0.8, label="depth = 1 (query only)")
    ax.set_yscale("log")
    ax.set_xlabel("Query length (aa)", fontweight="bold")
    ax.set_ylabel("MSA depth", fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(**LEG)
    _save(fig, out)




# --------------------------------------------------------------------------
def plot_method(df: pd.DataFrame, plot_dir: Path, method: str, dataset: str,
                baseline_tag: str, formats) -> pd.DataFrame:
    out = plot_dir / method
    out.mkdir(parents=True, exist_ok=True)
    depth_vs_qlen(df, out / "msa_depth_vs_qlen.pdf", baseline_tag)

    fds = FULL_DB_SIZE.get(dataset)
    for metric in METRICS:
        kw = {}
        if metric == "n_hits" and fds is not None:
            kw = dict(add_vanilla=True, vanilla_value=fds, brackets=False)
        barplot_metric(df, metric, out / f"bar_{metric}.pdf", baseline_tag, method, **kw)

    return stats_table(df, baseline_tag)


def stats_table(df: pd.DataFrame, baseline_tag: str) -> pd.DataFrame:
    """The numbers behind the stars: per (metric, q) paired t-test, Holm-adjusted."""
    rows = []
    for metric in METRICS:
        qs, ps, deltas, ns = [], [], [], []
        for q in EFDR:
            tag = f"efdr_{q:.2f}".replace(".", "p")
            a = df[df["tag"] == tag][["query_id", metric]]
            b = df[df["tag"] == baseline_tag][["query_id", metric]]
            m = a.merge(b, on="query_id", suffixes=("_q", "_base")).dropna()
            qs.append(q); ns.append(len(m))
            if len(m) < 2:
                ps.append(np.nan); deltas.append(np.nan); continue
            x = m[f"{metric}_q"].to_numpy(float)
            y = m[f"{metric}_base"].to_numpy(float)
            ps.append(_ttest_rel_p(x, y)); deltas.append(float((x - y).mean()))
        adj = _holm_adjust(ps)
        for q, p, pa, d, n in zip(qs, ps, adj, deltas, ns):
            rows.append({"metric": metric, "q": q, "n_pairs": n, "mean_delta": d,
                         "p_raw": p, "p_holm": pa, "stars": _sig_stars(pa)})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 6: every ColabFold figure, plus the paired t-tests behind them.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--methods", nargs="+", default=["dhr_postprocess", "plm", "tmvec"])
    p.add_argument("--jack-iters", type=int, default=1)
    p.add_argument("--metrics-table", type=Path, default=None,
                   help="single metrics.tsv; default: step 5's, per method")
    p.add_argument("--baseline-tag", default=None,
                   help="default: the dataset's first baseline")
    p.add_argument("--formats", nargs="+", default=["pdf"], choices=["pdf", "png"])
    p.add_argument("--plot-dir", type=Path, default=None)
    p.add_argument("--stats-out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    plot_dir = a.plot_dir or (DEFAULT_DATA_DIR / "figures" / a.dataset)

    all_stats = []
    for method in a.methods:
        path = a.metrics_table or metrics_path(a.dataset, method, a.jack_iters)
        if not path.exists():
            print(f"[skip] {method}: missing {path}")
            continue
        df = pd.read_csv(path, sep="\t", dtype={"query_id": str, "tag": str})
        if "af2_status" in df.columns:
            df = df[df["af2_status"].isin(["ok", "reused", "skipped"])]
        baseline = a.baseline_tag or infer_baseline(df)
        print(f"\n>>> {a.dataset}/{method}  ({len(df)} rows, baseline={baseline})")
        st = plot_method(df, plot_dir, method, a.dataset, baseline, a.formats)
        st.insert(0, "method", method)
        all_stats.append(st)

    if all_stats:
        stats = pd.concat(all_stats, ignore_index=True)
        out = a.stats_out or (plot_dir / "paired_ttest.tsv")
        out.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(out, sep="\t", index=False)
        print(f"\n[step6] paired t-tests -> {out}")
        sig = stats[stats["p_holm"] < 0.05]
        print(f"[step6] {len(sig)}/{len(stats)} (metric, q) cells significant after Holm")


if __name__ == "__main__":
    main()
