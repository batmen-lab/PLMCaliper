#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import style 
from paths import experiment_dir

DEFAULT_DATA_DIR = experiment_dir("ColabFold")

DATASETS = {
    "astral_db": dict(methods=["plm", "tmvec", "dhr_postprocess"], vanilla_tag="vanilla"),
    "ur50_db": dict(methods=["plm", "dhr_postprocess"], vanilla_tag="base_alldb_iter1"),
}

FULL_DB_SIZE = {"ur50_db": 38_794_121, "astral_db": 15_177}

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


def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 300, 3e-14, 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _ttest_rel_p(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    n = len(d)
    if n < 2:
        return float("nan")
    md = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        return 1.0 if md == 0.0 else 0.0
    t = md / (sd / math.sqrt(n))
    df = n - 1
    return _betai(0.5 * df, 0.5, df / (df + t * t))


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


def plot_method(dataset_dir: Path, plot_dir: Path, method: str, vanilla_tag: str):
    summary = dataset_dir / f"metrics_{method}.tsv"
    if not summary.exists():
        print(f"[skip] {method}: missing {summary.name}")
        return
    df = pd.read_csv(summary, sep="\t")
    if "af2_status" in df.columns:
        df = df[df["af2_status"].isin(["ok", "skipped"])]
    out = plot_dir / method
    print(f"\n>>> {dataset_dir.name}/{method}  ({summary.name}, {len(df)} rows, "
          f"vanilla={vanilla_tag})")
    depth_vs_qlen(df, out / "msa_depth_vs_qlen.pdf", vanilla_tag)

    for metric in ["meff", "msa_seconds", "plddt", "rmsd", "tm_score"]:
        barplot_metric(df, metric, out / f"bar_{metric}.pdf", vanilla_tag, method)

    fds = FULL_DB_SIZE.get(dataset_dir.name)
    barplot_metric(df, "n_hits", out / "bar_n_hits.pdf", vanilla_tag, method,
                   add_vanilla=fds is not None, vanilla_value=fds, brackets=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--methods", nargs="+", default=None,
                   help="override the dataset's default methods")
    p.add_argument("--vanilla-tag", default=None,
                   help="override the dataset's default baseline tag")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--plot-dir", type=Path, default=DEFAULT_DATA_DIR / "figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[info] data={args.data_dir}  out={args.plot_dir}  datasets={args.datasets}")
    for dataset in args.datasets:
        preset = DATASETS[dataset]
        methods = args.methods or preset["methods"]
        vanilla_tag = args.vanilla_tag or preset["vanilla_tag"]
        for method in methods:
            plot_method(args.data_dir / dataset, args.plot_dir / dataset, method, vanilla_tag)


if __name__ == "__main__":
    main()
