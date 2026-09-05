import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # headless; must precede the pyplot import
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from tsne import (BAGEL_PLOT_DIR, CLASS_COLORS, CLASS_MARKERS, DATA_DIR, FONT_LABEL,
                  FONT_LEGEND, FONT_TICK, FONT_TITLE, METHOD_DISK, SUMMARY_AX_SIZE,
                  SUMMARY_BG_ALPHA, SUMMARY_BG_SIZE, SUMMARY_STAR_EDGE_LW,
                  SUMMARY_STAR_SIZE, class_from_qid, load_putative_ids)

CLASSES = ("1", "2", "3")
METHOD_LABEL = {"plm": "PLMsearch", "tmvec": "TMvec", "dhr": "DHR", "blastp": "BLASTp"}

METHOD_DECOY = {"plm": "extended_mkv2", "tmvec": "extended_mkv2",
                "dhr": "extended_mkv2", "blastp": "mkv2"}
QUERY_NAME, TARGET_NAME, WEIGHT_METHOD = "putative", "class_all", "AdaptiveBell"
TIE_RTOL = 1e-9
STAR_Z = 100

DEFAULT_DATA_DIR = BAGEL_PLOT_DIR / "extended_mkv2_tau025" / "tsne_per_query"
DEFAULT_CUTOFF = 0.6
Q_LEVELS_FALLBACK = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
FIGURE_KINDS = ("bar", "map", "grid")
Q_SOURCES = ("rank", "grid")

NO_HIT_COLOR = "#DDDDDD"
UNASSIGNED_COLOR = "#FFFFFF"
LABEL_WIDTH = 28

LABEL_EM_PER_CHAR = 0.62
LABEL_PAD_IN = 0.12

BAR_AX_W, BAR_AX_H = 5.2, 1.9
GRID_QUERY_W, GRID_CLASS_H = BAR_AX_W / 30.0, 0.24
GRID_DOT_MIN, GRID_DOT_MAX = 10.0, 110.0
GRID_FILL_MIN = 0.28
GRID_RIM_LW = 0.4
GRID_SEL_LW = 1.0
GRID_SEL_DARKEN = 0.55
GRID_BAND_COLOR = "#EAF0F7"
GRID_SEP_COLOR = "#999999"

KEY_GAP = 0.34
KEY_COLOR = "#767676"
KEY_SAMPLES = 512
KEY_WIDTH = 2.0 * float(np.sqrt(GRID_DOT_MAX / np.pi)) / 72.0


def short_label(qid, width=LABEL_WIDTH):
    return qid if len(qid) <= width else qid[: width - 1] + "…"


def label_margin(labels, fontsize=FONT_TICK, minimum=0.0):
    longest = max((len(str(s)) for s in labels), default=0)
    return max(minimum, longest * LABEL_EM_PER_CHAR * fontsize / 72.0 + LABEL_PAD_IN)


def load_context(data_dir):
    coords = np.load(data_dir / "coords.npy")
    ids = np.array([s for s in (data_dir / "ids.txt").read_text().split("\n") if s])
    dfc = pd.read_csv(data_dir / "id2class.tsv", sep="\t", dtype=str)
    id2class = dict(zip(dfc["id"], dfc["class"]))
    putative = sorted(load_putative_ids())
    missing = [p for p in putative if p not in set(ids)]
    if missing:
        print(f"[WARN] {len(missing)} putative id(s) absent from {data_dir.name}/ids.txt")
    return coords, ids, id2class, putative


def load_q_levels(data_dir):
    meta = data_dir / "meta.json"
    if meta.exists():
        import json
        levels = json.loads(meta.read_text()).get("q_levels")
        if levels:
            return sorted(float(q) for q in levels)
    return list(Q_LEVELS_FALLBACK)


def load_hits(data_dir, method, cutoff):
    path = data_dir / f"hits_{method}.tsv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t", dtype={"qid": str, "tid": str})
    df["q"] = pd.to_numeric(df["q"], errors="coerce")
    df = df[df["q"].notna() & (df["q"] <= cutoff + 1e-9)]
    df = df.sort_values("q").drop_duplicates(["qid", "tid"], keep="first")
    return df[["qid", "tid", "q"]]


# Per-hit q-values from raw rankings.
HIT_COLUMNS = ["qid", "rep_id", "tid", "homo_type_real", "score_real", "score_decoy",
               "hit_rank", "n_target_ge", "n_decoy_ge", "efdp_at_score", "q_hit",
               "T_cutoff", "eFDP_at_T_cutoff"]


def score_paths(method):
    disk = METHOD_DISK.get(method, method)
    if method not in METHOD_DECOY:
        raise KeyError(f"no decoy database registered for method {method!r}")
    decoy = METHOD_DECOY[method]
    return (DATA_DIR / f"result_{disk}_{QUERY_NAME}_target_noisy_{TARGET_NAME}.txt",
            DATA_DIR / f"result_{disk}_{QUERY_NAME}_{decoy}"
                       f"_calibrated_gam_{WEIGHT_METHOD}_{TARGET_NAME}.txt")


def _strip_decoy_suffix(decoy_qid, real_qids):
    for i in range(len(decoy_qid), 0, -1):
        if decoy_qid[:i] in real_qids:
            return decoy_qid[:i]
    return None


def load_pair_table(method):
    real_path, decoy_path = score_paths(method)
    for path in (real_path, decoy_path):
        if not path.exists():
            raise FileNotFoundError(f"{method}: missing score table {path}")

    real = pd.read_csv(real_path, sep="\t", dtype={"qid": str, "tid": str})
    decoy = pd.read_csv(decoy_path, sep="\t", dtype={"qid": str, "tid": str})

    known = set(real["qid"])
    strip = {q: _strip_decoy_suffix(q, known) for q in decoy["qid"].unique()}
    orphans = sorted(q for q, root in strip.items() if root is None)
    if orphans:
        raise ValueError(f"{method}: {len(orphans)} decoy query id(s) extend no real query, "
                         f"e.g. {orphans[0]!r}")

    pair = (real.rename(columns={"score": "score_real", "homo_type": "homo_type_real"})
                .merge(decoy.assign(qid=decoy["qid"].map(strip))
                            .rename(columns={"score": "score_decoy"})
                            [["qid", "tid", "rep_id", "score_decoy"]],
                       on=["qid", "tid", "rep_id"], how="inner"))
    return pair[["qid", "rep_id", "tid", "homo_type_real", "score_real", "score_decoy"]]


def efdp_curve(scores_t, scores_d):
    cand = np.sort(np.unique(np.concatenate([scores_t, scores_d])))
    n_target_ge = len(scores_t) - np.searchsorted(np.sort(scores_t), cand, side="left")
    n_decoy_ge = len(scores_d) - np.searchsorted(np.sort(scores_d), cand, side="left")
    efdp = (n_decoy_ge + 1.0) / np.maximum(n_target_ge, 1.0)
    # A hit's q is the best eFDP reachable at or below its score.
    return {"cand": cand, "efdp": efdp, "q_curve": np.minimum.accumulate(efdp),
            "n_target_ge": n_target_ge, "n_decoy_ge": n_decoy_ge}


def per_hit_qvalues(pair, cutoff, tol=1e-12):
    frames = []
    for (qid, rep_id), grp in pair.groupby(["qid", "rep_id"], sort=False):
        scores_t = grp["score_real"].to_numpy(dtype=float)
        if scores_t.size == 0:
            continue
        curve = efdp_curve(scores_t, grp["score_decoy"].to_numpy(dtype=float))

        valid = np.flatnonzero(curve["efdp"] <= cutoff + tol)
        if valid.size == 0:
            continue
        cut_i = valid[np.argmin(curve["cand"][valid])]
        t_cutoff = float(curve["cand"][cut_i])

        at = np.searchsorted(curve["cand"], scores_t, side="right") - 1
        q_hit = curve["q_curve"][at]
        keep = q_hit <= cutoff + tol

        if not np.array_equal(keep, scores_t >= t_cutoff):
            raise AssertionError(
                f"{qid} rep {rep_id}: q_hit <= {cutoff} keeps {int(keep.sum())} hits but "
                f"score >= T_cutoff ({t_cutoff:.6g}) keeps {int((scores_t >= t_cutoff).sum())}")
        if not keep.any():
            continue

        sub = grp[keep].copy()
        sub["q_hit"] = q_hit[keep]
        sub["efdp_at_score"] = curve["efdp"][at][keep]
        sub["n_target_ge"] = curve["n_target_ge"][at][keep]
        sub["n_decoy_ge"] = curve["n_decoy_ge"][at][keep]
        sub["T_cutoff"] = t_cutoff
        sub["eFDP_at_T_cutoff"] = float(curve["efdp"][cut_i])
        sub = sub.sort_values("score_real", ascending=False)
        sub["hit_rank"] = np.arange(1, len(sub) + 1)
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=HIT_COLUMNS)
    return pd.concat(frames, ignore_index=True)[HIT_COLUMNS]


def collapse_reps(hits):
    return hits.groupby(["qid", "tid"], as_index=False).agg(q=("q_hit", "mean"),
                                                            n_reps_hit=("rep_id", "nunique"))


def load_q_table(data_dir, method, cutoff, q_source):
    if q_source == "grid":
        return load_hits(data_dir, method, cutoff), None
    try:
        detail = per_hit_qvalues(load_pair_table(method), cutoff)
    except (FileNotFoundError, KeyError) as exc:
        print(f"[skip] {method}: {exc}")
        return None, None
    return collapse_reps(detail), detail


def per_class_tables(hits, id2class, queries, cutoff):
    h = hits.assign(cls=hits["tid"].map(id2class))
    n_unknown = int(h["cls"].isna().sum())
    if n_unknown:
        print(f"  [WARN] {n_unknown} hit(s) to targets with no class code, dropped")
    h = h[h["cls"].notna()]

    counts = (h.pivot_table(index="qid", columns="cls", values="tid", aggfunc="count")
                .reindex(index=queries, columns=list(CLASSES)).fillna(0.0))
    pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100.0

    sums = (h.groupby(["qid", "cls"])["q"].sum().unstack()
             .reindex(index=queries, columns=list(CLASSES)).fillna(0.0))

    n_pad = counts.max(axis=1).replace(0, np.nan)
    n_missing = counts.rsub(n_pad, axis=0).clip(lower=0)
    wq = (sums + cutoff * n_missing).div(n_pad, axis=0).where(counts > 0)
    return counts, pct, wq


def audit_tables(hits, detail, id2class, queries, counts, wq, tied, cutoff):
    h = hits.assign(cls=hits["tid"].map(id2class))
    h = h[h["cls"].notna()]
    if detail is not None and not detail.empty:
        score = detail.groupby(["qid", "tid"])["score_real"].mean().rename("score")
        h = h.merge(score, on=["qid", "tid"], how="left")
    else:
        h = h.assign(score=np.nan)

    h = h.assign(_n=h["qid"].map(_qid_number))
    hits_long = (h.sort_values(["_n", "qid", "cls", "score", "q", "tid"],
                               ascending=[True, True, True, False, True, True])
                  .drop(columns="_n"))
    hits_long.insert(0, "rank_in_class",
                     hits_long.groupby(["qid", "cls"]).cumcount() + 1)
    hits_long = hits_long.rename(columns={"cls": "class"})[
        ["qid", "class", "rank_in_class", "tid", "score", "q"]]

    sums = h.groupby(["qid", "cls"])["q"].sum()
    n_pad = counts.max(axis=1)
    rows = []
    for qid in queries:
        winners = tied.get(qid, ())
        pad = float(n_pad.get(qid, 0.0))
        for cls in CLASSES:
            n = float(counts.loc[qid, cls]) if qid in counts.index else 0.0
            n_missing = max(pad - n, 0.0)
            s = float(sums.get((qid, cls), 0.0))
            rows.append({
                "qid": qid, "class": cls, "n_hits": int(n),
                "n_pad": int(pad), "n_missing": int(n_missing),
                "sum_q_hits": round(s, 9) if n else np.nan,
                "q_pad": cutoff if n else np.nan,
                "sum_q_pad": round(cutoff * n_missing, 9) if n else np.nan,
                "numerator": round(s + cutoff * n_missing, 9) if n else np.nan,
                "denominator": int(pad) if n else np.nan,
                "weighted_q": round((s + cutoff * n_missing) / pad, 9) if n and pad else np.nan,
                "excluded_no_hits": int(n == 0),
                "is_winner": int(cls in winners),
                "tie_size": len(winners),
            })
    terms = pd.DataFrame(rows).sort_values(
        ["qid", "class"], key=lambda c: c.map(_qid_number) if c.name == "qid" else c)

    ref = wq.stack(dropna=True)
    got = terms.set_index(["qid", "class"])["weighted_q"].dropna()
    shared = ref.index.intersection(got.index)
    delta = float((ref.loc[shared] - got.loc[shared]).abs().max()) if len(shared) else 0.0
    if len(shared) != len(ref) or delta > 1e-9:
        raise AssertionError(f"audit disagrees with the plotted weighted q "
                             f"(max delta {delta:.3g}, {len(shared)}/{len(ref)} cells matched)")
    return hits_long, terms


def best_classes(counts, wq, how, rtol=TIE_RTOL):
    table = wq if how == "weighted" else counts.replace(0, np.nan)
    out = {}
    for qid, row in table.iterrows():
        row = row.dropna()
        if row.empty:
            out[qid] = ()
            continue
        best = row.min() if how == "weighted" else row.max()
        out[qid] = tuple(str(c) for c in row.index
                         if np.isclose(row[c], best, rtol=rtol, atol=1e-12))
    return pd.Series(out, index=table.index, dtype=object)


def assign_class(counts, wq, how):
    tied = best_classes(counts, wq, how)
    return pd.Series({qid: (t[0] if t else None) for qid, t in tied.items()},
                     index=tied.index, dtype=object)


def _qid_number(qid):
    head = qid.split(";", 1)[0].split(".")[0]
    return int(head) if head.isdigit() else 10 ** 9


def order_queries(queries, assigned, wq, how):
    if how == "true-class":
        def key(qid):
            cls = class_from_qid(qid)
            return (CLASSES.index(cls) if cls in CLASSES else 9, _qid_number(qid), qid)
    else:
        def key(qid):
            cls = assigned.get(qid)
            if cls not in CLASSES:
                return (9, 9.0, qid)
            return (CLASSES.index(cls), float(wq.loc[qid, cls]), qid)
    return sorted(queries, key=key)


def assignment_table(counts, pct, wq, tied, majority, *, q_source, detail=None):
    out = pd.DataFrame(index=counts.index)
    out["n_hits"] = counts.sum(axis=1).astype(int)
    for cls in CLASSES:
        out[f"n_class{cls}"] = counts[cls].astype(int)
    for cls in CLASSES:
        out[f"pct_class{cls}"] = pct[cls].round(2)
    for cls in CLASSES:
        out[f"weighted_q_class{cls}"] = wq[cls].round(6)
    out["majority_class"] = majority
    out["assigned_class"] = pd.Series({q: (t[0] if t else None) for q, t in tied.items()})
    out["tied_classes"] = pd.Series({q: "|".join(t) for q, t in tied.items()})
    out["n_tied"] = pd.Series({q: len(t) for q, t in tied.items()})
    if detail is not None and not detail.empty:
        # Threshold diagnostics.
        thr = detail.groupby("qid")[["T_cutoff", "eFDP_at_T_cutoff"]].mean()
        out["T_cutoff"] = thr["T_cutoff"].round(6)
        out["eFDP_at_T_cutoff"] = thr["eFDP_at_T_cutoff"].round(6)
    out["q_source"] = q_source
    return out.reset_index().rename(columns={"index": "qid"})


# Stacked bar.
def fig_class_composition(pct, counts, order, out_path, *, method_label, cutoff, dpi=400):
    left, right, top = 0.55, 0.10, 0.45
    bottom = label_margin(order, minimum=1.50)
    fig_w, fig_h = BAR_AX_W + left + right, BAR_AX_H + top + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, bottom / fig_h, BAR_AX_W / fig_w, BAR_AX_H / fig_h])

    x = np.arange(len(order))
    stack = np.zeros(len(order))
    for cls in CLASSES:
        vals = pct.loc[order, cls].fillna(0.0).to_numpy(float)
        ax.bar(x, vals, bottom=stack, width=0.78, color=CLASS_COLORS[cls],
               edgecolor="white", linewidth=0.3, zorder=2)
        stack += vals

    no_hit = counts.loc[order].sum(axis=1).to_numpy() == 0
    if no_hit.any():
        ax.bar(x[no_hit], np.full(int(no_hit.sum()), 100.0), width=0.78,
               color=NO_HIT_COLOR, edgecolor="white", linewidth=0.3, zorder=2)

    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(x)
    ax.set_xticklabels(list(order), rotation=90, fontsize=FONT_TICK)
    ax.set_ylabel("hits per class (%)", fontsize=FONT_LABEL, fontweight="bold")
    ax.tick_params(labelsize=FONT_TICK, length=2.0)
    ax.set_title(f"{method_label}  —  Class composition of accepted hits "
                 f"(eFDP threshold = {cutoff:g})",
                 fontsize=FONT_TITLE, fontweight="bold", pad=4)

    handles = [mpatches.Patch(facecolor=CLASS_COLORS[c], label=f"Class {c}") for c in CLASSES]
    if no_hit.any():
        handles.append(mpatches.Patch(facecolor=NO_HIT_COLOR, label="no hit"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0 + 0.24 / BAR_AX_H),
              ncol=len(handles), fontsize=FONT_LEGEND, frameon=False,
              handlelength=1.2, handleheight=1.0, columnspacing=1.2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  [OK] stacked bar   -> {out_path.name}")


# t-SNE view.
def star_pie(ax, x, y, colors, *, size, edge_lw, z):
    # Split-star outlines need isolated z-order.
    if len(colors) > 1:
        trans = (mtransforms.Affine2D().scale(1.0 / 72.0) + ax.figure.dpi_scale_trans
                 + mtransforms.ScaledTranslation(x, y, ax.transData))
        r = 2.0 * np.sqrt(size)
        for k, color in enumerate(colors):
            a = np.radians(np.linspace(90.0 + k * 360.0 / len(colors),
                                       90.0 + (k + 1) * 360.0 / len(colors), 33))
            sector = mpatches.Polygon(
                np.vstack([[0.0, 0.0], np.column_stack([r * np.cos(a), r * np.sin(a)])]),
                closed=True, transform=trans)
            ax.scatter([x], [y], marker="*", s=size, c=[color],
                       linewidths=0, zorder=z).set_clip_path(sector)
        colors = ["none"]
    ax.scatter([x], [y], marker="*", s=size, facecolors=colors[0], edgecolors="black",
               linewidths=edge_lw, zorder=z + 1)


def fig_dataset_view(coords, ids, id2class, putative, tied, out_path, *,
                     method_label, cutoff, dpi=400):
    seq2idx = {s: i for i, s in enumerate(ids)}
    putative_set = set(putative)

    left, right, top, bottom = 0.58, 0.08, 0.30, 0.95
    fig_w, fig_h = SUMMARY_AX_SIZE + left + right, SUMMARY_AX_SIZE + top + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, bottom / fig_h,
                       SUMMARY_AX_SIZE / fig_w, SUMMARY_AX_SIZE / fig_h])

    n_bg = 0
    for cls, marker in list(CLASS_MARKERS.items()) + [("Unknown", "o")]:
        idxs = [i for s, i in seq2idx.items()
                if s not in putative_set and id2class.get(s, "Unknown") == cls]
        if not idxs:
            continue
        n_bg += len(idxs)
        idxs = np.array(idxs)
        ax.scatter(coords[idxs, 0], coords[idxs, 1], c=CLASS_COLORS[cls], marker=marker,
                   s=SUMMARY_BG_SIZE, alpha=SUMMARY_BG_ALPHA, linewidths=0, zorder=1)

    n_star = n_split = 0
    drawn = []
    for k, qid in enumerate(putative):
        if qid not in seq2idx:
            continue
        n_star += 1
        won = [c for c in tied.get(qid, ()) if c in CLASSES]
        n_split += len(won) > 1
        x, y = coords[seq2idx[qid]]
        star_pie(ax, x, y, [CLASS_COLORS[c] for c in won] or [UNASSIGNED_COLOR],
                 size=SUMMARY_STAR_SIZE, edge_lw=SUMMARY_STAR_EDGE_LW, z=STAR_Z + 2 * k)
        drawn.append((qid, x, y))

    ax.autoscale_view()
    (x0, x1), d = ax.get_xlim(), np.sqrt(SUMMARY_STAR_SIZE) / 72.0 / SUMMARY_AX_SIZE
    d_star = d * (x1 - x0)
    covered = [q for i, (q, x, y) in enumerate(drawn)
               if any(np.hypot(x - bx, y - by) < 0.5 * d_star for _, bx, by in drawn[i + 1:])]

    ax.set_xlabel("t-SNE 1", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("t-SNE 2", fontsize=FONT_LABEL, fontweight="bold")
    ax.tick_params(labelsize=FONT_TICK)
    ax.set_title(f"{method_label}  —  Putative queries by assigned class "
                 f"(eFDP threshold = {cutoff:g})",
                 fontsize=FONT_TITLE, fontweight="bold", pad=4)

    bits = ["fill = assigned class"]
    if n_split:
        bits.append("split = tied classes")
    if any(not tied.get(q) for q in putative):
        bits.append("white = none")
    star_label = f"putative ({', '.join(bits)})"
    handles = [
        mlines.Line2D([], [], marker=CLASS_MARKERS[c], linestyle="None",
                      markerfacecolor=CLASS_COLORS[c], markeredgecolor=CLASS_COLORS[c],
                      markersize=5, alpha=0.7, label=f"Class {c}")
        for c in CLASSES
    ] + [mlines.Line2D([], [], marker="*", linestyle="None", markerfacecolor=UNASSIGNED_COLOR,
                       markeredgecolor="black", markeredgewidth=SUMMARY_STAR_EDGE_LW,
                       markersize=8, label=star_label)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11),
              ncol=2, fontsize=FONT_LEGEND, frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  [OK] dataset view  -> {out_path.name}  ({n_bg} class + {n_star} putative, "
          f"{n_split} split)")
    if covered:
        print(f"    [WARN] {len(covered)} star(s) mostly hidden under a later one: "
              + ", ".join(short_label(q) for q in covered))


# Weighted-q scorecard.
def _darker(color, factor=GRID_SEL_DARKEN):
    return tuple(c * factor for c in mcolors.to_rgb(color))


def q_to_frac(q, q_lo, q_hi):
    return np.clip((q_hi - np.asarray(q, dtype=float)) / max(q_hi - q_lo, 1e-9), 0.0, 1.0)


def q_to_area(q, q_lo, q_hi):
    return GRID_DOT_MIN + (GRID_DOT_MAX - GRID_DOT_MIN) * q_to_frac(q, q_lo, q_hi)


def _tint(color, frac):
    t = GRID_FILL_MIN + (1.0 - GRID_FILL_MIN) * float(frac)
    return tuple(1.0 - t + t * c for c in mcolors.to_rgb(color))


def draw_q_key(fig, rect, q_lo, q_hi):
    if not q_hi > q_lo + 1e-9:
        return None
    kax = fig.add_axes(rect)
    qs = np.linspace(q_lo, q_hi, KEY_SAMPLES)
    radius = np.sqrt(q_to_area(qs, q_lo, q_hi) / np.pi)
    r_max = float(radius[0])

    ramp = np.array([_tint(KEY_COLOR, f) for f in q_to_frac(qs, q_lo, q_hi)])
    img = kax.imshow(ramp.reshape(-1, 1, 3), extent=(-r_max, r_max, q_hi, q_lo),
                     origin="upper", aspect="auto", interpolation="bilinear", zorder=1)
    wedge = np.concatenate([np.column_stack([radius, qs]),
                            np.column_stack([-radius[::-1], qs[::-1]])])
    img.set_clip_path(mpatches.Polygon(wedge, closed=True, transform=kax.transData))
    kax.add_patch(mpatches.Polygon(wedge, closed=True, facecolor="none", edgecolor=KEY_COLOR,
                                   linewidth=GRID_RIM_LW, clip_on=False, zorder=2))

    kax.set_xlim(-r_max, r_max)
    kax.set_ylim(q_hi, q_lo)
    kax.set_xticks([])
    kax.set_yticks(np.linspace(q_lo, q_hi, 5))
    kax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    kax.yaxis.tick_right()
    kax.tick_params(axis="y", length=1.5, width=0.4, pad=1.5, labelsize=FONT_TICK)
    for spine in kax.spines.values():
        spine.set_visible(False)
    kax.set_title("weighted q", fontsize=FONT_TICK, pad=4)
    return kax


def fig_qvalue_grid(wq, tied, order, out_path, *, method_label, cutoff, q_lo, dpi=400):
    n = len(order)
    ax_w, ax_h = GRID_QUERY_W * n, GRID_CLASS_H * len(CLASSES)
    left, right, top = 0.62, 1.45, 0.30
    bottom = label_margin(order, minimum=1.45)
    fig_w, fig_h = ax_w + left + right, ax_h + top + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, bottom / fig_h, ax_w / fig_w, ax_h / fig_h])

    for j in range(0, n, 2):
        ax.axvspan(j - 0.5, j + 0.5, facecolor=GRID_BAND_COLOR, edgecolor="none", zorder=0)
    for i in range(1, len(CLASSES)):
        ax.axhline(i - 0.5, color=GRID_SEP_COLOR, linewidth=0.5, linestyle=(0, (4, 3)),
                   zorder=2)

    x = np.arange(n)
    for i, cls in enumerate(CLASSES):
        q = wq.loc[order, cls].to_numpy(dtype=float)
        ok = ~np.isnan(q)
        if not ok.any():
            continue
        sel = ok & np.array([cls in tied.get(qid, ()) for qid in order])
        for mask, edge, lw, z in ((ok & ~sel, CLASS_COLORS[cls], GRID_RIM_LW, 3),
                                  (sel, _darker(CLASS_COLORS[cls]), GRID_SEL_LW, 4)):
            if mask.any():
                frac = q_to_frac(q[mask], q_lo, cutoff)
                ax.scatter(x[mask], np.full(int(mask.sum()), i),
                           s=q_to_area(q[mask], q_lo, cutoff),
                           c=[_tint(CLASS_COLORS[cls], f) for f in frac],
                           edgecolors=edge, linewidths=lw, zorder=z)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(len(CLASSES) - 0.5, -0.5)              # Class 1 on the top row
    ax.set_xticks(range(n))
    ax.set_xticklabels(list(order), rotation=90, fontsize=FONT_TICK)
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels([f"Class {c}" for c in CLASSES], fontsize=FONT_TICK)
    for tick, cls in zip(ax.get_yticklabels(), CLASSES):
        tick.set_color(CLASS_COLORS[cls])
        tick.set_fontweight("bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{method_label}  —  Weighted q-value per class "
                 f"(eFDP threshold = {cutoff:g})",
                 fontsize=FONT_TITLE, fontweight="bold", pad=5)

    draw_q_key(fig, [(left + ax_w + KEY_GAP) / fig_w, bottom / fig_h,
                     KEY_WIDTH / fig_w, ax_h / fig_h], q_lo, cutoff)

    # Mark selected winners.
    n_split = sum(len(tied.get(qid, ())) > 1 for qid in order)
    ax.legend(handles=[mlines.Line2D([], [], marker="o", linestyle="None",
                                     markerfacecolor="#888888",
                                     markeredgecolor=_darker("#888888"),
                                     markeredgewidth=GRID_SEL_LW, markersize=6,
                                     label="assigned class" + (" (a tie rings every winner)"
                                                               if n_split else ""))],
              loc="upper left", frameon=False, fontsize=FONT_LEGEND,
              handletextpad=0.8, borderpad=0.0,
              bbox_to_anchor=(1.0 + KEY_GAP / ax_w, -0.35))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  [OK] q-value grid  -> {out_path.name}")


# Driver.
def run_method(method, *, data_dir, out_dir, cutoff, assign_how, order_how, kinds, q_lo,
               q_source, coords, ids, id2class, putative, dpi):
    hits, detail = load_q_table(data_dir, method, cutoff, q_source)
    if hits is None:
        if q_source == "grid":
            print(f"[skip] {method}: no hits_{method}.tsv in {data_dir}")
        return
    if hits.empty:
        print(f"[skip] {method}: no hits at q <= {cutoff}")
        return

    counts, pct, wq = per_class_tables(hits, id2class, putative, cutoff)
    tied = best_classes(counts, wq, assign_how)
    assigned = pd.Series({q: (t[0] if t else None) for q, t in tied.items()}, dtype=object)
    majority = assign_class(counts, wq, "majority")
    order = order_queries(putative, assigned, wq, order_how)

    n_assigned = int(assigned.notna().sum())
    tally = {c: int((assigned == c).sum()) for c in CLASSES}
    print(f"\n=== {method}: {len(hits):,} hits ({q_source} q), {n_assigned}/{len(putative)} "
          f"queries assigned ({assign_how}) -> class {tally}")
    if detail is not None and not detail.empty:
        nq = detail["q_hit"].round(9).nunique()
        print(f"    q per hit: {detail['q_hit'].min():.4f} .. {detail['q_hit'].max():.4f}, "
              f"{nq} distinct value(s)")
    if assign_how == "weighted":
        n_diff = int((assigned.notna() & (assigned != majority)).sum())
        print(f"    weighted-q and majority disagree on {n_diff} query(ies)")
    ties = {q: t for q, t in tied.items() if len(t) > 1}
    if ties:
        print(f"    {len(ties)} tie(s): " +
              ", ".join(f"{short_label(q)}[{'='.join(t)}]" for q, t in sorted(ties.items())))

    mlabel = METHOD_LABEL.get(method, method)
    out_dir.mkdir(parents=True, exist_ok=True)
    if detail is not None and not detail.empty:
        hit_tsv = out_dir / f"{method}_hit_qvalues.tsv"
        (detail.assign(target_class=detail["tid"].map(id2class))
               .sort_values(["qid", "hit_rank"])
               .to_csv(hit_tsv, sep="\t", index=False))
        print(f"  [OK] per-hit q     -> {hit_tsv.name} ({len(detail):,} rows)")
    tsv = out_dir / f"{method}_class_assignment.tsv"
    assignment_table(counts, pct, wq, tied, majority,
                     q_source=q_source, detail=detail).to_csv(tsv, sep="\t", index=False)
    print(f"  [OK] table         -> {tsv.name}")

    hits_long, terms = audit_tables(hits, detail, id2class, putative,
                                    counts, wq, tied, cutoff)
    hits_tsv = out_dir / f"{method}_wq_audit_hits.tsv"
    terms_tsv = out_dir / f"{method}_wq_audit_terms.tsv"
    hits_long.to_csv(hits_tsv, sep="\t", index=False, float_format="%.9g")
    terms.to_csv(terms_tsv, sep="\t", index=False, float_format="%.9g")
    print(f"  [OK] wq audit      -> {hits_tsv.name} ({len(hits_long):,} rows), "
          f"{terms_tsv.name} ({len(terms):,} rows)")

    if q_lo is None:
        q_lo = float(np.nanmin(wq.to_numpy(dtype=float))) if wq.notna().any().any() else 0.0

    if "bar" in kinds:
        fig_class_composition(pct, counts, order, out_dir / f"{method}_class_composition.pdf",
                              method_label=mlabel, cutoff=cutoff, dpi=dpi)
    if "map" in kinds:
        fig_dataset_view(coords, ids, id2class, putative, tied,
                         out_dir / f"{method}_dataset_view.pdf",
                         method_label=mlabel, cutoff=cutoff, dpi=dpi)
    if "grid" in kinds:
        fig_qvalue_grid(wq, tied, order, out_dir / f"{method}_qvalue_grid.pdf",
                        method_label=mlabel, cutoff=cutoff, q_lo=q_lo, dpi=dpi)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help="tsne_per_query dir holding hits_*.tsv / coords.npy / id2class.tsv")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: data/plot_data/bagel/addon/<source dir name>")
    ap.add_argument("--methods", nargs="+", default=["plm", "tmvec", "dhr", "blastp"])
    ap.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF, help="q cutoff (default 0.6)")
    ap.add_argument("--assign", choices=("weighted", "majority"), default="weighted")
    ap.add_argument("--order", choices=("true-class", "assigned"), default="true-class",
                    help="column order of the bar and the scorecard (they always match)")
    ap.add_argument("--only", nargs="+", default=list(FIGURE_KINDS), choices=FIGURE_KINDS,
                    help="which figure kinds to draw")
    ap.add_argument("--q-source", choices=Q_SOURCES, default="rank",
                    help="rank: each hit's own q from the score ranking (default); "
                         "grid: the legacy q quantised to the six scanned levels")
    ap.add_argument("--q-lo", default="auto",
                    help="q mapped to the largest/darkest dot in the grid figure "
                         "(default auto: that method's own best weighted q)")
    ap.add_argument("--dpi", type=int, default=400)
    return ap


def main():
    args = build_parser().parse_args()
    data_dir = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data dir: {data_dir}")

    tag = data_dir.parent.name
    out_dir = args.out_dir or (BAGEL_PLOT_DIR / "addon" / tag)
    coords, ids, id2class, putative = load_context(data_dir)
    q_lo = None if args.q_lo == "auto" else float(args.q_lo)
    if q_lo is None and args.q_source == "grid":
        q_lo = min(load_q_levels(data_dir))

    print(f"[addon] source={data_dir}\n[addon] out={out_dir}\n"
          f"[addon] cutoff q<={args.cutoff}  q-source={args.q_source}  assign={args.assign}  "
          f"order={args.order}  queries={len(putative)}  figures={','.join(args.only)}")

    for method in args.methods:
        run_method(method, data_dir=data_dir, out_dir=out_dir, cutoff=args.cutoff,
                   assign_how=args.assign, order_how=args.order, kinds=set(args.only), q_lo=q_lo,
                   q_source=args.q_source, coords=coords, ids=ids, id2class=id2class,
                   putative=putative, dpi=args.dpi)
    print(f"\n[DONE] {out_dir}")


if __name__ == "__main__":
    main()
