import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
DB_DIR = DATA_DIR / "db"
BAGEL_PLOT_DIR = DATA_DIR / "bagel"
PLOT_DATA_DIR = BAGEL_PLOT_DIR / "tsne_per_query"
CLASS_ALL_FA = DATA_DIR / "class_all.fa"
PUTATIVE_FA = DATA_DIR / "putative.fa"

TMVEC_SRC = BASE_DIR / "libs" / "tm-vec-master"
BUILD_DB_SCRIPT = TMVEC_SRC / "scripts" / "tmvec-build-database"
PROTRANS_MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"

DECOY_METHOD = "extended_shuf"
Q_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
METHOD_DISK = {"plm": "plm", "tmvec": "tmvec", "dhr": "dhr_postprocess",
               "blastp": "blastp_postprocessed"}

FONT_LABEL, FONT_TITLE, FONT_TICK, FONT_LEGEND = 7, 7, 6, 6
_SANS_PREF = ["Helvetica", "Arial", "Nimbus Sans", "Liberation Sans",
              "TeX Gyre Heros", "DejaVu Sans"]
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif", "axes.unicode_minus": False,
    "font.sans-serif": _SANS_PREF + list(matplotlib.rcParams["font.sans-serif"]),
    "font.size": FONT_TICK, "axes.labelsize": FONT_LABEL, "axes.titlesize": FONT_TITLE,
    "xtick.labelsize": FONT_TICK, "ytick.labelsize": FONT_TICK,
    "legend.fontsize": FONT_LEGEND, "axes.linewidth": 0.6, "lines.linewidth": 0.9,
    "lines.markersize": 3.0, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "savefig.dpi": 300, "legend.frameon": True, "legend.framealpha": 1.0,
})

CLASS_MARKERS = {"1": "o", "2": "s", "3": "^"}
CLASS_COLORS = {"1": "#0173B2", "2": "#DE8F05", "3": "#029E73", "Unknown": "#BBBBBB"}
Q_BAND_CMAP = "plasma"
ACCEPTED_HIT_SIZE = 18
QUERY_STAR_SIZE = 90
SUMMARY_AX_SIZE = 3.4
SUMMARY_BG_SIZE = 18
SUMMARY_BG_ALPHA = 0.18
SUMMARY_STAR_SIZE = QUERY_STAR_SIZE
SUMMARY_STAR_EDGE_LW = 0.5


def safe_filename(text):
    return re.sub(r"[^\w.\-]+", "_", str(text))


def fasta_ids(path):
    """Sequence ids in file order, description dropped."""
    with open(path) as fh:
        return [line[1:].split(None, 1)[0].strip() for line in fh if line.startswith(">")]


def load_id2class_codes(fastas=None):
    """id -> BAGEL class code ("1"/"2"/"3")"""
    fastas = list(fastas or (CLASS_ALL_FA, PUTATIVE_FA))
    ids = []
    for fa in fastas:
        if not Path(fa).exists():
            raise SystemExit(f"[tsne] missing {fa}; needed for the BAGEL class labels")
        ids.extend(fasta_ids(fa))
    df = pd.DataFrame({"ids": ids}).drop_duplicates("ids", ignore_index=True)
    return df, df["ids"].map(class_from_qid)


def build_or_load_embeddings(fasta, db_out, device):
    db_npy, meta_npy = db_out / "db.npy", db_out / "meta.npy"
    if not (db_npy.exists() and meta_npy.exists()):
        print(f"[build] TM-Vec embedding {fasta.name} -> {db_out}")
        subprocess.run([
            sys.executable, str(BUILD_DB_SCRIPT),
            "--input-fasta", str(fasta),
            "--tm-vec-model", str(TMVEC_SRC / "model" / "tm_vec_cath_model.ckpt"),
            "--tm-vec-config-path", str(TMVEC_SRC / "model" / "tm_vec_cath_model_params.json"),
            "--protrans-model", PROTRANS_MODEL,
            "--device", device,
            "--output", str(db_out),
        ], check=True)
    else:
        print(f"[cache] reusing {db_out}")
    emb = np.load(db_npy)
    ids = [str(x) for x in np.load(meta_npy, allow_pickle=True)]
    print(f"        {fasta.name}: {emb.shape[0]} seqs, dim={emb.shape[1]}")
    return emb, ids


def run_coords(out_dir, putative_fa, class_all_fa, *, seed, perplexity, init,
               learning_rate, scale, device, force, l2_normalize=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [out_dir / "X_2d_all.npy", out_dir / "seqs_all.npy", out_dir / "labels_all.npy"]
    if all(t.exists() for t in targets) and not force:
        print(f"[cache] t-SNE coordinates already in {out_dir}; use --force to recompute")
        return

    # putative first, then class_all -- this ordering defines seqs_all
    emb_p, ids_p = build_or_load_embeddings(putative_fa, DB_DIR / "db_putative_tmvec", device)
    emb_c, ids_c = build_or_load_embeddings(class_all_fa, DB_DIR / "db_class_all_tmvec", device)

    X = np.vstack([emb_p, emb_c])
    ids = ids_p + ids_c

    # The published map was built from unit-length TM-Vec vectors: data/*_embeddings.npy
    # in the source tree match db.npy row-normalised, to float32. TM-Vec scores by cosine
    # similarity, so the norms carry no signal anyway.
    if l2_normalize:
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

    df, code = load_id2class_codes()
    id2class = dict(zip(df["ids"].astype(str), code))
    labels = ["Putative"] * len(ids_p) + [f"Class {id2class.get(i, '?')}" for i in ids_c]

    from sklearn.manifold import TSNE
    if scale:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)
    print(f"[tsne] N={X.shape[0]} dim={X.shape[1]} perplexity={perplexity} "
          f"seed={seed} init={init} scale={scale}")
    X_2d = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate,
                init=init, random_state=seed).fit_transform(X)

    np.save(out_dir / "X_2d_all.npy", X_2d.astype(np.float32))
    np.save(out_dir / "seqs_all.npy", np.asarray(ids, dtype=object))
    np.save(out_dir / "labels_all.npy", np.asarray(labels, dtype=object))
    print(f"[OK] wrote X_2d_all/seqs_all/labels_all.npy ({X_2d.shape[0]} points) -> {out_dir}")


def write_hits_for_method(out_dir, label, decoy_method, scan_dir=DATA_DIR):
    path = scan_dir / f"putative_discovery_hits_scan_{METHOD_DISK[label]}_{decoy_method}.tsv"
    if not path.exists():
        print(f"  [WARN] {label}: missing {path.name}; skip")
        return 0
    df = pd.read_csv(path, sep="\t")
    df["q"] = pd.to_numeric(df["q"], errors="coerce")
    df = df[df["q"].notna()]
    df = df[df["q"].apply(lambda q: any(np.isclose(q, ql) for ql in Q_LEVELS))]
    hits = (df[["qid", "q", "tid"]].astype({"qid": str, "tid": str})
            .drop_duplicates().sort_values(["qid", "q", "tid"]))
    hits.to_csv(out_dir / f"hits_{label}.tsv", sep="\t", index=False)
    return hits["qid"].nunique()


def run_prep(out_dir, methods, decoy_method, coords_dir=DATA_DIR, scan_dir=DATA_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)

    coords = np.load(Path(coords_dir) / "X_2d_all.npy")
    ids = np.load(Path(coords_dir) / "seqs_all.npy", allow_pickle=True).astype(str)
    if coords.shape[0] != ids.shape[0]:
        raise ValueError(f"coords/ids length mismatch: {coords.shape} vs {ids.shape}")
    np.save(out_dir / "coords.npy", coords)
    (out_dir / "ids.txt").write_text("\n".join(ids) + "\n")

    df, code = load_id2class_codes()
    df = df.assign(class_code=code)
    df[["ids", "class_code"]].to_csv(out_dir / "id2class.tsv", sep="\t",
                                     index=False, header=["id", "class"])
    print(f"[OK] coords + ids: {len(ids)} points   id2class: {len(df)} targets")

    for label in methods:
        print(f"[OK] hits_{label}.tsv: "
              f"{write_hits_for_method(out_dir, label, decoy_method, Path(scan_dir))} queries")

    meta_path = out_dir / "meta.json"
    existing = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta_path.write_text(json.dumps({
        "q_levels": Q_LEVELS,
        "decoy_method": decoy_method,
        "methods": sorted(set(existing.get("methods", [])) | set(methods)),
    }, indent=2))
    print(f"[DONE] plotting data -> {out_dir}")


def class_from_qid(qid):
    parts = qid.split(";", 1)[0].split(".")
    return parts[1] if len(parts) >= 2 and parts[1] in {"1", "2", "3"} else "?"


def _short_qid(qid):
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


def partition_hits_by_first_q(hits_by_q, q_levels):
    """Each hit is attributed to the lowest q at which it first appears."""
    q_levels = sorted(q_levels)
    tid2first_q, new_by_q, seen = {}, {q: set() for q in q_levels}, set()
    for q in q_levels:
        new_tids = hits_by_q.get(q, set()) - seen
        new_by_q[q] = new_tids
        for tid in new_tids:
            tid2first_q[tid] = q
        seen |= hits_by_q.get(q, set())
    return tid2first_q, new_by_q


def _plot_q_color_bars(ax_qbar, q_levels, q_colors, new_by_q):
    q_levels = sorted(q_levels)
    counts = [len(new_by_q.get(q, set())) for q in q_levels]
    y_pos = np.arange(len(q_levels))
    ax_qbar.barh(y_pos, counts, color=[q_colors[q] for q in q_levels],
                 edgecolor="black", linewidth=0.4, height=0.72)
    ax_qbar.set_yticks(y_pos)
    ax_qbar.set_yticklabels([f"q = {q:.2f}" for q in q_levels], fontsize=FONT_TICK)
    ax_qbar.set_title("q band colors", fontsize=FONT_TITLE)
    ax_qbar.invert_yaxis()
    if max(counts, default=0) == 0:
        ax_qbar.set_xlim(0, 1)
        ax_qbar.text(0.5, 0.5, "no hits", transform=ax_qbar.transAxes,
                     ha="center", va="center", fontsize=FONT_TICK, color="gray")


def _draw_query_tsne_q_bands(ax, ax_qbar, qid, hits_by_q, q_levels, X_2d, seq2idx,
                             id2class, *, bg_size=18):
    if qid not in seq2idx:
        return False
    q_levels = sorted(q_levels)
    cmap = plt.get_cmap(Q_BAND_CMAP)
    q_colors = {q: cmap(i / max(len(q_levels) - 1, 1)) for i, q in enumerate(q_levels)}
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
    _plot_q_color_bars(ax_qbar, q_levels, q_colors, new_by_q)
    ax_qbar.set_xlabel("# new hits", fontsize=FONT_TICK)
    ax_qbar.tick_params(labelsize=FONT_TICK)
    return True


def plot_query(qid, hits_by_q, q_levels, X_2d, ids, id2class, out_path, dpi=400):
    seq2idx = {s: i for i, s in enumerate(ids)}
    if qid not in seq2idx:
        print(f"  [WARN] {qid} not in ids; skip")
        return False

    fig = plt.figure(figsize=(7.2, 5.0))            # Nature double column
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.0], wspace=0.22)
    ax, ax_qbar = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
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


def hits_by_q_per_query(data_dir, label, q_levels):
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


def run_plot(data_dir, plot_dir, methods, decoy_method):
    coords = np.load(data_dir / "coords.npy")
    ids = np.array([s for s in (data_dir / "ids.txt").read_text().split("\n") if s]).astype(str)
    dfc = pd.read_csv(data_dir / "id2class.tsv", sep="\t", dtype=str)
    id2class = dict(zip(dfc["id"], dfc["class"]))
    meta = json.loads((data_dir / "meta.json").read_text())
    q_levels = meta["q_levels"]

    for label in methods or meta["methods"]:
        if not (data_dir / f"hits_{label}.tsv").exists():
            print(f"[skip] {label}: no hits_{label}.tsv")
            continue
        hits = hits_by_q_per_query(data_dir, label, q_levels)
        print(f"\n=== {label}: {len(hits)} queries ===")
        n_ok = 0
        for qid, hits_by_q in hits.items():
            q_min = min_discoverable_q(hits_by_q, q_levels)
            subdir = (f"q_{q_min:.2f}" if q_min is not None
                      else f"no_hit_above_q{max(q_levels):.2f}")
            out_path = (plot_dir / label / subdir
                        / f"tsne_{decoy_method}_{safe_filename(qid)}.pdf")
            n_ok += int(plot_query(qid, hits_by_q, q_levels, coords, ids, id2class, out_path))
        print(f"[OK] {label}: {n_ok} figures -> {plot_dir / label}")


def load_putative_ids():
    """Ids of the putative queries, in the order the figures lay them out."""
    if IDS_NPY.exists() and LABELS_NPY.exists():
        ids = np.load(IDS_NPY, allow_pickle=True).astype(str)
        labels = np.load(LABELS_NPY, allow_pickle=True).astype(str)
        putative = {i for i, lab in zip(ids, labels) if lab == "Putative"}
        if putative:
            return putative
    if PUTATIVE_FA.exists():
        return {ln[1:].split()[0] for ln in PUTATIVE_FA.read_text().splitlines()
                if ln.startswith(">")}
    raise FileNotFoundError(
        f"Cannot tell which points are putative: need {LABELS_NPY} or {PUTATIVE_FA}")


def plot_summary(X_2d, ids, id2class, putative_ids, out_path, *,
                 bg_alpha=SUMMARY_BG_ALPHA, star_size=SUMMARY_STAR_SIZE, dpi=400):
    """All BAGEL classes as backdrop, with putative queries starred by BAGEL class."""
    seq2idx = {s: i for i, s in enumerate(ids)}

    # margins in inches: ylabel/xlabel gutters plus room for the legend strip below
    left, right, top, bottom = 0.58, 0.08, 0.12, 0.95
    fig_w, fig_h = SUMMARY_AX_SIZE + left + right, SUMMARY_AX_SIZE + top + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, bottom / fig_h,
                       SUMMARY_AX_SIZE / fig_w, SUMMARY_AX_SIZE / fig_h])
    n_bg = 0
    for cls, marker in list(CLASS_MARKERS.items()) + [("Unknown", "o")]:
        idxs = [i for s, i in seq2idx.items()
                if s not in putative_ids and id2class.get(s, "Unknown") == cls]
        if not idxs:
            continue
        idxs = np.array(idxs)
        n_bg += len(idxs)
        ax.scatter(X_2d[idxs, 0], X_2d[idxs, 1], c=CLASS_COLORS[cls], marker=marker,
                   s=SUMMARY_BG_SIZE, alpha=bg_alpha, linewidths=0, zorder=1)

    missing = sorted(p for p in putative_ids if p not in seq2idx)
    if missing:
        print(f"  [WARN] {len(missing)} putative id(s) absent from the coords: {missing[:3]}...")
    n_putative = 0
    for cls in ("1", "2", "3", "Unknown"):
        pidx = np.array([
            seq2idx[p] for p in sorted(putative_ids)
            if p in seq2idx and id2class.get(p, "Unknown") == cls
        ])
        if not pidx.size:
            continue
        n_putative += pidx.size
        ax.scatter(X_2d[pidx, 0], X_2d[pidx, 1], c=CLASS_COLORS[cls],
                   marker="*", s=star_size, edgecolors="black",
                   linewidths=SUMMARY_STAR_EDGE_LW, zorder=100)

    ax.set_xlabel("t-SNE 1", fontsize=FONT_LABEL, fontweight="bold")
    ax.set_ylabel("t-SNE 2", fontsize=FONT_LABEL, fontweight="bold")
    ax.tick_params(labelsize=FONT_TICK)

    handles = [
        mlines.Line2D([], [], marker=CLASS_MARKERS[c], linestyle="None",
                      markerfacecolor=CLASS_COLORS[c], markeredgecolor=CLASS_COLORS[c],
                      markersize=5, alpha=0.7, label=f"Class {c}")
        for c in ("1", "2", "3")
    ] + [mlines.Line2D([], [], marker="*", linestyle="None", markerfacecolor="white",
                       markeredgecolor="black", markeredgewidth=SUMMARY_STAR_EDGE_LW,
                       markersize=8, label="putative query")]

    # anchored to the axes (not the figure) so it stays centred under the map
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11),
              ncol=4, fontsize=FONT_LEGEND, frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"[OK] summary map: {n_bg} class + {n_putative} putative points -> {out_path}")


def run_summary(data_dir, out_path, *, bg_alpha=SUMMARY_BG_ALPHA,
                star_size=SUMMARY_STAR_SIZE):
    coords = np.load(data_dir / "coords.npy")
    ids = np.array([s for s in (data_dir / "ids.txt").read_text().split("\n") if s]).astype(str)
    dfc = pd.read_csv(data_dir / "id2class.tsv", sep="\t", dtype=str)
    id2class = dict(zip(dfc["id"], dfc["class"]))
    plot_summary(coords, ids, id2class, load_putative_ids(), Path(out_path),
                 bg_alpha=bg_alpha, star_size=star_size)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", default="all",
                    choices=["all", "coords", "prep", "plot", "summary"])
    ap.add_argument("--force", action="store_true", help="recompute the t-SNE even if cached")
    ap.add_argument("--methods", nargs="+", default=["plm", "tmvec", "dhr"],
                    choices=("plm", "tmvec", "dhr", "blastp"))
    ap.add_argument("--decoy-method", default=DECOY_METHOD)
    ap.add_argument("--coords-dir", type=Path, default=DATA_DIR,
                    help="where X_2d_all/seqs_all/labels_all.npy are written and read")
    ap.add_argument("--scan-dir", type=Path, default=DATA_DIR,
                    help="where discover wrote putative_discovery_hits_scan_*.tsv")
    ap.add_argument("--data-dir", type=Path, default=PLOT_DATA_DIR)
    ap.add_argument("--plot-dir", type=Path, default=PLOT_DATA_DIR / "figures")
    ap.add_argument("--summary-out", type=Path,
                    default=BAGEL_PLOT_DIR / "tsne_summary_putative.pdf")
    ap.add_argument("--bg-alpha", type=float, default=SUMMARY_BG_ALPHA,
                    help="transparency of the class backdrop in the summary map")
    ap.add_argument("--star-size", type=float, default=SUMMARY_STAR_SIZE)
    ap.add_argument("--putative-fa", type=Path, default=DATA_DIR / "putative.fa")
    ap.add_argument("--class-all-fa", type=Path, default=DATA_DIR / "class_all.fa")
    # These are the settings the published bagel map was computed with:
    # TSNE(n_components=2, perplexity=15, random_state=0) on the raw TM-Vec
    # embeddings, so init/learning_rate are sklearn's pre-1.2 defaults and the
    # features are not standardised. The coordinates themselves are not
    # bit-reproducible -- the TM-Vec embeddings have drifted since, and
    # sklearn's t-SNE has changed -- so the map is equivalent, not identical.
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perplexity", type=float, default=15.0)
    ap.add_argument("--init", default="random", choices=("pca", "random"))
    ap.add_argument("--learning-rate", default=200.0)
    ap.add_argument("--scale", action="store_true",
                    help="standardise the embeddings first (the published map did not)")
    ap.add_argument("--no-l2", action="store_true",
                    help="skip the unit-length normalisation the published map used")
    ap.add_argument("--device", default="gpu", choices=("gpu", "cpu"))
    return ap


def main():
    args = build_parser().parse_args()

    if args.stage in ("all", "coords"):
        run_coords(args.coords_dir, args.putative_fa, args.class_all_fa,
                   seed=args.seed, perplexity=args.perplexity, init=args.init,
                   learning_rate=args.learning_rate, scale=args.scale,
                   device=args.device, force=args.force, l2_normalize=not args.no_l2)
    if args.stage in ("all", "prep"):
        run_prep(args.data_dir, args.methods, args.decoy_method,
                 args.coords_dir, args.scan_dir)
    if args.stage in ("all", "plot"):
        run_plot(args.data_dir, args.plot_dir, args.methods, args.decoy_method)
    if args.stage in ("all", "summary"):
        run_summary(args.data_dir, args.summary_out,
                    bg_alpha=args.bg_alpha, star_size=args.star_size)


if __name__ == "__main__":
    main()
