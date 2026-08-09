import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# Subclass definitions (Class I)
# ============================================================

SUBCLASS_LABELS = (
    "lanthipeptide_A",
    "lanthipeptide_B",
    "Cyanobactin",
    "Linardin",
    "Other_ClassI",
)

SUBCLASS_MARKERS = {
    "lanthipeptide_A": "s",
    "lanthipeptide_B": "^",
    "Cyanobactin": "D",
    "Linardin": "p",
    "Other_ClassI": "o",
}

SUBCLASS_COLORS = {
    "lanthipeptide_A": "#7B2D8E",   # purple
    "lanthipeptide_B": "#0173B2",   # blue
    "Cyanobactin": "#DE8F05",       # orange
    "Linardin": "#029E73",          # dark green
    "Other_ClassI": "#999999",      # gray
    "Unknown": "#BBBBBB",
}


# ============================================================
# Helpers
# ============================================================

def _get_subclass_from_mapping(qid, id2subclass):
    return id2subclass.get(qid, "Unknown")


def remove_decoy_suffix(qid, decoy_suffix):
    if qid.endswith(decoy_suffix):
        return qid[: -len(decoy_suffix)]
    return qid


def make_pair_table_keep_tid(path_real, path_decoy, decoy_suffix):
    df_real = pd.read_csv(path_real, sep="\t")
    df_decoy = pd.read_csv(path_decoy, sep="\t", on_bad_lines="skip")

    # Drop rows where qid literally equals "qid" (duplicate header rows)
    df_real = df_real[df_real["qid"] != "qid"].copy()
    df_decoy = df_decoy[df_decoy["qid"] != "qid"].copy()

    has_rep_real = "rep_id" in df_real.columns
    has_rep_decoy = "rep_id" in df_decoy.columns

    if not has_rep_real:
        df_real["rep_id"] = 0
    if not has_rep_decoy:
        df_decoy["rep_id"] = 0

    df_real = df_real[["qid", "tid", "homo_type", "score", "rep_id"]].copy()
    df_real = df_real.rename(columns={
        "score": "score_real",
        "homo_type": "homo_type_real",
    })
    df_real["score_real"] = pd.to_numeric(df_real["score_real"], errors="coerce")
    df_real["homo_type_real"] = pd.to_numeric(df_real["homo_type_real"], errors="coerce")

    df_decoy = df_decoy[["qid", "tid", "score", "rep_id"]].copy()
    df_decoy["score"] = pd.to_numeric(df_decoy["score"], errors="coerce")
    df_decoy["qid_orig"] = df_decoy["qid"].apply(
        lambda x: remove_decoy_suffix(x, decoy_suffix)
    )
    df_decoy = df_decoy.rename(columns={"score": "score_decoy"})

    df_merge = df_real.merge(
        df_decoy[["qid_orig", "tid", "score_decoy", "rep_id"]],
        left_on=["qid", "tid", "rep_id"],
        right_on=["qid_orig", "tid", "rep_id"],
        how="inner",
    )
    df_merge = df_merge.drop(columns=["qid_orig"], errors="ignore")

    df_merge["score_real"] = df_merge["score_real"].astype(np.float32)
    df_merge["score_decoy"] = df_merge["score_decoy"].astype(np.float32)
    if df_merge["homo_type_real"].dtype != object:
        df_merge["homo_type_real"] = df_merge["homo_type_real"].astype(np.int8)
    df_merge["rep_id"] = pd.to_numeric(df_merge["rep_id"], downcast="integer")
    return df_merge


# ============================================================
# Core: per-rep FDR-controlled selection + subclass voting
# ============================================================

def _empty_select_result():
    return {
        "T_q": float("nan"),
        "n_pass": 0,
        "n_subclass": {c: 0 for c in SUBCLASS_LABELS},
        "rep_pred": None,
        "eFDP_at_T": float("nan"),
    }


def _select_and_count(scores_t, scores_d, tid_subclasses, q):
    n_t = len(scores_t)
    n_d = len(scores_d)
    if n_t == 0:
        return _empty_select_result()

    cand_t = np.sort(np.unique(np.concatenate([scores_t, scores_d])))
    scores_t_sorted = np.sort(scores_t)
    scores_d_sorted = np.sort(scores_d)

    target_count = n_t - np.searchsorted(scores_t_sorted, cand_t, side="left")
    decoy_count = n_d - np.searchsorted(scores_d_sorted, cand_t, side="left")
    est_fdp = (decoy_count + 1.0) / np.maximum(target_count, 1.0)

    valid = np.where(est_fdp <= q)[0]
    if len(valid) == 0:
        return _empty_select_result()

    best_idx = valid[np.argmin(cand_t[valid])]
    T_q = float(cand_t[best_idx])
    eFDP_at_T = float(est_fdp[best_idx])
    n_pass = int(target_count[best_idx])

    pass_mask = scores_t >= T_q
    passing_subclasses = tid_subclasses[pass_mask]
    n_subclass = {
        c: int(np.sum(passing_subclasses == c)) for c in SUBCLASS_LABELS
    }

    if max(n_subclass.values()) == 0:
        rep_pred = None
    else:
        rep_pred = max(n_subclass, key=lambda k: n_subclass[k])

    return {
        "T_q": T_q,
        "n_pass": n_pass,
        "n_subclass": n_subclass,
        "rep_pred": rep_pred,
        "eFDP_at_T": eFDP_at_T,
    }


def _find_threshold(scores_t, scores_d, q):
    n_t = len(scores_t)
    if n_t == 0:
        return None

    cand_t = np.sort(np.unique(np.concatenate([scores_t, scores_d])))
    scores_t_sorted = np.sort(scores_t)
    scores_d_sorted = np.sort(scores_d)
    n_d = len(scores_d)

    target_count = n_t - np.searchsorted(scores_t_sorted, cand_t, side="left")
    decoy_count = (
        n_d - np.searchsorted(scores_d_sorted, cand_t, side="left")
        if n_d > 0 else np.zeros_like(cand_t)
    )
    est_fdp = (decoy_count + 1.0) / np.maximum(target_count, 1.0)

    valid = np.where(est_fdp <= q)[0]
    if len(valid) == 0:
        return None

    best_idx = valid[np.argmin(cand_t[valid])]
    return float(cand_t[best_idx])


def select_accepted_tids(df_qr, q):
    scores_t = df_qr["score_real"].values
    scores_d = df_qr["score_decoy"].values
    t_q = _find_threshold(scores_t, scores_d, q)
    if t_q is None:
        return []
    return df_qr.loc[df_qr["score_real"].values >= t_q, "tid"].astype(str).tolist()


def collect_accepted_hits_by_q(df_merge, q_levels, pool_reps="union"):
    q_levels = sorted(q_levels)
    hits = {}
    grouped = df_merge.groupby("qid", sort=False, observed=True)

    for qid, df_q in grouped:
        hits[qid] = {}
        for q in q_levels:
            tids = set()
            rep_ids = sorted(df_q["rep_id"].unique().tolist())
            if pool_reps == "rep0":
                rep_ids = [rep_ids[0]]
            for rep_id in rep_ids:
                df_qr = df_q[df_q["rep_id"] == rep_id]
                tids.update(select_accepted_tids(df_qr, q))
            hits[qid][q] = tids
    return hits


def safe_filename(text):
    return re.sub(r"[^\w.\-]+", "_", str(text))


def classify_putative_subclass(df_merge, id2subclass, q):
    rows = []
    grouped = df_merge.groupby("qid", sort=False, observed=True)

    for qid, df_q in tqdm(grouped, desc=f"Classify subclass @ q={q:.2f}"):
        rep_ids = sorted(df_q["rep_id"].unique().tolist())

        per_rep_n_subclass = []
        per_rep_pred = []
        per_rep_Tq = []
        per_rep_efdp = []
        per_rep_n_pass = []

        for rep_id in rep_ids:
            df_qr = df_q[df_q["rep_id"] == rep_id]
            tid_subclasses = df_qr["tid"].map(id2subclass).fillna("Unknown").values
            res = _select_and_count(
                df_qr["score_real"].values,
                df_qr["score_decoy"].values,
                tid_subclasses,
                q,
            )
            per_rep_n_subclass.append(res["n_subclass"])
            per_rep_pred.append(res["rep_pred"])
            per_rep_Tq.append(res["T_q"])
            per_rep_efdp.append(res["eFDP_at_T"])
            per_rep_n_pass.append(res["n_pass"])

        n_subclass_mean = {
            c: float(np.mean([d[c] for d in per_rep_n_subclass]))
            for c in SUBCLASS_LABELS
        }
        n_total_mean = sum(n_subclass_mean.values())

        if n_total_mean == 0:
            pred_subclass = "Unclassified"
        else:
            pred_subclass = max(n_subclass_mean, key=lambda k: n_subclass_mean[k])

        rep_votes = [p for p in per_rep_pred if p is not None]
        if len(rep_votes) == 0:
            subclass_agreement = 0.0
        else:
            subclass_agreement = sum(1 for v in rep_votes if v == pred_subclass) / len(rep_votes)

        if n_total_mean > 0 and pred_subclass != "Unclassified":
            pct_dominant = n_subclass_mean[pred_subclass] / n_total_mean
        else:
            pct_dominant = 0.0

        row = {
            "qid": qid,
            "pred_subclass": pred_subclass,
        }
        for sc in SUBCLASS_LABELS:
            row[f"n_{sc}_mean"] = n_subclass_mean[sc]
        row.update({
            "n_total_mean": n_total_mean,
            "pct_dominant": pct_dominant,
            "subclass_agreement": subclass_agreement,
            "n_reps_passed": len(rep_votes),
            "T_q_mean": float(np.nanmean(per_rep_Tq)),
            "eFDP_at_T_mean": float(np.nanmean(per_rep_efdp)),
            "bagel_subclass": _get_subclass_from_mapping(qid, id2subclass),
        })
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# t-SNE projection
# ============================================================

def _safe_load_pkl_emb(pkl_path):
    if not os.path.exists(pkl_path):
        return None
    try:
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
    except ModuleNotFoundError as e:
        print(f"[WARN] cannot unpickle {pkl_path}: {e}")
        return None

    out = {}
    for k, v in d.items():
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        elif hasattr(v, "numpy"):
            v = v.numpy()
        out[str(k)] = np.asarray(v).ravel()
    return out


def get_tsne_projection(
    plm_putative_pkl,
    plm_target_pkl,
    fallback_X_2d_path,
    fallback_seqs_path,
    cache_X_2d_path,
    cache_seqs_path,
    perplexity=15,
    seed=0,
):
    if os.path.exists(cache_X_2d_path) and os.path.exists(cache_seqs_path):
        print(f"[INFO] Loading cached PLM t-SNE: {cache_X_2d_path}")
        X_2d = np.load(cache_X_2d_path)
        ids = np.load(cache_seqs_path, allow_pickle=True)
        return X_2d, np.asarray(ids).astype(str), "PLM (cached)"

    putative_emb = _safe_load_pkl_emb(plm_putative_pkl)
    target_emb = _safe_load_pkl_emb(plm_target_pkl)
    if putative_emb is not None and target_emb is not None:
        try:
            from sklearn.manifold import TSNE
        except Exception as e:
            print(f"[WARN] sklearn unavailable: {e}")
            putative_emb = None

    if putative_emb is not None and target_emb is not None:
        print(f"[INFO] Computing PLM t-SNE from {len(putative_emb)} putative + "
              f"{len(target_emb)} target embeddings ...")
        ids = list(putative_emb.keys()) + list(target_emb.keys())
        X = np.vstack(
            [putative_emb[i] for i in putative_emb]
            + [target_emb[i] for i in target_emb]
        )
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed)
        X_2d = tsne.fit_transform(X)
        np.save(cache_X_2d_path, X_2d)
        np.save(cache_seqs_path, np.asarray(ids))
        print(f"[OK] Saved PLM t-SNE -> {cache_X_2d_path}")
        return X_2d, np.asarray(ids), "PLM (newly computed)"

    if os.path.exists(fallback_X_2d_path) and os.path.exists(fallback_seqs_path):
        print(f"[WARN] PLM embeddings unavailable; falling back to TM-Vec t-SNE "
              f"({fallback_X_2d_path}).")
        X_2d = np.load(fallback_X_2d_path)
        ids = np.load(fallback_seqs_path, allow_pickle=True)
        return X_2d, np.asarray(ids).astype(str), "TM-Vec (fallback)"

    raise FileNotFoundError(
        "No t-SNE source found. Provide PLM embedding .pkl files or "
        "TM-Vec X_2d_all.npy + seqs_all.npy."
    )


# ============================================================
# Cumulative fraction + eFDP
# ============================================================

def plot_cumulative_fraction_for_query(
    df_qr,
    qid,
    out_path,
    score_col_real="score_real",
    score_col_decoy="score_decoy",
    homo_col="homo_type_real",
    dpi=300,
):
    df_q = df_qr.sort_values(score_col_real, ascending=False).copy()
    if len(df_q) == 0:
        print(f"  [WARN] No data for cumulative plot: {qid}")
        return False

    scores_t = df_q[score_col_real].values
    scores_d = df_q[score_col_decoy].values
    homo_types = df_q[homo_col].values
    is_homo = np.isin(homo_types, [1, 2])

    n_target = len(scores_t)
    n_decoy = len(scores_d)
    if n_target == 0 or n_decoy == 0:
        print(f"  [WARN] Empty target/decoy for cumulative plot: {qid}")
        return False

    counts_t = np.arange(1, n_target + 1)
    target_frac = counts_t / float(n_target)

    scores_d_asc = np.sort(scores_d)
    counts_d = n_decoy - np.searchsorted(scores_d_asc, scores_t, side="left")
    decoy_frac = counts_d / float(n_decoy)

    efdp = (counts_d * (n_target / n_decoy) + 1) / counts_t

    min_frac = 1e-6
    target_frac_plot = np.maximum(target_frac, min_frac)
    decoy_frac_plot = np.maximum(decoy_frac, min_frac)

    fig, ax1 = plt.subplots(figsize=(6, 6))
    ax1.plot(
        [min_frac, 1], [min_frac, 1],
        color="gray", linestyle="--", linewidth=1.5, zorder=1,
    )
    ax1.scatter(
        target_frac_plot[~is_homo],
        decoy_frac_plot[~is_homo],
        color="steelblue", marker="o", s=20, alpha=0.6,
        label="Non-homolog", zorder=2,
    )
    if np.any(is_homo):
        ax1.scatter(
            target_frac_plot[is_homo],
            decoy_frac_plot[is_homo],
            color="red", marker="*", s=120,
            edgecolors="darkred", linewidths=0.8,
            label="Homolog", zorder=3,
        )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    pad_factor = 2.0
    ax1.set_xlim(min_frac / pad_factor, 1 * pad_factor)
    ax1.set_ylim(min_frac / pad_factor, 1 * pad_factor)
    ticks = [10 ** i for i in range(-6, 1)]
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax1.set_xlabel("Cumulative target fraction\nup to threshold", fontsize=14)
    ax1.set_ylabel("Cumulative decoy fraction\nup to threshold", fontsize=14)
    ax1.legend(loc="lower right", fontsize=11)

    ax2 = ax1.twinx()
    ax2.plot(
        target_frac, efdp,
        color="mediumseagreen", linestyle="-", linewidth=2, alpha=0.9,
        label="eFDP", zorder=4,
    )
    ax2.legend(loc="upper left", fontsize=11)
    ax2.set_yscale("linear")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_yticks(np.linspace(0, 1, 6))
    ax2.set_ylabel("eFDP", fontsize=14, rotation=270, labelpad=18)
    ax2.tick_params(axis="y", labelsize=11)

    rep_label = ""
    if "rep_id" in df_q.columns and df_q["rep_id"].nunique() == 1:
        rep_label = f"\nrep {int(df_q['rep_id'].iloc[0])}"

    ax1.text(
        0.95, 0.05, f"Query:\n{_short_qid(qid)}{rep_label}",
        transform=ax1.transAxes,
        fontsize=12, horizontalalignment="right", verticalalignment="bottom",
    )

    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
    ax1.tick_params(width=1.5, length=5, labelsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def plot_all_queries_cumfrac(
    df_merge,
    out_dir,
    decoy_method,
    q_levels,
    rep_id=0,
):
    os.makedirs(out_dir, exist_ok=True)
    q_levels = sorted(q_levels)
    hits_all = collect_accepted_hits_by_q(df_merge, q_levels, pool_reps="union")
    qids = df_merge["qid"].astype(str).unique().tolist()

    n_ok = 0
    for qid in tqdm(qids, desc="Cumulative per query"):
        hits_by_q = hits_all[qid]
        q_min = min_discoverable_q(hits_by_q, q_levels)
        subdir = q_discovery_folder_name(q_min, q_levels)
        subdir_path = os.path.join(out_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)

        df_q = df_merge[df_merge["qid"].astype(str) == qid]
        if rep_id is not None:
            df_q = df_q[df_q["rep_id"] == rep_id]
        if len(df_q) == 0:
            continue

        out_path = os.path.join(
            subdir_path,
            f"cumfrac_{decoy_method}_{safe_filename(qid)}.png",
        )
        if plot_cumulative_fraction_for_query(df_q, qid, out_path):
            n_ok += 1

    print(f"[OK] Per-query cumulative plots -> {out_dir} ({n_ok}/{len(qids)} saved)")
    return n_ok


def load_raw_scores(path_raw):
    df = pd.read_csv(path_raw, sep="\t")
    df = df[df["qid"] != "qid"].copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return dict(zip(zip(df["qid"], df["tid"]), df["score"]))


def plot_score_by_subclass_for_query(
    df_qr,
    qid,
    id2subclass,
    raw_score_map,
    out_path,
    dpi=300,
    bins=45,
):
    df_q = df_qr.copy()
    if len(df_q) == 0:
        print(f"  [WARN] No data for score-by-subclass plot: {qid}")
        return False

    tids_unique = df_q["tid"].astype(str).unique()
    rows = []
    for tid in tids_unique:
        raw = raw_score_map.get((qid, tid))
        if raw is None:
            continue
        rows.append({"tid": tid, "score_raw": float(raw)})

    if not rows:
        print(f"  [WARN] No raw scores found for: {qid}")
        return False

    df_raw = pd.DataFrame(rows)
    df_raw["subclass_code"] = df_raw["tid"].map(id2subclass).fillna("Unknown")

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for sc in SUBCLASS_LABELS:
        scores = df_raw.loc[df_raw["subclass_code"] == sc, "score_raw"].values
        if len(scores) == 0:
            continue
        ax.hist(
            scores,
            bins=bins,
            density=True,
            alpha=0.35,
            color=SUBCLASS_COLORS[sc],
            edgecolor="none",
            label=f"{sc} (n={len(scores)})",
        )
        plotted = True

    unknown = df_raw.loc[df_raw["subclass_code"] == "Unknown", "score_raw"].values
    if len(unknown) > 0:
        ax.hist(
            unknown,
            bins=bins,
            density=True,
            alpha=0.25,
            color=SUBCLASS_COLORS["Unknown"],
            edgecolor="none",
            label=f"Unknown (n={len(unknown)})",
        )
        plotted = True

    if not plotted:
        print(f"  [WARN] No subclass-labeled scores for: {qid}")
        plt.close(fig)
        return False

    ax.set_xlabel("Raw target score (original)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"{_short_qid(qid)}", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.tick_params(labelsize=10)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def plot_all_queries_score_by_subclass(
    df_merge,
    id2subclass,
    raw_score_map,
    out_dir,
    decoy_method,
    q_levels,
):
    os.makedirs(out_dir, exist_ok=True)
    q_levels = sorted(q_levels)
    hits_all = collect_accepted_hits_by_q(df_merge, q_levels, pool_reps="union")
    qids = df_merge["qid"].astype(str).unique().tolist()

    n_ok = 0
    for qid in tqdm(qids, desc="Score-by-subclass per query"):
        hits_by_q = hits_all[qid]
        q_min = min_discoverable_q(hits_by_q, q_levels)
        subdir = q_discovery_folder_name(q_min, q_levels)
        subdir_path = os.path.join(out_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)

        df_q = df_merge[df_merge["qid"].astype(str) == qid]
        if len(df_q) == 0:
            continue

        out_path = os.path.join(
            subdir_path,
            f"score_by_subclass_{decoy_method}_{safe_filename(qid)}.png",
        )
        if plot_score_by_subclass_for_query(
            df_q, qid, id2subclass, raw_score_map, out_path
        ):
            n_ok += 1

    print(f"[OK] Per-query score-by-subclass plots -> {out_dir} ({n_ok}/{len(qids)} saved)")
    return n_ok


# ============================================================
# t-SNE: one figure per query, segmented q-band colors on accepted hits
# ============================================================

Q_BAND_CMAP = "plasma"
ACCEPTED_HIT_SIZE = 35
QUERY_STAR_SIZE = 180


def _short_qid(qid):
    if not isinstance(qid, str):
        return str(qid)
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
    q_levels = sorted(q_levels)
    tid2first_q = {}
    new_by_q = {q: set() for q in q_levels}
    seen = set()

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


def _plot_q_color_bars(ax_qbar, q_levels, q_colors, new_by_q, *, label_fontsize=9, title_fontsize=10):
    q_levels = sorted(q_levels)
    counts = [len(new_by_q.get(q, set())) for q in q_levels]
    y_pos = np.arange(len(q_levels))

    ax_qbar.barh(
        y_pos,
        counts,
        color=[q_colors[q] for q in q_levels],
        edgecolor="black",
        linewidth=0.4,
        height=0.72,
    )
    ax_qbar.set_yticks(y_pos)
    ax_qbar.set_yticklabels([f"q = {q:.2f}" for q in q_levels], fontsize=label_fontsize)
    ax_qbar.set_title("q band colors", fontsize=title_fontsize)
    ax_qbar.invert_yaxis()
    if max(counts, default=0) == 0:
        ax_qbar.set_xlim(0, 1)
        ax_qbar.text(
            0.5, 0.5, "no hits",
            transform=ax_qbar.transAxes,
            ha="center", va="center", fontsize=label_fontsize, color="gray",
        )


def _draw_query_tsne_q_bands(
    ax,
    ax_qbar,
    qid,
    hits_by_q,
    q_levels,
    X_2d,
    seq2idx,
    id2subclass,
    bagel_subclass=None,
    *,
    title_fontsize=9,
    label_fontsize=8,
    bg_size=25,
    hit_size=None,
    star_size=None,
    qbar_title_fontsize=8,
    qbar_label_fontsize=7,
    show_xlabel=True,
    use_square_aspect=True,
):
    if qid not in seq2idx:
        return False

    hit_size = ACCEPTED_HIT_SIZE if hit_size is None else hit_size
    star_size = QUERY_STAR_SIZE if star_size is None else star_size

    q_levels = sorted(q_levels)
    q_colors = _q_band_colors(q_levels)
    tid2first_q, new_by_q = partition_hits_by_first_q(hits_by_q, q_levels)
    highlighted = set(tid2first_q.keys())

    q_min = min_discoverable_q(hits_by_q, q_levels)

    for sc, marker in SUBCLASS_MARKERS.items():
        idxs = [
            seq2idx[s] for s in seq2idx
            if s != qid and id2subclass.get(s, "Unknown") == sc and s not in highlighted
        ]
        if not idxs:
            continue
        idxs = np.array(idxs)
        ax.scatter(
            X_2d[idxs, 0], X_2d[idxs, 1],
            c=SUBCLASS_COLORS[sc], marker=marker,
            s=bg_size, alpha=0.18, linewidths=0, zorder=1,
        )

    for q in q_levels:
        tids = new_by_q.get(q, set())
        if not tids:
            continue
        for sc, marker in SUBCLASS_MARKERS.items():
            idxs = [
                seq2idx[tid] for tid in tids
                if tid in seq2idx and id2subclass.get(tid, "Unknown") == sc
            ]
            if not idxs:
                continue
            idxs = np.array(idxs)
            ax.scatter(
                X_2d[idxs, 0], X_2d[idxs, 1],
                c=[q_colors[q]], marker=marker,
                s=hit_size,
                alpha=0.90,
                edgecolors="black",
                linewidths=0.35,
                zorder=10,
            )

    qi = seq2idx[qid]
    ax.scatter(
        X_2d[qi, 0], X_2d[qi, 1],
        c="red", marker="*",
        s=star_size, edgecolors="black", linewidths=0.5,
        zorder=100,
    )

    title = _short_qid(qid)
    if bagel_subclass and bagel_subclass != "Unknown":
        title += f" | {bagel_subclass}"
    if q_min is not None:
        title += f" | min q = {q_min:.2f}"
    else:
        title += f" | no hit"
    ax.set_title(title, fontsize=title_fontsize)
    if show_xlabel:
        ax.set_xlabel("t-SNE 1", fontsize=label_fontsize)
    ax.set_ylabel("t-SNE 2", fontsize=label_fontsize)
    ax.tick_params(labelsize=label_fontsize - 1)
    if use_square_aspect:
        ax.set_box_aspect(1)
        ax_qbar.set_box_aspect(1)

    _plot_q_color_bars(
        ax_qbar, q_levels, q_colors, new_by_q,
        label_fontsize=qbar_label_fontsize,
        title_fontsize=qbar_title_fontsize,
    )
    if show_xlabel:
        ax_qbar.set_xlabel("# new hits", fontsize=qbar_label_fontsize)
    ax_qbar.tick_params(labelsize=qbar_label_fontsize)
    return True


def plot_tsne_single_query_q_bands(
    qid,
    hits_by_q,
    q_levels,
    X_2d,
    ids,
    id2subclass,
    out_path,
    bagel_subclass=None,
    source_tag="",
    dpi=400,
):
    ids = np.asarray(ids).astype(str)
    seq2idx = {s: i for i, s in enumerate(ids)}

    if qid not in seq2idx:
        print(f"  [WARN] {qid} not in t-SNE ids; skip.")
        return False

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.0], wspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    ax_qbar = fig.add_subplot(gs[0, 1])

    _draw_query_tsne_q_bands(
        ax, ax_qbar, qid, hits_by_q, q_levels, X_2d, seq2idx, id2subclass,
        bagel_subclass=bagel_subclass,
        title_fontsize=11,
        label_fontsize=10,
        bg_size=35,
        qbar_title_fontsize=10,
        qbar_label_fontsize=9,
        show_xlabel=True,
        use_square_aspect=True,
    )

    shape_handles = [
        mlines.Line2D([], [], marker=SUBCLASS_MARKERS[sc], linestyle="None",
                      markerfacecolor=SUBCLASS_COLORS[sc],
                      markeredgecolor=SUBCLASS_COLORS[sc],
                      markersize=7, alpha=0.7, label=sc)
        for sc in SUBCLASS_LABELS
    ]
    star_handle = mlines.Line2D([], [], marker="*", linestyle="None",
                                markerfacecolor="red", markeredgecolor="black",
                                markeredgewidth=0.5, markersize=12, label="query")
    fig.legend(
        handles=shape_handles + [star_handle],
        loc="lower center", ncol=3, fontsize=7,
        bbox_to_anchor=(0.42, -0.04),
    )

    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def q_discovery_folder_name(q_min, q_levels):
    if q_min is not None:
        return f"q_{q_min:.2f}"
    return f"no_hit_above_q{max(q_levels):.2f}"


def plot_all_queries_tsne_q_bands(
    df_merge,
    X_2d,
    ids,
    id2subclass,
    q_levels,
    out_dir,
    decoy_method,
    pool_reps="union",
    source_tag="",
):
    os.makedirs(out_dir, exist_ok=True)
    q_levels = sorted(q_levels)
    hits_all = collect_accepted_hits_by_q(df_merge, q_levels, pool_reps=pool_reps)
    qids = df_merge["qid"].astype(str).unique().tolist()

    n_ok = 0
    folder_counts = {}
    for qid in tqdm(qids, desc="t-SNE per query"):
        hits_by_q = hits_all[qid]
        q_min = min_discoverable_q(hits_by_q, q_levels)
        subdir = q_discovery_folder_name(q_min, q_levels)
        subdir_path = os.path.join(out_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)

        out_path = os.path.join(
            subdir_path,
            f"tsne_{decoy_method}_{safe_filename(qid)}.png",
        )
        ok = plot_tsne_single_query_q_bands(
            qid=qid,
            hits_by_q=hits_by_q,
            q_levels=q_levels,
            X_2d=X_2d,
            ids=ids,
            id2subclass=id2subclass,
            out_path=out_path,
            bagel_subclass=_get_subclass_from_mapping(qid, id2subclass),
            source_tag=source_tag,
        )
        if ok:
            n_ok += 1
            folder_counts[subdir] = folder_counts.get(subdir, 0) + 1

    print(f"[OK] Per-query t-SNE plots -> {out_dir} ({n_ok}/{len(qids)} saved)")
    for subdir in sorted(folder_counts):
        print(f"     {subdir}: {folder_counts[subdir]} queries")
    return hits_all


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA = os.path.join(BASE_DIR, "data")
    OUT_DIR = os.path.join(BASE_DIR, "results")
    os.makedirs(OUT_DIR, exist_ok=True)

    SEARCH_METHOD = "plm"
    QUERY_NAME = "putative"
    TARGET_NAME = "class_1"
    DECOY_METHOD = "extended_shuf"
    WEIGHT_METHOD = "AdaptiveBell"

    path_real = os.path.join(
        DATA,
        f"result_{SEARCH_METHOD}_{QUERY_NAME}_target_noisy_{TARGET_NAME}.txt",
    )
    path_decoy = os.path.join(
        DATA,
        f"result_{SEARCH_METHOD}_{QUERY_NAME}_{DECOY_METHOD}"
        f"_calibrated_gam_{WEIGHT_METHOD}_{TARGET_NAME}.txt",
    )
    path_raw = os.path.join(
        DATA,
        f"result_{SEARCH_METHOD}_{QUERY_NAME}_{TARGET_NAME}.txt",
    )
    decoy_suffix = "_shuf"
    subclass_info_file = os.path.join(DATA, "bagel_boa_subclass.txt")

    print(f"[INFO] target file (noisy):  {path_real}")
    print(f"[INFO] decoy file (calibr.): {path_decoy}")
    print(f"[INFO] raw score file:       {path_raw}")
    print(f"[INFO] subclass info:        {subclass_info_file}")

    df_merge = make_pair_table_keep_tid(
        path_real=path_real,
        path_decoy=path_decoy,
        decoy_suffix=decoy_suffix,
    )
    print(f"[INFO] merged paired rows: {len(df_merge)}")

    raw_score_map = load_raw_scores(path_raw)
    print(f"[INFO] raw score entries loaded: {len(raw_score_map)}")

    df_subclass = pd.read_csv(subclass_info_file, sep="\t")
    id2subclass = dict(zip(df_subclass["ids"], df_subclass["class"]))
    print(f"[INFO] subclass mapping entries: {len(id2subclass)}")
    print(f"[INFO] subclass distribution:")
    print(df_subclass["class"].value_counts().to_string())

    # ----- Single q -----
    Q_SINGLE = 0.20
    print(f"\n=== Fixed q = {Q_SINGLE} (subclass classification) ===")
    df_pred = classify_putative_subclass(df_merge, id2subclass, q=Q_SINGLE)

    out_tsv = os.path.join(
        DATA,
        f"putative_subclass_classification_{DECOY_METHOD}_q{Q_SINGLE:.2f}.txt",
    )
    df_pred.to_csv(out_tsv, sep="\t", index=False)
    print(f"[OK] Per-putative subclass classification saved to {out_tsv}")

    print(f"\n[INFO] Predicted subclass distribution at q={Q_SINGLE:.2f}:")
    print(df_pred["pred_subclass"].value_counts())

    print(
        f"\n[INFO] Mean subclass_agreement (rep consistency): "
        f"{df_pred['subclass_agreement'].mean():.3f}"
    )
    print(
        f"[INFO] Mean n_total passing hits per putative:  "
        f"{df_pred['n_total_mean'].mean():.1f}"
    )
    df_pred_cl = df_pred[df_pred["pred_subclass"] != "Unclassified"]
    if len(df_pred_cl) > 0:
        has_bagel = df_pred_cl["bagel_subclass"] != "Unknown"
        if has_bagel.any():
            agree = (
                df_pred_cl.loc[has_bagel, "pred_subclass"]
                == df_pred_cl.loc[has_bagel, "bagel_subclass"]
            ).mean()
            print(
                f"[INFO] Agreement with BAGEL's own subclass: "
                f"{agree:.3f} (over {has_bagel.sum()} with known subclass)"
            )

    # ----- Per-query t-SNE: segmented q-band colors -----
    PLOT_TSNE_PER_QUERY = True
    PLOT_CUMFRAC_PER_QUERY = True
    PLOT_SCORE_BY_SUBCLASS = True
    CUMFRAC_REP_ID = 0
    Q_LEVELS_TO_PLOT = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    POOL_REPS_FOR_HITS = "union"
    PERPLEXITY = 15

    tsne_out_dir = os.path.join(
        OUT_DIR, f"tsne_per_query_subclass_{DECOY_METHOD}"
    )

    if PLOT_SCORE_BY_SUBCLASS:
        print("\n=== Per-query raw score by subclass (grouped by min discoverable q) ===")
        plot_all_queries_score_by_subclass(
            df_merge=df_merge,
            id2subclass=id2subclass,
            raw_score_map=raw_score_map,
            out_dir=tsne_out_dir,
            decoy_method=DECOY_METHOD,
            q_levels=Q_LEVELS_TO_PLOT,
        )

    if PLOT_CUMFRAC_PER_QUERY:
        print("\n=== Per-query cumulative fraction + eFDP (grouped by min discoverable q) ===")
        plot_all_queries_cumfrac(
            df_merge=df_merge,
            out_dir=tsne_out_dir,
            decoy_method=DECOY_METHOD,
            q_levels=Q_LEVELS_TO_PLOT,
            rep_id=CUMFRAC_REP_ID,
        )

    if PLOT_TSNE_PER_QUERY:
        print("\n=== Per-query t-SNE (grouped by min discoverable q) ===")
        plm_putative_pkl = os.path.join(
            DATA, "db", "db_astral_plm", f"db_{QUERY_NAME}_embedding.pkl"
        )
        plm_target_pkl = os.path.join(
            DATA, "db", f"db_{TARGET_NAME}_plm", f"{TARGET_NAME}_embedding.pkl"
        )
        fallback_X_2d_path = os.path.join(DATA, "X_2d_all.npy")
        fallback_seqs_path = os.path.join(DATA, "seqs_all.npy")
        cache_X_2d_path = os.path.join(DATA, "X_2d_plm_subclass.npy")
        cache_seqs_path = os.path.join(DATA, "seqs_plm_subclass.npy")

        try:
            X_2d, ids_all, source_tag = get_tsne_projection(
                plm_putative_pkl=plm_putative_pkl,
                plm_target_pkl=plm_target_pkl,
                fallback_X_2d_path=fallback_X_2d_path,
                fallback_seqs_path=fallback_seqs_path,
                cache_X_2d_path=cache_X_2d_path,
                cache_seqs_path=cache_seqs_path,
                perplexity=PERPLEXITY,
            )

            plot_all_queries_tsne_q_bands(
                df_merge=df_merge,
                X_2d=X_2d,
                ids=ids_all,
                id2subclass=id2subclass,
                q_levels=Q_LEVELS_TO_PLOT,
                out_dir=tsne_out_dir,
                decoy_method=DECOY_METHOD,
                pool_reps=POOL_REPS_FOR_HITS,
                source_tag=source_tag,
            )
        except Exception as e:
            print(f"[WARN] Could not produce per-query t-SNE: {e}")
