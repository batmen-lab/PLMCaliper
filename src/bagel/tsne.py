import os
import ast
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm


# ============================================================
# Helpers
# ============================================================

def _get_class_from_name(name):
    if not isinstance(name, str):
        return "Unknown"
    left = name.split(";", 1)[0].strip()
    parts = left.split(".")
    if len(parts) < 2:
        return "Unknown"
    class_code = parts[1]
    mapping = {"1": "1", "2": "2", "3": "3"}
    return mapping.get(class_code, "0")


def remove_decoy_suffix(qid, decoy_suffix):
    if qid.endswith(decoy_suffix):
        return qid[: -len(decoy_suffix)]
    return qid


def make_pair_table_keep_tid(path_real, path_decoy, decoy_suffix):
    df_real = pd.read_csv(path_real, sep="\t")
    df_decoy = pd.read_csv(path_decoy, sep="\t")

    df_real = df_real[["qid", "tid", "homo_type", "score", "rep_id"]].copy()
    df_real = df_real.rename(columns={
        "score": "score_real",
        "homo_type": "homo_type_real",
    })

    df_decoy = df_decoy[["qid", "tid", "score", "rep_id"]].copy()
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

    # downcast for memory
    df_merge["score_real"] = df_merge["score_real"].astype(np.float32)
    df_merge["score_decoy"] = df_merge["score_decoy"].astype(np.float32)
    if df_merge["homo_type_real"].dtype != object:
        df_merge["homo_type_real"] = df_merge["homo_type_real"].astype(np.int8)
    df_merge["rep_id"] = pd.to_numeric(df_merge["rep_id"], downcast="integer")
    return df_merge


def _paired_topk_efdp(scores_t, scores_d, K):
    scores_t = np.asarray(scores_t, dtype=float)
    scores_d = np.asarray(scores_d, dtype=float)

    if len(scores_t) < K or len(scores_t) == 0:
        return np.nan

    # K-th largest target score
    t_star = np.partition(scores_t, -K)[-K]

    decoy_count = int(np.sum(scores_d >= t_star))
    efdp = (decoy_count + 1.0) / float(K)
    return float(efdp)


def _topk_real_tids(df_qrc, K):
    return (
        df_qrc.sort_values("score_real", ascending=False)
        .head(K)["tid"]
        .tolist()
    )


# ============================================================
# Core: per-class top-K eFDP across reps
# ============================================================

def get_efdp_per_class(df_merge, id2class, K=3, rep0_for_topk=True):
    CLASS_LABELS = ["1", "2", "3"]
    rows = []

    grouped = df_merge.groupby("qid", sort=False, observed=True)

    for qid, df_q in tqdm(grouped, desc="Per-query eFDP"):
        rep_ids = sorted(df_q["rep_id"].unique().tolist())
        # Per-rep eFDPs: shape (n_reps, 3)
        per_rep_efdps = []
        # rep0's top-K tids per class for plotting
        topk_rep0 = {c: [] for c in CLASS_LABELS}

        for rep_id in rep_ids:
            df_qr = df_q[df_q["rep_id"] == rep_id]
            tid_class = df_qr["tid"].map(id2class).fillna("Unknown")

            this_rep = []
            for c in CLASS_LABELS:
                df_qrc = df_qr[tid_class == c]

                if len(df_qrc) < K:
                    this_rep.append(np.nan)
                    if rep_id == rep_ids[0] and rep0_for_topk:
                        topk_rep0[c] = _topk_real_tids(df_qrc, len(df_qrc))
                    continue

                efdp = _paired_topk_efdp(
                    df_qrc["score_real"].values,
                    df_qrc["score_decoy"].values,
                    K=K,
                )
                this_rep.append(efdp)

                if rep_id == rep_ids[0] and rep0_for_topk:
                    topk_rep0[c] = _topk_real_tids(df_qrc, K)

            per_rep_efdps.append(this_rep)

        per_rep_arr = np.asarray(per_rep_efdps, dtype=float)  # (n_reps, 3)
        efdp_mean = np.nanmean(per_rep_arr, axis=0).tolist()  # 3-vector

        # Argmin over classes that have at least one valid rep
        if np.all(np.isnan(efdp_mean)):
            best_class = "Unknown"
        else:
            best_idx = int(np.nanargmin(efdp_mean))
            best_class = CLASS_LABELS[best_idx]

        rows.append({
            "qid": qid,
            "pred_class": best_class,
            "efdp_mean": efdp_mean,
            "efdp_per_rep": per_rep_arr.tolist(),
            "topk1": topk_rep0["1"],
            "topk2": topk_rep0["2"],
            "topk3": topk_rep0["3"],
            "bagel_predicted_class": _get_class_from_name(qid),
        })

    result_df = pd.DataFrame(rows)

    if len(result_df) > 0:
        agree = (result_df["pred_class"] == result_df["bagel_predicted_class"]).mean()
    else:
        agree = float("nan")

    return result_df, agree


# ============================================================
# Optional: t-SNE visualization (kept from the old version)
# ============================================================

def main_plot(df_topk, tsne_emb_path, tsne_labels_path, tsne_seqs_path, out_path):
    if not (os.path.exists(tsne_emb_path)
            and os.path.exists(tsne_labels_path)
            and os.path.exists(tsne_seqs_path)):
        print(f"[SKIP] t-SNE artifacts not found; skipping plot.\n"
              f"       expected: {tsne_emb_path}, {tsne_labels_path}, {tsne_seqs_path}")
        return

    tsne_embeddings = np.load(tsne_emb_path)
    tsne_labels = np.load(tsne_labels_path, allow_pickle=True)
    tsne_seqs = np.load(tsne_seqs_path, allow_pickle=True)

    qids = df_topk["qid"].unique()
    print(f"Number of qids: {len(qids)}")

    seq2idx = {seq: i for i, seq in enumerate(tsne_seqs)}

    def rank_to_alpha(rank, K):
        return 0.3 + 0.7 * (1 - rank / max(K - 1, 1))

    ncols = 3
    nrows = len(qids)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 10, nrows * 8))
    axes = np.atleast_1d(axes).flatten()

    for i, q in tqdm(enumerate(qids), total=len(qids), desc="Plotting"):
        df_q = df_topk[df_topk["qid"] == q].iloc[0]

        topk_dict = {
            "1": ast.literal_eval(df_q["topk1"]) if isinstance(df_q["topk1"], str) else df_q["topk1"],
            "2": ast.literal_eval(df_q["topk2"]) if isinstance(df_q["topk2"], str) else df_q["topk2"],
            "3": ast.literal_eval(df_q["topk3"]) if isinstance(df_q["topk3"], str) else df_q["topk3"],
        }

        pred_class = df_q["pred_class"]
        bagel_class = df_q["bagel_predicted_class"]

        if q not in seq2idx:
            continue
        q_idx = seq2idx[q]

        all_topk_idx = set()
        for v in topk_dict.values():
            all_topk_idx |= {seq2idx[s] for s in v if s in seq2idx}

        for c, class_label in enumerate(["1", "2", "3"]):
            ax = axes[i * 3 + c]

            bg_mask = np.ones(len(tsne_seqs), dtype=bool)
            bg_mask[list(all_topk_idx)] = False
            bg_mask[q_idx] = False
            ax.scatter(
                tsne_embeddings[bg_mask, 0],
                tsne_embeddings[bg_mask, 1],
                c="lightgray", s=80, alpha=0.6, linewidths=0,
            )

            color_map = {"1": "red", "2": "blue", "3": "green"}
            marker_map = {"1": "o", "2": "s", "3": "^"}

            topk_seqs = topk_dict[class_label]
            for rank, seq in enumerate(topk_seqs):
                if seq not in seq2idx:
                    continue
                idx = seq2idx[seq]
                ax.scatter(
                    tsne_embeddings[idx, 0], tsne_embeddings[idx, 1],
                    c=color_map[class_label],
                    s=80,
                    alpha=rank_to_alpha(rank, len(topk_seqs)),
                    marker=marker_map[class_label],
                    edgecolors="k",
                )

            ax.scatter(
                tsne_embeddings[q_idx, 0], tsne_embeddings[q_idx, 1],
                c="red", marker="*", s=250, edgecolors="k", zorder=10,
            )

            ax.set_title(
                f"Class {class_label}\nqid={q}\nPred={pred_class} / BAGEL={bagel_class}",
                fontsize=17,
            )
            ax.set_xlabel("Dim 1", fontsize=17)
            ax.set_ylabel("Dim 2", fontsize=17)

            legend_handles = [
                mlines.Line2D([], [], color="black", marker=marker_map["1"],
                              linestyle="None", markersize=10, label="Class 1"),
                mlines.Line2D([], [], color="black", marker=marker_map["2"],
                              linestyle="None", markersize=10, label="Class 2"),
                mlines.Line2D([], [], color="black", marker=marker_map["3"],
                              linestyle="None", markersize=10, label="Class 3"),
                mlines.Line2D([], [], color="red", marker="*",
                              linestyle="None", markersize=12, label="Query"),
            ]
            ax.legend(
                handles=legend_handles, loc="center left",
                bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=12,
            )

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved to {out_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # ------------- config -------------
    DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA = os.path.join(DATA_DIR, "data")
    OUT_DIR = os.path.join(DATA_DIR, "results")
    os.makedirs(OUT_DIR, exist_ok=True)

    SEARCH_METHOD = "plm"
    QUERY_NAME = "putative"
    TARGET_NAME = "class_all"
    DECOY_METHOD = "extended_shuf"     # or "shuf"
    WEIGHT_METHOD = "AdaptiveBell"
    K = 4                              # top-K real hits per class for eFDP

    # ------------- inputs -------------
    path_real = os.path.join(
        DATA,
        f"result_{SEARCH_METHOD}_{QUERY_NAME}_target_noisy_{TARGET_NAME}.txt",
    )
    path_decoy = os.path.join(
        DATA,
        f"result_{SEARCH_METHOD}_{QUERY_NAME}_{DECOY_METHOD}"
        f"_calibrated_gam_{WEIGHT_METHOD}_{TARGET_NAME}.txt",
    )

    # decoy_suffix: the trailing token on decoy qid AFTER the underscore.
    # extended_shuf decoy file has qid like "<orig>_shuf" (see putative_shuf.fa).
    decoy_suffix = "_shuf"

    homo_info_file = os.path.join(DATA, "bagel_class.txt")

    print(f"[INFO] target file:  {path_real}")
    print(f"[INFO] decoy file:   {path_decoy}")
    print(f"[INFO] homo info:    {homo_info_file}")

    # ------------- build merged paired table -------------
    df_merge = make_pair_table_keep_tid(
        path_real=path_real,
        path_decoy=path_decoy,
        decoy_suffix=decoy_suffix,
    )
    print(f"[INFO] merged paired rows: {len(df_merge)}")

    # ------------- tid -> BAGEL class -------------
    df_class = pd.read_csv(homo_info_file, sep="\t")
    # Normalize class labels to "1" / "2" / "3"
    class_str_map = {"ClassI": "1", "ClassII": "2", "ClassIII": "3"}
    df_class["class_code"] = (
        df_class["class"].astype(str).map(class_str_map).fillna(df_class["class"].astype(str))
    )
    id2class = dict(zip(df_class["ids"], df_class["class_code"]))

    # ------------- per-class top-K eFDP -> predict class -------------
    result_df, agree = get_efdp_per_class(df_merge, id2class, K=K)
    print(f"[INFO] putative classified: {len(result_df)}")
    print(f"[INFO] agreement with BAGEL's own predicted class: {agree:.3f}")

    out_tsv = os.path.join(
        DATA,
        f"putative_efdp_{DECOY_METHOD}_K{K}.txt",
    )
    result_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"[OK] Saved per-qid classification to {out_tsv}")

    # ------------- per-class distribution summary -------------
    print("\n[INFO] Predicted class distribution:")
    print(result_df["pred_class"].value_counts())
    print("\n[INFO] BAGEL predicted class distribution:")
    print(result_df["bagel_predicted_class"].value_counts())

    # ------------- optional t-SNE plot -------------
    # Only runs if PLM-based t-SNE artifacts exist.
    PLOT_TSNE = True
    if PLOT_TSNE:
        main_plot(
            df_topk=result_df,
            tsne_emb_path=os.path.join(DATA, "X_2d_all.npy"),
            tsne_labels_path=os.path.join(DATA, "labels_all.npy"),
            tsne_seqs_path=os.path.join(DATA, "seqs_all.npy"),
            out_path=os.path.join(OUT_DIR, f"tsne_diff_classes_topk_{DECOY_METHOD}_K{K}.png"),
        )
