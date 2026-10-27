import csv
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import groupby
import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_Q_LEVELS = np.arange(0.05, 1.00, 0.05)


class FDREvaluator:
    def __init__(self, decoy_suffix="", q_levels=DEFAULT_Q_LEVELS):
        self.decoy_suffix = decoy_suffix
        self.q_levels = q_levels

    def pair_table(self, df_real, df_decoy):
        df_real = df_real.assign(rep_id=df_real.get("rep_id", 0))
        df_decoy = df_decoy.assign(rep_id=df_decoy.get("rep_id", 0))
        df_real = df_real.rename(columns={"score": "score_real", "homo_type": "homo_type_real"})
        df_decoy = df_decoy.rename(columns={"score": "score_decoy"})
        df_decoy["qid"] = df_decoy["qid"].str.removesuffix(self.decoy_suffix)
        return df_real[["qid", "tid", "homo_type_real", "score_real", "rep_id"]].merge(
            df_decoy[["qid", "tid", "score_decoy", "rep_id"]], on=["qid", "tid", "rep_id"])

    def _iter_qid_blocks(self, path):
        with open(path, newline="") as handle:
            for _, rows in groupby(csv.DictReader(handle, delimiter="\t"), key=lambda row: row["qid"]):
                yield pd.DataFrame(list(rows))

    def iter_pair_tables(self, path_real, path_decoy):
        for df_real, df_decoy in zip(self._iter_qid_blocks(path_real), self._iter_qid_blocks(path_decoy), strict=True):
            yield self.pair_table(df_real.astype({"score": np.float32, "homo_type": int}),
                                  df_decoy.astype({"score": np.float32})).astype({"rep_id": int}).drop(columns="tid")

    def efdp_curve(self, scores_t, scores_d):
        cand = np.unique(np.concatenate([scores_t, scores_d]))
        n_target_ge = len(scores_t) - np.searchsorted(np.sort(scores_t), cand, side="left")
        n_decoy_ge = len(scores_d) - np.searchsorted(np.sort(scores_d), cand, side="left")
        return {"cand": cand, "efdp": (n_decoy_ge + 1.0) / np.maximum(n_target_ge, 1.0)}

    # evaluation against the homology labels
    def label_curve(self, df_q):
        scores_t = df_q["score_real"].values
        curve = self.efdp_curve(scores_t, df_q["score_decoy"].values)
        cand_t = curve["cand"]

        is_homo_t = df_q["homo_type_real"].isin([1, 2]).values
        is_strict_nonhomo = (df_q["homo_type_real"] == -1).values
        is_annotated = is_homo_t | is_strict_nonhomo

        scores_t_homo_only = np.sort(scores_t[is_homo_t])
        scores_t_false_sorted = np.sort(scores_t[is_strict_nonhomo])
        scores_t_annotated_sorted = np.sort(scores_t[is_annotated])

        true_pos_count = len(scores_t_homo_only) - np.searchsorted(scores_t_homo_only, cand_t, side='left')
        false_count = len(scores_t_false_sorted) - np.searchsorted(scores_t_false_sorted, cand_t, side='left')
        annotated_count = len(scores_t_annotated_sorted) - np.searchsorted(scores_t_annotated_sorted, cand_t, side='left')

        return pd.DataFrame({
            "qid": df_q["qid"].iloc[0],
            "rep_id": df_q["rep_id"].iloc[0],
            "threshold": cand_t,
            "est_fdp": curve["efdp"],
            "real_fdp": false_count / np.maximum(annotated_count, 1.0),
            "n_true_pos": true_pos_count,
            "n_false": false_count,
            "total_homo": is_homo_t.sum(),
            "n_neg_total": int(is_strict_nonhomo.sum()),
        })

    def _curves_for_block(self, df_qall):
        return [self.label_curve(df_q) for _, df_q in df_qall.groupby(["qid", "rep_id"], sort=False)]

    def curves(self, path_real, path_decoy, n_jobs=1):
        curve_list, pending = [], set()
        with ProcessPoolExecutor(n_jobs) as pool, tqdm(desc="  -> Computing TDA curves", unit="query", colour="cyan") as pbar:
            for df_qall in self.iter_pair_tables(path_real, path_decoy):
                pending.add(pool.submit(self._curves_for_block, df_qall))
                if len(pending) >= 2 * n_jobs:  # bound the number of queries held in memory
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    curve_list += [c for fut in done for c in fut.result()]
                    pbar.update(len(done))
            curve_list += [c for fut in pending for c in fut.result()]
            pbar.update(len(pending))
        return pd.concat(curve_list, ignore_index=True)

    def summarize(self, df_curve_all):
        grouped_curves = list(df_curve_all.groupby(["qid", "rep_id"], sort=False))
        rows = []
        for q in self.q_levels:
            for (qid, rep_id), grp in grouped_curves:
                total_homo, n_neg_total = grp["total_homo"].iloc[0], grp["n_neg_total"].iloc[0]
                valid_rows = grp[grp["est_fdp"] <= q]
                if len(valid_rows) > 0:
                    row = valid_rows.loc[valid_rows["threshold"].idxmin()]
                    tp, fp, est_fdp, real_fdp = row["n_true_pos"], row["n_false"], row["est_fdp"], row["real_fdp"]
                else: # no hit accepted
                    tp, fp, est_fdp, real_fdp = 0, 0, np.nan, np.nan
                denom = total_homo + n_neg_total
                rows.append({
                    "qid": qid,
                    "rep_id": rep_id,
                    "q": float(q),
                    "est_fdp": est_fdp,
                    "real_fdp": real_fdp,
                    "power": tp / total_homo if total_homo > 0 else 0.0,
                    # Accuracy = (TP + TN) / (TP + FP + TN + FN); TN = n_neg_total - FP.
                    "accuracy": (tp + (n_neg_total - fp)) / denom if denom > 0 else np.nan,
                    "valid": int(len(valid_rows) > 0),
                })
        return pd.DataFrame(rows)

    def aggregate(self, df_summary):
        q_levels = np.sort(df_summary["q"].unique())
        agg = {"q_levels": q_levels, "mean_real_fdr": [], "mean_power": [], "mean_accuracy": [], "valid_count": [],
               "all_real_fdps_per_q": [], "all_powers_per_q": [], "all_accuracy_per_q": [],
               "total_curves": df_summary.groupby(["qid", "rep_id"]).ngroups, "n_unique_qids": df_summary["qid"].nunique()}
        for q in q_levels:
            sub = df_summary[df_summary["q"] == q]
            agg["mean_real_fdr"].append(float(np.nanmean(sub["real_fdp"].to_numpy())))
            agg["mean_power"].append(float(np.nanmean(sub["power"].to_numpy())))
            agg["mean_accuracy"].append(float(np.nanmean(sub["accuracy"].to_numpy())))
            agg["valid_count"].append(int(sub["valid"].sum()))
            agg["all_real_fdps_per_q"].append(sub.loc[sub["valid"] == 1, "real_fdp"].dropna().tolist())
            agg["all_powers_per_q"].append(sub["power"].dropna().tolist())
            agg["all_accuracy_per_q"].append(sub["accuracy"].dropna().tolist())
        return agg

    def evaluate(self, path_real, path_decoy, n_jobs=1):
        return self.aggregate(self.summarize(self.curves(path_real, path_decoy, n_jobs=n_jobs)))
