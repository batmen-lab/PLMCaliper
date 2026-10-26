"""Target-decoy FDR evaluation shared by the tutorials.

FDREvaluator pairs the noisy-target and calibrated-decoy tables, computes the eFDP curve
of each (query, rep), picks the per-query cutoff (smallest threshold with eFDP <= q) and,
where homology labels are present, scores that cutoff against them (real FDP, power,
accuracy).

    ev = FDREvaluator(decoy_suffix="_mkv", q_levels=[0.1, 0.2])
    agg = ev.evaluate("target_noisy.tsv", "calibrated_decoy.tsv")
"""
import csv
import gc
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

STREAMING_THRESHOLD_GB = 1.0
DEFAULT_Q_LEVELS = np.arange(0.05, 1.00, 0.05)


def _with_rep_id(df: pd.DataFrame) -> pd.DataFrame:
    """Raw search tables (no calibration) are a single replicate and carry no rep_id."""
    if "rep_id" not in df.columns:
        df["rep_id"] = 0
    return df


class FDREvaluator:
    def __init__(self, decoy_suffix="", q_levels=DEFAULT_Q_LEVELS):
        self.decoy_suffix = decoy_suffix
        self.q_levels = q_levels

    # ---- pair tables -------------------------------------------------------

    def pair_table(self, path_real, path_decoy, keep_tid=False):
        """Real (with homo_type) and decoy scores paired on (qid, tid, rep_id). Without
        keep_tid, tid is dropped and qid stored as a category to keep big tables small."""
        df_real = _with_rep_id(pd.read_csv(path_real, sep="\t"))
        df_decoy = _with_rep_id(pd.read_csv(path_decoy, sep="\t"))

        df_real = df_real[["qid", "tid", "homo_type", "score", "rep_id"]].copy()
        df_real = df_real.rename(columns={"score": "score_real", "homo_type": "homo_type_real"})

        df_decoy = df_decoy[["qid", "tid", "score", "rep_id"]].copy()
        df_decoy["qid_orig"] = df_decoy["qid"].str.removesuffix(self.decoy_suffix)
        df_decoy = df_decoy.rename(columns={"score": "score_decoy"})

        df_merge = df_real.merge(
            df_decoy[["qid_orig", "tid", "score_decoy", "rep_id"]],
            left_on=["qid", "tid", "rep_id"],
            right_on=["qid_orig", "tid", "rep_id"],
            how="inner",
        )
        df_merge = df_merge.drop(columns=["qid_orig"] if keep_tid else ["tid", "qid_orig"],
                                 errors="ignore")

        # Downcast numeric types to reduce memory footprint
        df_merge["score_real"] = df_merge["score_real"].astype(np.float32)
        df_merge["score_decoy"] = df_merge["score_decoy"].astype(np.float32)
        if df_merge["homo_type_real"].dtype != object:
            df_merge["homo_type_real"] = df_merge["homo_type_real"].astype(np.int8)
        df_merge["rep_id"] = pd.to_numeric(df_merge["rep_id"], downcast="integer")
        if not keep_tid:
            df_merge["qid"] = df_merge["qid"].astype("category")
        return df_merge

    def _iter_qid_blocks(self, path) -> Iterator[tuple[str, list[dict]]]:
        current_qid = None
        block: list[dict] = []
        with Path(path).open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                qid = row["qid"]
                if current_qid is not None and qid != current_qid:
                    yield current_qid, block
                    block = []
                current_qid = qid
                block.append(row)
        if block and current_qid is not None:
            yield current_qid, block

    def _rows_to_pair_frame(self, real_rows: list[dict], decoy_rows: list[dict]) -> pd.DataFrame:
        df_real = _with_rep_id(pd.DataFrame(real_rows))
        df_decoy = _with_rep_id(pd.DataFrame(decoy_rows))
        if df_real.empty:
            return pd.DataFrame()

        df_real = df_real[["qid", "tid", "homo_type", "score", "rep_id"]].copy()
        df_real = df_real.rename(columns={"score": "score_real", "homo_type": "homo_type_real"})
        df_real["score_real"] = df_real["score_real"].astype(np.float32)

        df_decoy = df_decoy[["qid", "tid", "score", "rep_id"]].copy()
        df_decoy["qid_orig"] = df_decoy["qid"].str.removesuffix(self.decoy_suffix)
        df_decoy = df_decoy.rename(columns={"score": "score_decoy"})
        df_decoy["score_decoy"] = df_decoy["score_decoy"].astype(np.float32)

        df_merge = df_real.merge(
            df_decoy[["qid_orig", "tid", "score_decoy", "rep_id"]],
            left_on=["qid", "tid", "rep_id"],
            right_on=["qid_orig", "tid", "rep_id"],
            how="inner",
        )
        df_merge = df_merge.drop(columns=["qid_orig", "tid"], errors="ignore")
        df_merge["score_real"] = df_merge["score_real"].astype(np.float32)
        df_merge["score_decoy"] = df_merge["score_decoy"].astype(np.float32)
        if df_merge["homo_type_real"].dtype != object:
            df_merge["homo_type_real"] = df_merge["homo_type_real"].astype(np.int8)
        df_merge["rep_id"] = pd.to_numeric(df_merge["rep_id"], downcast="integer")
        return df_merge

    def iter_pair_tables(self, path_real, path_decoy) -> Iterator[tuple[str, pd.DataFrame]]:
        """Pair tables one query at a time, for tables too big to load at once. Both files
        must hold each query's rows contiguously and in the same query order."""
        real_iter = self._iter_qid_blocks(path_real)
        decoy_iter = self._iter_qid_blocks(path_decoy)

        for (real_qid, real_rows), (decoy_qid, decoy_rows) in zip(real_iter, decoy_iter, strict=True):
            qid = decoy_qid.removesuffix(self.decoy_suffix)
            if qid != real_qid:
                raise ValueError(f"QID mismatch while streaming: real={real_qid!r} decoy={decoy_qid!r}")
            df_merge = self._rows_to_pair_frame(real_rows, decoy_rows)
            if not df_merge.empty:
                yield real_qid, df_merge

    # ---- eFDP and the per-query cutoff -------------------------------------

    def efdp_curve(self, scores_t, scores_d):
        """eFDP at every candidate threshold (the union of target and decoy scores)."""
        cand = np.sort(np.unique(np.concatenate([scores_t, scores_d])))
        n_target_ge = len(scores_t) - np.searchsorted(np.sort(scores_t), cand, side="left")
        n_decoy_ge = len(scores_d) - np.searchsorted(np.sort(scores_d), cand, side="left")
        # +1 is the standard TDA conservative correction (else eFDP reads exactly 0 in the tail)
        efdp = (n_decoy_ge + 1.0) / np.maximum(n_target_ge, 1.0)
        return {"cand": cand, "efdp": efdp, "n_target_ge": n_target_ge, "n_decoy_ge": n_decoy_ge}

    def find_threshold(self, scores_t, scores_d, q, tol=0.0):
        """Smallest threshold with eFDP <= q (the most permissive cutoff within budget).
        None when the query has no target score or no threshold reaches q."""
        if len(scores_t) == 0:
            return None
        curve = self.efdp_curve(scores_t, scores_d)
        valid = np.flatnonzero(curve["efdp"] <= q + tol)
        if valid.size == 0:
            return None
        i = valid[np.argmin(curve["cand"][valid])]
        return {"T_q": float(curve["cand"][i]), "eFDP_at_T": float(curve["efdp"][i]),
                "n_pass": int(curve["n_target_ge"][i])}

    # ---- evaluation against the homology labels ----------------------------

    def label_curve(self, df_q):
        """eFDP and real FDP at every candidate threshold for one (query, rep).
        homo_type 1/2 = homolog, -1 = non-homolog; anything else is unannotated and left
        out of the real FDP."""
        scores_t = df_q["score_real"].values
        curve = self.efdp_curve(scores_t, df_q["score_decoy"].values)
        cand_t = curve["cand"]

        is_homo_t = df_q["homo_type_real"].isin([1, 2]).values
        is_strict_nonhomo = (df_q["homo_type_real"] == -1).values
        is_annotated = is_homo_t | is_strict_nonhomo

        total_homo = is_homo_t.sum()

        scores_t_homo_only = np.sort(scores_t[is_homo_t])
        scores_t_false_sorted = np.sort(scores_t[is_strict_nonhomo])
        scores_t_annotated_sorted = np.sort(scores_t[is_annotated])

        true_pos_count = len(scores_t_homo_only) - np.searchsorted(scores_t_homo_only, cand_t, side='left')
        false_count = len(scores_t_false_sorted) - np.searchsorted(scores_t_false_sorted, cand_t, side='left')
        annotated_count = len(scores_t_annotated_sorted) - np.searchsorted(scores_t_annotated_sorted, cand_t, side='left')

        real_fdp = false_count / np.maximum(annotated_count, 1.0)

        return pd.DataFrame({
            "qid": df_q["qid"].iloc[0],
            "rep_id": df_q["rep_id"].iloc[0],
            "threshold": cand_t,
            "est_fdp": curve["efdp"],
            "real_fdp": real_fdp,
            "n_selected": curve["n_target_ge"],
            "n_true_pos": true_pos_count,
            "n_false": false_count,
            "total_homo": total_homo,
            "n_neg_total": int(is_strict_nonhomo.sum()),
        })

    def _curves_for_block(self, df_qall: pd.DataFrame) -> list[pd.DataFrame]:
        curves: list[pd.DataFrame] = []
        for _, df_q in df_qall.groupby(["qid", "rep_id"], sort=False):
            df_curve = self.label_curve(df_q)
            if not df_curve.empty:
                curves.append(df_curve)
        return curves

    def _maybe_flush(self, curve_list: list[pd.DataFrame], flush_every: int) -> list[pd.DataFrame]:
        # periodically merge the list to release memory held by many small frames
        if len(curve_list) >= flush_every:
            merged = pd.concat(curve_list, ignore_index=True)
            gc.collect()
            return [merged]
        return curve_list

    def _curves_in_memory(self, df_merge, flush_every=2000):
        curve_list = []
        grouped = df_merge.groupby(["qid", "rep_id"], sort=False)
        for (qid, rep_id), df_q in tqdm(grouped, desc="  -> Computing TDA curves", leave=False, colour='cyan'):
            df_curve = self.label_curve(df_q)
            if not df_curve.empty:
                curve_list.append(df_curve)
            curve_list = self._maybe_flush(curve_list, flush_every)
        return pd.concat(curve_list, ignore_index=True) if curve_list else pd.DataFrame()

    def _curves_streaming(self, path_real, path_decoy, flush_every=2000, n_jobs=1):
        curve_list: list[pd.DataFrame] = []
        pbar = tqdm(desc="  -> Computing TDA curves", unit="query", colour="cyan")
        try:
            if n_jobs <= 1:
                for _, df_qall in self.iter_pair_tables(path_real, path_decoy):
                    curve_list.extend(self._curves_for_block(df_qall))
                    pbar.update(1)
                    curve_list = self._maybe_flush(curve_list, flush_every)
            else:
                max_inflight = max(n_jobs * 2, n_jobs + 1)
                with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                    pending: set = set()
                    for _, df_qall in self.iter_pair_tables(path_real, path_decoy):
                        pending.add(pool.submit(self._curves_for_block, df_qall))
                        if len(pending) >= max_inflight:
                            done, pending = wait(pending, return_when=FIRST_COMPLETED)
                            for fut in done:
                                curve_list.extend(fut.result())
                                pbar.update(1)
                            curve_list = self._maybe_flush(curve_list, flush_every)
                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for fut in done:
                            curve_list.extend(fut.result())
                            pbar.update(1)
                        curve_list = self._maybe_flush(curve_list, flush_every)
        finally:
            pbar.close()
        return pd.concat(curve_list, ignore_index=True) if curve_list else pd.DataFrame()

    def curves(self, path_real, path_decoy, n_jobs=1):
        """label_curve for every (query, rep). Tables over STREAMING_THRESHOLD_GB are read one
        query at a time (n_jobs > 1 spreads those queries over worker processes)."""
        gb_real = Path(path_real).stat().st_size / 1024**3
        gb_decoy = Path(path_decoy).stat().st_size / 1024**3
        if gb_real > STREAMING_THRESHOLD_GB or gb_decoy > STREAMING_THRESHOLD_GB:
            print(f"  streaming the pair table ({gb_real + gb_decoy:.1f} GB)", flush=True)
            return self._curves_streaming(path_real, path_decoy, n_jobs=n_jobs)
        df_merge = self.pair_table(str(path_real), str(path_decoy))
        print(f"  {len(df_merge):,} paired scores", flush=True)
        return self._curves_in_memory(df_merge)

    def summarize(self, df_curve_all):
        """Per (query, rep, q): real FDP, power and accuracy at the eFDP <= q cutoff."""
        grouped_curves = list(df_curve_all.groupby(["qid", "rep_id"], sort=False))
        rows = []

        for q in self.q_levels:
            q = float(q)
            for (_qid, _rep_id), grp in grouped_curves:
                valid_rows = grp[grp["est_fdp"] <= q]
                n_neg_total = float(grp["n_neg_total"].iloc[0]) if "n_neg_total" in grp else 0.0
                if len(valid_rows) > 0:
                    row = valid_rows.loc[valid_rows["threshold"].idxmin()]
                    total_homo = row["total_homo"]
                    power = row["n_true_pos"] / total_homo if total_homo > 0 else 0.0
                    tp = float(row["n_true_pos"])
                    fp = float(row.get("n_false", np.nan))
                    denom = total_homo + n_neg_total
                    # Accuracy = (TP + TN) / (TP + FP + TN + FN); TN = n_neg_total - FP.
                    accuracy = (tp + (n_neg_total - fp)) / denom if denom > 0 else np.nan
                    rows.append({
                        "qid": _qid,
                        "rep_id": _rep_id,
                        "q": q,
                        "est_fdp": row["est_fdp"],
                        "real_fdp": row["real_fdp"],
                        "power": power,
                        "accuracy": accuracy,
                        "valid": 1,
                    })
                else:
                    total_homo = float(grp["total_homo"].iloc[0]) if "total_homo" in grp else 0.0
                    denom = total_homo + n_neg_total
                    # No hit accepted -> TP=FP=0, TN=n_neg_total, FN=total_homo.
                    accuracy = n_neg_total / denom if denom > 0 else np.nan
                    rows.append({
                        "qid": _qid,
                        "rep_id": _rep_id,
                        "q": q,
                        "est_fdp": np.nan,
                        "real_fdp": np.nan,
                        "power": 0.0,
                        "accuracy": accuracy,
                        "valid": 0,
                    })

        return pd.DataFrame(rows)

    def aggregate(self, df_summary):
        """Means over queries per q, plus the per-query values behind each mean."""
        q_levels = np.sort(df_summary["q"].unique())
        total_curves = df_summary.groupby(["qid", "rep_id"]).ngroups
        n_unique_qids = df_summary["qid"].nunique()

        has_accuracy = "accuracy" in df_summary.columns

        mean_real_fdr_list = []
        mean_power_list = []
        mean_accuracy_list = []
        valid_count_list = []
        all_real_fdps_per_q = []
        all_powers_per_q = []
        all_accuracy_per_q = []

        for q in q_levels:
            sub = df_summary[df_summary["q"] == q]
            real_fdps = sub["real_fdp"].to_numpy()
            powers = sub["power"].to_numpy()
            valid_count = int(sub["valid"].sum())

            mean_real_fdr_list.append(float(np.nanmean(real_fdps)))
            mean_power_list.append(float(np.nanmean(powers)))
            valid_count_list.append(valid_count)
            all_real_fdps_per_q.append(sub.loc[sub["valid"] == 1, "real_fdp"].dropna().tolist())
            all_powers_per_q.append([float(p) for p in powers if not np.isnan(p)])

            if has_accuracy:
                accs = sub["accuracy"].to_numpy()
                mean_accuracy_list.append(float(np.nanmean(accs)))
                all_accuracy_per_q.append([float(a) for a in accs if not np.isnan(a)])
            else:
                mean_accuracy_list.append(float("nan"))
                all_accuracy_per_q.append([])

        return {
            "q_levels": q_levels,
            "mean_real_fdr": mean_real_fdr_list,
            "mean_power": mean_power_list,
            "mean_accuracy": mean_accuracy_list,
            "valid_count": valid_count_list,
            "all_real_fdps_per_q": all_real_fdps_per_q,
            "all_powers_per_q": all_powers_per_q,
            "all_accuracy_per_q": all_accuracy_per_q,
            "total_curves": total_curves,
            "n_unique_qids": n_unique_qids,
        }

    def evaluate(self, path_real, path_decoy, n_jobs=1):
        """curves -> summarize -> aggregate for one noisy-target / calibrated-decoy pair."""
        return self.aggregate(self.summarize(self.curves(path_real, path_decoy, n_jobs=n_jobs)))
