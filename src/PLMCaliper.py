import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random
from collections import defaultdict
from types import SimpleNamespace
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from patsy import dmatrix, build_design_matrices
import cvxpy as cp


# Decoy types <"decoy_design": ("suffix", "file_suffix")>
DECOY_DESIGNS = {
    "extended_mkv": ("_mkv",    "_extended_mkv{k}"),
    "mkv":          ("_mkv{k}", "_mkv{k}"),
    "shuf":         ("_shuf",   "_shuf"),
    "rev":          ("_rev",    "_rev"),
}


class PLMCaliper:
    """
    PLM-Caliper: model-agnostic, per-query FDR control for protein retrieval with protein
    language models, using query-based decoys together with score calibration.
    https://doi.org/10.5281/zenodo.22436908

    Input for PLM-Caliper: the search scores of the real queries and of their decoys against
    the same target database. A decoy qid is its real qid plus a fixed suffix (build_decoy
    writes e.g. "_mkv" or "_shuf"); PLM-Caliper reads that suffix off the two tables.

    Usage:
        # Option 1:
        pc = PLMCaliper(tau=0.2)
        decoy_fa = pc.build_decoy("queries.fa", design="extended_mkv")
        result = pc.plmcaliper(df_real, df_decoy, q_levels=[0.5])
        result.cutoffs, result.hits    # also result.calibrated_decoy, result.noisy_target, result.efdps
        # Each step can also be run on its own:
        calibrated_decoy, noisy_target = pc.calibration(df_real, df_decoy)
        efdps = pc.compute_efdp(calibrated_decoy, noisy_target)
        cutoffs = pc.compute_cutoff(efdps, q_levels=[0.5])
        hits = pc.obtain_hits(noisy_target, efdps, cutoffs)

        # Option 2:
        # If you want the results written to files, use the command line:
        python run_PLMCaliper.py calibrate --real real.tsv --decoy decoy.tsv --target-fdr 0.5 --out cutoffs.tsv --hits-out hits.tsv

    For more examples, refer to the tutorials folder.

    params:
        - **tau**:
            Quantile level of the pinball loss in the per-query spline fit. Default: 0.2.
        - **n_splines**:
            Degrees of freedom of the cubic B-spline that maps decoy scores onto the real null. Default: 12.
        - **lam**:
            Weight of the smoothness penalty on neighbouring spline coefficients. Default: 1.0.
        - **num_points**:
            Number of quantile anchors used in the spline fit. Default: 1000.
        - **tol**:
            Tail-match tolerance: the fit stops once the calibrated decoy matches the real scores on the top 5% tail within this distance. Default: 0.2.
        - **max_iter**:
            Maximum number of spline fits per query, each one tilting the anchors further to the tail. Default: 2.
        - **noise_lambda**:
            Scale of the gaussian noise added to both the calibrated decoy and the target, relative to each query's real score range. Default: 0.02.
        - **n_reps**:
            Number of noise replicates. Default: 1.
        - **seed**:
            Random seed for the noise. Default: 43.

    data:
        - **df_real**: search results of the real queries.
        - **df_decoy**: search results of the decoy queries.
        - **q_levels**: list of target FDR levels.

    return (plmcaliper):
        A SimpleNamespace object with the following attributes:
        - calibrated_decoy: calibrated decoy scores.
        - noisy_target: real scores with the noise for breaking tie purpose.
        - efdps: eFDP at every candidate threshold.
        - cutoffs: per-query score cutoff for each target FDR and cutoff_score is NaN when the FDR is unreachable.
        - hits: selected target hits with the eFDP at each hit's score.
    """

    def __init__(self, tau=0.2, n_splines=12, lam=1.0, num_points=1000,
                 tol=0.2, max_iter=2, noise_lambda=0.02, n_reps=1, seed=43):
        self.tau = tau
        self.n_splines = n_splines
        self.lam = lam
        self.num_points = num_points
        self.tol = tol
        self.max_iter = max_iter
        self.noise_lambda = noise_lambda
        self.n_reps = n_reps
        self.seed = seed

    def build_decoy(self, fasta_path, out_path=None, design="extended_mkv", markov_order=2, seed=0):
        """Decoy types: extended_mkv, mkv, shuf, and rev"""
        id_suffix, stem_suffix = DECOY_DESIGNS[design]
        suffix = id_suffix.format(k=markov_order)
        out_path = out_path or f"{os.path.splitext(fasta_path)[0]}{stem_suffix.format(k=markov_order)}.fa"
        records = list(SeqIO.parse(fasta_path, "fasta"))
        decoys = []

        # rev
        if design == "rev":
            decoys = [str(record.seq)[::-1] for record in records]

        # shuf
        elif design == "shuf":
            rng = np.random.default_rng(seed=seed)
            for record in records:
                seq_list = list(str(record.seq))
                rng.shuffle(seq_list)
                decoys.append("".join(seq_list))

        # extended_mkv / mkv
        else:
            random.seed(seed)
            k = markov_order
            for record in records:
                sequence = str(record.seq)
                if design == "extended_mkv":
                    sequence = sequence.upper()
                seq_len = len(sequence)

                if seq_len < k + 1:
                    result = list(sequence)
                    random.shuffle(result)
                else:
                    model = defaultdict(lambda: defaultdict(int))
                    for i in range(seq_len - k):
                        prefix = sequence[i:(i + k)]
                        next_char = sequence[i + k]
                        model[prefix][next_char] += 1

                    markov_model = {}
                    for prefix, suffix_counts in model.items():
                        total = sum(suffix_counts.values())
                        chars, probs = [], []
                        for char, count in suffix_counts.items():
                            chars.append(char)
                            probs.append(count * 1.0 / total)
                        markov_model[prefix] = (chars, probs)

                    start = random.choice(list(markov_model.keys()))
                    result = list(start)
                    for _ in range(seq_len - k):
                        prefix = ''.join(result[-k:])
                        if prefix in markov_model:
                            chars, probs = markov_model[prefix]
                            next_char = random.choices(chars, probs)[0]
                        else:
                            next_char = random.choice(sequence)
                        result.append(next_char)

                decoy = ''.join(result)
                if design == "extended_mkv":
                    decoy = sequence + decoy
                decoys.append(decoy)

        SeqIO.write([SeqRecord(Seq(decoy), id=record.id + suffix, description="")
                     for record, decoy in zip(records, decoys)], out_path, "fasta")
        print(f"[PLMCaliper] built {len(records)} '{design}' decoy sequences (suffix = {suffix})")
        return out_path

    def fit_weights(self, df_merge):
        """fit the weight function"""

        scores_t = df_merge["score_real"].to_numpy(dtype=float)
        scores_d = df_merge["score_decoy"].to_numpy(dtype=float)

        X = np.concatenate([scores_t, scores_d]).reshape(-1, 1)
        y = np.concatenate([np.ones(len(scores_t)), np.zeros(len(scores_d))])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(
            solver="liblinear", class_weight="balanced",
            max_iter=1000, random_state=0,
        )
        model.fit(X_scaled, y)

        p_target_global = model.predict_proba(scaler.transform(scores_t.reshape(-1, 1)))[:, 1]
        p_decoy_global = model.predict_proba(scaler.transform(scores_d.reshape(-1, 1)))[:, 1]

        df_out = df_merge.copy()
        df_out["p_target"] = p_target_global
        df_out["p_decoy"] = p_decoy_global
        df_out["weight"] = np.zeros(len(df_out), dtype=float)

        # asymmetric Gaussian
        for qid, df_q in df_out.groupby("qid"):
            p_t_q = df_q["p_target"].values
            p_d_q = df_q["p_decoy"].values
            p_center = np.percentile(p_d_q, 95)
            sigma_base = np.maximum(np.std(p_d_q), 0.05)
            right_stretch = 2.0
            diff = p_t_q - p_center
            sigma_vec = np.where(diff > 0, sigma_base * right_stretch, sigma_base)
            df_out.loc[df_q.index, "weight"] = np.exp(-0.5 * (diff / sigma_vec) ** 2)

        return df_out.drop(columns=["p_target", "p_decoy"])

    def _decoy_suffix(self, real_qids, decoy_qids):
        # The suffix turns a real qid into its decoy qid (e.g. "_mkv")
        real_qids = set(real_qids)
        for qid in pd.unique(decoy_qids):
            for i in range(len(qid), 0, -1):
                if qid[:i] in real_qids:
                    return qid[i:]

    def calibration(self, df_real, df_decoy):
        tau, n_splines, lam, num_points = self.tau, self.n_splines, self.lam, self.num_points
        tol, max_iter, noise_lambda, n_reps = self.tol, self.max_iter, self.noise_lambda, self.n_reps

        decoy_suffix = self._decoy_suffix(df_real["qid"], df_decoy["qid"])
        df_decoy = df_decoy.assign(qid=df_decoy["qid"].str.removesuffix(decoy_suffix))
        df_merge = df_real.merge(df_decoy[["qid", "tid", "score"]], on=["qid", "tid"],
                                 suffixes=("_real", "_decoy"))
        
        print("[PLMCaliper] fitting weights function")
        df_merge = self.fit_weights(df_merge)

        print(f"[PLMCaliper] doing per-query calibration (tau={tau})")
        rng = np.random.default_rng(self.seed)
        out_cols = ["qid", "tid", "score", "rep_id"]
        decoy_blocks, target_blocks = [], []

        n_queries = int(df_merge["qid"].nunique())
        query_iter = tqdm(df_merge.groupby("qid"), total=n_queries, desc="  -> calibration", unit="query", leave=False)
        for qid, df_q in query_iter:
            idx = df_q.index

            raw_real = df_q["score_real"].to_numpy(dtype=float)
            raw_decoy = df_q["score_decoy"].to_numpy(dtype=float)

            real_min = float(np.min(raw_real))
            real_max = float(np.max(raw_real))
            real_range = (real_max - real_min) + 1e-8

            df_q = df_q.copy()
            df_q["score_real"] = (raw_real - real_min) / real_range
            df_q["score_decoy"] = (raw_decoy - real_min) / real_range

            x_all = df_q["score_decoy"].to_numpy(dtype=float)
            y_true = df_q["score_real"].values

            n_pts = len(x_all)
            x_all_dps = [x_all + noise_lambda * rng.standard_normal(n_pts) for _ in range(n_reps)]
            y_real_noisys = [y_true + noise_lambda * rng.standard_normal(n_pts) for _ in range(n_reps)]
            x_all_dp = x_all_dps[0]
            y_real_noisy = y_real_noisys[0]

            decoy_for_fit_0 = x_all + noise_lambda * rng.standard_normal(n_pts)
            real_for_fit_0 = y_true + noise_lambda * rng.standard_normal(n_pts)

            boundary_pad = 5.0 * noise_lambda
            x_min_global = float(np.min(x_all)) - boundary_pad
            x_max_global = float(np.max(x_all)) + boundary_pad

            df_q_sorted = df_q.sort_values("score_real")
            sorted_weights = df_q_sorted["weight"].values

            def try_fit(k_val):
                t = np.linspace(0.0, 1.0, num_points)
                probs = 1.0 - (1.0 - t) ** k_val 

                x_bulk = np.quantile(decoy_for_fit_0, probs)
                y_bulk = np.quantile(real_for_fit_0, probs)

                indices = (probs * (len(df_q_sorted) - 1)).astype(int)
                w_bulk = np.maximum(sorted_weights[indices], 1e-6)

                formula = (
                    f"0 + bs(x, df={n_splines}, degree=3, "
                    f"include_intercept=True, lower_bound={x_min_global}, upper_bound={x_max_global})"
                )

                x_bulk_eval = np.clip(x_bulk, x_min_global, x_max_global)
                B_bulk_df = dmatrix(formula, {"x": x_bulk_eval}, return_type="dataframe")
                B_bulk = np.asarray(B_bulk_df)

                x_all_dp_eval = np.clip(x_all_dp, x_min_global, x_max_global)
                B_all = np.asarray(
                    build_design_matrices([B_bulk_df.design_info], {"x": x_all_dp_eval})[0]
                )

                c = cp.Variable(B_bulk.shape[1])
                residual = y_bulk - B_bulk @ c
                pinball = cp.maximum(tau * residual, (tau - 1.0) * residual)
                data_loss = cp.sum(cp.multiply(w_bulk, pinball))
                smooth_loss = lam * cp.sum_squares(cp.diff(c))

                problem = cp.Problem(cp.Minimize(data_loss + smooth_loss), [])

                for solver in ["CLARABEL", "ECOS", "SCS"]:
                    if solver in cp.installed_solvers():
                        try:
                            problem.solve(solver=solver, verbose=False)
                            if c.value is not None:
                                return True, np.asarray(c.value).reshape(-1), B_bulk_df.design_info, B_all
                        except Exception:
                            pass
                return False, None, None, None

            best_loss = float("inf")
            best_model = None
            k_val = 1.0

            for _ in range(max_iter):
                success, coeffs, design_info, B_all = try_fit(k_val)
                if not success:
                    k_val += 1.0
                    continue

                y_cal_tmp = B_all @ coeffs

                # score the fit on the top 5% tail only
                probs = np.linspace(0.95, 1.0, 50)
                y_cal_std = (y_cal_tmp - np.mean(y_cal_tmp)) / (np.std(y_cal_tmp) + 1e-8)
                y_true_std = (y_real_noisy - np.mean(y_real_noisy)) / (np.std(y_real_noisy) + 1e-8)
                current_loss = float(np.mean(np.abs(np.quantile(y_cal_std, probs) - np.quantile(y_true_std, probs))))

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_model = (coeffs, design_info)

                if current_loss <= tol:
                    break
                k_val += 1.0

            final_success = best_model is not None
            if final_success:
                coeffs, design_info = best_model
            else:
                print(f"[PLMCaliper] quantile solver failed for {qid}")

            df_q_raw = df_merge.loc[idx]
            for r in range(n_reps):
                xr_dp = x_all_dps[r]
                yr_noisy = y_real_noisys[r]

                if final_success:
                    B_r = np.asarray(build_design_matrices(
                        [design_info], {"x": np.clip(xr_dp, x_min_global, x_max_global)})[0])
                    yc_r = B_r @ coeffs
                else:
                    yc_r = np.full(n_pts, float(np.mean(yr_noisy)), dtype=float)

                yc_r = np.clip(yc_r, float(np.min(yr_noisy)), float(np.max(yr_noisy)))

                d_block = pd.DataFrame({
                    "qid": str(qid) + decoy_suffix,
                    "tid": df_q_raw["tid"].values,
                    "score": yc_r * real_range + real_min,
                }).sort_values("score", ascending=False)
                d_block["rep_id"] = r
                decoy_blocks.append(d_block[out_cols])

                t_block = pd.DataFrame({
                    "qid": str(qid),
                    "tid": df_q_raw["tid"].values,
                    "score": yr_noisy * real_range + real_min,
                }).sort_values("score", ascending=False)
                t_block["rep_id"] = r
                target_blocks.append(t_block[out_cols])

        return (pd.concat(decoy_blocks, ignore_index=True),
                pd.concat(target_blocks, ignore_index=True))

    def compute_efdp(self, calibrated_decoy, noisy_target):
        decoy_suffix = self._decoy_suffix(noisy_target["qid"], calibrated_decoy["qid"])
        decoy = calibrated_decoy.assign(qid=calibrated_decoy["qid"].str.removesuffix(decoy_suffix))
        paired = noisy_target.merge(decoy, on=["qid", "tid", "rep_id"], suffixes=("_real", "_decoy"))

        curves = []
        for (qid, rep_id), grp in paired.groupby(["qid", "rep_id"], sort=False):
            s_t = np.sort(grp["score_real"].to_numpy(dtype=float))
            s_d = np.sort(grp["score_decoy"].to_numpy(dtype=float))

            cand_t = np.unique(np.concatenate([s_t, s_d]))

            n_real = len(s_t) - np.searchsorted(s_t, cand_t, side="left")
            n_decoy = len(s_d) - np.searchsorted(s_d, cand_t, side="left")

            est_fdp = (n_decoy + 1.0) / np.maximum(n_real, 1.0)
            curves.append(pd.DataFrame({"qid": qid, "rep_id": rep_id, "threshold": cand_t,
                                        "est_fdp": est_fdp, "n_selected": n_real}))
        return pd.concat(curves, ignore_index=True)

    def compute_cutoff(self, efdps, q_levels):
        """the most permissive threshold"""
        rows = []
        for (qid, rep_id), curve in efdps.groupby(["qid", "rep_id"], sort=False):
            cand_t = curve["threshold"].to_numpy()
            est_fdp = curve["est_fdp"].to_numpy()
            n_real = curve["n_selected"].to_numpy()

            for target_fdr in q_levels:
                ok = np.flatnonzero(est_fdp <= target_fdr)
                if ok.size:
                    i = ok[int(np.argmin(cand_t[ok]))]   # choose the most permissive cutoff
                    rows.append({
                        "qid": qid, "rep_id": rep_id,
                        "target_fdr": target_fdr,
                        "cutoff_score": float(cand_t[i]),
                        "est_fdp": float(est_fdp[i]),
                        "n_selected": int(n_real[i]),
                    })
                else:
                    rows.append({
                        "qid": qid, "rep_id": rep_id,
                        "target_fdr": target_fdr,
                        "cutoff_score": np.nan,
                        "est_fdp": np.nan,
                        "n_selected": 0,
                    })
        return pd.DataFrame(rows)

    def obtain_hits(self, noisy_target, efdps, cutoffs):
        hits = noisy_target.merge(cutoffs.dropna(subset=["cutoff_score"]), on=["qid", "rep_id"])
        hits = hits[hits["score"] >= hits["cutoff_score"]]
        hits = hits[["qid", "tid", "rep_id", "score"]].drop_duplicates()
        hits = hits.merge(efdps[["qid", "rep_id", "threshold", "est_fdp"]],
                          left_on=["qid", "rep_id", "score"], right_on=["qid", "rep_id", "threshold"])
        hits = hits.sort_values(["qid", "rep_id", "score", "tid"], ascending=[True, True, False, True])
        return hits[["qid", "tid", "rep_id", "score", "est_fdp"]].reset_index(drop=True)

    def plmcaliper(self, df_real, df_decoy, q_levels):
        """run full procedures"""
        calibrated_decoy, noisy_target = self.calibration(df_real, df_decoy)
        efdps = self.compute_efdp(calibrated_decoy, noisy_target)
        cutoffs = self.compute_cutoff(efdps, q_levels)
        hits = self.obtain_hits(noisy_target, efdps, cutoffs)
        return SimpleNamespace(calibrated_decoy=calibrated_decoy, noisy_target=noisy_target,
                               efdps=efdps, cutoffs=cutoffs, hits=hits)
