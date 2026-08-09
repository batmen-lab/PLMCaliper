import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pygam import LinearGAM, s
from patsy import dmatrix, build_design_matrices
import cvxpy as cp

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else _TqdmNoOp(**kwargs)


class _TqdmNoOp:
    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable) if self.iterable is not None else iter(())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, n=1):
        return None

    def close(self):
        return None

warnings.filterwarnings("ignore", category=FutureWarning)


def remove_decoy_suffix(qid: str, decoy_suffix: str) -> str:
    if qid.endswith(decoy_suffix):
        return qid[:-len(decoy_suffix)]
    return qid


def make_calibration_table(path_real, path_decoy, decoy_suffix):
    df_real = pd.read_csv(path_real, sep="\t")
    df_decoy = pd.read_csv(path_decoy, sep="\t")

    required_cols = {"qid", "tid", "score", "homo_type"}
    missing_real = required_cols - set(df_real.columns)
    missing_decoy = required_cols - set(df_decoy.columns)
    if missing_real:
        raise ValueError(
            f"[make_calibration_table] {path_real} missing columns "
            f"{missing_real}. Did you forget to run parse_bagel.py on it?"
        )
    if missing_decoy:
        raise ValueError(
            f"[make_calibration_table] {path_decoy} missing columns "
            f"{missing_decoy}. Did you forget to run parse_bagel.py on it?"
        )

    # real table
    df_real = df_real[["qid", "tid", "score", "homo_type"]].copy()
    df_real = df_real.rename(columns={"score": "score_real", "homo_type": "homo_type_real"})

    # decoy table: map decoy qid back to original qid
    df_decoy = df_decoy[["qid", "tid", "score", "homo_type"]].copy()
    df_decoy["qid_orig"] = df_decoy["qid"].apply(lambda x: remove_decoy_suffix(x, decoy_suffix))
    df_decoy = df_decoy.rename(columns={"score": "score_decoy", "homo_type": "homo_type_decoy"})

    # merge by original qid + tid
    df_merge = df_real.merge(
        df_decoy[["qid_orig", "tid", "score_decoy", "homo_type_decoy"]],
        left_on=["qid", "tid"],
        right_on=["qid_orig", "tid"],
        how="inner"
    )

    df_merge["delta"] = df_merge["score_real"] - df_merge["score_decoy"]
    
    # label: homolog = 1 if homo_type in {1,2}, else 0
    df_merge["y"] = df_merge["homo_type_real"].isin([1, 2]).astype(int)
    
    return df_merge




def fit_logistic_and_get_pqt(df_merge, method="Target"):
    '''
    method: "Decoy", "Target", "Barbell", "Bell", "Constant", "AdaptiveBell"
    '''
    scores_t = df_merge["score_real"].to_numpy(dtype=float)
    scores_d = df_merge["score_decoy"].to_numpy(dtype=float)
    
    if method == "Constant":
        df_out = df_merge.copy()
        df_out["weight"] = 1.0
        return df_out, None, None

    X = np.concatenate([scores_t, scores_d]).reshape(-1, 1)
    y = np.concatenate([np.ones(len(scores_t)), np.zeros(len(scores_d))])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=0
    )
    model.fit(X_scaled, y)

    scores_for_weight = scores_t
    X_real_scaled = scaler.transform(scores_for_weight.reshape(-1, 1))
    p_target_global = model.predict_proba(X_real_scaled)[:, 1]

    scores_d_for_weight = scores_d
    X_d_scaled = scaler.transform(scores_d_for_weight.reshape(-1, 1))
    p_decoy_global = model.predict_proba(X_d_scaled)[:, 1]

    df_out = df_merge.copy()
    df_out["p_target"] = p_target_global
    df_out["p_decoy"] = p_decoy_global
    df_out["weight"] = np.zeros(len(df_out), dtype=float)

    # Weight Methods
    if method == "Target":
        df_out["weight"] = p_target_global
    elif method == "Decoy":
        df_out["weight"] = 1 - p_target_global
    elif method == "Bell":
        df_out["weight"] = 4 * p_target_global * (1 - p_target_global)
    elif method == "Barbell":
        df_out["weight"] = 1 - (4 * p_target_global * (1 - p_target_global))
    elif method == "AdaptiveBell":
        # query-specific adaptive bell
        for qid, df_q in df_out.groupby("qid"):
            p_t_q = df_q["p_target"].values
            p_d_q = df_q["p_decoy"].values
            p_center = np.percentile(p_d_q, 95)
            sigma_base = np.maximum(np.std(p_d_q), 0.05)
            right_stretch = 2.0
            diff = p_t_q - p_center
            sigma_vec = np.where(diff > 0, sigma_base * right_stretch, sigma_base)
            w_q = np.exp(-0.5 * (diff / sigma_vec)**2)

            df_out.loc[df_q.index, "weight"] = w_q

    df_out = df_out.drop(columns=["p_target", "p_decoy"])
    
    return df_out, model, scaler





def nonlinear_weighted_quantile_fit(x, y, w, tau=0.5, degree=3, n_splines=12, lam=1.0):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x = x[mask]
    y = y[mask]
    w = w[mask]


    # sort by x so monotonicity is meaningful and prediction is stable
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    w = w[order]

    X = x.reshape(-1, 1)

    gam = LinearGAM(
        s(0, n_splines=n_splines, constraints="monotonic_inc"),
        fit_intercept=True,
        lam=lam
    )

    gam.fit(X, y, weights=w)

    return gam, None




def _bad_records_for_query(qid, rep_id, df_q, q_min, q_max, use_strict):

    df_q = df_q.sort_values("score_real_noisy", ascending=False).copy()

    scores_t = df_q["score_real_noisy"].values
    scores_d = df_q["score_decoy_cal"].values
    homo_types = df_q["homo_type_real"].values
    is_homo = np.isin(homo_types, [1, 2])
    is_strict_non_homo = (homo_types == -1)

    N_target = len(scores_t)
    N_decoy = len(scores_d)
    if N_target == 0 or N_decoy == 0:
        return False, []

    counts_t = np.arange(1, N_target + 1)
    scores_d_asc = np.sort(scores_d)

    counts_d = N_decoy - np.searchsorted(scores_d_asc, scores_t, side='left')

    efdp = (counts_d * (N_target / N_decoy) + 1) / counts_t

    counts_non_homo_all = np.cumsum(~is_homo)
    real_fdp_all = counts_non_homo_all / counts_t

    counts_strict_non_homo = np.cumsum(is_strict_non_homo)
    counts_annotated = np.cumsum(is_homo | is_strict_non_homo)

    real_fdp_strict = np.full_like(real_fdp_all, np.nan, dtype=float)
    valid_idx = counts_annotated > 0
    real_fdp_strict[valid_idx] = (
        counts_strict_non_homo[valid_idx] / counts_annotated[valid_idx]
    )
    real_fdp = real_fdp_strict if use_strict else real_fdp_all

    mask = (efdp >= q_min) & (efdp <= q_max) & np.isfinite(real_fdp) & (real_fdp > efdp)
    if not np.any(mask):
        return False, []

    out = []
    for i in np.where(mask)[0]:
        out.append({
            "qid": qid,
            "rep_id": int(rep_id),
            "rank": int(i + 1),
            "score_threshold": float(scores_t[i]),
            "q_est": float(efdp[i]),
            "real_fdp": float(real_fdp[i]),
            "gap": float(real_fdp[i] - efdp[i]),
        })
    return True, out




def apply_gam_calibration(
    df_merge,
    tau=0.5,
    n_splines=12,
    lam=1.0,
    num_points=500,
    tol=0.024,
    max_iter=5,
    noise_lambda=0.05,
    seed=0,
    n_reps=5,
    out_decoy_path=None,
    out_target_path=None,
    decoy_suffix="",
    plot_qids=None,
    bad_q_min=None,
    bad_q_max=None,
    bad_use_strict=True,
):
    rng = np.random.default_rng(seed)

    plot_qids_set = set(plot_qids) if plot_qids else set()

    keep_records = []          # for plot_qids only
    bad_records = []           # per-(qid, rep_id) point details
    bad_summary_records = []   # per-qid aggregated summary (all-reps-fail rule)

    params_list = []
    models_dict = {}

    out_cols = ["qid", "tid", "score", "homo_type", "rank", "rep_id"]
    f_decoy = None
    f_target = None
    if out_decoy_path is not None:
        f_decoy = open(out_decoy_path, "w")
        f_decoy.write("\t".join(out_cols) + "\n")
    if out_target_path is not None:
        f_target = open(out_target_path, "w")
        f_target.write("\t".join(out_cols) + "\n")

    n_queries = int(df_merge["qid"].nunique())
    query_iter = tqdm(
        df_merge.groupby("qid"),
        total=n_queries,
        desc="  -> GAM calibration",
        unit="query",
        colour="magenta",
        leave=False,
    )
    for qid, df_q in query_iter:

        idx = df_q.index

        # Per-query min-max normalization to [0, 1]
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

        # Per-query independent noise 
        n_pts = len(x_all)
        x_all_dps = [
            x_all + noise_lambda * rng.standard_normal(n_pts) for _ in range(n_reps)
        ]
        y_real_noisys = [
            y_true + noise_lambda * rng.standard_normal(n_pts) for _ in range(n_reps)
        ]
        x_all_dp = x_all_dps[0]          # rep 0 used for fit-iteration selection
        y_real_noisy = y_real_noisys[0]

        decoy_for_fit_0 = x_all + noise_lambda * rng.standard_normal(n_pts)
        real_for_fit_0 = y_true + noise_lambda * rng.standard_normal(n_pts)

        boundary_pad = 5.0 * noise_lambda
        x_min_global = float(np.min(x_all)) - boundary_pad
        x_max_global = float(np.max(x_all)) + boundary_pad

        df_q_sorted = df_q.sort_values("score_real")
        sorted_weights = df_q_sorted["weight"].values

        y_min = float(np.min(y_real_noisy))
        y_max = float(np.max(y_real_noisy))

        def try_fit_gam(k_val):
            t = np.linspace(0.0, 1.0, num_points)

            probs = 1.0 - (1.0 - t)**k_val

            x_bulk = np.quantile(decoy_for_fit_0, probs)
            y_bulk = np.quantile(real_for_fit_0, probs)

            # weight
            indices = (probs * (len(df_q_sorted) - 1)).astype(int)
            w_bulk = sorted_weights[indices]
            w_bulk = np.maximum(w_bulk, 1e-6)

            formula = (
                f"0 + bs(x, df={n_splines}, degree=3, "
                f"include_intercept=True, lower_bound={x_min_global}, upper_bound={x_max_global})"
            )

            x_bulk_eval = np.clip(x_bulk, x_min_global, x_max_global)
            B_bulk_df = dmatrix(formula, {"x": x_bulk_eval}, return_type="dataframe")
            B_bulk = np.asarray(B_bulk_df)

            # Apply the fitted mapping to d'' (raw decoy + independent noise).
            x_all_dp_eval = np.clip(x_all_dp, x_min_global, x_max_global)
            B_all = np.asarray(
                build_design_matrices([B_bulk_df.design_info], {"x": x_all_dp_eval})[0]
            )

            n_basis = B_bulk.shape[1]
            c = cp.Variable(n_basis)
            y_hat_bulk = B_bulk @ c
            residual = y_bulk - y_hat_bulk

            pinball = cp.maximum(tau * residual, (tau - 1.0) * residual)
            data_loss = cp.sum(cp.multiply(w_bulk, pinball))
            smooth_loss = lam * cp.sum_squares(cp.diff(c))

            problem = cp.Problem(cp.Minimize(data_loss + smooth_loss), [])

            solved = False
            for solver in ["CLARABEL", "ECOS", "SCS"]:
                if solver in cp.installed_solvers():
                    try:
                        problem.solve(solver=solver, verbose=False)
                        if c.value is not None:
                            solved = True
                            break
                    except Exception:
                        pass
            
            if not solved or c.value is None:
                return False, None, None, None, None

            coeffs = np.asarray(c.value).reshape(-1)
            return True, coeffs, formula, B_bulk_df.design_info, B_all

        # iteration
        best_loss = float('inf')
        best_model = None
        best_k = np.nan
        final_success = False
        
        k_val = 1.0 
        
        for i in range(max_iter):
            success, coeffs, formula, design_info, B_all = try_fit_gam(k_val)
            if not success:
                k_val += 1.0
                continue

            y_cal_tmp = B_all @ coeffs

            # top 5% tail
            probs = np.linspace(0.95, 1.0, 50)

            # standardization
            y_cal_std = (y_cal_tmp - np.mean(y_cal_tmp)) / (np.std(y_cal_tmp) + 1e-8)
            y_true_std = (y_real_noisy - np.mean(y_real_noisy)) / (np.std(y_real_noisy) + 1e-8)
            
            y_tail_pred = np.quantile(y_cal_std, probs)
            y_tail_true = np.quantile(y_true_std, probs)
            
            current_loss = float(np.mean(np.abs(y_tail_pred - y_tail_true)))
            # print(f"query: {qid}, k_val: {k_val}, current_loss: {current_loss}")
            
            # save the current best result
            if current_loss < best_loss:
                best_loss = current_loss
                best_model = (coeffs, formula, design_info, B_all)
                best_k = k_val
                
            # check if the tolerance is satisfied
            if current_loss <= tol:
                final_success = True
                break
                
            # otherwise, increase k, and enter the next round of non-uniform sampling
            k_val += 1.0

        # pack the results (best GAM selected on rep 0)
        if best_model is not None:
            final_success = True
            coeffs, formula, design_info, _ = best_model
            used_k = best_k
            w_dist = best_loss
            models_dict[qid] = {
                "type": f"spline_k{used_k}",
                "coeffs": coeffs,
                "x_min": x_min_global,
                "x_max": x_max_global,
                "formula": formula,
                "design_info": design_info,
            }
        else:
            print(f"[Warning] Quantile solver failed for {qid}, fallback to mean.")
            models_dict[qid] = {"type": "constant_fallback"}
            used_k = np.nan
            w_dist = np.nan

        # Apply the chosen GAM to all n_reps mapping-time realizations
        df_q_raw = df_merge.loc[idx].copy()  
        per_rep_dfs = [] 
        for r in range(n_reps):
            xr_dp = x_all_dps[r]
            yr_noisy = y_real_noisys[r]
            yr_min = float(np.min(yr_noisy))
            yr_max = float(np.max(yr_noisy))

            if final_success:
                xr_dp_eval = np.clip(xr_dp, x_min_global, x_max_global)
                B_r = np.asarray(
                    build_design_matrices([design_info], {"x": xr_dp_eval})[0]
                )
                yc_r = B_r @ coeffs
            else:
                yc_r = np.full(n_pts, float(np.mean(yr_noisy)), dtype=float)

            yc_r = np.clip(yc_r, yr_min, yr_max)

            df_rep = df_q_raw.copy()
            df_rep["score_decoy_cal"]   = yc_r * real_range + real_min
            df_rep["score_real_noisy"]  = yr_noisy * real_range + real_min
            df_rep["score_decoy_noisy"] = xr_dp * real_range + real_min
            df_rep["rep_id"] = r
            df_rep["tid_orig"] = df_rep["tid"].values
            per_rep_dfs.append(df_rep)

            if f_decoy is not None:
                d_block = pd.DataFrame({
                    "qid": str(qid) + decoy_suffix,
                    "tid": df_rep["tid"].values,
                    "score": df_rep["score_decoy_cal"].values,
                    "homo_type": df_rep["homo_type_decoy"].values,
                })
                d_block = d_block.sort_values("score", ascending=False)
                d_block["rank"] = np.arange(1, len(d_block) + 1)
                d_block["rep_id"] = r
                d_block[out_cols].to_csv(f_decoy, sep="\t", index=False, header=False)

            if f_target is not None:
                t_block = pd.DataFrame({
                    "qid": str(qid),
                    "tid": df_rep["tid"].values,
                    "score": df_rep["score_real_noisy"].values,
                    "homo_type": df_rep["homo_type_real"].values,
                })
                t_block = t_block.sort_values("score", ascending=False)
                t_block["rank"] = np.arange(1, len(t_block) + 1)
                t_block["rep_id"] = r
                t_block[out_cols].to_csv(f_target, sep="\t", index=False, header=False)

        if bad_q_min is not None and bad_q_max is not None:
            n_total = len(per_rep_dfs)
            n_failed = 0
            for r_idx, df_rep in enumerate(per_rep_dfs):
                rep_failed, rep_bad_points = _bad_records_for_query(
                    qid, r_idx, df_rep, bad_q_min, bad_q_max, bad_use_strict,
                )
                if rep_failed:
                    n_failed += 1
                    bad_records.extend(rep_bad_points)
            bad_summary_records.append({
                "qid": qid,
                "n_failed_reps": int(n_failed),
                "n_total_reps": int(n_total),
                "all_repeats_failed": bool(n_total > 0 and n_failed == n_total),
            })

        # Retain full per-rep DataFrames ONLY for the small set of plot_qids.
        if qid in plot_qids_set:
            keep_records.extend(per_rep_dfs)

        # Drop per-query intermediates before moving on — critical for memory.
        del per_rep_dfs

        params_list.append({
            "qid": qid,
            "tau": tau,
            "n_splines": n_splines,
            "lam": lam,
            "used_k": round(used_k, 2) if pd.notnull(used_k) else np.nan,
            "w_dist": round(w_dist, 5) if pd.notnull(w_dist) else np.nan
        })

    if f_decoy is not None:
        f_decoy.close()
    if f_target is not None:
        f_target.close()

    df_cal_subset = (
        pd.concat(keep_records, ignore_index=True) if keep_records else pd.DataFrame()
    )
    df_params = pd.DataFrame(params_list)
    df_bad = pd.DataFrame(bad_records)
    df_bad_summary = pd.DataFrame(bad_summary_records)
    return df_cal_subset, df_params, models_dict, df_bad, df_bad_summary



def plot_score_distributions_for_query(
    df_cal,
    target_qid,
    out_path,
    title=None,
):
    df_plot = df_cal[df_cal["qid"] == target_qid].copy()

    plt.figure(figsize=(6, 6))
    is_homo = df_plot["homo_type_real"].isin([1, 2])
    df_nonhomo = df_plot[~is_homo]
    df_homo = df_plot[is_homo]

    if not df_nonhomo.empty:
        sns.histplot(
            df_nonhomo["score_real_noisy"],
            bins=50,
            stat="density",
            kde=True,
            alpha=0.35,
            edgecolor=None,
            label="<Query,Non-homo> (target')",
            color="blue"
        )

    if not df_plot.empty:
        sns.histplot(
            df_plot["score_decoy"],
            bins=50,
            stat="density",
            kde=True,
            alpha=0.35,
            label="<Query_decoy,Target> (raw)",
            color="orange",
            edgecolor=None,
        )

        sns.histplot(
            df_plot["score_decoy_cal"],
            bins=50,
            stat="density",
            kde=True,
            alpha=0.35,
            label="<Calibrated Query_decoy,Target>",
            color="green",
            edgecolor=None,
        )

    if not df_homo.empty:
        first_line = True
        for score in df_homo["score_real_noisy"]:
            plt.axvline(
                x=score,
                color="red",
                linestyle="--",
                linewidth=0.5,
                alpha=0.8,
                label="<Query,Homolog> (target')" if first_line else ""
            )
            first_line = False

    plt.xlabel("Score", fontsize=13)
    plt.ylabel("Density", fontsize=13)
    plt.title(title, fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] Saved plot to {out_path}")




def plot_cumulative_fraction_for_query(
    df_cal,
    target_qid,
    out_path,
    score_col_decoy="score_decoy_cal",
):

    df_q = df_cal[df_cal["qid"] == target_qid].copy()

    if len(df_q) == 0:
        print(f"[Warning] No data found for query: {target_qid}")
        return

    df_q = df_q.sort_values("score_real_noisy", ascending=False)

    scores_t = df_q["score_real_noisy"].values
    scores_d = df_q[score_col_decoy].values
    
    homo_types = df_q["homo_type_real"].values
    is_homo = np.isin(homo_types, [1, 2])
    
    is_strict_non_homo = (homo_types == -1)

    N_target = len(scores_t)
    N_decoy = len(scores_d)

    counts_t = np.arange(1, N_target + 1)
    target_frac = counts_t / float(N_target)

    scores_d_asc = np.sort(scores_d)

    counts_d = N_decoy - np.searchsorted(scores_d_asc, scores_t, side='left')

    decoy_frac = counts_d / float(N_decoy)

    # eFDP: (#decoy * (N_target/N_decoy) + 1) / #target
    efdp = (counts_d * (N_target / N_decoy) + 1) / counts_t

    # 1. Real FDP with 0
    counts_non_homo_all = np.cumsum(~is_homo) 
    real_fdp_all = counts_non_homo_all / counts_t

    # 2. Real FDP without 0
    counts_strict_non_homo = np.cumsum(is_strict_non_homo)
    counts_annotated = np.cumsum(is_homo | is_strict_non_homo)

    real_fdp_strict = np.zeros_like(target_frac)
    valid_idx = counts_annotated > 0
    real_fdp_strict[valid_idx] = counts_strict_non_homo[valid_idx] / counts_annotated[valid_idx]

    min_frac = 1e-6
    target_frac = np.maximum(target_frac, min_frac)
    decoy_frac = np.maximum(decoy_frac, min_frac)

    # plot
    fig, ax1 = plt.subplots(figsize=(6, 6)) 
    ax1.plot([min_frac, 1], [min_frac, 1], color="gray", linestyle="--", linewidth=1.5, zorder=1)

    # plot Non-homo (blue dots)
    ax1.scatter(
        target_frac[~is_homo],
        decoy_frac[~is_homo],
        color="steelblue",
        marker="o",
        s=20,
        alpha=0.6,
        label="Non-homolog",
        zorder=2
    )

    # plot Homolog (red stars)
    ax1.scatter(
        target_frac[is_homo],
        decoy_frac[is_homo],
        color="red",
        marker="*",
        s=120,
        edgecolors="darkred",
        linewidths=0.8,
        label="Homolog",
        zorder=3
    )

    ax1.set_xscale("log")
    ax1.set_yscale("log")

    pad_factor = 2.0  
    ax1.set_xlim(min_frac / pad_factor, 1 * pad_factor)
    ax1.set_ylim(min_frac / pad_factor, 1 * pad_factor)

    ticks = [10**i for i in range(-6, 1)]
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)

    ax1.set_xlabel("Cumulative target fraction\nup to threshold", fontsize=16)
    ax1.set_ylabel("Cumulative decoy fraction\nup to threshold", fontsize=16)
    
    # eFDP & Real FDP
    ax2 = ax1.twinx()
    
    ax2.plot(target_frac, efdp, color="mediumseagreen", linestyle="-", linewidth=2, alpha=0.8, label="eFDP", zorder=4)
    
    # Real FDP with 0
    ax2.plot(target_frac, real_fdp_all, color="purple", linestyle="-", linewidth=2, alpha=0.3, label="Real FDP (inc. 0)", zorder=4)
    
    # Real FDP without 0
    ax2.plot(target_frac, real_fdp_strict, color="purple", linestyle="--", linewidth=2.5, alpha=0.9, label="Real FDP (strict -1)", zorder=4)
    
    ax2.legend(loc="upper left", fontsize=12)
    ax2.set_yscale("linear")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_yticks(np.linspace(0, 1, 6))
    ax2.set_ylabel("FDP (Estimated & Real)", fontsize=16, color="black", rotation=270, labelpad=20)
    ax2.tick_params(axis='y', colors="black", width=2, length=6, labelsize=12)
    
    # plot text
    ax1.text(0.95, 0.05, f"Query:\n{target_qid}", transform=ax1.transAxes, 
             fontsize=20, horizontalalignment='right', verticalalignment='bottom')

    # plot border
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    ax1.tick_params(width=2, length=6, labelsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] Saved Cumulative Fraction plot to {out_path}")





def plot_background_score_distributions(
    df_cal,
    target_qid,
    out_path,
    title=None,
):

    df_plot = df_cal[df_cal["qid"] == target_qid].copy()

    if len(df_plot) == 0:
        print(f"[Warning] No data found for query: {target_qid}")
        return

    if title is None:
        title = f"Background Score Distributions - {target_qid}"

    plt.figure(figsize=(6, 6))

    # Non-homologs
    is_homo = df_plot["homo_type_real"].isin([1, 2])
    df_nonhomo = df_plot[~is_homo]

    if not df_nonhomo.empty:
        sns.histplot(
            df_nonhomo["score_real_noisy"],
            bins=50,
            stat="density",
            kde=True,
            color="blue",
            alpha=0.3,
            label="Real Non-homolog (target')",
            edgecolor=None
        )

    if not df_plot.empty:
        sns.histplot(
            df_plot["score_decoy"],
            bins=50,
            stat="density",
            kde=True,
            color="gold",
            alpha=0.3,
            label="Uncalibrated Decoy",
            edgecolor=None
        )

        sns.histplot(
            df_plot["score_decoy_cal"],
            bins=50,
            stat="density",
            kde=True,
            color="green",
            alpha=0.3,
            label="Calibrated Decoy",
            edgecolor=None
        )

    plt.xlabel("Score", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title(title, fontsize=15, fontweight='bold')
    plt.legend(fontsize=12, loc="upper right")
    
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.5, length=5, labelsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] Saved Background Distribution plot to {out_path}")




def plot_noise_comparison_for_query(
    df_cal,
    target_qid,
    out_path,
    col_clean,
    col_noisy,
    label_clean,
    label_noisy,
    title=None,
):
    df_plot = df_cal[df_cal["qid"] == target_qid].copy()
    if df_plot.empty:
        print(f"[Warning] No data found for query: {target_qid}")
        return

    plt.figure(figsize=(6, 6))
    sns.histplot(
        df_plot[col_clean], bins=50, stat="density", kde=True,
        alpha=0.35, edgecolor=None, label=label_clean, color="steelblue",
    )
    sns.histplot(
        df_plot[col_noisy], bins=50, stat="density", kde=True,
        alpha=0.35, edgecolor=None, label=label_noisy, color="orange",
    )
    plt.xlabel("Score", fontsize=13)
    plt.ylabel("Density", fontsize=13)
    if title:
        plt.title(title, fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Saved noise comparison plot to {out_path}")




def find_bad_queries_by_q_range(
    df_cal,
    q_min=0.0,
    q_max=0.4,
    use_strict=True,
    out_txt=None,
):
    has_rep_id = "rep_id" in df_cal.columns

    bad_records = []
    summary_records = []

    for qid, df_q in df_cal.groupby("qid"):
        if has_rep_id:
            rep_ids = sorted(df_q["rep_id"].unique().tolist())
        else:
            rep_ids = [0]

        n_total = len(rep_ids)
        n_failed = 0

        for r in rep_ids:
            df_rep = df_q[df_q["rep_id"] == r] if has_rep_id else df_q
            rep_failed, rep_bad_points = _bad_records_for_query(
                qid, r, df_rep, q_min, q_max, use_strict,
            )
            if rep_failed:
                n_failed += 1
                bad_records.extend(rep_bad_points)

        summary_records.append({
            "qid": qid,
            "n_failed_reps": int(n_failed),
            "n_total_reps": int(n_total),
            "all_repeats_failed": bool(n_total > 0 and n_failed == n_total),
        })

    df_debug = pd.DataFrame(bad_records)
    df_summary = pd.DataFrame(summary_records)

    if not df_summary.empty:
        bad_queries = sorted(
            df_summary.loc[df_summary["all_repeats_failed"], "qid"].tolist()
        )
    else:
        bad_queries = []

    if out_txt is not None:
        with open(out_txt, "w") as f:
            for qid in bad_queries:
                f.write(f"{qid}\n")
        print(f"[OK] Saved bad query list to: {out_txt}")

        if not df_debug.empty:
            out_detail = out_txt.replace(".txt", "_details.txt")
            df_debug.to_csv(out_detail, sep="\t", index=False)
            print(f"[OK] Saved bad query details (per-rep) to: {out_detail}")

        if not df_summary.empty:
            out_summary = out_txt.replace(".txt", "_summary.txt")
            df_summary.to_csv(out_summary, sep="\t", index=False)
            print(f"[OK] Saved bad query summary (per-qid) to: {out_summary}")

    print(f"[INFO] Found {len(bad_queries)} bad queries (all reps failed) in q range [{q_min}, {q_max}].")
    return bad_queries, df_debug, df_summary



if __name__ == "__main__":

    method2type = {
        "shuf": "shuf",
        "shufpartly": "shuf",
        "shufpartly_60": "shufpartly",
        "shufpartly_40": "shufpartly",
        "shufpartly_80": "shufpartly",
        "rev": "rev",
        "mkv1": "mkv1",
        "mkv2": "mkv2",
        "dplm": "dplm",
        "PGmsk25pct": "PGmsk25pct",
        "copy": "copy",
        "extended_dup": "dup",
        "extended_shuf": "shuf",
        "extended_rev": "rev",
        "extended_then_shuf": "shuf",
        "adp_Region_W5S1P1T0_extended": "adp",
        "adp_Region_W5S1P1T1_extended": "adp",
        "extended_shufpartly40": "shufpartly",
        "adp_Region_W5S1P1w1T0": "adp",
        "extended_mkv1": "mkv",
        "extended_mkv2": "mkv",
    }

    methods = ['dhr_postprocess']
    decoy_methods = ['extended_shuf']

    for SEARCH_METHOD in methods:
        for DECOY_METHOD in decoy_methods:
            WEIGHT_METHOD = 'AdaptiveBell'
            tau = 0.2
            tol = 0.2
            max_iter = 2
            noise_lambda = 0.02   # noise
            n_reps = 1            # number of independent (d'', target') 

            DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
            OTHER_PLOT_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'plots')
            SEQ_NAME_QUERY = 'class_all'
            SEQ_NAME_TARGET = 'class_all'
            TYPE = method2type[DECOY_METHOD]

            path_real = f'{DATA_DIR}/result_{SEARCH_METHOD}_{SEQ_NAME_QUERY}_{SEQ_NAME_TARGET}.txt'
            path_decoy = f'{DATA_DIR}/result_{SEARCH_METHOD}_{SEQ_NAME_QUERY}_{DECOY_METHOD}_{SEQ_NAME_TARGET}.txt'

            decoy_suffix = f"_{TYPE}"
            # target_qids = ['d1a12a_', 'd1a0ia1', 'd1ahoa_', 'd1a79a1', 'd1bxda_', 'd1j2ra_', 'd1c48a', 'd1acwa_', 'd4hswa_', 'd1aoza3']
            target_qids = [
                            # 'd1gy7a_',
                            # 'd2csua3',
                            # 'd4k3wa_',
                            # 'd7cohm_',
                            # 'd1a12a_',
                            # 'd1a79a2',
                            # 'd1amma2',
                            # 'd1au1a_',
                            # 'd1bd8a_',
                            # 'd1ctfa_'
                            ]

            # 1. merge real and decoy tables
            df_merge = make_calibration_table(
                path_real=path_real,
                path_decoy=path_decoy,
                decoy_suffix=decoy_suffix
            )
            print("[INFO] merged rows:", len(df_merge))

            # 2. calculate weights
            df_out, model, scaler = fit_logistic_and_get_pqt(df_merge, method=WEIGHT_METHOD) 


            # 3. apply GAM calibration
            out_decoy_txt = f'{DATA_DIR}/result_{SEARCH_METHOD}_{SEQ_NAME_QUERY}_{DECOY_METHOD}_calibrated_gam_{WEIGHT_METHOD}_{SEQ_NAME_TARGET}.txt'
            out_target_txt = f'{DATA_DIR}/result_{SEARCH_METHOD}_{SEQ_NAME_QUERY}_target_noisy_{SEQ_NAME_TARGET}.txt'
            out_bad_query_txt = f"{DATA_DIR}/bad_queries_{SEARCH_METHOD}_{SEQ_NAME_QUERY}_{DECOY_METHOD}_{WEIGHT_METHOD}_{SEQ_NAME_TARGET}.txt"

            df_cal, df_params, gam_models, df_bad, df_bad_summary = apply_gam_calibration(
                df_out,
                tau=tau,
                n_splines=12,
                lam=1.0,
                num_points=1000,
                tol=tol,
                max_iter=max_iter,
                noise_lambda=noise_lambda,
                seed=43,
                n_reps=n_reps,
                # streaming output
                out_decoy_path=out_decoy_txt,
                out_target_path=out_target_txt,
                decoy_suffix=decoy_suffix,
                plot_qids=target_qids,
                # inline bad-query detection (was: post-hoc find_bad_queries_by_q_range)
                bad_q_min=0.0,
                bad_q_max=0.05,
                bad_use_strict=True,
            )
            print(f"[OK] Calibrated decoy data successfully saved to: {out_decoy_txt}")
            print(f"[OK] Noisy target data successfully saved to: {out_target_txt}\n")

            df_params.to_csv(
                f"{DATA_DIR}/gam_params_{SEARCH_METHOD}_{SEQ_NAME_QUERY}_{DECOY_METHOD}_{WEIGHT_METHOD}_{SEQ_NAME_TARGET}.txt",
                sep="\t", index=False,
            )
            print(f"[DEBUG] Saved gam parameters to: {DATA_DIR}/gam_params_{SEARCH_METHOD}_{SEQ_NAME_QUERY}_{DECOY_METHOD}_{WEIGHT_METHOD}_{SEQ_NAME_TARGET}.txt")

            # 4. persist inline-computed bad queries
            # Final bad list = qids that failed in EVERY rep.
            if not df_bad_summary.empty:
                bad_queries = sorted(
                    df_bad_summary.loc[df_bad_summary["all_repeats_failed"], "qid"].tolist()
                )
            else:
                bad_queries = []
            with open(out_bad_query_txt, "w") as f:
                for q in bad_queries:
                    f.write(f"{q}\n")
            print(f"[OK] Saved bad query list (all-reps-failed) to: {out_bad_query_txt}")

            if not df_bad.empty:
                out_bad_detail = out_bad_query_txt.replace(".txt", "_details.txt")
                df_bad.to_csv(out_bad_detail, sep="\t", index=False)
                print(f"[OK] Saved bad query per-rep details to: {out_bad_detail}")

            if not df_bad_summary.empty:
                out_bad_summary = out_bad_query_txt.replace(".txt", "_summary.txt")
                df_bad_summary.to_csv(out_bad_summary, sep="\t", index=False)
                print(f"[OK] Saved bad query per-qid summary to: {out_bad_summary}")

            print(f"[INFO] Found {len(bad_queries)} bad queries (all reps failed) in q range [0.0, 0.05].")

            ## Debug: save intermediate results for each iteration
            # max_iter = 5  
            # for k in range(1, max_iter + 1):
            #     k_col = f"score_decoy_cal_k{float(k)}"
            #     if k_col in df_cal.columns:
            #         df_k = df_cal.copy()
                    
            #         df_k = df_k.dropna(subset=[k_col])
                    
            #         if df_k.empty:
            #             print(f"[INFO] No queries reached k={float(k)}, skipping file generation.")
            #             continue
                    
            #         df_k["qid"] = df_k["qid"] + decoy_suffix
            #         df_k = df_k[["qid", "tid", k_col, "homo_type_decoy"]]
            #         df_k = df_k.rename(columns={
            #             k_col: "score",
            #             "homo_type_decoy": "homo_type"
            #         })

            #         df_k = df_k.sort_values(by=["qid", "score"], ascending=[True, False])
            #         df_k["rank"] = df_k.groupby("qid").cumcount() + 1
            #         df_k = df_k[["qid", "tid", "score", "homo_type", "rank"]]

            #         out_k_path = f'{DATA_DIR}/result_{SEARCH_METHOD}_{SEQ_NAME_QUERY}_{DECOY_METHOD}_calibrated_gam_{WEIGHT_METHOD}_tau{tau}_iter_k{float(k)}_{SEQ_NAME_TARGET}.txt'
            #         df_k.to_csv(out_k_path, sep="\t", index=False)
            #         print(f"[OK] Intermediate result (k={float(k)}) saved to: {out_k_path}")
            # print("\n")

            # 5. plot diagnostic plots for each query
            for target_qid in target_qids:
                print("===========================================")
                print(f"Generating Plots for: {target_qid}")
                print("===========================================")

                # Plot specific query distribution (all n_reps pooled)
                out_path_all = f"{OTHER_PLOT_OUT_DIR}/dist_{SEQ_NAME_QUERY}_{SEQ_NAME_TARGET}_{target_qid}_cal_{DECOY_METHOD}_{target_qid}.png"
                plot_score_distributions_for_query(
                    df_cal,
                    target_qid=target_qid,
                    out_path=out_path_all,
                    title=f"Score Dist ({DECOY_METHOD}) - {target_qid}"
                )

                # Plot Cumulative Fraction (uses full n_reps-expanded df_cal)
                out_path_cumfrac = f"{OTHER_PLOT_OUT_DIR}/cumfrac_{SEQ_NAME_QUERY}_{SEQ_NAME_TARGET}_{target_qid}_cal_{DECOY_METHOD}_{target_qid}.png"
                plot_cumulative_fraction_for_query(
                    df_cal,
                    target_qid=target_qid,
                    out_path=out_path_cumfrac,
                    score_col_decoy="score_decoy_cal",
                )

                # Plot target vs target' (all n_reps pooled)
                out_path_target_noise = f"{OTHER_PLOT_OUT_DIR}/noise_target_{SEQ_NAME_QUERY}_{SEQ_NAME_TARGET}_{target_qid}_{DECOY_METHOD}.png"
                plot_noise_comparison_for_query(
                    df_cal,
                    target_qid=target_qid,
                    out_path=out_path_target_noise,
                    col_clean="score_real",
                    col_noisy="score_real_noisy",
                    label_clean="Target (raw)",
                    label_noisy="Target' (noisy)",
                    title=f"Target vs Target' - {target_qid}",
                )

                # Plot decoy vs d'' (all n_reps pooled)
                out_path_decoy_noise = f"{OTHER_PLOT_OUT_DIR}/noise_decoy_{SEQ_NAME_QUERY}_{SEQ_NAME_TARGET}_{target_qid}_{DECOY_METHOD}.png"
                plot_noise_comparison_for_query(
                    df_cal,
                    target_qid=target_qid,
                    out_path=out_path_decoy_noise,
                    col_clean="score_decoy",
                    col_noisy="score_decoy_noisy",
                    label_clean="Decoy (raw)",
                    label_noisy="d'' (noisy)",
                    title=f"Decoy vs d'' - {target_qid}",
                )

                # # Plot Cumulative Fraction (Original)
                # out_path_cumfrac_orig = f"{OTHER_PLOT_OUT_DIR}/cumfrac_{SEQ_NAME_QUERY}_{SEQ_NAME_TARGET}_{target_qid}_ORIGINAL_{DECOY_METHOD}_{target_qid}.png"
                # plot_cumulative_fraction_for_query(
                #     df_cal,
                #     target_qid=target_qid,
                #     out_path=out_path_cumfrac_orig,
                #     score_col_decoy="score_decoy",
                # )
                # print("===========================================")