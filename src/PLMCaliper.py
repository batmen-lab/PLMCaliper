import argparse
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# id suffix that the extended-Markov decoy generator puts on every decoy
DECOY_SUFFIX = "_mkv"
DEFAULT_DESIGN = "extended_mkv"
DEFAULT_Q_LEVELS = [0.50]
SCORE_COLS = ["qid", "tid", "score"]
HIT_COLS = ["qid", "tid", "rep_id", "score", "est_fdp"]
CUTOFF_COLS = ["qid", "rep_id", "target_fdr", "cutoff_score", "est_fdp", "n_selected"]


def _tqdm(iterable=None, **kwargs):
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, **kwargs)


def protein_markovgen(sequence, k, on_short="shuffle"):
    """Resample a sequence from an order-k Markov chain fitted on itself.

    A sequence shorter than k+1 has no order-k statistics to fit. `on_short`
    picks what to do then: "shuffle" falls back to a plain shuffle (what the
    extended-Markov design does), "raise" rejects the sequence (what the plain
    Markov design does).
    """
    seq_len = len(sequence)
    if seq_len < k + 1:
        if on_short == "raise":
            raise ValueError(
                f"sequence of length {seq_len} is too short for Markov order {k}")
        seq_list = list(sequence)
        random.shuffle(seq_list)
        return ''.join(seq_list)

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

    return ''.join(result)


# ---------------------------------------------------------------------------
# Decoy designs
#
# A builder takes the parsed input records and returns (decoy_id, decoy_seq)
# pairs; make_decoy() turns those into the output FASTA. Biopython is imported
# lazily there, so the builders never touch it and stay easy to reuse.
#
# The suffix is what marks a query as a decoy downstream -- pass the same string
# to `calibrate --decoy-suffix`. The ablation in src/analysis/ablation compares
# exactly these designs.
# ---------------------------------------------------------------------------


def _build_extended_mkv(records, markov_order=2, seed=0):
    """seq + markov_k(seq): the decoy keeps the real sequence as a prefix."""
    random.seed(seed)
    out = []
    for record in records:
        seq = str(record.seq).upper()
        out.append((record.id + "_mkv", seq + protein_markovgen(seq, markov_order)))
    return out


def _build_mkv(records, markov_order=2, seed=0):
    """markov_k(seq) alone, same length as the query."""
    random.seed(seed)
    out = []
    for record in records:
        seq = str(record.seq)
        try:
            decoy = protein_markovgen(seq, markov_order, on_short="raise")
        except ValueError as err:
            raise SystemExit(
                f"[PLMCaliper] {record.id}: {err}.\n"
                f"             Lower --markov-order, drop the short sequences, or use "
                f"--design extended_mkv, which shuffles them instead.") from None
        out.append((f"{record.id}_mkv{markov_order}", decoy))
    return out


def _build_shuf(records, markov_order=2, seed=0):
    """shuffle(seq): same residue composition, no order."""
    rng = np.random.default_rng(seed=seed)
    out = []
    for record in records:
        seq_list = list(str(record.seq))
        rng.shuffle(seq_list)
        out.append((record.id + "_shuf", "".join(seq_list)))
    return out


def _build_rev(records, markov_order=2, seed=0):
    """reverse(seq): composition and local windows preserved, direction flipped."""
    return [(record.id + "_rev", str(record.seq)[::-1]) for record in records]


def _build_dplm(records, markov_order=2, seed=0):
    """Decoys sampled from the DPLM protein language model. Not bundled here."""
    raise SystemExit(
        "[PLMCaliper] the 'dplm' design samples decoys from the DPLM protein "
        "language model, which is not part of this release.\n"
        "             Generate the decoy FASTA with DPLM separately, give every "
        "id the '_dplm' suffix, then run\n"
        "             PLMCaliper.py calibrate --decoy-suffix _dplm ...")


# name -> (id suffix, builder, default output stem suffix)
DECOY_DESIGNS = {
    "extended_mkv": ("_mkv",  _build_extended_mkv, "_extended_mkv{k}"),
    "mkv":          ("_mkv{k}", _build_mkv,        "_mkv{k}"),
    "shuf":         ("_shuf", _build_shuf,         "_shuf"),
    "rev":          ("_rev",  _build_rev,          "_rev"),
    "dplm":         ("_dplm", _build_dplm,         "_dplm"),
}


def decoy_suffix_for(design, markov_order=2):
    """The id suffix a design stamps on its decoys -- calibrate's --decoy-suffix."""
    return DECOY_DESIGNS[design][0].format(k=markov_order)


def make_decoy(fasta_path, out_path=None, markov_order=2, seed=0, design=DEFAULT_DESIGN):
    """Write one decoy per query, using the named decoy design."""
    from Bio import SeqIO          # biopython is only needed for this step
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    if design not in DECOY_DESIGNS:
        raise SystemExit(f"[PLMCaliper] unknown decoy design {design!r}; expected one of "
                         f"{', '.join(sorted(DECOY_DESIGNS))}")
    id_suffix, build, stem_suffix = DECOY_DESIGNS[design]
    out_path = out_path or (f"{os.path.splitext(fasta_path)[0]}"
                            f"{stem_suffix.format(k=markov_order)}.fa")

    records = build(list(SeqIO.parse(fasta_path, "fasta")),
                    markov_order=markov_order, seed=seed)
    SeqIO.write([SeqRecord(Seq(seq), id=rid, description="") for rid, seq in records],
                out_path, "fasta")
    print(f"[PLMCaliper] {len(records)} '{design}' decoy sequences written "
          f"(id suffix {id_suffix.format(k=markov_order)})")
    return out_path


def load_score_table(path, what):
    """Read a qid/tid/score table. Separator is sniffed; extra columns are dropped."""
    df = pd.read_csv(path, sep=None, engine="python")
    missing = set(SCORE_COLS) - set(df.columns)
    if missing:
        raise SystemExit(
            f"[PLMCaliper] {what} table {path} is missing column(s) {sorted(missing)}.\n"
            f"             Found: {list(df.columns)}\n"
            f"             Expected a header with at least: qid, tid, score"
        )
    df = df[SCORE_COLS].copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    n_bad = int(df["score"].isna().sum())
    if n_bad:
        print(f"[PLMCaliper] dropping {n_bad} row(s) in the {what} table with a non-numeric score")
        df = df.dropna(subset=["score"])
    return df


def strip_decoy_suffix(qid, decoy_suffix):
    return qid[: -len(decoy_suffix)] if qid.endswith(decoy_suffix) else qid


def require_unique_pair_keys(df, what, keys):
    dup = df.duplicated(keys, keep=False)
    if not bool(dup.any()):
        return

    examples = df.loc[dup, keys].value_counts().head(3)
    example_text = "; ".join(
        f"{tuple(key) if isinstance(key, tuple) else key} x{count:,}"
        for key, count in examples.items()
    )
    raise SystemExit(
        f"[PLMCaliper] {what} table has duplicated pairing keys ({', '.join(keys)}), "
        "which would expand the target-decoy merge.\n"
        f"             Duplicate rows: {int(dup.sum()):,}; examples: {example_text}\n"
        "             Check the retrieval output for repeated or invalid hits."
    )


def build_pair_table(df_real, df_decoy, decoy_suffix):
    """Pair on the same target, which controls for target-side properties so that the
    only thing varying is real query vs its decoy."""
    real = df_real.rename(columns={"score": "score_real"})
    decoy = df_decoy.rename(columns={"score": "score_decoy"}).copy()
    decoy["qid_orig"] = decoy["qid"].map(lambda x: strip_decoy_suffix(x, decoy_suffix))
    require_unique_pair_keys(real, "real", ["qid", "tid"])
    require_unique_pair_keys(decoy, "decoy", ["qid_orig", "tid"])

    df = real.merge(
        decoy[["qid_orig", "tid", "score_decoy"]],
        left_on=["qid", "tid"], right_on=["qid_orig", "tid"], how="inner",
    ).drop(columns=["qid_orig"])

    if df.empty:
        raise SystemExit(
            "[PLMCaliper] no (qid, tid) pair matched between the two tables.\n"
            f"             Decoy ids are expected to end with '{decoy_suffix}' and to\n"
            "             otherwise equal the real query id. Check --decoy-suffix."
        )
    return df


def fit_importance_weights(df_merge):
    """Adaptive Bell weights: where on the score axis the calibration must be accurate."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

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

    # asymmetric Gaussian on this query's decoy 95th pct, stretched right so the fit
    # stays accurate above the top of the null -- where the threshold lands
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


def apply_calibration(df_merge, out_decoy_path, out_target_path, *, decoy_suffix="",
                      tau=0.2, n_splines=12, lam=1.0, num_points=1000, tol=0.2,
                      max_iter=2, noise_lambda=0.02, seed=43, n_reps=1):
    """Fit and apply the per-query decoy->real-null map, writing the calibrated decoy
    and noise-matched target tables (matched noise keeps the competition fair)."""
    from patsy import dmatrix, build_design_matrices
    import cvxpy as cp

    rng = np.random.default_rng(seed)
    out_cols = ["qid", "tid", "score", "rep_id"]
    params_list = []

    f_decoy = open(out_decoy_path, "w")
    f_target = open(out_target_path, "w")
    f_decoy.write("\t".join(out_cols) + "\n")
    f_target.write("\t".join(out_cols) + "\n")

    n_queries = int(df_merge["qid"].nunique())
    query_iter = _tqdm(df_merge.groupby("qid"), total=n_queries,
                       desc="  -> calibration", unit="query", leave=False)

    for qid, df_q in query_iter:
        idx = df_q.index

        # normalise both sides on the *real* scale so knots/anchors compare across queries
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
        x_all_dp = x_all_dps[0]          # rep 0 drives the fit-iteration selection
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
            probs = 1.0 - (1.0 - t) ** k_val            # sample density tilted to the tail

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

            # no hard monotonicity constraint: monotone Q-Q anchors + smoothness suffice
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
        best_k = np.nan
        k_val = 1.0

        for _ in range(max_iter):
            success, coeffs, design_info, B_all = try_fit(k_val)
            if not success:
                k_val += 1.0
                continue

            y_cal_tmp = B_all @ coeffs

            # score the fit on the top 5% tail only -- where the threshold is read off
            probs = np.linspace(0.95, 1.0, 50)
            y_cal_std = (y_cal_tmp - np.mean(y_cal_tmp)) / (np.std(y_cal_tmp) + 1e-8)
            y_true_std = (y_real_noisy - np.mean(y_real_noisy)) / (np.std(y_real_noisy) + 1e-8)
            current_loss = float(np.mean(np.abs(np.quantile(y_cal_std, probs)
                                                - np.quantile(y_true_std, probs))))

            if current_loss < best_loss:
                best_loss = current_loss
                best_model = (coeffs, design_info)
                best_k = k_val

            if current_loss <= tol:
                break
            k_val += 1.0

        final_success = best_model is not None
        if final_success:
            coeffs, design_info = best_model
        else:
            print(f"[PLMCaliper] quantile solver failed for {qid}; falling back to the mean.")

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
            d_block[out_cols].to_csv(f_decoy, sep="\t", index=False, header=False)

            t_block = pd.DataFrame({
                "qid": str(qid),
                "tid": df_q_raw["tid"].values,
                "score": yr_noisy * real_range + real_min,
            }).sort_values("score", ascending=False)
            t_block["rep_id"] = r
            t_block[out_cols].to_csv(f_target, sep="\t", index=False, header=False)

        params_list.append({
            "qid": qid,
            "used_k": round(best_k, 2) if pd.notnull(best_k) else np.nan,
            "tail_dist": round(best_loss, 5) if np.isfinite(best_loss) else np.nan,
        })

    f_decoy.close()
    f_target.close()
    return pd.DataFrame(params_list)


def efdp_curve(scores_real, scores_decoy):
    """eFDP at every candidate threshold for one query."""
    s_t = np.sort(np.asarray(scores_real, dtype=float))
    s_d = np.sort(np.asarray(scores_decoy, dtype=float))

    cand_t = np.unique(np.concatenate([s_t, s_d]))

    n_real = len(s_t) - np.searchsorted(s_t, cand_t, side="left")
    n_decoy = len(s_d) - np.searchsorted(s_d, cand_t, side="left")

    # +1 is the standard TDA conservative correction (else eFDP reads exactly 0 in the tail)
    est_fdp = (n_decoy + 1.0) / np.maximum(n_real, 1.0)
    return cand_t, est_fdp, n_real


def cutoffs_for_query(scores_real, scores_decoy, q_levels):
    """Smallest threshold with eFDP <= target_fdr. NaN when unreachable."""
    cand_t, est_fdp, n_real = efdp_curve(scores_real, scores_decoy)

    rows = []
    for target_fdr in q_levels:
        ok = np.flatnonzero(est_fdp <= target_fdr)
        if ok.size:
            i = ok[int(np.argmin(cand_t[ok]))]   # most permissive cutoff within budget
            rows.append({
                "target_fdr": target_fdr,
                "cutoff_score": float(cand_t[i]),
                "est_fdp": float(est_fdp[i]),
                "n_selected": int(n_real[i]),
            })
        else:
            rows.append({
                "target_fdr": target_fdr,
                "cutoff_score": np.nan,
                "est_fdp": np.nan,
                "n_selected": 0,
            })
    return rows


def hits_for_cutoffs(paired, cutoffs):
    """Selected target hits, with eFDP computed at each hit's score."""
    grouped = {}
    for key, grp in paired.groupby(["qid", "rep_id"], sort=False):
        cand, est_fdp, _ = efdp_curve(grp["score_real"].values, grp["score_decoy"].values)
        hit = grp[["qid", "tid", "rep_id", "score_real"]].rename(
            columns={"score_real": "score"}).copy()
        at = np.searchsorted(cand, hit["score"].to_numpy(dtype=float), side="right") - 1
        hit["est_fdp"] = est_fdp[np.maximum(at, 0)]
        grouped[key] = hit.sort_values(["score", "tid"], ascending=[False, True])

    frames = []
    for _, row in cutoffs.sort_values(["qid", "rep_id", "target_fdr"]).iterrows():
        cutoff = row["cutoff_score"]
        if pd.isna(cutoff):
            continue
        grp = grouped.get((row["qid"], row["rep_id"]))
        if grp is None:
            continue
        hit = grp[grp["score"] >= cutoff].copy()
        if hit.empty:
            continue
        frames.append(hit[HIT_COLS])

    if not frames:
        return pd.DataFrame(columns=HIT_COLS)
    return pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)


def default_hits_path(out_path):
    out_path = Path(out_path)
    if out_path.suffix:
        return out_path.with_name(f"{out_path.stem}_hits{out_path.suffix}")
    return out_path.with_name(f"{out_path.name}_hits.tsv")


def calibrate(real_path, decoy_path, out_path, *, decoy_suffix=DECOY_SUFFIX,
              q_levels=None, weight_method="AdaptiveBell", tau=0.2, tol=0.2,
              max_iter=2, noise_lambda=0.02, n_reps=1, seed=43,
              n_splines=12, lam=1.0, num_points=1000, keep_intermediates=None,
              write_hits=False, hits_out_path=None):
    q_levels = sorted(q_levels or DEFAULT_Q_LEVELS)
    if weight_method != "AdaptiveBell":
        raise ValueError("PLM-Caliper uses AdaptiveBell weights; other weight methods are not supported.")

    print("[PLMCaliper] reading score tables")
    df_real = load_score_table(real_path, "real")
    df_decoy = load_score_table(decoy_path, "decoy")

    df_merge = build_pair_table(df_real, df_decoy, decoy_suffix)
    print(f"[PLMCaliper] {len(df_merge):,} paired scores over {df_merge['qid'].nunique():,} queries")

    print("[PLMCaliper] fitting importance weights (AdaptiveBell)")
    df_weighted = fit_importance_weights(df_merge)

    workdir = Path(keep_intermediates) if keep_intermediates else Path(tempfile.mkdtemp(prefix="plmcaliper_"))
    workdir.mkdir(parents=True, exist_ok=True)
    cal_decoy = workdir / "calibrated_decoy.tsv"
    cal_target = workdir / "target_noisy.tsv"

    print(f"[PLMCaliper] per-query calibration (tau={tau}, n_reps={n_reps})")
    apply_calibration(
        df_weighted, str(cal_decoy), str(cal_target), decoy_suffix=decoy_suffix,
        tau=tau, n_splines=n_splines, lam=lam, num_points=num_points, tol=tol,
        max_iter=max_iter, noise_lambda=noise_lambda, seed=seed, n_reps=n_reps,
    )

    # re-pair the calibrated decoy against the noise-matched target
    df_t = pd.read_csv(cal_target, sep="\t")
    df_d = pd.read_csv(cal_decoy, sep="\t")
    df_d["qid"] = df_d["qid"].map(lambda x: strip_decoy_suffix(x, decoy_suffix))
    paired = df_t.merge(df_d, on=["qid", "tid", "rep_id"], suffixes=("_real", "_decoy"))

    print(f"[PLMCaliper] target-decoy competition over {paired['qid'].nunique():,} queries")
    rows = []
    for (qid, rep_id), grp in paired.groupby(["qid", "rep_id"], sort=False):
        for r in cutoffs_for_query(grp["score_real"].values, grp["score_decoy"].values, q_levels):
            rows.append({"qid": qid, "rep_id": rep_id, **r})
    out = pd.DataFrame(rows)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out = out[CUTOFF_COLS]

    # cutoff_score must round-trip exactly: rounded to 6 significant digits it no
    # longer selects the hits it was computed for. target_fdr stays short and readable.
    written = out.copy()
    written["target_fdr"] = [f"{v:g}" for v in written["target_fdr"]]
    written["cutoff_score"] = [("" if v != v else f"{v:.17g}") for v in written["cutoff_score"]]
    written["est_fdp"] = [("" if v != v else f"{v:.6g}") for v in written["est_fdp"]]
    written.to_csv(out_path, sep="\t", index=False)
    print(f"[PLMCaliper] wrote per-query cutoffs -> {out_path}")

    if write_hits:
        hits_path = Path(hits_out_path) if hits_out_path else default_hits_path(out_path)
        hits_path.parent.mkdir(parents=True, exist_ok=True)
        hits = hits_for_cutoffs(paired, out)
        written_hits = hits.copy()
        if not written_hits.empty:
            written_hits["score"] = [f"{v:.17g}" for v in written_hits["score"]]
            written_hits["est_fdp"] = [f"{v:.6g}" for v in written_hits["est_fdp"]]
        written_hits.to_csv(hits_path, sep="\t", index=False)
        print(f"[PLMCaliper] wrote selected hits -> {hits_path} ({len(hits):,} rows)")

    print(f"\n{'target_fdr':>10} {'queries_controlled':>19} {'median_cutoff':>15} {'median_n_hits':>15}")
    for target_fdr in q_levels:
        s = out[out["target_fdr"] == target_fdr]
        ok = s["cutoff_score"].notna()
        med_c = s.loc[ok, "cutoff_score"].median() if ok.any() else float("nan")
        med_n = s.loc[ok, "n_selected"].median() if ok.any() else 0
        print(f"{target_fdr:>10.2f} {f'{int(ok.sum())}/{len(s)}':>19} {med_c:>15.4f} {med_n:>15.0f}")

    if keep_intermediates:
        print(f"\n[PLMCaliper] intermediates kept in {workdir}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="PLMCaliper",
        description="Label-free per-query FDR control for homolog search. Run make-decoy, "
                    "search the decoy against the same target DB, then calibrate.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-decoy", help="build the decoy FASTA to search")
    p.add_argument("--fasta", required=True, help="query FASTA")
    p.add_argument("--out", default=None, help="output FASTA (default: <stem><design suffix>.fa)")
    p.add_argument("--design", default=DEFAULT_DESIGN, choices=sorted(DECOY_DESIGNS),
                   help=f"decoy design (default: {DEFAULT_DESIGN}, the one used in the paper)")
    p.add_argument("--markov-order", type=int, default=2,
                   help="k for the extended_mkv and mkv designs")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("calibrate", help="score tables -> per-query cutoffs")
    p.add_argument("--real", required=True, help="search results for the real queries (qid/tid/score)")
    p.add_argument("--decoy", required=True, help="search results for the decoy queries (qid/tid/score)")
    p.add_argument("--out", required=True, help="output TSV of per-query cutoffs")
    p.add_argument("--decoy-suffix", default=DECOY_SUFFIX,
                   help=f"id suffix identifying decoy queries (default: {DECOY_SUFFIX})")
    p.add_argument("--target-fdr", nargs="+", type=float, required=True,
                   help="target FDR level(s), for example: --target-fdr 0.2")
    p.add_argument("--tau", type=float, default=0.2, help="quantile level of the pinball loss")
    p.add_argument("--num-points", type=int, default=1000,
                   help="quantile anchors per query; 1000 is what the paper used")
    p.add_argument("--tol", type=float, default=0.2, help="tail-match tolerance for the fit")
    p.add_argument("--max-iter", type=int, default=2)
    p.add_argument("--noise-lambda", type=float, default=0.02)
    p.add_argument("--n-reps", type=int, default=1)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--keep-intermediates", default=None, metavar="DIR",
                   help="keep the calibrated-decoy / noisy-target tables in DIR")
    p.add_argument("-hits", "--hits", action="store_true",
                   help="also write selected hits with eFDP at each hit's score")
    p.add_argument("--hits-out", default=None,
                   help="output TSV for -hits/--hits (default: <out stem>_hits.tsv)")

    args = ap.parse_args(argv)

    if args.cmd == "make-decoy":
        out = make_decoy(args.fasta, args.out, args.markov_order, args.seed, args.design)
        suffix = decoy_suffix_for(args.design, args.markov_order)
        print(f"[PLMCaliper] decoy FASTA -> {out}")
        print("[PLMCaliper] next: search this file against the same target DB, then run\n"
              "             PLMCaliper.py calibrate --real <real.tsv> --decoy <decoy.tsv> "
              f"--decoy-suffix {suffix} --target-fdr <fdr> --out cutoffs.tsv")
    elif args.cmd == "calibrate":
        calibrate(args.real, args.decoy, args.out,
                  decoy_suffix=args.decoy_suffix, q_levels=args.target_fdr,
                  tau=args.tau, tol=args.tol,
                  max_iter=args.max_iter, noise_lambda=args.noise_lambda,
                  n_reps=args.n_reps, seed=args.seed, num_points=args.num_points,
                  keep_intermediates=args.keep_intermediates,
                  write_hits=args.hits, hits_out_path=args.hits_out)


if __name__ == "__main__":
    main()
