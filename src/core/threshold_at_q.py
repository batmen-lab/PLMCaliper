import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CORE = Path(__file__).resolve().parent
sys.path.insert(0, str(CORE))
from fdr import make_pair_table, estimate_fdr_curves_all_queries  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-method", required=True, help="plm | tmvec | dhr_postprocess | ...")
    ap.add_argument("--query-name", default="astral")
    ap.add_argument("--target-name", default="astral")
    ap.add_argument("--decoy-method", default="extended_shuf")
    ap.add_argument("--decoy-suffix", default="_shuf", help="suffix on decoy query ids")
    ap.add_argument("--weight-method", default="AdaptiveBell")
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--q", nargs="+", type=float,
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    help="target FDR levels (default 0.1 .. 0.9)")
    ap.add_argument("--out", default=None, help="write the (qid, q, threshold) table to this CSV")
    args = ap.parse_args()

    dd = Path(args.data_dir)
    m, qn, tn = args.search_method, args.query_name, args.target_name
    path_real = dd / f"result_{m}_{qn}_target_noisy_{tn}.txt"
    path_decoy = dd / f"result_{m}_{qn}_{args.decoy_method}_calibrated_gam_{args.weight_method}_{tn}.txt"
    for p in (path_real, path_decoy):
        if not p.exists():
            sys.exit(f"[fatal] missing calibrated file: {p}\n"
                     f"        run core/run_calibration_fdr_pipeline.py first (or --keep-intermediates).")

    df_merge = make_pair_table(str(path_real), str(path_decoy), args.decoy_suffix)
    df_curve, _ = estimate_fdr_curves_all_queries(df_merge)   # threshold-level eFDP curves

    rows = []
    for (qid, rep_id), grp in df_curve.groupby(["qid", "rep_id"], sort=False):
        for q in args.q:
            valid = grp[grp["est_fdp"] <= q]
            if len(valid):
                r = valid.loc[valid["threshold"].idxmin()]   # smallest score threshold with eFDP<=q
                th = float(r["threshold"]); n = int(r["n_selected"]); efdp = float(r["est_fdp"])
            else:
                th, n, efdp = np.nan, 0, np.nan               # FDR not controllable at this q
            rows.append({"qid": qid, "rep_id": rep_id, "q": q,
                         "threshold": th, "n_selected": n, "est_fdp": efdp})
    out = pd.DataFrame(rows)

    # console summary: median operating threshold across queries at each q
    print(f"[threshold_at_q] {m} / {qn} vs {tn}  ({out['qid'].nunique()} queries)")
    print(f"{'q':>5} {'median_threshold':>18} {'valid_queries':>14} {'median_n_selected':>18}")
    for q in args.q:
        s = out[out["q"] == q]
        valid = s["threshold"].notna()
        med_th = s.loc[valid, "threshold"].median() if valid.any() else float("nan")
        med_n = s.loc[valid, "n_selected"].median() if valid.any() else 0
        print(f"{q:>5.2f} {med_th:>18.4f} {int(valid.sum()):>14} {med_n:>18.0f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"[saved] per-(query, q) thresholds -> {args.out}")


if __name__ == "__main__":
    main()
