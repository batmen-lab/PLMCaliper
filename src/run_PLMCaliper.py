import argparse
from pathlib import Path
import pandas as pd
from PLMCaliper import PLMCaliper, DECOY_DESIGNS


def main(args):
    if args.cmd == "make-decoy":
        out = PLMCaliper().build_decoy(args.fasta, args.out, design=args.design,
                                       markov_order=args.markov_order, seed=args.seed)
        suffix = DECOY_DESIGNS[args.design][0].format(k=args.markov_order)
        print(f"[PLMCaliper] saved decoy FASTA to {out}")

    elif args.cmd == "calibrate":
        q_levels = sorted(args.target_fdr)

        print("[PLMCaliper] reading score tables")
        df_real = pd.read_csv(args.real, sep=None, engine="python").dropna(subset=["score"])
        df_decoy = pd.read_csv(args.decoy, sep=None, engine="python").dropna(subset=["score"])

        calibrated_decoy, noisy_target, efdps, cutoffs, hits = PLMCaliper(
            decoy_suffix=args.decoy_suffix, tau=args.tau, tol=args.tol,
            max_iter=args.max_iter, noise_lambda=args.noise_lambda,
            n_reps=args.n_reps, seed=args.seed, num_points=args.num_points,
        ).plmcaliper(df_real, df_decoy, q_levels)

        if args.keep_intermediates:
            workdir = Path(args.keep_intermediates)
            workdir.mkdir(parents=True, exist_ok=True)
            calibrated_decoy.to_csv(workdir / "calibrated_decoy.tsv", sep="\t", index=False)
            noisy_target.to_csv(workdir / "target_noisy.tsv", sep="\t", index=False)
            print(f"[PLMCaliper] saved calibrated decoy and noisy target to {workdir}")

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)

        written = cutoffs.copy()
        written["target_fdr"] = [f"{v:g}" for v in written["target_fdr"]]
        written["cutoff_score"] = [("" if v != v else f"{v:.17g}") for v in written["cutoff_score"]]
        written["est_fdp"] = [("" if v != v else f"{v:.6g}") for v in written["est_fdp"]]
        written.to_csv(args.out, sep="\t", index=False)
        print(f"[PLMCaliper] saved per-query cutoffs to {args.out}")

        if args.hits_out:
            hits_path = Path(args.hits_out)
            hits_path.parent.mkdir(parents=True, exist_ok=True)
            written_hits = hits.copy()
            written_hits["score"] = [f"{v:.17g}" for v in written_hits["score"]]
            written_hits["est_fdp"] = [f"{v:.6g}" for v in written_hits["est_fdp"]]
            written_hits.to_csv(hits_path, sep="\t", index=False)
            print(f"[PLMCaliper] saved selected hits to {hits_path}")

        print(f"\n{'target_fdr':>10} {'queries_controlled':>19} {'median_cutoff':>15} {'median_n_hits':>15}")
        for target_fdr in q_levels:
            s = cutoffs[cutoffs["target_fdr"] == target_fdr]
            ok = s["cutoff_score"].notna()
            med_c = s.loc[ok, "cutoff_score"].median() if ok.any() else float("nan")
            med_n = s.loc[ok, "n_selected"].median() if ok.any() else 0
            print(f"{target_fdr:>10.2f} {f'{int(ok.sum())}/{len(s)}':>19} {med_c:>15.4f} {med_n:>15.0f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        prog="PLMCaliper",
        description="Label-free per-query FDR control for protein retrieval. Make decoy, "
                    "search the decoy against the same target DB, then calibrate.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-decoy", help="build the decoy FASTA")
    p.add_argument("--fasta", required=True, help="query FASTA")
    p.add_argument("--out", default=None, help="output FASTA")
    p.add_argument("--design", default="extended_mkv", choices=sorted(DECOY_DESIGNS), help="decoy designs")
    p.add_argument("--markov-order", type=int, default=2, help="Markov order used for extended_mkv and mkv")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("calibrate", help="run PLM-Caliper calibration")
    p.add_argument("--real", required=True, help="search results for the real queries")
    p.add_argument("--decoy", required=True, help="search results for the decoy queries")
    p.add_argument("--out", required=True, help="output TSV of per-query cutoffs")
    p.add_argument("--decoy-suffix", default="_mkv", help="id suffix identifying decoy queries")
    p.add_argument("--target-fdr", nargs="+", type=float, required=True, help="target FDR level(s)")
    p.add_argument("--tau", type=float, default=0.2, help="quantile level of the pinball loss")
    p.add_argument("--num-points", type=int, default=1000, help="number of quantile anchors used in the spline fit.")
    p.add_argument("--tol", type=float, default=0.2, help="tail-match tolerance for the fit")
    p.add_argument("--max-iter", type=int, default=2, help="maximum spline fits per query")
    p.add_argument("--noise-lambda", type=float, default=0.02, help="gaussian noise scale")
    p.add_argument("--n-reps", type=int, default=1, help="number of noise replicates")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--keep-intermediates", default=None, help="save the calibrated-decoy and noisy-target tables")
    p.add_argument("--hits-out", default=None, help="save selected hits with eFDP at each hit's score to this TSV")

    main(ap.parse_args())
