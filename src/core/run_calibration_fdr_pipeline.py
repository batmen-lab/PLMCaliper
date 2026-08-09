import argparse
import gc
import sys
import types
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

CORE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CORE_DIR.parents[1]
sys.path.insert(0, str(CORE_DIR))

try:
    import tqdm  # noqa: F401
except ImportError:
    class _FakeTqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, n=1):
            return None

        def close(self):
            return None

    _fake_tqdm_mod = types.ModuleType("tqdm")
    _fake_tqdm_mod.tqdm = _FakeTqdm
    sys.modules["tqdm"] = _fake_tqdm_mod

from calibration import (  # noqa: E402
    apply_gam_calibration,
    fit_logistic_and_get_pqt,
    make_calibration_table,
)
from fdr import (  # noqa: E402
    estimate_fdr_curves_all_queries,
    estimate_fdr_curves_streaming,
    file_size_gb,
    make_pair_table,
    plot_q_vs_real_fdr_and_discoveries_from_summary,
    should_stream_pair_table,
    summarize_fdr_curves_at_q_levels,
)

METHOD2TYPE = {
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
    "adp201515k1_cont_incrsalpha": "adp",
    "adp201515k1_cont_incrsalpha_esm1b": "adp",
    "adp201515k1_cont_incrsalpha_protbert": "adp",
    "adp105015klist2_rand_incrsalpha_impute2L": "adpk2",
    "adp105015klist2_rand_incrsalpha_impute2L_esm1b": "adpk2",
    "adp105015klist2_rand_incrsalpha_impute2L_protbert": "adpk2",
    "adp201515k1_rand_test_opt": "adp",
    "adp201515k1_rand_test_ctl": "adp",
    "adp401515k1_rand_test_ctl": "adp",
    "adp401515k1_rand_test_opt": "adp",
    "adp601515k1_rand_test_ctl": "adp",
    "adp601515k1_rand_test_opt": "adp",
    "adp801515k1_rand_test_ctl": "adp",
    "adp801515k1_rand_test_opt": "adp",
    "copy": "copy",
    "adp_Region_W5S1P1w1": "adp",
    "adp_Region_W5S1P1w2": "adp",
    "adp_Region_W5S1P1w5": "adp",
    "adp_Region_W5S1P1w1T0": "adp",
    "adp_Region_W5S1P1w1_var": "adp",
    "adp_Region_W5S1P1w0.5_var": "adp",
    "adp_Region_W5S1P1w3_var": "adp",
    "adp_Region_W5S1P1w0_var": "adp",
    "adp_Region_W5S1P1w0T0_var": "adp",
    "adp_Region_W5S1P1w1T0_var": "adp",
    "adp_Region_W5S1P1w1T1_var": "adp",
    "adp_Region_W5S1P1w1T2_var": "adp",
    "adp_Region_W5S1P1w0T1_var": "adp",
    "adp_Region_W5S1P1w1_v1": "adp",
    "adp_Region_W5S1P1w1_v2": "adp",
    "adp_Region_W10S1P1w1_v1": "adp",
    "adp_Region_W10S1P1w1_v2": "adp",
    "adp_Region_W5S1P1T1_sensloc": "adp",
    "adp_Region_W5S1P1T0_sensloc": "adp",
    "extended_dup": "dup",
    "extended_shuf": "shuf",
    "extended_rev": "rev",
    "extended_then_shuf": "shuf",
    "adp_Region_W5S1P1T0_extended": "adp",
    "adp_Region_W5S1P1T1_extended": "adp",
    "extended_shufpartly40": "shufpartly",
    "extended_mkv1": "mkv",
    "extended_mkv2": "mkv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibration + FDR pipeline")
    parser.add_argument(
        "--search-methods",
        nargs="+",
        default=["plm", "tmvec"],
        help="Searching methods to run (default: plm tmvec)",
    )
    parser.add_argument(
        "--decoy-methods",
        nargs="+",
        default=["extended_shuf", "shuf"],
        help="Decoy types to run (default: extended_shuf shuf)",
    )
    parser.add_argument("--query-name", default="putative")
    parser.add_argument("--target-name", default="class_all")
    parser.add_argument("--weight-method", default="AdaptiveBell")
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--tol", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=2)
    parser.add_argument("--noise-lambda", type=float, default=0.02)
    parser.add_argument("--n-reps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
    )
    parser.add_argument(
        "--curve-archive-dir",
        type=Path,
        default=PROJECT_ROOT / "results_final" / "performance" / "data",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=PROJECT_ROOT / "results_final" / "performance",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip calibration; expect calibrated + target_noisy files to exist",
    )
    parser.add_argument(
        "--skip-fdr",
        action="store_true",
        help="Run calibration only, skip FDR and plotting",
    )
    parser.add_argument(
        "--force-target-noisy",
        action="store_true",
        help="Regenerate target_noisy even when the file already exists",
    )
    parser.add_argument(
        "--save-calibrated",
        action="store_true",
        default=True,
        help="Write calibrated decoy txt to data/ (default: True)",
    )
    parser.add_argument(
        "--save-full-curves",
        action="store_true",
        help="Also save full threshold-level df_curve archive as plain TSV (very large)",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep calibrated decoy and target-noisy score files after each combo",
    )
    parser.add_argument(
        "--force-load-all",
        action="store_true",
        help="Load full pair table into memory for FDR (may OOM on large datasets)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="CPU workers for streaming FDR (default: 1). Use CPU core count, not GPU count.",
    )
    return parser.parse_args()


def _stem(
    search_method: str,
    query_name: str,
    decoy_method: str,
    weight_method: str,
    target_name: str,
) -> str:
    return (
        f"df_curve_all_{search_method}_{query_name}_{decoy_method}_"
        f"{weight_method}_{target_name}"
    )


def _plot_summary_stem(
    search_method: str,
    query_name: str,
    decoy_method: str,
    weight_method: str,
    target_name: str,
) -> str:
    return (
        f"fdr_plot_summary_{search_method}_{query_name}_{decoy_method}_"
        f"{weight_method}_{target_name}"
    )


def path_raw_real(data_dir: Path, search_method: str, query_name: str, target_name: str) -> Path:
    return data_dir / f"result_{search_method}_{query_name}_{target_name}.txt"


def path_raw_decoy(
    data_dir: Path,
    search_method: str,
    query_name: str,
    decoy_method: str,
    target_name: str,
) -> Path:
    return data_dir / f"result_{search_method}_{query_name}_{decoy_method}_{target_name}.txt"


def path_target_noisy(data_dir: Path, search_method: str, query_name: str, decoy_method: str, target_name: str) -> Path:
    return data_dir / f"result_{search_method}_{query_name}_{decoy_method}_target_noisy_{target_name}.txt"


def path_calibrated_decoy(
    data_dir: Path,
    search_method: str,
    query_name: str,
    decoy_method: str,
    weight_method: str,
    target_name: str,
) -> Path:
    return data_dir / (
        f"result_{search_method}_{query_name}_{decoy_method}_calibrated_gam_"
        f"{weight_method}_{target_name}.txt"
    )


def save_dataframe_tsv(df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix != ".txt":
        out_path = out_path.with_suffix(".txt")
    df.to_csv(out_path, sep="\t", index=False)
    return out_path


def cleanup_intermediate_scores(
    *,
    data_dir: Path,
    search_method: str,
    query_name: str,
    decoy_method: str,
    target_name: str,
    weight_method: str,
) -> None:
    for path in (
        path_calibrated_decoy(
            data_dir, search_method, query_name, decoy_method, weight_method, target_name
        ),
        path_target_noisy(data_dir, search_method, query_name, decoy_method, target_name),
    ):
        if path.exists():
            path.unlink()
            print(f"[cleanup] removed {path.name}")


def run_calibration(
    *,
    data_dir: Path,
    search_method: str,
    decoy_method: str,
    query_name: str,
    target_name: str,
    weight_method: str,
    tau: float,
    tol: float,
    max_iter: int,
    noise_lambda: float,
    n_reps: int,
    seed: int,
    write_target_noisy: bool,
    save_calibrated: bool,
) -> None:
    if decoy_method not in METHOD2TYPE:
        raise ValueError(f"Unknown decoy method '{decoy_method}'. Add it to METHOD2TYPE.")

    path_real = path_raw_real(data_dir, search_method, query_name, target_name)
    path_decoy = path_raw_decoy(data_dir, search_method, query_name, decoy_method, target_name)
    out_target = path_target_noisy(data_dir, search_method, query_name, decoy_method, target_name)
    out_decoy = path_calibrated_decoy(
        data_dir, search_method, query_name, decoy_method, weight_method, target_name
    )

    for p in (path_real, path_decoy):
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    decoy_suffix = f"_{METHOD2TYPE[decoy_method]}"
    print(f"[calibration] merge tables: {path_real.name} + {path_decoy.name}")
    df_merge = make_calibration_table(
        path_real=str(path_real),
        path_decoy=str(path_decoy),
        decoy_suffix=decoy_suffix,
    )
    print(f"[calibration] merged rows: {len(df_merge)}")

    df_out, _, _ = fit_logistic_and_get_pqt(df_merge, method=weight_method)

    out_target_path = str(out_target) if write_target_noisy else None
    out_decoy_path = str(out_decoy) if save_calibrated else None

    print(f"[calibration] GAM fit ({weight_method}, tau={tau}, n_reps={n_reps})")
    apply_gam_calibration(
        df_out,
        tau=tau,
        n_splines=12,
        lam=1.0,
        num_points=1000,
        tol=tol,
        max_iter=max_iter,
        noise_lambda=noise_lambda,
        seed=seed,
        n_reps=n_reps,
        out_decoy_path=out_decoy_path,
        out_target_path=out_target_path,
        decoy_suffix=decoy_suffix,
        plot_qids=[],
        bad_q_min=0.0,
        bad_q_max=0.05,
        bad_use_strict=True,
    )

    if out_decoy_path:
        print(f"[calibration] saved calibrated decoy -> {out_decoy}")
    if out_target_path:
        print(f"[calibration] saved target_noisy -> {out_target}")


def run_fdr_and_plot(
    *,
    data_dir: Path,
    curve_archive_dir: Path,
    plot_dir: Path,
    search_method: str,
    decoy_method: str,
    query_name: str,
    target_name: str,
    weight_method: str,
    tau: float,
    save_full_curves: bool = False,
    force_load_all: bool = False,
    n_jobs: int = 1,
) -> tuple[Path, Path, Path, Path, Path | None]:
    if decoy_method not in METHOD2TYPE:
        raise ValueError(f"Unknown decoy method '{decoy_method}'.")

    path_real = path_target_noisy(data_dir, search_method, query_name, decoy_method, target_name)
    path_decoy = path_calibrated_decoy(
        data_dir, search_method, query_name, decoy_method, weight_method, target_name
    )

    for p in (path_real, path_decoy):
        if not p.exists():
            raise FileNotFoundError(f"Missing calibrated input: {p}")

    decoy_suffix = f"_{METHOD2TYPE[decoy_method]}"
    print(f"[fdr] pair table: {path_real.name} + {path_decoy.name}")
    if should_stream_pair_table(path_real, path_decoy, force_load_all=force_load_all):
        print(
            f"[fdr] streaming by qid "
            f"(real={file_size_gb(path_real):.1f} GB, decoy={file_size_gb(path_decoy):.1f} GB, "
            f"n_jobs={n_jobs})"
        )
        df_curve_all, _ = estimate_fdr_curves_streaming(
            path_real,
            path_decoy,
            decoy_suffix,
            desc="  -> Computing TDA curves",
            n_jobs=n_jobs,
        )
    else:
        df_merge = make_pair_table(str(path_real), str(path_decoy), decoy_suffix)
        print(f"[fdr] merged pairs: {len(df_merge)}")
        df_curve_all, _ = estimate_fdr_curves_all_queries(df_merge)
        del df_merge
        gc.collect()

    df_summary = summarize_fdr_curves_at_q_levels(df_curve_all)

    full_archive_path: Path | None = None
    if save_full_curves:
        stem = _stem(search_method, query_name, decoy_method, weight_method, target_name)
        full_archive_path = curve_archive_dir / f"{stem}.txt"
        print(f"[fdr] saving full curve archive -> {full_archive_path}")
        save_dataframe_tsv(df_curve_all, full_archive_path)
        print(f"[fdr] full archive size: {full_archive_path.stat().st_size / 1e9:.2f} GB")

    del df_curve_all
    gc.collect()

    summary_stem = _plot_summary_stem(
        search_method, query_name, decoy_method, weight_method, target_name
    )
    summary_path = curve_archive_dir / f"{summary_stem}.txt"
    print(f"[fdr] saving plot summary -> {summary_path}")
    save_dataframe_tsv(df_summary, summary_path)
    print(f"[fdr] plot summary size: {summary_path.stat().st_size / 1e6:.1f} MB")

    plot_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{search_method}_{query_name}_{decoy_method}_{weight_method}"
    out_fdr = plot_dir / f"{tag}_FDR.pdf"
    out_power = plot_dir / f"{tag}_Power.pdf"
    out_valid = plot_dir / f"{tag}_ValidCount.pdf"

    title_suffix = f"({search_method} - {decoy_method})"
    fdr_boxplot, power_plot, valid_plot = plot_q_vs_real_fdr_and_discoveries_from_summary(
        df_summary,
        out_png_fdr=str(out_fdr),
        out_png_power=str(out_power),
        out_png_valid=str(out_valid),
        title_suffix=title_suffix,
        weight_method=weight_method,
        tau=tau,
    )

    del df_summary
    gc.collect()

    return Path(fdr_boxplot), Path(power_plot), Path(valid_plot), summary_path, full_archive_path


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.curve_archive_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Calibration + FDR pipeline")
    print(f"  data dir     : {args.data_dir}")
    print(f"  curve archive: {args.curve_archive_dir}")
    print(f"  plot dir     : {args.plot_dir}")
    print(f"  search       : {args.search_methods}")
    print(f"  decoy        : {args.decoy_methods}")
    print(f"  keep interm. : {args.keep_intermediates}")
    print("=" * 60)

    summary_rows = []

    for search_method in args.search_methods:
        for decoy_method in args.decoy_methods:
            combo = f"{search_method} / {decoy_method}"
            print(f"\n{'=' * 60}\nProcessing {combo}\n{'=' * 60}")

            try:
                if not args.skip_calibration:
                    target_path = path_target_noisy(
                        args.data_dir, search_method, args.query_name, decoy_method, args.target_name
                    )
                    write_target = args.force_target_noisy or not target_path.exists()

                    run_calibration(
                        data_dir=args.data_dir,
                        search_method=search_method,
                        decoy_method=decoy_method,
                        query_name=args.query_name,
                        target_name=args.target_name,
                        weight_method=args.weight_method,
                        tau=args.tau,
                        tol=args.tol,
                        max_iter=args.max_iter,
                        noise_lambda=args.noise_lambda,
                        n_reps=args.n_reps,
                        seed=args.seed,
                        write_target_noisy=write_target,
                        save_calibrated=args.save_calibrated,
                    )

                if args.skip_fdr:
                    print(f"[skip] FDR step skipped for {combo}")
                    continue

                fdr_plot, power_plot, valid_plot, plot_summary, full_archive = run_fdr_and_plot(
                    data_dir=args.data_dir,
                    curve_archive_dir=args.curve_archive_dir,
                    plot_dir=args.plot_dir,
                    search_method=search_method,
                    decoy_method=decoy_method,
                    query_name=args.query_name,
                    target_name=args.target_name,
                    weight_method=args.weight_method,
                    tau=args.tau,
                    save_full_curves=args.save_full_curves,
                    force_load_all=args.force_load_all,
                    n_jobs=args.n_jobs,
                )
                summary_rows.append(
                    {
                        "search_method": search_method,
                        "decoy_method": decoy_method,
                        "fdr_boxplot": str(fdr_plot),
                        "power_plot": str(power_plot),
                        "valid_count_plot": str(valid_plot),
                        "plot_summary": str(plot_summary),
                        "curve_archive": str(full_archive) if full_archive else "",
                        "status": "ok",
                    }
                )
                print(f"[done] {combo}")
                print(f"       FDR boxplot : {fdr_plot}")
                print(f"       Power plot  : {power_plot}")
                print(f"       Valid plot  : {valid_plot}")
                print(f"       Plot summary: {plot_summary}")
                if full_archive:
                    print(f"       Full archive: {full_archive}")

                if not args.keep_intermediates:
                    cleanup_intermediate_scores(
                        data_dir=args.data_dir,
                        search_method=search_method,
                        query_name=args.query_name,
                        decoy_method=decoy_method,
                        target_name=args.target_name,
                        weight_method=args.weight_method,
                    )

            except Exception as exc:
                print(f"[ERROR] {combo}: {exc}")
                summary_rows.append(
                    {
                        "search_method": search_method,
                        "decoy_method": decoy_method,
                        "fdr_boxplot": "",
                        "power_plot": "",
                        "valid_count_plot": "",
                        "plot_summary": "",
                        "curve_archive": "",
                        "status": f"error: {exc}",
                    }
                )

    summary_path = args.plot_dir / "pipeline_summary.tsv"
    pd.DataFrame(summary_rows).to_csv(summary_path, sep="\t", index=False)
    print(f"\n[OK] Pipeline summary -> {summary_path}")


if __name__ == "__main__":
    main()
