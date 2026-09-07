import argparse
import importlib.util
import shutil
from pathlib import Path

BAGEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BAGEL_DIR.parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PLMCALIPER_PY = PROJECT_ROOT / "src" / "PLMCaliper" / "PLMCaliper.py"
ABLATION_CALIBRATE_PY = PROJECT_ROOT / "src" / "analysis" / "ablation" / "calibrate.py"

# bagel_main.py's own constants; the installed filenames have to match them
QUERY_NAME, TARGET_NAME, WEIGHT_METHOD = "putative", "class_all", "AdaptiveBell"
DEFAULTS = dict(tau=0.25, tol=0.2, max_iter=2, noise_lambda=0.02, n_reps=1,
                seed=43, num_points=1000)


def _load(path: Path, name: str):
    """Import a single file by path, so this arm and the released tool cannot diverge."""
    if not path.exists():
        raise SystemExit(f"[bagel-calib] missing {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def raw_paths(search_method: str, decoy_method: str, data_dir: Path):
    """The two retrieval score tables, as run_search.sh (plus any postprocess) leaves them."""
    return (data_dir / f"result_{search_method}_{QUERY_NAME}_{TARGET_NAME}.txt",
            data_dir / f"result_{search_method}_{QUERY_NAME}_{decoy_method}_{TARGET_NAME}.txt")


def installed_paths(search_method: str, decoy_method: str, weight_method: str, data_dir: Path):
    """The two names bagel_main.py discover builds internally -- keep them in sync."""
    return (data_dir / f"result_{search_method}_{QUERY_NAME}_target_noisy_{TARGET_NAME}.txt",
            data_dir / f"result_{search_method}_{QUERY_NAME}_{decoy_method}"
                       f"_calibrated_gam_{weight_method}_{TARGET_NAME}.txt")


def parse_args():
    p = argparse.ArgumentParser(
        description="BAGEL step 0: calibrate one (search method, decoy) pair with PLM-Caliper.")
    p.add_argument("--search-method", required=True,
                   help="table prefix, e.g. plm, tmvec, dhr_postprocess, blastp_postprocessed")
    p.add_argument("--decoy-method", default="extended_mkv2")
    p.add_argument("--decoy-suffix", default="_mkv")
    p.add_argument("--target-fdr", type=float, default=0.20,
                   help="only sets the cutoff table; discover runs its own q scan")
    p.add_argument("--weight-method", default=WEIGHT_METHOD)
    p.add_argument("--tau", type=float, default=DEFAULTS["tau"])
    p.add_argument("--tol", type=float, default=DEFAULTS["tol"])
    p.add_argument("--max-iter", type=int, default=DEFAULTS["max_iter"])
    p.add_argument("--noise-lambda", type=float, default=DEFAULTS["noise_lambda"])
    p.add_argument("--n-reps", type=int, default=DEFAULTS["n_reps"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--num-points", type=int, default=DEFAULTS["num_points"])
    p.add_argument("--data-dir", type=Path, default=DATA_DIR,
                   help="where the retrieval score tables are (shared, default: data/)")
    p.add_argument("--out-data-dir", type=Path, default=None,
                   help="where the calibrated tables go (default: same as --data-dir)")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="where the raw calibration intermediates go "
                        "(default: <repo>/results/bagel/<method>_<decoy>)")
    p.add_argument("--force", action="store_true",
                   help="recalibrate even when the installed tables already exist")
    return p.parse_args()


def main():
    a = parse_args()
    out_data_dir = a.out_data_dir or a.data_dir
    out_data_dir.mkdir(parents=True, exist_ok=True)
    raw_real, raw_decoy = raw_paths(a.search_method, a.decoy_method, a.data_dir)
    out_real, out_decoy = installed_paths(a.search_method, a.decoy_method,
                                          a.weight_method, out_data_dir)

    if out_real.exists() and out_decoy.exists() and not a.force:
        print(f"[bagel-calib] {a.search_method}/{a.decoy_method}: already installed, "
              f"skipping (use --force to recalibrate)")
        print(f"              {out_real.name}\n              {out_decoy.name}")
        return

    missing = [p for p in (raw_real, raw_decoy) if not p.exists()]
    if missing:
        raise SystemExit(
            f"[bagel-calib] {a.search_method}/{a.decoy_method}: missing "
            + ", ".join(p.name for p in missing)
            + "\n              Run the retrieval step first (and dhr-postprocess / "
              "blastp-densify for those two methods).")

    work_dir = a.work_dir or (PROJECT_ROOT / "results" / "bagel"
                              / f"{a.search_method}_{a.decoy_method}")
    work_dir.mkdir(parents=True, exist_ok=True)

    pc = _load(PLMCALIPER_PY, "PLMCaliper")
    abl = _load(ABLATION_CALIBRATE_PY, "ablation_calibrate")

    print(f"[bagel-calib] === {a.search_method} / {a.decoy_method} "
          f"(decoy suffix {a.decoy_suffix}) ===")
    pc.calibrate(
        str(raw_real), str(raw_decoy), str(work_dir / "cutoffs.tsv"),
        decoy_suffix=a.decoy_suffix, q_levels=[a.target_fdr],
        weight_method=a.weight_method, tau=a.tau, tol=a.tol,
        max_iter=a.max_iter, noise_lambda=a.noise_lambda, n_reps=a.n_reps,
        seed=a.seed, num_points=a.num_points, keep_intermediates=str(work_dir),
    )

    target_noisy = work_dir / "target_noisy.tsv"
    # the calibration writes qid/tid/score/rep_id; discover's pair table also
    # needs homo_type, so join it back from the raw real table
    abl.attach_labels(target_noisy, raw_real)

    shutil.copyfile(target_noisy, out_real)
    shutil.copyfile(work_dir / "calibrated_decoy.tsv", out_decoy)
    print(f"[OK] {out_real.name}\n[OK] {out_decoy.name}")
    print(f"     intermediates in {work_dir}")


if __name__ == "__main__":
    main()
