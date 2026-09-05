import argparse
import importlib.util
from itertools import product
from pathlib import Path

ABLATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ABLATION_DIR.parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PLMCALIPER_PY = PROJECT_ROOT / "src" / "PLMCaliper" / "PLMCaliper.py"

# Decoy id suffix per generator; the calibration strips it to pair decoy with real.
DECOY_SUFFIX = {
    "shuf": "_shuf", "rev": "_rev", "dplm": "_dplm", "mkv2": "_mkv2",
    "extended_shuf": "_shuf", "extended_mkv2": "_mkv",
}

# The published ablation ran at tau = 0.25 (the figures are named ..._tau0.25.pdf).
DEFAULTS = dict(tau=0.25, tol=0.2, max_iter=2, noise_lambda=0.02, n_reps=1,
                seed=43, num_points=1000, weight_method="AdaptiveBell")
Q_LEVELS = [round(0.1 * i, 1) for i in range(1, 10)]


def load_plmcaliper():
    """Import the released single-file tool, so this arm and it cannot diverge."""
    if not PLMCALIPER_PY.exists():
        raise SystemExit(f"[ablation] missing {PLMCALIPER_PY}")
    spec = importlib.util.spec_from_file_location("PLMCaliper", PLMCALIPER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def combo_tag(search_method, query_name, decoy_method, weight_method, target_name):
    return f"{search_method}_{query_name}_{decoy_method}_{weight_method}_{target_name}"


def raw_paths(search_method, decoy_method, query_name, target_name, data_dir):
    real = data_dir / f"result_{search_method}_{query_name}_{target_name}.txt"
    stem = f"result_{search_method}_{query_name}_{decoy_method}_{target_name}.txt"
    for cand in (data_dir / stem,
                 data_dir / "_hf_decoy" / "raw" / "scores" / "astral_main" / stem,
                 data_dir / "search_data" / stem):
        if cand.exists():
            return real, cand
    return real, data_dir / stem


def attach_labels(target_noisy: Path, raw_real: Path) -> None:
    import pandas as pd

    noisy = pd.read_csv(target_noisy, sep="\t")
    if "homo_type" in noisy.columns:
        return
    labels = pd.read_csv(raw_real, sep="\t", usecols=["qid", "tid", "homo_type"],
                         dtype={"qid": str, "tid": str})
    noisy["qid"] = noisy["qid"].astype(str)
    noisy["tid"] = noisy["tid"].astype(str)
    merged = noisy.merge(labels, on=["qid", "tid"], how="left")
    if len(merged) != len(noisy):
        raise SystemExit(f"[ablation] label join changed the row count "
                         f"({len(noisy)} -> {len(merged)}); duplicate (qid, tid)?")
    n_missing = int(merged["homo_type"].isna().sum())
    if n_missing:
        print(f"  [warn] {n_missing} pairs had no label; marked unannotated (0)")
        merged["homo_type"] = merged["homo_type"].fillna(0)
    merged["homo_type"] = merged["homo_type"].astype(int)
    merged.to_csv(target_noisy, sep="\t", index=False)


def calibrate_one(pc, real, decoy, out_dir, *, decoy_suffix, q_levels, **kw):
    out_dir.mkdir(parents=True, exist_ok=True)
    pc.calibrate(
        str(real), str(decoy), str(out_dir / "cutoffs.tsv"),
        decoy_suffix=decoy_suffix, q_levels=q_levels,
        weight_method=kw["weight_method"], tau=kw["tau"], tol=kw["tol"],
        max_iter=kw["max_iter"], noise_lambda=kw["noise_lambda"],
        n_reps=kw["n_reps"], seed=kw["seed"], num_points=kw["num_points"],
        keep_intermediates=str(out_dir),
    )
    target_noisy = out_dir / "target_noisy.tsv"
    attach_labels(target_noisy, Path(real))
    return target_noisy, out_dir / "calibrated_decoy.tsv"


def parse_args():
    p = argparse.ArgumentParser(
        description="Ablation step 1: calibrate each decoy generator with PLM-Caliper.")
    p.add_argument("--search-methods", nargs="+",
                   default=["plm", "tmvec", "dhr_postprocess", "blastp_postprocessed"])
    p.add_argument("--decoy-methods", nargs="+",
                   default=["shuf", "rev", "dplm", "mkv2"])
    p.add_argument("--query-name", default="astral")
    p.add_argument("--target-name", default="astral")
    p.add_argument("--weight-method", default=DEFAULTS["weight_method"])
    p.add_argument("--tau", type=float, default=DEFAULTS["tau"])
    p.add_argument("--tol", type=float, default=DEFAULTS["tol"])
    p.add_argument("--max-iter", type=int, default=DEFAULTS["max_iter"])
    p.add_argument("--noise-lambda", type=float, default=DEFAULTS["noise_lambda"])
    p.add_argument("--n-reps", type=int, default=DEFAULTS["n_reps"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--num-points", type=int, default=DEFAULTS["num_points"],
                   help="quantile anchors per query; 1000 is what the paper used")
    p.add_argument("--q-levels", nargs="+", type=float, default=Q_LEVELS)
    p.add_argument("--real", type=Path, default=None, help="one explicit real table")
    p.add_argument("--decoy", type=Path, default=None, help="one explicit decoy table")
    p.add_argument("--decoy-suffix", default=None, help="override the generator's suffix")
    p.add_argument("--tag", default=None, help="output subfolder for an explicit pair")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path,
                   default=PROJECT_ROOT / "results" / "ablation")
    return p.parse_args()


def main():
    a = parse_args()
    pc = load_plmcaliper()
    kw = dict(weight_method=a.weight_method, tau=a.tau, tol=a.tol,
              max_iter=a.max_iter, noise_lambda=a.noise_lambda,
              n_reps=a.n_reps, seed=a.seed, num_points=a.num_points)

    if a.real and a.decoy:
        jobs = [(a.tag or "explicit", a.real, a.decoy,
                 a.decoy_suffix or DECOY_SUFFIX.get(a.tag, "_mkv"))]
    else:
        jobs = []
        for m, d in product(a.search_methods, a.decoy_methods):
            real, decoy = raw_paths(m, d, a.query_name, a.target_name, a.data_dir)
            tag = combo_tag(m, a.query_name, d, a.weight_method, a.target_name)
            jobs.append((tag, real, decoy, a.decoy_suffix or DECOY_SUFFIX.get(d, f"_{d}")))

    print(f"[ablation] {len(jobs)} combination(s), tau={a.tau}, "
          f"weights={a.weight_method}, num_points={a.num_points}")
    ok = 0
    for tag, real, decoy, suffix in jobs:
        missing = [p for p in (real, decoy) if not Path(p).exists()]
        if missing:
            print(f"[skip] {tag}: missing {', '.join(p.name for p in missing)}")
            continue
        print(f"\n=== {tag}  (decoy suffix {suffix}) ===")
        t, c = calibrate_one(pc, real, decoy, a.out_dir / tag,
                             decoy_suffix=suffix, q_levels=sorted(a.q_levels), **kw)
        print(f"[OK] {t.name} + {c.name} -> {a.out_dir / tag}")
        ok += 1

    print(f"\n[DONE] calibrated {ok}/{len(jobs)} -> {a.out_dir}")
    if ok:
        print("       Next: python fdr.py --out-dir <same out-dir>")


if __name__ == "__main__":
    main()
