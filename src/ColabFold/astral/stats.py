import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from config import ASTRAL_BATCH_DIR, EFDR_THRESHOLDS, VANILLA_TAG


def efdr_tag(efdr: float) -> str:
    return f"efdr_{efdr:.2f}".replace(".", "p")


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run pipeline_astral.py / pipeline_ur50.py (or parallel_fold.py) first.")
    df = pd.read_csv(path, sep="\t")
    df = df[df["af2_status"] == "ok"].copy()
    return df


def _betacf(a: float, b: float, x: float) -> float:
    max_iter, eps, fpmin = 300, 3e-14, 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def paired_ttest_rel(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = a - b
    n = len(d)
    if n < 2:
        return np.nan, np.nan
    mean_delta = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        return (0.0, 1.0) if mean_delta == 0.0 else (math.copysign(math.inf, mean_delta), 0.0)
    t_stat = mean_delta / (sd / math.sqrt(n))
    df = n - 1
    p_value = _betai(0.5 * df, 0.5, df / (df + t_stat * t_stat))
    return float(t_stat), float(p_value)


def paired_test(
    df: pd.DataFrame, q: float, value_col: str, alpha: float
) -> dict | None:
    cond = df[df["tag"] == efdr_tag(q)][["query_id", value_col]]
    base = df[df["tag"] == VANILLA_TAG][["query_id", value_col]]
    merged = cond.merge(base, on="query_id", suffixes=("_q", "_base")).dropna()
    n = len(merged)
    if n < 2:
        return {"n_pairs": n, "mean_q": np.nan, "mean_base": np.nan,
                "mean_delta": np.nan, "t_stat": np.nan, "p_value": np.nan}
    a = merged[f"{value_col}_q"].to_numpy(float)
    b = merged[f"{value_col}_base"].to_numpy(float)
    t_stat, p_value = paired_ttest_rel(a, b)
    return {
        "n_pairs": n,
        "mean_q": float(a.mean()),
        "mean_base": float(b.mean()),
        "mean_delta": float((a - b).mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def build_summary(
    df: pd.DataFrame, thresholds: list[float], time_col: str, alpha: float
) -> pd.DataFrame:
    rows = []
    for q in thresholds:
        pl = paired_test(df, q, "plddt", alpha)
        tm = paired_test(df, q, time_col, alpha)
        if pl is None or tm is None:
            continue
        # quality not significantly worse than vanilla
        quality_ok = bool(pl["p_value"] > alpha or pl["mean_delta"] >= 0)
        # prediction significantly faster than vanilla
        time_better = bool(tm["p_value"] < alpha and tm["mean_delta"] < 0)
        speedup = (tm["mean_base"] / tm["mean_q"]) if tm["mean_q"] else np.nan
        rows.append({
            "q": q,
            "n_pairs": pl["n_pairs"],
            "mean_plddt_q": round(pl["mean_q"], 3),
            "mean_plddt_vanilla": round(pl["mean_base"], 3),
            "delta_plddt": round(pl["mean_delta"], 3),
            "p_plddt": pl["p_value"],
            f"mean_{time_col}_q": round(tm["mean_q"], 2),
            f"mean_{time_col}_vanilla": round(tm["mean_base"], 2),
            f"delta_{time_col}": round(tm["mean_delta"], 2),
            "speedup": round(speedup, 2) if speedup == speedup else np.nan,
            "p_time": tm["p_value"],
            "quality_ok": quality_ok,
            "time_better": time_better,
            "recommended": quality_ok and time_better,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired t-tests: eFDR q vs vanilla")
    parser.add_argument("--summary", type=Path, default=ASTRAL_BATCH_DIR / "metrics.tsv")
    parser.add_argument("--out", type=Path, default=ASTRAL_BATCH_DIR / "paired_ttest_summary.tsv")
    parser.add_argument("--time-col", type=str, default="total_seconds",
                        choices=["total_seconds", "predict_seconds", "msa_seconds"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--q-levels", nargs="+", type=float, default=EFDR_THRESHOLDS)
    args = parser.parse_args()

    df = load_table(args.summary)
    summary = build_summary(df, sorted(set(args.q_levels)), args.time_col, args.alpha)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, sep="\t", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(summary.to_string(index=False))

    rec = summary[summary["recommended"]]
    print()
    if rec.empty:
        print(f"[RESULT] No eFDR threshold matches vanilla pLDDT (p>{args.alpha}) "
              f"while being significantly faster (p<{args.alpha}).")
    else:
        best = rec.sort_values("q").iloc[0]
        print(f"[RESULT] Recommended empirical threshold: q={best['q']:.1f}  "
              f"(delta_pLDDT={best['delta_plddt']:+.2f}, p_plddt={best['p_plddt']:.3f}; "
              f"speedup={best['speedup']:.2f}x, p_time={best['p_time']:.1e})")
    print(f"[OK] Summary -> {args.out}")


if __name__ == "__main__":
    main()
