import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "results/ColabFold/ur50_db/tmvec/iter1"
OUT = ROOT / "data/plot_data/ColabFold/ur50_db/metrics_tmvec.tsv"

fold = pd.read_csv(BASE / "metrics.tsv", sep="\t")
build = pd.read_csv(BASE / "msa_metrics.tsv", sep="\t")

key = ["query_id", "tag"]
# take real msa_seconds + meff from the build table
bcols = key + ["msa_seconds"] + (["meff"] if "meff" in build.columns else [])
b = build[bcols].rename(columns={"msa_seconds": "msa_seconds_build"})
m = fold.merge(b, on=key, how="left")
m["msa_seconds"] = m["msa_seconds_build"].fillna(m["msa_seconds"])
m = m.drop(columns=["msa_seconds_build"])
m["total_seconds"] = m["msa_seconds"].fillna(0) + m["predict_seconds"].fillna(0)

OUT.parent.mkdir(parents=True, exist_ok=True)
m.to_csv(OUT, sep="\t", index=False)
# also refresh the canonical results copy
m.to_csv(BASE / "metrics.tsv", sep="\t", index=False)
print(f"[done] {len(m)} rows -> {OUT}")
print("cols:", list(m.columns))
for tag in ["base_alldb_iter1", "base_alldb_iter2", "base_top400k_iter2", "efdr_0p50", "efdr_0p90"]:
    s = m[m.tag == tag]
    if len(s):
        mv = f" meff={s.meff.median():.0f}" if "meff" in s else ""
        print(f"  {tag:18} plddt={s.plddt.median():.1f} tm={s.tm_score.median():.3f} "
              f"rmsd={s.rmsd.median():.2f} msa_s={s.msa_seconds.median():.0f} "
              f"total_s={s.total_seconds.median():.0f} n_hits={s.n_hits.median():.0f}{mv}")
