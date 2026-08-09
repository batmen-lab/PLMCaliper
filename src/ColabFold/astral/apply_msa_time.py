import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RETIME = ROOT / "results/ColabFold/astral_db/retime_msa.tsv"
METHODS = {"DHR": "dhr_postprocess", "PLMsearch": "plm", "TMvec": "tmvec"}


def main():
    rt = pd.read_csv(RETIME, sep="\t")
    if not rt["identical"].all():
        raise SystemExit(f"[abort] {(~rt['identical']).sum()} MSAs differ; not replacing times")

    for disp, mdir in METHODS.items():
        sub = rt[rt["method"] == disp]
        lut = {(r.query_id, r.tag): r.msa_seconds_new for r in sub.itertuples()}
        shards = sorted(glob.glob(str(ROOT / f"results/ColabFold/astral_db/{mdir}/iter1/parallel/out_gpu*.tsv")))
        for f in shards:
            f = Path(f)
            d = pd.read_csv(f, sep="\t")
            backup = f.with_suffix(".tsv.orig")
            if not backup.exists():
                backup.write_text(f.read_text())     # one-time backup of the pristine shard
            new_msa = [lut.get((qid, tag), msa) for qid, tag, msa in
                       zip(d["query_id"], d["tag"], d["msa_seconds"])]
            d["msa_seconds"] = [round(float(x), 3) for x in new_msa]
            d["total_seconds"] = [round(float(np.nansum([m, p])), 2)
                                  for m, p in zip(d["msa_seconds"], d["predict_seconds"])]
            d.to_csv(f, sep="\t", index=False)
        print(f"[{disp}] updated {len(shards)} shards ({len(sub)} retimed rows)")

    print("[DONE] msa_seconds/total_seconds replaced; originals saved as *.tsv.orig")


if __name__ == "__main__":
    main()
