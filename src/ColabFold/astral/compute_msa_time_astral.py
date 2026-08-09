import subprocess
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ASTRAL_FASTA = ROOT / "data/astral.fa"
QJACKHMMER = ROOT / "libs/Dense-Homolog-Retrieval-main/bin/qjackhmmer"
JH = dict(cpu=8, F1="0.0005", F2="5e-05", F3="5e-07")
VANILLA_INCE, FDR_INCE, JACK_ITERS = "1e-3", "1e9", 1
EFDR = [round(0.1 * i, 1) for i in range(1, 10)]

METHODS = {"DHR": "dhr_postprocess", "PLMsearch": "plm", "TMvec": "tmvec"}
RUNS = {k: ROOT / f"results/ColabFold/astral_db/{v}/iter1/parallel/runs" for k, v in METHODS.items()}
OUT = ROOT / "results/ColabFold/astral_db/retime_msa.tsv"


def efdr_tag(q):
    return f"efdr_{q:.2f}".replace(".", "p")


def a3m_seqset(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return set()
    seqs, name, buf = {}, None, []
    for ln in path.read_text().splitlines():
        if ln.startswith(">"):
            if name is not None:
                seqs[name] = "".join(buf)
            name, buf = ln[1:], []
        else:
            buf.append(ln.strip())
    if name is not None:
        seqs[name] = "".join(buf)
    return set(seqs.items())


def run_qjackhmmer(qfa: Path, target: Path, out_a3m: Path, incE: str):
    cmd = [str(QJACKHMMER), "-B", str(out_a3m), "--noali",
           "--incE", incE, "--incdomE", incE, "-E", incE, "--domE", incE,
           "--cpu", str(JH["cpu"]), "-N", str(JACK_ITERS), "--F1", JH["F1"],
           "--F2", JH["F2"], "--F3", JH["F3"], "-o", str(out_a3m.with_suffix(".jh.log")),
           str(qfa), str(target)]
    subprocess.run(cmd, check=True)


def retime_one(method: str, qid: str, tag: str, q):
    d = RUNS[method] / qid / tag
    qfa, old_a3m = d / "query.fa", d / "msa.a3m"
    if not qfa.exists():
        return None
    if tag == "vanilla":
        target, incE = ASTRAL_FASTA, VANILLA_INCE
    else:
        hits = d / "hits.fa"
        if not hits.exists():            # query-only (no candidate hits) -> no MSA build
            return dict(method=method, query_id=qid, tag=tag, efdr_limit=q,
                        msa_seconds_new=0.0, n_old=len(a3m_seqset(old_a3m)), n_new=0,
                        identical=(a3m_seqset(old_a3m) == set()))
        target, incE = hits, FDR_INCE
    new_a3m = d / "msa_retime.a3m"
    t0 = time.perf_counter()
    run_qjackhmmer(qfa, target, new_a3m, incE)
    dt = time.perf_counter() - t0
    old_set, new_set = a3m_seqset(old_a3m), a3m_seqset(new_a3m)
    new_a3m.unlink(missing_ok=True)
    new_a3m.with_suffix(".jh.log").unlink(missing_ok=True)
    return dict(method=method, query_id=qid, tag=tag, efdr_limit=q,
                msa_seconds_new=round(dt, 3), n_old=len(old_set), n_new=len(new_set),
                identical=(old_set == new_set))


def main():
    qids = sorted(p.name for p in (RUNS["DHR"]).iterdir() if p.is_dir())
    conditions = [(efdr_tag(q), q) for q in EFDR] + [("vanilla", None)]
    rows, t_start = [], time.time()
    for i, qid in enumerate(qids, 1):
        for tag, q in conditions:
            for method in METHODS:                      # 3 methods back-to-back (same load)
                r = retime_one(method, qid, tag, q)
                if r:
                    rows.append(r)
        if i % 10 == 0 or i == len(qids):
            ident = sum(r["identical"] for r in rows)
            print(f"[{i}/{len(qids)} queries] rows={len(rows)} identical={ident}/{len(rows)} "
                  f"elapsed={time.time()-t_start:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, sep="\t", index=False)
    print(f"\n[written] {OUT}  ({len(df)} rows)")
    print(f"[identity] {df['identical'].sum()}/{len(df)} rebuilt MSAs match the original")
    bad = df[~df["identical"]]
    if len(bad):
        print(f"[WARN] {len(bad)} mismatches (showing up to 15):")
        print(bad.head(15).to_string(index=False))
    print("\n[old vs new median msa_seconds per method, FDR conditions only]")
    for m in METHODS:
        sub = df[(df["method"] == m) & df["efdr_limit"].notna()]
        print(f"  {m:10s} new median={sub['msa_seconds_new'].median():.3f}  "
              f"new mean={sub['msa_seconds_new'].mean():.3f}")


if __name__ == "__main__":
    main()
