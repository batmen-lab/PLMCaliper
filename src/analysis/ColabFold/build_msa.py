import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
LIBS = PROJECT_ROOT / "libs"
QJACKHMMER = LIBS / "Dense-Homolog-Retrieval-main" / "bin" / "qjackhmmer"

DATASETS = ("astral", "ur50")
RESULTS_DIR = {"astral": PROJECT_ROOT / "results" / "ColabFold" / "astral_db",
               "ur50": PROJECT_ROOT / "results" / "ur50_exp"}

QUERY_SOURCE = {"astral": ("fasta", DATA_DIR / "astral.fa"),
                "ur50": ("tsv", DATA_DIR / "astral4f_query.tsv")}
DB_FASTA = {"astral": DATA_DIR / "astral.fa",
            "ur50": DATA_DIR / "uniref50" / "uniref50.fasta"}
DB_TSV = {"astral": None, "ur50": DATA_DIR / "uniref50" / "uniref50.tsv"}
CANDIDATE_SOURCE = {"astral": "query_fasta", "ur50": "db_tsv"}
TOP400K_DIR = DATA_DIR / "ur50_top400k"          # <method>/<qid>.fa, prebuilt

# baseline arms: (tag, kind, jackhmmer iterations). step 6 reads them back off the
# metrics table, so this list is the only place they are declared.
BASELINES = {"astral": [("vanilla", "alldb", 1)],
             "ur50": [("base_alldb_iter1", "alldb", 1),
                      ("base_alldb_iter2", "alldb", 2),
                      ("base_top400k_iter2", "top400k", 2)]}

EFDR_THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 10)]   # 0.1 .. 0.9
# DHR-repo JackHMMER prefilters (Methods)
JACKHMMER = dict(cpu=8, F1="0.0005", F2="5e-05", F3="5e-07")
BASE_INCE = "1e-3"    # baselines: DHR default inclusion
FDR_INCE = "1e9"      # FDR arm: no E-value filter, so the selection is ours alone
MEFF_SEQID = 0.8


def efdr_tag(q: float) -> str:
    return f"efdr_{q:.2f}".replace(".", "p")


from calibration import hits_path  # noqa: E402



# --------------------------------------------------------------------------
# helpers: sequence and MSA IO, JackHMMER, Meff
# --------------------------------------------------------------------------
def write_fa(path: Path, recs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f">{i}\n{s}\n" for i, s in recs))


def read_fasta(path: Path) -> dict:
    seqs, name, buf = {}, None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                seqs[name] = "".join(buf)
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def read_seq_tsv(path: Path) -> dict:
    return {l.split("\t")[0]: l.rstrip("\n").split("\t")[1]
            for l in open(path) if l.strip()}


def a3m_depth(a3m: Path) -> int:
    if not Path(a3m).exists():
        return 0
    return sum(1 for l in Path(a3m).read_text().splitlines() if l.startswith(">"))


def a3m_matrix(path: Path):
    """a3m -> uint8 matrix of match-state columns, or None if unusable."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    seqs, buf, in_seq = [], [], False
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if in_seq:
                seqs.append("".join(buf))
            buf, in_seq = [], True
        else:
            buf.append("".join(c for c in line.strip() if c == "-" or c.isupper()))
    if in_seq:
        seqs.append("".join(buf))
    seqs = [s for s in seqs if s]
    if not seqs:
        return None
    length = len(seqs[0])
    seqs = [s for s in seqs if len(s) == length]
    if not seqs:
        return None
    return np.frombuffer("".join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), length)


def meff(path: Path, chunk: int = 2048) -> float:
    """Effective sequence count of an MSA at MEFF_SEQID identity, computed exactly."""
    M = a3m_matrix(path)
    if M is None:
        return 0.0
    n, length = M.shape
    if n == 1:
        return 1.0
    thr = MEFF_SEQID * length
    total = 0.0
    for start in range(0, n, chunk):
        block = M[start:start + chunk]
        ident = np.zeros((block.shape[0], n), dtype=np.float32)
        for symbol in np.unique(M):
            ident += (block == symbol).astype(np.float32) @ (M == symbol).astype(np.float32).T
        total += float((1.0 / (ident >= thr).sum(axis=1)).sum())
    return total


def run_qjackhmmer(query_fa: Path, target_fa: Path, out_a3m: Path, inc_e: str,
                   iters: int) -> int:
    """Build an a3m; returns its depth.  inc_e also sets -E/--domE so reporting and
    inclusion agree."""
    cmd = [str(QJACKHMMER), "-B", str(out_a3m), "--noali",
           "--incE", inc_e, "--incdomE", inc_e, "-E", inc_e, "--domE", inc_e,
           "--cpu", str(JACKHMMER["cpu"]), "-N", str(iters),
           "--F1", JACKHMMER["F1"], "--F2", JACKHMMER["F2"], "--F3", JACKHMMER["F3"],
           "-o", str(out_a3m.with_suffix(".jh.log")), str(query_fa), str(target_fa)]
    subprocess.run(cmd, check=True)
    if not out_a3m.exists() or out_a3m.stat().st_size == 0:
        return 0
    return a3m_depth(out_a3m)


def write_metrics(rows: list[dict], out: Path, columns: list[str]) -> None:
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    ordered = columns + [c for c in df.columns if c not in columns]
    out.parent.mkdir(parents=True, exist_ok=True)
    df[ordered].to_csv(out, sep="\t", index=False)


def read_query_ids(path: Path) -> list[str]:
    ids, seen = [], set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        qid = line.split("\t")[0].strip()
        if qid and qid not in seen:
            ids.append(qid)
            seen.add(qid)
    return ids


    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return set()
    return set(read_fasta(path).items())


def a3m_seqset(path: Path) -> set:
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return set()
    return set(read_fasta(path).items())


def retime_one(ds, method: str, runs_dir: Path, qid: str, tag: str, q, inc_e, iters):
    d = runs_dir / qid / tag
    qfa, old_a3m = d / "query.fa", d / "msa.a3m"
    if not qfa.exists():
        return None
    if q is None:
        target = ds.db_fasta
    else:
        target = d / "hits.fa"
        if not target.exists():                    # query-only arm: nothing to rebuild
            return dict(method=method, query_id=qid, tag=tag, efdr_limit=q,
                        msa_seconds_new=0.0, n_old=len(a3m_seqset(old_a3m)), n_new=0,
                        identical=(a3m_seqset(old_a3m) == set()))
    new_a3m = d / "msa_retime.a3m"
    t0 = time.perf_counter()
    run_qjackhmmer(qfa, target, new_a3m, inc_e, iters)
    dt = time.perf_counter() - t0
    old_set, new_set = a3m_seqset(old_a3m), a3m_seqset(new_a3m)
    new_a3m.unlink(missing_ok=True)
    new_a3m.with_suffix(".jh.log").unlink(missing_ok=True)
    return dict(method=method, query_id=qid, tag=tag, efdr_limit=q,
                msa_seconds_new=round(dt, 3), n_old=len(old_set), n_new=len(new_set),
                identical=(old_set == new_set))



MSA_COLUMNS = ["method", "query_id", "tag", "efdr_limit", "qlen", "n_hits",
               "msa_depth", "meff", "msa_seconds", "msa_status"]


def conditions(dataset: str):
    """(tag, kind, efdr_limit, inc_e, jackhmmer_iters) for every arm."""
    fdr = [(efdr_tag(q), "fdr", q, FDR_INCE, 1) for q in EFDR_THRESHOLDS]
    base = [(tag, kind, None, BASE_INCE, iters) for tag, kind, iters in BASELINES[dataset]]
    return fdr + base


def load_query_seqs(dataset: str) -> dict:
    kind, path = QUERY_SOURCE[dataset]
    return read_fasta(path) if kind == "fasta" else read_seq_tsv(path)


def load_candidate_seqs(dataset: str, tids: set[str], prebuilt: Path | None) -> dict:
    """Sequences of every target any arm selected, so hits.fa can be written."""
    if CANDIDATE_SOURCE[dataset] == "query_fasta":
        return read_fasta(DB_FASTA[dataset])
    if prebuilt and Path(prebuilt).exists():
        print(f"[step3] reading prebuilt candidates from {prebuilt}", flush=True)
        src = open(prebuilt)
    else:
        print(f"[step3] scanning {DB_TSV[dataset]} for {len(tids):,} candidates ...", flush=True)
        src = open(DB_TSV[dataset])
    out = {}
    with src as fh:
        for line in fh:
            i = line.find("\t")
            if i > 0 and line[:i] in tids:
                out[line[:i]] = line[i + 1:].rstrip("\n")
                if len(out) == len(tids):
                    break
    return out


def build_one(dataset, method, qid, qseq, hits_by_tag, cand, runs_dir, force):
    nat_rows = []
    for tag, kind, q, inc_e, iters in conditions(dataset):
        d = runs_dir / qid / tag
        d.mkdir(parents=True, exist_ok=True)
        qfa = d / "query.fa"
        write_fa(qfa, [(qid, qseq)])
        a3m = d / "msa.a3m"
        built = a3m.exists() and a3m.stat().st_size > 0
        n_hits, target, status = 0, None, "pending"

        t0 = time.perf_counter()
        try:
            if kind == "fdr":
                recs = [(t, cand[t]) for t in hits_by_tag.get(tag, []) if t in cand]
                n_hits = len(recs)
                if n_hits == 0:
                    status = "query_only"
                else:
                    target = d / "hits.fa"
                    write_fa(target, recs)
            elif kind == "alldb":
                target, n_hits = DB_FASTA[dataset], -1
            elif kind == "top400k":
                target, n_hits = TOP400K_DIR / method / f"{qid}.fa", -2
                if not target.exists():
                    target, status = None, "top400k_missing"
            if target is not None and (force or not built):
                n = run_qjackhmmer(qfa, target, a3m, inc_e, iters)
                status = "jackhmmer_ok" if n > 1 else "thin_msa"
            elif built:
                status = "reused"
        except Exception as exc:                    # keep going; record the failure
            status = f"msa_failed:{exc}"
        msa_s = time.perf_counter() - t0

        nat_rows.append({
            "method": method, "query_id": qid, "tag": tag, "efdr_limit": q,
            "qlen": len(qseq), "n_hits": n_hits, "msa_depth": a3m_depth(a3m),
            "meff": round(meff(a3m), 3), "msa_seconds": round(msa_s, 2),
            "msa_status": status,
        })
        print(f"  [{qid}] {tag}: hits={n_hits} depth={nat_rows[-1]['msa_depth']} "
              f"{status}", flush=True)
    return nat_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 3: turn each step-2 hit set into an MSA with qjackhmmer.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--method", default="dhr_postprocess")
    p.add_argument("--hits", type=Path, default=None, help="default: step 2's table")
    p.add_argument("--query-list", type=Path, default=None)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--cand-seqs", type=Path, default=None,
                   help="prebuilt tid<TAB>seq subset, avoids rescanning the UR50 TSV")
    p.add_argument("--jack-iters", type=int, default=1, help="also names iter<N>/")
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--runs-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    base = RESULTS_DIR[a.dataset] / a.method / f"iter{a.jack_iters}"
    out = a.out or base / "msa.tsv"
    runs_dir = a.runs_dir or base / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    hits = pd.read_csv(a.hits or hits_path(a.dataset, a.method), sep="\t",
                       dtype={"qid": str, "tid": str, "tag": str})
    qseqs = load_query_seqs(a.dataset)
    qids = read_query_ids(a.query_list) if a.query_list else sorted(hits["qid"].unique())
    qids = [q for q in qids if q in qseqs][: a.max_queries]
    cand = load_candidate_seqs(a.dataset, set(hits["tid"]), a.cand_seqs)
    print(f"[step3] {len(qids)} queries x {len(conditions(a.dataset))} conditions", flush=True)

    rows = []
    for qid in qids:
        by_tag = (hits[hits["qid"] == qid].groupby("tag")["tid"].apply(list).to_dict())
        rows += build_one(a.dataset, a.method, qid, qseqs[qid], by_tag, cand,
                          runs_dir, a.force_rebuild)
    write_metrics(rows, out, MSA_COLUMNS)
    print(f"[step3] -> {out}")


if __name__ == "__main__":
    main()
