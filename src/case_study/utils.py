import csv
import re
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO


def remove_decoy_suffix(qid: str, decoy_suffix: str) -> str:
    if qid.endswith(decoy_suffix):
        return qid[: -len(decoy_suffix)]
    return qid


def file_size_gb(path: str | Path) -> float:
    return Path(path).stat().st_size / (1024**3)


def _rows_to_pair_frame(real_rows: list[dict], decoy_rows: list[dict], decoy_suffix: str) -> pd.DataFrame:
    df_real = pd.DataFrame(real_rows)
    df_decoy = pd.DataFrame(decoy_rows)
    if df_real.empty:
        return pd.DataFrame()

    df_real = df_real[["qid", "tid", "homo_type", "score", "rep_id"]].copy()
    df_real = df_real.rename(columns={"score": "score_real", "homo_type": "homo_type_real"})
    df_real["score_real"] = df_real["score_real"].astype(np.float32)

    df_decoy = df_decoy[["qid", "tid", "score", "rep_id"]].copy()
    df_decoy["qid_orig"] = df_decoy["qid"].apply(lambda x: remove_decoy_suffix(x, decoy_suffix))
    df_decoy = df_decoy.rename(columns={"score": "score_decoy"})
    df_decoy["score_decoy"] = df_decoy["score_decoy"].astype(np.float32)

    df_merge = df_real.merge(
        df_decoy[["qid_orig", "tid", "score_decoy", "rep_id"]],
        left_on=["qid", "tid", "rep_id"],
        right_on=["qid_orig", "tid", "rep_id"],
        how="inner",
    )
    df_merge = df_merge.drop(columns=["qid_orig"], errors="ignore")
    df_merge["rep_id"] = pd.to_numeric(df_merge["rep_id"], downcast="integer")
    return df_merge


def _iter_qid_blocks(path: str | Path) -> Iterator[tuple[str, list[dict]]]:
    current_qid: str | None = None
    block: list[dict] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            qid = row["qid"]
            if current_qid is not None and qid != current_qid:
                yield current_qid, block
                block = []
            current_qid = qid
            block.append(row)
    if block and current_qid is not None:
        yield current_qid, block


def iter_pair_tables_by_qid(
    path_real: str | Path,
    path_decoy: str | Path,
    decoy_suffix: str,
    *,
    qid_allowlist: set[str] | None = None,
) -> Iterator[tuple[str, pd.DataFrame]]:
    real_iter = _iter_qid_blocks(path_real)
    decoy_iter = _iter_qid_blocks(path_decoy)

    for (real_qid, real_rows), (decoy_qid, decoy_rows) in zip(real_iter, decoy_iter, strict=True):
        qid = remove_decoy_suffix(decoy_qid, decoy_suffix)
        if qid != real_qid:
            raise ValueError(f"QID mismatch while streaming: real={real_qid!r} decoy={decoy_qid!r}")
        if qid_allowlist is not None and real_qid not in qid_allowlist:
            continue
        df_merge = _rows_to_pair_frame(real_rows, decoy_rows, decoy_suffix)
        if not df_merge.empty:
            yield real_qid, df_merge


def load_pair_table_for_qids(
    path_real: str | Path,
    path_decoy: str | Path,
    decoy_suffix: str,
    qids: set[str],
) -> pd.DataFrame:
    chunks = [df for qid, df in iter_pair_tables_by_qid(path_real, path_decoy, decoy_suffix, qid_allowlist=qids)]
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def make_pair_table_keep_tid(path_real: str | Path, path_decoy: str | Path, decoy_suffix: str) -> pd.DataFrame:
    df_real = pd.read_csv(path_real, sep="\t")
    df_decoy = pd.read_csv(path_decoy, sep="\t")

    df_real = df_real[["qid", "tid", "homo_type", "score", "rep_id"]].copy()
    df_real = df_real.rename(columns={"score": "score_real", "homo_type": "homo_type_real"})

    df_decoy = df_decoy[["qid", "tid", "score", "rep_id"]].copy()
    df_decoy["qid_orig"] = df_decoy["qid"].apply(lambda x: remove_decoy_suffix(x, decoy_suffix))
    df_decoy = df_decoy.rename(columns={"score": "score_decoy"})

    df_merge = df_real.merge(
        df_decoy[["qid_orig", "tid", "score_decoy", "rep_id"]],
        left_on=["qid", "tid", "rep_id"],
        right_on=["qid_orig", "tid", "rep_id"],
        how="inner",
    )
    df_merge = df_merge.drop(columns=["qid_orig"], errors="ignore")
    df_merge["score_real"] = df_merge["score_real"].astype(np.float32)
    df_merge["score_decoy"] = df_merge["score_decoy"].astype(np.float32)
    df_merge["rep_id"] = pd.to_numeric(df_merge["rep_id"], downcast="integer")
    return df_merge


def load_fasta_dict(fasta_path: str | Path) -> dict[str, str]:
    return {rec.id: str(rec.seq) for rec in SeqIO.parse(str(fasta_path), "fasta")}


def run_pairewise_seq_identity_score(sequence1: str, sequence2: str) -> tuple[float, str | None]:
    from Bio import pairwise2 as pw2

    if not sequence1 or not sequence2:
        return float("nan"), None

    best_sequence_identity = -1.0
    best_align = None
    for aln in pw2.align.globalxx(sequence1, sequence2):
        span = aln[4] - aln[3]
        if span <= 0:
            continue
        sid = aln[2] / span
        if sid > best_sequence_identity:
            best_sequence_identity = sid
            best_align = aln

    if best_align is None:
        return float("nan"), None
    return float(best_sequence_identity), None


def sequence_identity(seq_a: str, seq_b: str) -> float:
    sid, _ = run_pairewise_seq_identity_score(seq_a, seq_b)
    return sid


def select_threshold_at_q(df_curve: pd.DataFrame, q: float) -> float | None:
    valid = df_curve[df_curve["est_fdp"] <= q]
    if valid.empty:
        return None
    return float(valid.loc[valid["threshold"].idxmin(), "threshold"])


def export_structure_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def est_fdp_at_score(df_curve: pd.DataFrame, score: float) -> float:
    if df_curve.empty:
        return float("nan")
    thr = df_curve["threshold"].to_numpy(dtype=np.float64)
    efdp = df_curve["est_fdp"].to_numpy(dtype=np.float64)
    idx = int(np.searchsorted(thr, score, side="right") - 1)
    if idx < 0:
        return float(efdp[0])
    return float(efdp[idx])


def disagreement_case(
    score_real: float,
    est_fdp_at_score: float,
    *,
    score_cutoff: float = 0.5,
    efdp_cutoff: float = 0.5,
    max_score_cutoff: float | None = None,
) -> str | None:
    if np.isnan(est_fdp_at_score):
        return None
    within_high_band = score_real >= score_cutoff
    if max_score_cutoff is not None:
        within_high_band = within_high_band and score_real <= max_score_cutoff
    high_efdp = est_fdp_at_score >= efdp_cutoff
    low_score = score_real < score_cutoff
    low_efdp = est_fdp_at_score < efdp_cutoff
    if within_high_band and high_efdp:
        return "high_score_high_efdp"
    if low_score and low_efdp:
        return "low_score_low_efdp"
    return None


def est_fdp_at_scores(df_curve: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    if df_curve.empty:
        return np.full(len(scores), np.nan, dtype=np.float64)
    thr = df_curve["threshold"].to_numpy(dtype=np.float64)
    efdp = df_curve["est_fdp"].to_numpy(dtype=np.float64)
    idx = np.searchsorted(thr, scores, side="right") - 1
    idx = np.clip(idx, 0, len(efdp) - 1)
    return efdp[idx]


def collect_disagreement_pairs_from_merge(
    df_merge: pd.DataFrame,
    *,
    score_cutoff: float = 0.5,
    efdp_cutoff: float = 0.5,
    max_score_cutoff: float | None = None,
    case_types: set[str] | None = None,
    rep_ids: list[int] | None = None,
    exclude_self: bool = True,
    homo_types: set[int] | None = None,
    max_pairs_per_query: int = 50,
) -> list[dict]:
    rows: list[dict] = []
    rep_filter = set(rep_ids) if rep_ids is not None else None

    for (qid, rep_id), df_q in df_merge.groupby(["qid", "rep_id"], sort=False):
        if rep_filter is not None and int(rep_id) not in rep_filter:
            continue

        df = df_q.copy()
        if exclude_self:
            df = df[df["tid"].astype(str) != str(qid)]
        if homo_types is not None:
            df = df[df["homo_type_real"].isin(homo_types)]
        if df.empty:
            continue

        scores = df["score_real"].to_numpy(dtype=np.float64)
        df_curve = _compute_tda_fdr_per_query(df_q)
        efdp = est_fdp_at_scores(df_curve, scores)

        high_mask = (scores >= score_cutoff) & (efdp >= efdp_cutoff)
        if max_score_cutoff is not None:
            high_mask = high_mask & (scores <= max_score_cutoff)
        low_mask = (scores < score_cutoff) & (efdp < efdp_cutoff)

        cases = [
            ("high_score_high_efdp", high_mask),
            ("low_score_low_efdp", low_mask),
        ]
        if case_types is not None:
            cases = [(ct, m) for ct, m in cases if ct in case_types]

        for case_type, mask in cases:
            if not mask.any():
                continue
            sub = df.loc[mask].copy()
            sub["est_fdp_at_score"] = efdp[mask]
            if case_type == "high_score_high_efdp":
                sub = sub.sort_values(
                    ["score_real", "est_fdp_at_score"],
                    ascending=[False, False],
                )
            else:
                sub = sub.sort_values(
                    ["est_fdp_at_score", "score_real"],
                    ascending=[True, True],
                )
            if max_pairs_per_query > 0:
                sub = sub.head(max_pairs_per_query)

            for _, hit in sub.iterrows():
                rows.append(
                    {
                        "qid": qid,
                        "tid": str(hit["tid"]),
                        "rep_id": int(rep_id),
                        "score_real": float(hit["score_real"]),
                        "score_decoy": float(hit["score_decoy"]),
                        "homo_type_real": int(hit["homo_type_real"]),
                        "est_fdp_at_score": float(hit["est_fdp_at_score"]),
                        "case_type": case_type,
                        "score_cutoff": score_cutoff,
                        "efdp_cutoff": efdp_cutoff,
                        "max_score_cutoff": max_score_cutoff,
                    }
                )
    return rows


PAIR_EXTRACT_COLUMNS = [
    "qid",
    "tid",
    "rep_id",
    "score_real",
    "score_decoy",
    "homo_type_real",
    "seq_identity",
    "T_q",
    "est_fdp_at_Tq",
    "q_level",
    "fdr_pass",
]


class TsvAppendWriter:
    """Append rows to a TSV, writing the header once."""

    def __init__(self, path: Path, columns: list[str]) -> None:
        self.path = path
        self.columns = columns
        self._has_header = self.path.exists() and self.path.stat().st_size > 0

    def write_rows(self, rows: list[dict]) -> None:
        if not rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._has_header else "w"
        with self.path.open(mode, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns, delimiter="\t")
            if not self._has_header:
                writer.writeheader()
                self._has_header = True
            for row in rows:
                writer.writerow({col: row[col] for col in self.columns})


def _rows_for_pairs(
    pairs: pd.DataFrame,
    qid: str,
    rep_id: int,
    q_seq: str,
    seq_lookup: dict[str, str],
    *,
    t_q: float,
    est_fdp_at_t: float,
    q_level: float,
    fdr_pass: int,
) -> list[dict]:
    rows: list[dict] = []
    for _, hit in pairs.iterrows():
        tid = str(hit["tid"])
        t_seq = seq_lookup.get(tid)
        if t_seq is None:
            continue
        rows.append(
            {
                "qid": qid,
                "tid": tid,
                "rep_id": rep_id,
                "score_real": float(hit["score_real"]),
                "score_decoy": float(hit["score_decoy"]),
                "homo_type_real": int(hit["homo_type_real"]),
                "seq_identity": sequence_identity(q_seq, t_seq),
                "T_q": t_q,
                "est_fdp_at_Tq": est_fdp_at_t,
                "q_level": q_level,
                "fdr_pass": fdr_pass,
            }
        )
    return rows


def _extract_rep_pairs(
    df_q: pd.DataFrame,
    qid: str,
    rep_id: int,
    q_seq: str,
    seq_lookup: dict[str, str],
    *,
    q: float,
    max_nonhits_per_query: int,
) -> tuple[list[dict], list[dict], float | None, float | None]:
    df_curve = _compute_tda_fdr_per_query(df_q)
    t_q = select_threshold_at_q(df_curve, q)
    if t_q is None:
        return [], [], None, None

    est_fdp_at_t = float(df_curve.loc[df_curve["threshold"] == t_q, "est_fdp"].iloc[0])
    hits = df_q[df_q["score_real"] >= t_q].sort_values("score_real", ascending=False)
    nonhits = df_q[df_q["score_real"] < t_q].sort_values("score_real", ascending=False)
    if max_nonhits_per_query > 0:
        nonhits = nonhits.head(max_nonhits_per_query)

    hit_rows = _rows_for_pairs(
        hits, qid, rep_id, q_seq, seq_lookup,
        t_q=t_q, est_fdp_at_t=est_fdp_at_t, q_level=q, fdr_pass=1,
    )
    nonhit_rows = _rows_for_pairs(
        nonhits, qid, rep_id, q_seq, seq_lookup,
        t_q=t_q, est_fdp_at_t=est_fdp_at_t, q_level=q, fdr_pass=0,
    )
    return hit_rows, nonhit_rows, t_q, est_fdp_at_t


def extract_pairs_from_merge(
    df_merge: pd.DataFrame,
    seq_lookup: dict[str, str],
    *,
    q: float,
    max_nonhits_per_query: int = 100,
) -> tuple[list[dict], list[dict], list[dict]]:
    hit_rows: list[dict] = []
    nonhit_rows: list[dict] = []
    threshold_rows: list[dict] = []

    for qid, df_qall in df_merge.groupby("qid", sort=False):
        q_seq = seq_lookup.get(str(qid))
        if q_seq is None:
            continue

        rep_ids = sorted(pd.unique(df_qall["rep_id"]))
        for rid in rep_ids:
            rid_int = int(rid)
            df_q = df_qall[df_qall["rep_id"] == rid]
            hits, nonhits, t_q, est_fdp_at_t = _extract_rep_pairs(
                df_q,
                str(qid),
                rid_int,
                q_seq,
                seq_lookup,
                q=q,
                max_nonhits_per_query=max_nonhits_per_query,
            )
            hit_rows.extend(hits)
            nonhit_rows.extend(nonhits)
            if t_q is not None:
                threshold_rows.append(
                    {
                        "qid": qid,
                        "rep_id": rid_int,
                        "T_q": t_q,
                        "est_fdp_at_Tq": est_fdp_at_t,
                        "q_level": q,
                        "n_fdr_hits": len(hits),
                        "n_nonhits_saved": len(nonhits),
                    }
                )
    return hit_rows, nonhit_rows, threshold_rows


def extract_pairs_streaming(
    path_real: Path,
    path_decoy: Path,
    suffix: str,
    seq_lookup: dict[str, str],
    out_dir: Path,
    *,
    q: float,
    max_nonhits_per_query: int,
    max_queries: int | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    hits_writer = TsvAppendWriter(out_dir / "hits_per_rep.tsv", PAIR_EXTRACT_COLUMNS)
    nonhits_writer = TsvAppendWriter(out_dir / "nonhits_per_rep.tsv", PAIR_EXTRACT_COLUMNS)
    threshold_cols = [
        "qid", "rep_id", "T_q", "est_fdp_at_Tq", "q_level",
        "n_fdr_hits", "n_nonhits_saved",
    ]
    thresholds_writer = TsvAppendWriter(out_dir / "rep_thresholds.tsv", threshold_cols)

    n_seen = 0
    n_hit_rows = 0
    n_nonhit_rows = 0
    print("[extract] Scanning queries...", flush=True)
    for _qid, df_merge in iter_pair_tables_by_qid(path_real, path_decoy, suffix):
        if max_queries is not None and n_seen >= max_queries:
            break
        n_seen += 1
        if n_seen % 500 == 0:
            print(
                f"        queries: {n_seen:,}  "
                f"hits: {n_hit_rows:,}  nonhits: {n_nonhit_rows:,}",
                flush=True,
            )

        hit_rows, nonhit_rows, threshold_rows = extract_pairs_from_merge(
            df_merge,
            seq_lookup,
            q=q,
            max_nonhits_per_query=max_nonhits_per_query,
        )
        hits_writer.write_rows(hit_rows)
        nonhits_writer.write_rows(nonhit_rows)
        thresholds_writer.write_rows(threshold_rows)
        n_hit_rows += len(hit_rows)
        n_nonhit_rows += len(nonhit_rows)

    print(f"        total queries: {n_seen:,}")
    print(f"        saved hits: {n_hit_rows:,}  nonhits: {n_nonhit_rows:,}")
    print(f"[OK] Cache -> {out_dir}")


def filter_candidates_from_hits(
    df_hits: pd.DataFrame,
    *,
    max_identity: float,
    q: float | None = None,
) -> pd.DataFrame:
    if df_hits.empty:
        return pd.DataFrame()

    if q is not None and "q_level" in df_hits.columns:
        cached_q = float(df_hits["q_level"].iloc[0])
        if abs(cached_q - q) > 1e-9:
            raise ValueError(
                f"Cache q_level={cached_q} != filter q={q}. Re-run extract with --q {q}."
            )

    df = df_hits[df_hits["seq_identity"] < max_identity].copy()
    rows: list[dict] = []

    for qid, df_qall in df.groupby("qid", sort=False):
        rep_ids = sorted(pd.unique(df_qall["rep_id"]))
        if not rep_ids:
            continue
        n_reps_total = len(rep_ids)

        passing_by_rep = [
            set(df_qall.loc[df_qall["rep_id"] == rid, "tid"].astype(str))
            for rid in rep_ids
        ]
        common_tids = set.intersection(*passing_by_rep) if passing_by_rep else set()

        for tid in common_tids:
            for rid in rep_ids:
                sub = df_qall[(df_qall["rep_id"] == rid) & (df_qall["tid"].astype(str) == tid)]
                if sub.empty:
                    continue
                hit = sub.iloc[0]
                rows.append(
                    {
                        "qid": qid,
                        "tid": tid,
                        "rep_id": int(rid),
                        "n_reps_total": n_reps_total,
                        "score_real": float(hit["score_real"]),
                        "score_decoy": float(hit["score_decoy"]),
                        "homo_type_real": int(hit["homo_type_real"]),
                        "seq_identity": float(hit["seq_identity"]),
                        "T_q": float(hit["T_q"]),
                        "est_fdp_at_Tq": float(hit["est_fdp_at_Tq"]),
                        "q_level": float(hit["q_level"]),
                    }
                )
    return aggregate_candidates(rows)


def _passing_tids_for_rep(
    df_q: pd.DataFrame,
    q_seq: str,
    seq_lookup: dict[str, str],
    *,
    q: float,
    max_identity: float,
) -> tuple[set[str], float | None, float | None, dict[str, dict]]:
    df_curve = _compute_tda_fdr_per_query(df_q)
    t_q = select_threshold_at_q(df_curve, q)
    if t_q is None:
        return set(), None, None, {}

    est_fdp_at_t = float(df_curve.loc[df_curve["threshold"] == t_q, "est_fdp"].iloc[0])
    passing = df_q[df_q["score_real"] >= t_q].sort_values("score_real", ascending=False)

    passing_tids: set[str] = set()
    tid_meta: dict[str, dict] = {}
    for _, hit in passing.iterrows():
        tid = str(hit["tid"])
        t_seq = seq_lookup.get(tid)
        if t_seq is None:
            continue
        sid = sequence_identity(q_seq, t_seq)
        if sid >= max_identity:
            continue
        passing_tids.add(tid)
        tid_meta[tid] = {
            "score_real": float(hit["score_real"]),
            "score_decoy": float(hit["score_decoy"]),
            "homo_type_real": int(hit["homo_type_real"]),
            "seq_identity": sid,
        }
    return passing_tids, t_q, est_fdp_at_t, tid_meta


def collect_candidates_from_merge(
    df_merge: pd.DataFrame,
    seq_lookup: dict[str, str],
    *,
    q: float,
    max_identity: float,
) -> list[dict]:
    rows: list[dict] = []

    for qid, df_qall in df_merge.groupby("qid", sort=False):
        q_seq = seq_lookup.get(str(qid))
        if q_seq is None:
            continue

        rep_ids = sorted(pd.unique(df_qall["rep_id"]))
        if not len(rep_ids):
            continue
        n_reps_total = len(rep_ids)

        passing_by_rep: list[set[str]] = []
        rep_context: dict[int, tuple[float | None, float | None, dict[str, dict]]] = {}

        for rid in rep_ids:
            df_q = df_qall[df_qall["rep_id"] == rid]
            passing_tids, t_q, est_fdp_at_t, tid_meta = _passing_tids_for_rep(
                df_q,
                q_seq,
                seq_lookup,
                q=q,
                max_identity=max_identity,
            )
            passing_by_rep.append(passing_tids)
            rep_context[int(rid)] = (t_q, est_fdp_at_t, tid_meta)

        if not passing_by_rep:
            continue

        common_tids = set.intersection(*passing_by_rep)
        for tid in common_tids:
            for rid in rep_ids:
                rid_int = int(rid)
                t_q, est_fdp_at_t, tid_meta = rep_context[rid_int]
                meta = tid_meta[tid]
                rows.append(
                    {
                        "qid": qid,
                        "tid": tid,
                        "rep_id": rid_int,
                        "n_reps_total": n_reps_total,
                        "score_real": meta["score_real"],
                        "score_decoy": meta["score_decoy"],
                        "homo_type_real": meta["homo_type_real"],
                        "seq_identity": meta["seq_identity"],
                        "T_q": t_q,
                        "est_fdp_at_Tq": est_fdp_at_t,
                        "q_level": q,
                    }
                )
    return rows


def aggregate_candidates(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    out = (
        df.groupby(["qid", "tid"], as_index=False)
        .agg(
            n_reps_passing=("rep_id", "nunique"),
            n_reps_total=("n_reps_total", "first"),
            score_real_mean=("score_real", "mean"),
            score_real_min=("score_real", "min"),
            score_real_max=("score_real", "max"),
            score_decoy_mean=("score_decoy", "mean"),
            seq_identity=("seq_identity", "first"),
            homo_type_real=("homo_type_real", "first"),
            T_q_mean=("T_q", "mean"),
            est_fdp_at_Tq_mean=("est_fdp_at_Tq", "mean"),
            q_level=("q_level", "first"),
        )
    )
    out["passes_all_reps"] = out["n_reps_passing"] == out["n_reps_total"]
    return out[out["passes_all_reps"]].copy()


CANDIDATE_SUMMARY_COLUMNS = [
    "qid",
    "tid",
    "n_reps_passing",
    "n_reps_total",
    "score_real_mean",
    "score_real_min",
    "score_real_max",
    "score_decoy_mean",
    "seq_identity",
    "homo_type_real",
    "T_q_mean",
    "est_fdp_at_Tq_mean",
    "q_level",
    "passes_all_reps",
]


def collect_candidates_streaming(
    path_real: Path,
    path_decoy: Path,
    suffix: str,
    seq_lookup: dict[str, str],
    *,
    q: float,
    max_identity: float,
    max_queries: int | None,
) -> pd.DataFrame:
    rows: list[dict] = []
    n_seen = 0
    print("[step1] Scanning queries...", flush=True)
    for qid, df_merge in iter_pair_tables_by_qid(path_real, path_decoy, suffix):
        if max_queries is not None and n_seen >= max_queries:
            break
        n_seen += 1
        if n_seen % 500 == 0:
            print(f"        processed queries: {n_seen:,}  candidates so far: {len(rows):,}")

        block_rows = collect_candidates_from_merge(
            df_merge,
            seq_lookup,
            q=q,
            max_identity=max_identity,
        )
        rows.extend(block_rows)

    print(f"        total queries scanned: {n_seen:,}")
    return aggregate_candidates(rows)


def _compute_tda_fdr_per_query(df_q: pd.DataFrame) -> pd.DataFrame:
    import sys

    core = Path(__file__).resolve().parents[1] / "core"
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    from fdr import compute_tda_fdr_per_query

    return compute_tda_fdr_per_query(df_q)


def safe_id(seq_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", seq_id)


def detect_chain_id(pdb_path: Path) -> str:
    counts: dict[str, int] = {}
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            chain = line[21:22].strip() or (line.split()[4] if len(line.split()) >= 5 else "")
            if chain:
                counts[chain] = counts.get(chain, 0) + 1
    return max(counts, key=counts.get) if counts else "A"
