import argparse

import numpy as np
import pandas as pd
from Bio import SeqIO


def load_ids(fasta_path):
    return [rec.id for rec in SeqIO.parse(fasta_path, "fasta")]


def load_class_map(path):
    m = {}
    with open(path) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                m[p[0]] = p[1]
    return m


def remove_decoy_suffix(value, suffix):
    if suffix and value.endswith(suffix):
        return value[: -len(suffix)]
    return value


def fill_matrix(path, q2i, t2i, suffix, nq, nt):
    df = pd.read_csv(path, sep="\t", usecols=["qid", "tid", "score"])
    if suffix:
        df["qid"] = df["qid"].map(lambda x: remove_decoy_suffix(x, suffix))
    rows = df["qid"].map(q2i)
    cols = df["tid"].map(t2i)
    mask = rows.notna() & cols.notna()
    rows = rows[mask].astype(np.int64).to_numpy()
    cols = cols[mask].astype(np.int64).to_numpy()
    scores = df["score"][mask].astype(np.float64).to_numpy()
    mat = np.full((nq, nt), np.nan, dtype=np.float64)
    mat[rows, cols] = scores
    return mat


def pad_rows(mat, eps, fallback):
    with np.errstate(all="ignore"):
        row_min = np.nanmin(mat, axis=1)
    pad = np.round(row_min - eps, 6)
    pad[~np.isfinite(pad)] = fallback
    nan_idx = np.where(np.isnan(mat))
    mat[nan_idx] = pad[nan_idx[0]]
    return mat


def write_side(path, query_ids, target_ids, score_mat, homo_mat):
    nt = len(target_ids)
    order = np.argsort(-score_mat, axis=1, kind="stable")
    score_sorted = np.take_along_axis(score_mat, order, axis=1).ravel()
    homo_sorted = np.take_along_axis(homo_mat, order, axis=1).ravel()
    target_arr = np.asarray(target_ids, dtype=object)
    tid_sorted = target_arr[order].ravel()
    qid_col = np.repeat(np.asarray(query_ids, dtype=object), nt)
    rank_col = np.tile(np.arange(1, nt + 1, dtype=np.int32), len(query_ids))
    out = pd.DataFrame({
        "qid": qid_col,
        "tid": tid_sorted,
        "score": score_sorted,
        "homo_type": homo_sorted,
        "rank": rank_col,
    })
    out.to_csv(path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True)
    parser.add_argument("--decoy", required=True)
    parser.add_argument("--query-fasta", default="data/putative.fa")
    parser.add_argument("--target-fasta", default="data/class_all.fa")
    parser.add_argument("--class-file", default="data/bagel_class.txt")
    parser.add_argument("--decoy-suffix", default="_shuf")
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--fallback", type=float, default=0.0)
    parser.add_argument("--out-real", required=True)
    parser.add_argument("--out-decoy", required=True)
    args = parser.parse_args()

    query_ids = load_ids(args.query_fasta)
    target_ids = load_ids(args.target_fasta)
    cls = load_class_map(args.class_file)
    nq, nt = len(query_ids), len(target_ids)
    q2i = {v: i for i, v in enumerate(query_ids)}
    t2i = {v: i for i, v in enumerate(target_ids)}

    qcls = np.array([cls.get(q, "__NA_Q__") for q in query_ids], dtype=object)
    tcls = np.array([cls.get(t, "__NA_T__") for t in target_ids], dtype=object)
    homo = np.where(qcls[:, None] == tcls[None, :], 1, -1).astype(np.int8)

    real_mat = pad_rows(fill_matrix(args.real, q2i, t2i, "", nq, nt), args.eps, args.fallback)
    write_side(args.out_real, query_ids, target_ids, real_mat, homo)

    decoy_mat = pad_rows(fill_matrix(args.decoy, q2i, t2i, args.decoy_suffix, nq, nt), args.eps, args.fallback)
    decoy_ids = [q + args.decoy_suffix for q in query_ids]
    write_side(args.out_decoy, decoy_ids, target_ids, decoy_mat, homo)

    print(f"nq={nq} nt={nt} pairs_per_side={nq * nt}")


if __name__ == "__main__":
    main()
