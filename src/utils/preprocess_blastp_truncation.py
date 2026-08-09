import argparse

import numpy as np
import pandas as pd
from Bio import SeqIO


def load_ids_and_labels(fasta_path):
    ids = []
    folds = []
    supfs = []
    fams = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        label = rec.description.split(" ")[1]
        a = label.split(".")
        ids.append(rec.id)
        folds.append(".".join(a[:2]))
        supfs.append(".".join(a[:3]))
        fams.append(".".join(a[:4]))
    return ids, folds, supfs, fams


def code(values):
    _, inv = np.unique(np.asarray(values), return_inverse=True)
    return inv.astype(np.int32)


def remove_decoy_suffix(value, suffix):
    if suffix and value.endswith(suffix):
        return value[: -len(suffix)]
    return value


def fill_matrix(path, id2idx, suffix, n):
    df = pd.read_csv(path, sep="\t", usecols=["qid", "tid", "score"])
    if suffix:
        df["qid"] = df["qid"].map(lambda x: remove_decoy_suffix(x, suffix))
    rows = df["qid"].map(id2idx)
    cols = df["tid"].map(id2idx)
    mask = rows.notna() & cols.notna()
    rows = rows[mask].astype(np.int64).to_numpy()
    cols = cols[mask].astype(np.int64).to_numpy()
    scores = df["score"][mask].astype(np.float64).to_numpy()
    mat = np.full((n, n), np.nan, dtype=np.float64)
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
    n = len(target_ids)
    order = np.argsort(-score_mat, axis=1, kind="stable")
    score_sorted = np.take_along_axis(score_mat, order, axis=1).ravel()
    homo_sorted = np.take_along_axis(homo_mat, order, axis=1).ravel()
    target_arr = np.asarray(target_ids, dtype=object)
    tid_sorted = target_arr[order].ravel()
    qid_col = np.repeat(np.asarray(query_ids, dtype=object), n)
    rank_col = np.tile(np.arange(1, n + 1, dtype=np.int32), len(query_ids))
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
    parser.add_argument("--fasta", default="data/astral.fa")
    parser.add_argument("--decoy-suffix", default="_shuf")
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--fallback", type=float, default=0.0)
    parser.add_argument("--out-real", required=True)
    parser.add_argument("--out-decoy", required=True)
    args = parser.parse_args()

    ids, folds, supfs, fams = load_ids_and_labels(args.fasta)
    n = len(ids)
    id2idx = {v: i for i, v in enumerate(ids)}

    fold_c = code(folds)
    sf_c = code(supfs)
    fam_c = code(fams)
    homo = np.where(
        fold_c[:, None] != fold_c[None, :], -1,
        np.where(sf_c[:, None] != sf_c[None, :], 0,
                 np.where(fam_c[:, None] != fam_c[None, :], 1, 2)),
    ).astype(np.int8)

    real_mat = pad_rows(fill_matrix(args.real, id2idx, "", n), args.eps, args.fallback)
    write_side(args.out_real, ids, ids, real_mat, homo)
    del real_mat

    decoy_mat = pad_rows(fill_matrix(args.decoy, id2idx, args.decoy_suffix, n), args.eps, args.fallback)
    decoy_ids = [i + args.decoy_suffix for i in ids]
    write_side(args.out_decoy, decoy_ids, ids, decoy_mat, homo)
    del decoy_mat

    print(f"n={n} pairs_per_side={n * n}")


if __name__ == "__main__":
    main()
