import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO


def _parse_blast_data(url):
    # qid, tid, bit score, e-value
    qid, in_hits = "", False
    with open(url) as fp:
        for line in fp:
            line = line.rstrip()
            if line.startswith("Query="):
                qid = line.split()[1]
            elif line.startswith("Sequences"):
                in_hits = True
            elif line.startswith("Lambda"):
                in_hits = False
            elif in_hits and line:
                data = line.split()
                yield qid, data[0], float(data[-2]), float(data[-1])


def parse_scop(input_path, save_path, data_type, fasta_url):
    # homo_type: -1 different fold, 0 same fold, 1 same superfamily, 2 same family
    labels = {rec.id: tuple(".".join(rec.description.split(" ")[1].split(".")[:k]) for k in (2, 3, 4))
              for rec in SeqIO.parse(fasta_url, "fasta")}
    decoy = re.compile(r"_(rev|shuf|mkv|dplm).*$")
    save_url = os.path.join(save_path, f"parsed_result/{Path(input_path).name}.txt")
    os.makedirs(os.path.dirname(save_url), exist_ok=True)

    with open(save_url, "w") as out:
        out.write("qid\ttid\tscore\thomo_type\trank" + ("\tevalue\n" if data_type == "blastp" else "\n"))
        for file in tqdm(os.listdir(input_path), desc="Parsing batch..."):
            if not (".out" in file or ".txt" in file):
                continue
            url = os.path.join(input_path, file)
            if data_type == "blastp":
                rows = _parse_blast_data(url)
            else:
                with open(url) as handle:
                    lines = [line for line in handle if line.strip() and not line.startswith("query_id")]
                tcol = 2 if data_type == "tmvec" else 1
                rows = [(a[0], a[tcol], float(a[-1])) for a in (line.rstrip().split("," if data_type == "dhr" else None) for line in lines)]

            prev_qid, rank = None, 0
            for qid, tid, score, *evalue in rows:
                rank = rank + 1 if qid == prev_qid else 1
                prev_qid = qid
                q, t = labels[decoy.sub("", qid)], labels[decoy.sub("", tid)]
                homo_type = -1 if q[0] != t[0] else 0 if q[1] != t[1] else 1 if q[2] != t[2] else 2
                out.write("\t".join(map(str, (qid, tid, float(score), homo_type, rank, *evalue))) + "\n")


def dhr_postprocess(data_dir, query_name, db_name, decoy_types):
    # flip DHR distances into similarity scores
    for name in [query_name] + [f"{query_name}_{decoy_type}" for decoy_type in decoy_types]:
        df = pd.read_csv(f"{data_dir}/result_dhr_{name}_{db_name}.txt", sep="\t")
        df["score"] = 150 - df["score"]
        df.to_csv(f"{data_dir}/result_dhr_postprocess_{name}_{db_name}.txt", sep="\t", index=False)
        print(f"Saved to: {data_dir}/result_dhr_postprocess_{name}_{db_name}.txt")


def blastp_densify(real, decoy, fasta, decoy_suffix, eps, fallback, out_real, out_decoy):
    labels = {rec.id: tuple(".".join(rec.description.split(" ")[1].split(".")[:k]) for k in (2, 3, 4))
              for rec in SeqIO.parse(fasta, "fasta")}
    ids = list(labels)
    n = len(ids)
    id2idx = {v: i for i, v in enumerate(ids)}
    fold, supf, fam = (np.unique(level, return_inverse=True)[1] for level in zip(*labels.values()))
    homo = np.select([fold[:, None] != fold, supf[:, None] != supf, fam[:, None] != fam], [-1, 0, 1], 2).astype(np.int8)

    for path, suffix, out_path in ((real, "", out_real), (decoy, decoy_suffix, out_decoy)):
        df = pd.read_csv(path, sep="\t", usecols=["qid", "tid", "score"])
        rows, cols = df["qid"].str.removesuffix(suffix).map(id2idx), df["tid"].map(id2idx)
        mask = rows.notna() & cols.notna()
        mat = np.full((n, n), np.nan)
        mat[rows[mask].astype(np.int64), cols[mask].astype(np.int64)] = df["score"][mask].astype(np.float64)
        pad = np.round(np.nanmin(mat, axis=1) - eps, 6)
        pad[~np.isfinite(pad)] = fallback
        np.copyto(mat, pad[:, None], where=np.isnan(mat))

        order = np.argsort(-mat, axis=1, kind="stable")
        pd.DataFrame({
            "qid": np.repeat(np.array([i + suffix for i in ids], dtype=object), n),
            "tid": np.asarray(ids, dtype=object)[order].ravel(),
            "score": np.take_along_axis(mat, order, axis=1).ravel(),
            "homo_type": np.take_along_axis(homo, order, axis=1).ravel(),
            "rank": np.tile(np.arange(1, n + 1, dtype=np.int32), n),
        }).to_csv(out_path, sep="\t", index=False)
        del mat, order
        print(f"Saved to: {out_path}")


def parse_bagel(score_file, out_path, method):
    # homo_type: 1 if query and target share a BAGEL class, else -1
    if method == "tmvec":
        df = pd.read_csv(score_file, sep="\t").rename(columns={"query_id": "qid", "database_id": "tid", "tm-score": "score"})
    elif method == "plm":
        df = pd.read_csv(score_file, sep="\t", header=None).rename(columns={0: "qid", 1: "tid", 2: "score"})
    elif method == "dhr":  
        with open(score_file) as handle:
            df = pd.DataFrame([(p[0], ",".join(p[1:-1]), float(p[-1])) for p in (line.strip().split(",") for line in handle if line.strip())],
                              columns=["qid", "tid", "score"])
    else:
        df = pd.DataFrame(_parse_blast_data(score_file), columns=["qid", "tid", "score", "evalue"])

    if method != "tmvec":
        df["rank"] = df.groupby("qid")["score"].rank(ascending=method == "dhr", method="first").astype(int)
    bagel_class = lambda s: s.str.split(";").str[0].str.split(".").str[1]
    df["homo_type"] = np.where(bagel_class(df["qid"]) == bagel_class(df["tid"]), 1, -1)
    cols = ["qid", "tid", "score", "homo_type", "rank"]
    if method == "blastp":
        df = df.sort_values(["qid", "rank"])
        cols.append("evalue")
    df[cols].to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Parse raw search output, post-process DHR scores, densify blastp output.")
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("parse", help="native search output -> uniform score table")
    p.add_argument("--data_type", type=str, required=True, help="plm | tmvec | dhr | blastp")
    p.add_argument("--input_path", type=str, required=True, help="Directory of raw search output")
    p.add_argument("--save_path", type=str, required=True, help="Parsed result goes to <save_path>/parsed_result/")
    p.add_argument("--target-fasta", type=str, required=True, help="the target DB FASTA; SCOP labels are read from its headers")

    p = sub.add_parser("dhr-postprocess", help="flip DHR distance to a larger-is-better score")
    p.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[2] / "data"))
    p.add_argument("--query", default="class_all", help="Query set name")
    p.add_argument("--db", default="class_all", help="Target DB name")
    p.add_argument("--decoy", action="append", default=None, help="Decoy type; repeat for several (default: extended_shuf)")

    p = sub.add_parser("blastp-densify", help="fill blastp's sparse output into a dense table")
    p.add_argument("--real", required=True)
    p.add_argument("--decoy", required=True)
    p.add_argument("--fasta", default="data/astral.fa")
    p.add_argument("--decoy-suffix", default="_shuf", help="literal id suffix, WITH the leading underscore (e.g. `_mkv2`)")
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--fallback", type=float, default=0.0)
    p.add_argument("--out-real", required=True)
    p.add_argument("--out-decoy", required=True)

    p = sub.add_parser("parse-bagel", help="single-file parse, labelled by BAGEL class")
    p.add_argument("--method", required=True, choices=["plm", "tmvec", "dhr", "blastp"])
    p.add_argument("--score-file", required=True)
    p.add_argument("--out-path", required=True)

    args = ap.parse_args()
    if args.stage == "parse":
        parse_scop(args.input_path, args.save_path, args.data_type, args.target_fasta)
    elif args.stage == "dhr-postprocess":
        dhr_postprocess(args.data_dir, args.query, args.db, args.decoy or ["extended_shuf"])
    elif args.stage == "blastp-densify":
        blastp_densify(args.real, args.decoy, args.fasta, args.decoy_suffix, args.eps, args.fallback, args.out_real, args.out_decoy)
    else:
        parse_bagel(args.score_file, args.out_path, args.method)
