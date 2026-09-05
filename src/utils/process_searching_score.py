import sys, os
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

# DHR scores are distances; subtract from this to get larger-is-better. CORE_ALGORITHM.md §2.
DHR_SCORE_CEILING = 150.0


def get_origin_prot_id(prot_id):
    prot_id_origin = prot_id


    # window-decoy ids like "d4hswa_.1.10" -> "d4hswa_"
    m = re.match(r"^(.+)\.(\d+)\.(\d+)$", prot_id)
    if m:
        return m.group(1)

    if '_range' in prot_id: prot_id_origin = prot_id[:prot_id.find('_range')]
    elif '_rev' in prot_id: prot_id_origin = prot_id[:prot_id.find('_rev')]
    elif '_doub_shuf0' in prot_id: prot_id_origin = prot_id[:prot_id.find('_doub_shuf0')]
    elif '_doub_shuf1' in prot_id: prot_id_origin = prot_id[:prot_id.find('_doub_shuf1')]
    elif '_decoy' in prot_id: prot_id_origin = prot_id[:prot_id.find('_decoy')]
    elif '_trunc' in prot_id: prot_id_origin = prot_id[:prot_id.find('_trunc')]
    elif '_doub' in prot_id: prot_id_origin = prot_id[:prot_id.find('_doub')]
    elif '_mut' in prot_id: prot_id_origin = prot_id[:prot_id.find('_mut')]
    elif '_shuf' in prot_id: prot_id_origin = prot_id[:prot_id.find('_shuf')]
    elif '_mkv' in prot_id: prot_id_origin = prot_id[:prot_id.find('_mkv')]
    elif '_mkv2' in prot_id: prot_id_origin = prot_id[:prot_id.find('_mkv2')]
    elif '_dplm' in prot_id: prot_id_origin = prot_id[:prot_id.find('_dplm')]
    elif '_adp' in prot_id: prot_id_origin = prot_id[:prot_id.find('_adp')]
    elif '_PGlen' in prot_id: prot_id_origin = prot_id[:prot_id.find('_PGlen')]
    elif '_copy' in prot_id: prot_id_origin = prot_id[:prot_id.find('_copy')]
    elif '_dup' in prot_id: prot_id_origin = prot_id[:prot_id.find('_dup')]


    return prot_id_origin

def _scop_levels(description):
    """'d1abca_ a.1.1.1 (...)' -> ('a.1', 'a.1.1', 'a.1.1.1')."""
    scop_label = description.split(' ')[1]
    arr = scop_label.split('.')
    assert len(arr) == 4
    return '.'.join(arr[:2]), '.'.join(arr[:3]), '.'.join(arr[:4])

def get_seq_label_map(fasta_url):
    records = list(SeqIO.parse(fasta_url, "fasta"))

    fa_id_label_map = {}
    for record in tqdm(records):
        fa_id = record.id
        scop_fold, scop_supf, scop_fam = _scop_levels(record.description)

        assert fa_id not in fa_id_label_map
        fa_id_label_map[fa_id] = (scop_fold, scop_supf, scop_fam)

    return fa_id_label_map

def check_muti_level_homolog(src_labels, tgt_labels):
    src_scop_fold, src_scop_supf, src_scop_fam = src_labels
    tgt_scop_fold, tgt_scop_supf, tgt_scop_fam = tgt_labels

    if src_scop_fold != tgt_scop_fold:
        category = -1
    elif src_scop_supf != tgt_scop_supf:
        category = 0
    elif src_scop_fam != tgt_scop_fam:
        category = 1
    else:
        category = 2
    return category


def parse_plm_data(url):
    res = []
    with open(url, 'r') as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip().rstrip("\n")
            if len(line) <= 0: continue
            arr = line.split()
            res.append([arr[0], arr[1], float(arr[-1])])
    return res

def parse_tmvec_data(url):
    res = []
    with open(url, 'r') as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip().rstrip("\n")
            if len(line) <= 0: continue
            if line.startswith('query_id'): continue

            arr = line.split()
            res.append([arr[0], arr[2], float(arr[-1])])
    return res

def parse_dhr_data(url):
    res = []
    with open(url, 'r') as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip().rstrip("\n")
            if len(line) <= 0: continue

            arr = line.split(',')
            res.append([arr[0], arr[1], float(arr[-1])])
    return res

def parse_dctdomain_data(url):
    res = []
    with open(url, 'r') as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip().rstrip("\n")
            if len(line) <= 0: continue
            if not line.startswith('Query'): continue

            arr = line.split(':')
            assert len(arr) == 4

            curr_id = arr[1].strip().split()[0]
            tgt_id = arr[2].strip().split()[0]
            score = float(arr[3].strip())
            res.append([curr_id, tgt_id, score])
    return res

def parse_blast_data(url):
    curr_query_id = ''
    score_zone = False
    with open(url, 'r') as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip().rstrip("\n")
            if len(line) <= 0: continue

            if line.startswith("Results from"):
                curr_iter = int(line.split()[-1])
                print('curr_iter={}'.format(curr_iter))
                assert curr_iter <= 1

            elif line.startswith("Query="):
                curr_query_id = line.split()[1]

            elif line.startswith("Sequences"): score_zone = True
            elif line.startswith("Lambda"): score_zone = False
            else:
                if not score_zone: continue
                data = line.split()
                sid = data[0]
                evalue = float(data[-1])
                raw_bit = float(data[-2])

                yield (curr_query_id, sid, raw_bit, evalue)


ParserFunctionMap = {
    "plm": parse_plm_data,
    "tmvec": parse_tmvec_data,
    "dhr": parse_dhr_data,
    "dctdomain": parse_dctdomain_data,
    "blastp": parse_blast_data
}


def parse_search_result_by_type(input_path, save_path, data_type, fasta_url, max_hits):
    assert data_type in ParserFunctionMap.keys()

    fa_id_label_map = get_seq_label_map(fasta_url)
    print('fa_id_label_map={}'.format(len(fa_id_label_map)))

    fname = Path(input_path).name
    save_url = os.path.join(save_path, f'parsed_result/{fname}_hit{max_hits}.txt')

    if os.path.exists(save_url):
        with open(save_url, 'r') as f:
            qt_score = f.readlines()
        print(f"Parsed results already exists (line={len(qt_score)}): {save_url}!")
        print(f"Sample data: \n{qt_score[0]}\n{qt_score[1]}")

    else:
        assert os.path.exists(input_path)
        os.makedirs(os.path.dirname(save_url), exist_ok=True)

        parse_func = ParserFunctionMap.get(data_type)

        with open(save_url, 'w') as f:
            if data_type == 'blastp':
                f.write(f"qid\ttid\tscore\thomo_type\trank\tevalue\n")
            else:
                f.write(f"qid\ttid\tscore\thomo_type\trank\n")

        total_line = 0
        for file in tqdm(os.listdir(input_path), desc='Sparsing batch...'):
            if not ('.out' in file or '.txt' in file): continue

            url = os.path.join(input_path, file)
            print('url={}'.format(url))

            query_id_flag = None
            rank = None
            prev_score = None # for sanity check
            for row in tqdm(parse_func(url), desc='Sparsing file...'):
                curr_id, tgt_id, score, *rest = row
                score = float(score)

                if query_id_flag != curr_id:
                    if rank is not None and data_type not in ['dctdomain', 'blastp']: assert rank >= max_hits
                    query_id_flag = curr_id
                    rank = 1
                    prev_score = score
                else:
                    rank += 1
                    if data_type != 'blastp':
                        assert score >= prev_score if data_type == 'dhr' else score <= prev_score
                    prev_score = score

                if rank > max_hits: continue

                curr_id_origin = get_origin_prot_id(curr_id)
                assert curr_id_origin in fa_id_label_map, f"Error: {curr_id_origin} not found in label map!"

                tgt_id_origin = get_origin_prot_id(tgt_id)
                assert tgt_id_origin in fa_id_label_map, f"Error: {tgt_id_origin} not found in label map!"

                src_labels = fa_id_label_map[curr_id_origin]
                tgt_labels = fa_id_label_map[tgt_id_origin]
                homo_type = check_muti_level_homolog(src_labels, tgt_labels)

                with open(save_url, 'a') as f:
                    if rest:
                        f.write(f"{curr_id}\t{tgt_id}\t{score}\t{homo_type}\t{rank}\t{rest[0]}\n")
                    else:
                        f.write(f"{curr_id}\t{tgt_id}\t{score}\t{homo_type}\t{rank}\n")
                total_line += 1

        print(f"Parsed results (line={total_line}) saved to: {save_url}.")


def dhr_postprocess(data_dir, query_name, db_name, decoy_types,
                    ceiling=DHR_SCORE_CEILING, method="dhr"):
    """Flip DHR distances into larger-is-better scores."""
    def _flip(src, dst):
        df = pd.read_csv(src, sep="\t")
        df['score'] = ceiling - df['score']
        df.to_csv(dst, sep="\t", index=False)
        print(df.head())
        print(f'Saved to: {dst}')

    target_src = f"{data_dir}/result_{method}_{query_name}_{db_name}.txt"
    target_dst = f"{data_dir}/result_{method}_postprocess_{query_name}_{db_name}.txt"
    _flip(target_src, target_dst)

    for decoy_type in decoy_types:
        print(f"Processing {decoy_type}...")
        _flip(f"{data_dir}/result_{method}_{query_name}_{decoy_type}_{db_name}.txt",
              f"{data_dir}/result_{method}_postprocess_{query_name}_{decoy_type}_{db_name}.txt")


def load_ids_and_labels(fasta_path):
    ids = []
    folds = []
    supfs = []
    fams = []
    for rec in SeqIO.parse(fasta_path, "fasta"):
        fold, supf, fam = _scop_levels(rec.description)
        ids.append(rec.id)
        folds.append(fold)
        supfs.append(supf)
        fams.append(fam)
    return ids, folds, supfs, fams


def code(values):
    _, inv = np.unique(np.asarray(values), return_inverse=True)
    return inv.astype(np.int32)


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
    """A pair blastp never reported scores just below that query's weakest real hit."""
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


def blastp_densify(real, decoy, fasta, decoy_suffix, eps, fallback, out_real, out_decoy):
    ids, folds, supfs, fams = load_ids_and_labels(fasta)
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

    # one side at a time: each dense matrix is n^2 float64 (1.8 GB at n=15177)
    real_mat = pad_rows(fill_matrix(real, id2idx, id2idx, "", n, n), eps, fallback)
    write_side(out_real, ids, ids, real_mat, homo)
    del real_mat

    decoy_mat = pad_rows(fill_matrix(decoy, id2idx, id2idx, decoy_suffix, n, n), eps, fallback)
    decoy_ids = [i + decoy_suffix for i in ids]
    write_side(out_decoy, decoy_ids, ids, decoy_mat, homo)
    del decoy_mat

    print(f"n={n} pairs_per_side={n * n}")


def load_ids(fasta_path):
    return [rec.id for rec in SeqIO.parse(fasta_path, "fasta")]


def load_class_map(path):
    """bagel_class.txt: id<TAB>class, with a header row."""
    m = {}
    with open(path) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                m[p[0]] = p[1]
    return m


def clean_id(id_str, decoy_suffix):
    if not decoy_suffix:
        return id_str
    return re.sub(fr"_{decoy_suffix}.*$", "", id_str)


def _assign_homo_types(df, homo_dict, type_suffix, unclassified):
    """BAGEL homology is binary: same class -> 1, otherwise -1."""
    out = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing homo type"):
        tid_clean = clean_id(row["tid"], type_suffix)
        if unclassified:
            out.append(homo_dict.get(tid_clean, None))
            continue
        qid_clean = clean_id(row["qid"], type_suffix)
        out.append("1" if homo_dict.get(qid_clean) == homo_dict.get(tid_clean) else "-1")
    return out


def parse_bagel(score_file, homo_info_file, out_path, method, type_suffix,
                unclassified=False, max_hits=None):
    """Parse one BAGEL search table, labelling hits from bagel_class.txt."""
    homo_dict = dict(zip(*[pd.read_csv(homo_info_file, sep="\t", header=0)[c]
                           for c in ("ids", "class")]))

    if method == "tmvec":
        df = pd.read_csv(score_file, sep="\t", header=0).rename(
            columns={"query_id": "qid", "database_id": "tid", "tm-score": "score"})
        df["homo_type"] = _assign_homo_types(df, homo_dict, type_suffix, unclassified)
        cols = ["qid", "tid", "score", "homo_type", "rank"]

    elif method == "plm":
        df = pd.read_csv(score_file, sep="\t", header=None).rename(
            columns={0: "qid", 1: "tid", 2: "score"})
        df["rank"] = df.groupby("qid")["score"].rank(ascending=False, method="first").astype(int)
        df["homo_type"] = _assign_homo_types(df, homo_dict, type_suffix, unclassified)
        cols = ["qid", "tid", "score", "homo_type", "rank"]

    elif method == "dhr":
        rows = []
        with open(score_file) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                rows.append((parts[0], ",".join(parts[1:-1]), float(parts[-1])))
        df = pd.DataFrame(rows, columns=["qid", "tid", "score"])
        # DHR scores are distances: lower is better
        df["rank"] = df.groupby("qid")["score"].rank(ascending=True, method="first").astype(int)
        df["homo_type"] = _assign_homo_types(df, homo_dict, type_suffix, unclassified)
        cols = ["qid", "tid", "score", "homo_type", "rank"]

    elif method == "blastp":
        df = pd.DataFrame(parse_blast_data(score_file),
                          columns=["qid", "tid", "score", "evalue"])
        df["homo_type"] = _assign_homo_types(df, homo_dict, type_suffix, unclassified)
        df["rank"] = df.groupby("qid")["score"].rank(ascending=False, method="first").astype(int)
        if max_hits:
            df = df[df["rank"] <= max_hits]
        df = df.sort_values(["qid", "rank"])
        cols = ["qid", "tid", "score", "homo_type", "rank", "evalue"]

    else:
        raise ValueError(f"Unsupported method: {method}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df[cols].to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Saved to {out_path}")


def blastp_densify_bagel(real, decoy, query_fasta, target_fasta, class_file,
                         decoy_suffix, eps, fallback, out_real, out_decoy):
    """Densify sparse BLASTp output into a full query x target table for BAGEL."""
    query_ids = load_ids(query_fasta)
    target_ids = load_ids(target_fasta)
    cls = load_class_map(class_file)
    nq, nt = len(query_ids), len(target_ids)
    q2i = {v: i for i, v in enumerate(query_ids)}
    t2i = {v: i for i, v in enumerate(target_ids)}

    qcls = np.array([cls.get(q, "__NA_Q__") for q in query_ids], dtype=object)
    tcls = np.array([cls.get(t, "__NA_T__") for t in target_ids], dtype=object)
    homo = np.where(qcls[:, None] == tcls[None, :], 1, -1).astype(np.int8)

    real_mat = pad_rows(fill_matrix(real, q2i, t2i, "", nq, nt), eps, fallback)
    write_side(out_real, query_ids, target_ids, real_mat, homo)
    del real_mat

    decoy_mat = pad_rows(fill_matrix(decoy, q2i, t2i, decoy_suffix, nq, nt), eps, fallback)
    write_side(out_decoy, [q + decoy_suffix for q in query_ids], target_ids, decoy_mat, homo)
    del decoy_mat

    print(f"nq={nq} nt={nt} pairs_per_side={nq * nt}")


DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
BASEDB_FASTA = {
    'SCOPe40': os.path.join(DEFAULT_DATA_DIR, "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"),
    'SCOPe95': os.path.join(DEFAULT_DATA_DIR, "astral-scopedom-seqres-gd-sel-gs-bib-95-2.08.fa"),
}


def main():
    ap = argparse.ArgumentParser(
        description="Parse raw search output, post-process DHR scores, densify blastp output.")
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("parse", help="native search output -> uniform score table")
    p.add_argument("--data_type", type=str, required=True, help="plm | tmvec | dhr | blastp | dctdomain")
    p.add_argument("--input_path", type=str, required=True, help="Directory of raw search output")
    p.add_argument("--save_path", type=str, required=True, help="Parsed result goes to <save_path>/parsed_result/")
    p.add_argument("--max_hits", type=int, default=1000, help="Keep how many hits per query")
    p.add_argument("--basedb", type=str, required=True, help="SCOPe40 | SCOPe95")

    p = sub.add_parser("dhr-postprocess", help="flip DHR distance to a larger-is-better score")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--query", default="class_all", help="Query set name")
    p.add_argument("--db", default="class_all", help="Target DB name")
    p.add_argument("--decoy", action="append", default=None,
                   help="Decoy type; repeat for several (default: extended_shuf)")
    p.add_argument("--ceiling", type=float, default=DHR_SCORE_CEILING)

    p = sub.add_parser("blastp-densify", help="fill blastp's sparse output into a dense table")
    p.add_argument("--real", required=True)
    p.add_argument("--decoy", required=True)
    p.add_argument("--fasta", default="data/astral.fa")
    p.add_argument("--decoy-suffix", default="_shuf",
                   help="literal id suffix, WITH the leading underscore (e.g. `_mkv2`)")
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--fallback", type=float, default=0.0)
    p.add_argument("--out-real", required=True)
    p.add_argument("--out-decoy", required=True)

    p = sub.add_parser("parse-bagel", help="single-file parse using bagel_class.txt labels")
    p.add_argument("--method", required=True, choices=["plm", "tmvec", "dhr", "blastp"])
    p.add_argument("--score-file", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--homo-info", default=os.path.join(DEFAULT_DATA_DIR, "bagel_class.txt"))
    p.add_argument("--type-suffix", default="",
                   help="decoy tag to strip before the class lookup, WITHOUT the leading "
                        "underscore (e.g. `mkv`, which strips `_mkv...`)")
    p.add_argument("--max-hits", type=int, default=None, help="blastp only: keep top-N per query")
    p.add_argument("--unclassified", action="store_true")

    p = sub.add_parser("blastp-densify-bagel", help="densify blastp output against a BAGEL class DB")
    p.add_argument("--real", required=True)
    p.add_argument("--decoy", required=True)
    p.add_argument("--query-fasta", default="data/putative.fa")
    p.add_argument("--target-fasta", default="data/class_all.fa")
    p.add_argument("--class-file", default="data/bagel_class.txt")
    p.add_argument("--decoy-suffix", default="_shuf",
                   help="literal id suffix, WITH the leading underscore (e.g. `_mkv2`)")
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--fallback", type=float, default=0.0)
    p.add_argument("--out-real", required=True)
    p.add_argument("--out-decoy", required=True)

    args = ap.parse_args()

    if args.stage == "parse":
        if args.basedb not in BASEDB_FASTA:
            raise ValueError(f"Unsupported basedb: {args.basedb}")
        parse_search_result_by_type(args.input_path, args.save_path, args.data_type,
                                    BASEDB_FASTA[args.basedb], args.max_hits)
    elif args.stage == "dhr-postprocess":
        dhr_postprocess(args.data_dir, args.query, args.db,
                        args.decoy or ["extended_shuf"], args.ceiling)
    elif args.stage == "blastp-densify":
        blastp_densify(args.real, args.decoy, args.fasta, args.decoy_suffix,
                       args.eps, args.fallback, args.out_real, args.out_decoy)
    elif args.stage == "parse-bagel":
        parse_bagel(args.score_file, args.homo_info, args.out_path, args.method,
                    args.type_suffix, args.unclassified, args.max_hits)
    elif args.stage == "blastp-densify-bagel":
        blastp_densify_bagel(args.real, args.decoy, args.query_fasta, args.target_fasta,
                             args.class_file, args.decoy_suffix, args.eps, args.fallback,
                             args.out_real, args.out_decoy)


if __name__ == "__main__":
    main()
