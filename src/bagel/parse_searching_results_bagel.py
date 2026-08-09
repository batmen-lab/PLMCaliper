import argparse
import pandas as pd
import os
from tqdm import tqdm
import re


def load_db_to_dict(db_file, class_label):
    seq_dict = {}
    with open(db_file, "r") as f:
        lines = f.read().split("\n")

    current_seq = []
    for line in lines:
        if line.startswith(">"):
            if current_seq:
                seq = "".join(current_seq).strip()
                seq_dict[seq] = class_label
                current_seq = []
        else:
            current_seq.append(line.strip())

    # last one
    if current_seq:
        seq = "".join(current_seq).strip()
        seq_dict[seq] = class_label

    return seq_dict


def clean_id(id_str, decoy_suffix):
    if not decoy_suffix:
        return id_str
    return re.sub(fr"_{decoy_suffix}.*$", "", id_str)



def _assign_homo_types(df, homo_dict, type_suffix, Unclassified):
    if Unclassified:
        homo_type_list = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing homo type"):
            tid_clean = clean_id(row["tid"], type_suffix)
            homo_type_list.append(homo_dict.get(tid_clean, None))
        return homo_type_list

    homo_type_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing homo type"):
        qid_clean = clean_id(row["qid"], type_suffix)
        tid_clean = clean_id(row["tid"], type_suffix)
        class_qid = homo_dict.get(qid_clean, None)
        class_tid = homo_dict.get(tid_clean, None)
        if class_qid == class_tid:
            homo_type_list.append("1")
        else:
            homo_type_list.append("-1")
    return homo_type_list


def parse_bagel(score_file, homo_info_file, data_dir, file_name,
                method, type_suffix, Unclassified=False, out_path=None):

    if method == "tmvec":
        df = pd.read_csv(score_file, sep="\t", header=0)
        homo_info = pd.read_csv(homo_info_file, sep="\t")
        homo_dict = dict(zip(homo_info["ids"], homo_info["class"]))
        
        if Unclassified:
            homo_type_list = []
            for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing homo type"):
                tid_clean = clean_id(row["database_id"], type_suffix)
                class_tid = homo_dict.get(tid_clean, None)
                homo_type_list.append(class_tid)

        else:
            homo_type_list = []
            for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing homo type"):
                # Strip the decoy suffix from BOTH ids so the homology lookup
                # uses the underlying real qid / tid regardless of which side
                # is a decoy in this file.
                qid_clean = clean_id(row["query_id"], type_suffix)
                tid_clean = clean_id(row["database_id"], type_suffix)
                class_qid = homo_dict.get(qid_clean, None)
                class_tid = homo_dict.get(tid_clean, None)
                if class_qid == class_tid:
                    homo_type_list.append("1")
                else:
                    homo_type_list.append("-1")

        df["homo_type"] = homo_type_list
        df = df.rename(columns={
            "query_id": "qid",
            "database_id": "tid",
            "tm-score": "score",
            "homo_type": "homo_type",
            "rank": "rank"
            })
        df = df[["qid", "tid", "score", "homo_type", "rank"]]
        save_path = out_path or os.path.join(data_dir, f"{file_name}.txt")
        df.to_csv(save_path, sep="\t", index=False)
        print(f"[OK] Saved to {save_path}")

    elif method == "plm":
        df = pd.read_csv(score_file, sep="\t", header=None)
        df = df.rename(columns={0: "qid", 1: "tid", 2: "score"})
        df["rank"] = df.groupby("qid")["score"].rank(ascending=False, method="first").astype(int)
        print(df.head())
        homo_info = pd.read_csv(homo_info_file, sep="\t", header=0)
        homo_dict = dict(zip(homo_info["ids"], homo_info["class"]))
        df["homo_type"] = _assign_homo_types(df, homo_dict, type_suffix, Unclassified)
        df = df[["qid", "tid", "score", "homo_type", "rank"]]
        save_path = out_path or os.path.join(data_dir, f"{file_name}.txt")
        df.to_csv(save_path, sep="\t", index=False)
        print(f"[OK] Saved to {save_path}")

    elif method == "dhr":
        rows = []
        with open(score_file, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                rows.append((parts[0], ",".join(parts[1:-1]), float(parts[-1])))
        df = pd.DataFrame(rows, columns=["qid", "tid", "score"])
        # DHR outputs distance scores: lower is better.
        df["rank"] = df.groupby("qid")["score"].rank(ascending=True, method="first").astype(int)
        print(df.head())
        homo_info = pd.read_csv(homo_info_file, sep="\t", header=0)
        homo_dict = dict(zip(homo_info["ids"], homo_info["class"]))
        df["homo_type"] = _assign_homo_types(df, homo_dict, type_suffix, Unclassified)
        df = df[["qid", "tid", "score", "homo_type", "rank"]]
        save_path = out_path or os.path.join(data_dir, f"{file_name}.txt")
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        df.to_csv(save_path, sep="\t", index=False)
        print(f"[OK] Saved to {save_path}")

    else:
        raise ValueError(f"Unsupported method: {method}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse PLM/TMVec/DHR search results for BAGEL.")
    parser.add_argument("--method", default="tmvec", choices=["tmvec", "plm", "dhr"])
    parser.add_argument("--score-file", default=None, help="Single raw score file to parse")
    parser.add_argument("--out-path", default=None, help="Output TSV path (single-file mode)")
    parser.add_argument("--homo-info", default=None, help="bagel_class.txt path")
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data"))
    parser.add_argument("--file-name", default=None, help="Output stem when --out-path is omitted")
    parser.add_argument("--type-suffix", default="", help="Decoy suffix to strip during class lookup")
    parser.add_argument("--unclassified", action="store_true")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    HOMO_INFO_FILE = args.homo_info or os.path.join(DATA_DIR, "bagel_class.txt")

    if args.score_file:
        if not args.file_name and not args.out_path:
            parser.error("single-file mode requires --out-path or --file-name")
        parse_bagel(
            score_file=args.score_file,
            homo_info_file=HOMO_INFO_FILE,
            data_dir=DATA_DIR,
            file_name=args.file_name or "parsed_result",
            method=args.method,
            type_suffix=args.type_suffix,
            Unclassified=args.unclassified,
            out_path=args.out_path,
        )
    else:
        METHOD = args.method
        QUERY_NAME = "class_all"
        TARGET_NAME = "class_all"
        DECOY_METHOD = "extended_shuf"
        DECOY_SUFFIX = "shuf"

        runs = [
            (f"result_{METHOD}_{QUERY_NAME}_{TARGET_NAME}", ""),
            (f"result_{METHOD}_{QUERY_NAME}_{DECOY_METHOD}_{TARGET_NAME}", DECOY_SUFFIX),
        ]

        for file_stem, type_suffix in runs:
            score_file = os.path.join(DATA_DIR, f"{file_stem}.txt")
            if not os.path.exists(score_file):
                print(f"[SKIP] {score_file} not found.")
                continue

            print(f"\n[INFO] Parsing {score_file}  (type_suffix={type_suffix!r})")
            parse_bagel(
                score_file=score_file,
                homo_info_file=HOMO_INFO_FILE,
                data_dir=DATA_DIR,
                file_name=file_stem,
                method=METHOD,
                type_suffix=type_suffix,
                Unclassified=False,
            )