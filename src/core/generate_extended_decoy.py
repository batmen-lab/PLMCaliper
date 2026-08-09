import pandas as pd
import numpy as np
import copy
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
import random
from collections import defaultdict


def generate_duplicate_decoy(fasta_path, output_path):
    records = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        
        seq = record.seq
        seq_dup = seq + seq

        record.seq = seq_dup
        record.id = record.id + "_dup"
        record.description = ""
        records.append(record)

    SeqIO.write(records, output_path, "fasta")


def generate_shuf_decoy(fasta_path, output_path, seed=0):
    random.seed(seed)

    records = []

    for record in SeqIO.parse(fasta_path, "fasta"):

        seq = str(record.seq)
        seq_list = list(seq)
        random.shuffle(seq_list)
        shuf_seq = "".join(seq_list)

        record.seq = Seq(seq + shuf_seq)
        record.id = record.id + "_shuf"
        record.description = ""
        records.append(record)

    SeqIO.write(records, output_path, "fasta")


def generate_reverse_decoy(fasta_path, output_path):
    records = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq)
        reverse_seq = seq[::-1]

        record.seq = Seq(seq + reverse_seq)
        record.id = record.id + "_rev"
        record.description = ""
        records.append(record)

    SeqIO.write(records, output_path, "fasta")


def generate_extend_then_shuffle_decoy(fasta_path, output_path, seed=0):
    random.seed(seed)
    records = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq)

        # extend and shuffle
        extended_seq = seq + seq
        seq_list = list(extended_seq)
        random.shuffle(seq_list)
        shuffled_extended_seq = "".join(seq_list)

        record.seq = Seq(shuffled_extended_seq)
        record.id = record.id + "_shuf"
        record.description = ""
        records.append(record)

    SeqIO.write(records, output_path, "fasta")


def generate_extended_markov_decoy(fasta_path, output_path, markov_order=1, seed=0):
    random.seed(seed)

    def protein_markovgen(sequence: str, k: int):
        seq_len = len(sequence)
        if seq_len <= k:
            seq_list = list(sequence)
            random.shuffle(seq_list)
            return ''.join(seq_list)

        model = defaultdict(lambda: defaultdict(int))
        for i in range(seq_len - k):
            prefix = sequence[i:(i + k)]
            next_char = sequence[i + k]
            model[prefix][next_char] += 1

        markov_model = {}
        for prefix, suffix_counts in model.items():
            total = sum(suffix_counts.values())
            chars = list(suffix_counts.keys())
            probs = [cnt / total for cnt in suffix_counts.values()]
            markov_model[prefix] = (chars, probs)

        start = random.choice(list(markov_model.keys()))
        result = list(start)
        for _ in range(seq_len - k):
            prefix = ''.join(result[-k:])
            if prefix in markov_model:
                chars, probs = markov_model[prefix]
                next_char = random.choices(chars, weights=probs)[0]
            else:
                next_char = random.choice(sequence)
            result.append(next_char)

        return ''.join(result)

    # Save A+A_mkv decoy
    final_records = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper()

        # generate extended and markov shuffled sequence (length 2L)
        mkv_seq = protein_markovgen(seq, markov_order)
        extended_decoy_seq = seq + mkv_seq  
        
        decoy_record = SeqRecord(
            Seq(extended_decoy_seq),
            id=record.id + "_mkv",
            description=""
        )
        final_records.append(decoy_record)

    SeqIO.write(final_records, output_path, "fasta")
    print(f"[OK] Successfully generated A+A_mkv database. Total records: {len(final_records)}")


def partly_shuffle(seq, n=1, alpha=0.2):
    seq = list(seq)
    L = len(seq)
    region_len = max(1, int(L * alpha))

    for _ in range(n):
        if L == region_len:
            start = 0
        else:
            start = random.randint(0, L - region_len)
        end = start + region_len
        region = seq[start:end]
        random.shuffle(region)
        seq[start:end] = region

    return ''.join(seq)


def generate_partly_shuffle_decoy(fasta_path, output_path, n=1, alpha=0.2, seed=0):
    random.seed(seed)
    records = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq)

        extended_seq = partly_shuffle(seq, n=n, alpha=alpha)
        decoy_seq = seq + extended_seq
        
        record.seq = Seq(decoy_seq)
        record.id = record.id + "_shufpartly"
        record.description = ""
        records.append(record)
    
    SeqIO.write(records, output_path, "fasta")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build the extended-shuffle decoy: seq + shuffle(seq), "
                                             "id suffixed '_shuf'. Output: <fasta>_extended_shuf.fa")
    ap.add_argument("--fasta", required=True, help="input FASTA (e.g. data/db/astral.fa)")
    ap.add_argument("--out", default=None, help="output FASTA (default: <fasta stem>_extended_shuf.fa)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import os
    out = args.out or f"{os.path.splitext(args.fasta)[0]}_extended_shuf.fa"
    generate_shuf_decoy(args.fasta, out, args.seed)
    print(f"Saved to {out}")

    # --- other decoy variants (kept for reference) ---
    # markov_order = 2
    # generate_extended_markov_decoy(f"{data_dir}/{seq_name}.fa", f"{data_dir}/{seq_name}_extended_mkv{markov_order}.fa", markov_order=markov_order, seed=seed)
    # print(f"Saved to {data_dir}/{seq_name}_extended_mkv{markov_order}.fa")


    # generate_reverse_decoy(f"{data_dir}/{seq_name}.fa", f"{data_dir}/{seq_name}_extended_rev.fa")
    # print(f"Saved to {data_dir}/{seq_name}_extended_rev.fa")

    # generate_extend_then_shuffle_decoy(f"{data_dir}/{seq_name}.fa", f"{data_dir}/{seq_name}_extended_then_shuf.fa", seed)
    # print(f"Saved to {data_dir}/{seq_name}_extended_then_shuf.fa")

    # generate_partly_shuffle_decoy(f"{data_dir}/{seq_name}.fa", f"{data_dir}/{seq_name}_extended_shufpartly40.fa", n=2, alpha=0.2, seed=seed)
    # print(f"Saved to {data_dir}/{seq_name}_extended_shufpartly40.fa")