# Generate sequences

import os, sys, argparse, math, logging, subprocess, random
from collections import defaultdict
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
import torch
import esm
from statistics import mean
from typing import List, Tuple


# ========== Generate sequences ==========
## single sequence
def fasta_genshuf(fasta_url, output_url, include_origin=False, seed=None):
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))

    rng = np.random.default_rng(seed=seed)
    for record in tqdm(records):
        ID = record.id
        seq = record.seq
        desc = record.description
        # print('ID={}\tdesc={}'.format(ID, desc))

        seq_list = list(seq)
        rng.shuffle(seq_list)
        shuf_seq = "".join(seq_list)
        assert len(seq) == len(shuf_seq)
        line_num = math.ceil(len(seq) / 60)

        if include_origin:
            out_file.write(">{}\n".format(ID))
            for i in range(line_num):
                out_file.write("{}\n".format(seq[60 * i: 60 * (i + 1)]))

        out_file.write(">{}_shuf\n".format(ID))
        for i in range(line_num):
            out_file.write("{}\n".format(shuf_seq[60 * i: 60 * (i + 1)]))

    out_file.close()


def fasta_genreverse(fasta_url, output_url, include_origin=False):
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))

    for record in tqdm(records):
        ID = record.id
        seq = record.seq
        desc = record.description
        # print('ID={}\tdesc={}'.format(ID, desc))

        rev_seq = str(seq)[::-1]
        assert len(seq) == len(rev_seq)
        line_num = math.ceil(len(seq) / 60)

        if include_origin:
            out_file.write(">{}\n".format(ID))
            for i in range(line_num):
                out_file.write("{}\n".format(seq[60 * i: 60 * (i + 1)]))

        out_file.write(">{}_rev\n".format(ID))
        for i in range(line_num):
            out_file.write("{}\n".format(rev_seq[60 * i: 60 * (i + 1)]))

    out_file.close()


def fasta_genmarkov(fasta_url, output_url, markov_order, include_origin=False, seed=None):
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))

    rng = random.Random(seed)

    def protein_markovgen(sequence: str, k: int):
        seq_len = len(sequence)
        if seq_len < k + 1: raise ValueError("Input sequence is too short for the specified Markov order.")

        model = defaultdict(lambda: defaultdict(int))
        for i in range(seq_len - k):
            prefix = sequence[i:(i + k)]
            next_char = sequence[i + k]
            model[prefix][next_char] += 1

        markov_model = {}
        for prefix, suffix_counts in model.items():
            total = sum(suffix_counts.values())
            probs = []
            chars = []
            for char, count in suffix_counts.items():
                chars.append(char)
                probs.append(count * 1.0 / total)
            markov_model[prefix] = (chars, probs)

        start = random.choice(list(markov_model.keys()))
        result = list(start)
        # Generate sequence
        for _ in range(seq_len - k):
            prefix = ''.join(result[-k:])
            if prefix in markov_model:
                chars, probs = markov_model[prefix]
                next_char = random.choices(chars, probs)[0]
            else:
                # fallback: random choice from input sequence
                next_char = random.choice(sequence)
            result.append(next_char)

        return ''.join(result)

    for record in tqdm(records):
        ID = record.id
        seq = record.seq
        desc = record.description
        # print('ID={}\tdesc={}'.format(ID, desc))

        mkv_seq = protein_markovgen(seq, markov_order)
        assert len(seq) == len(mkv_seq)
        line_num = math.ceil(len(seq) / 60)

        if include_origin:
            out_file.write(">{}\n".format(ID))
            for i in range(line_num):
                out_file.write("{}\n".format(seq[60 * i: 60 * (i + 1)]))

        out_file.write(">{}_mkv{}\n".format(ID, markov_order))
        for i in range(line_num):
            out_file.write("{}\n".format(mkv_seq[60 * i: 60 * (i + 1)]))

    out_file.close()


## multiple sequences (make sure the small set is in the larger set)
def multi_shuffle_onefile(fasta_url, output_url, num_shuffles=5, include_origin=False):
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))
    
    # rng = np.random.default_rng(0)

    np.random.seed(0)
    random_seeds = np.random.randint(0, 1000000, len(records))
    
    for i, record in enumerate(tqdm(records)):
    # for record in tqdm(records):
        rng = np.random.default_rng(random_seeds[i])
        ID = record.id
        seq = record.seq
        line_num = math.ceil(len(seq) / 60)

        if include_origin:
            out_file.write(">{}\n".format(ID))
            for i in range(line_num):
                out_file.write("{}\n".format(seq[60 * i: 60 * (i + 1)]))

        # multiple shuffles
        for s in range(1, num_shuffles+1):
            seq_list = list(seq)
            rng.shuffle(seq_list)
            shuf_seq = "".join(seq_list)
            out_file.write(f">{ID}_shuf{s}\n")
            for i in range(line_num):
                out_file.write(f"{shuf_seq[60 * i: 60 * (i + 1)]}\n")
            assert len(seq) == len(shuf_seq)
    out_file.close()


def multi_markov_onefile(fasta_url, output_url, markov_order=2, num_samples=5, include_origin=False, seed=None):
    assert os.path.exists(fasta_url)
    records = list(SeqIO.parse(fasta_url, "fasta"))

    np.random.seed(0)
    random_seeds = np.random.randint(0, 1000000, len(records))

    with open(output_url, "w") as out_file:

        def protein_markovgen(sequence: str, k: int):
            seq_len = len(sequence)
            if seq_len < k + 1:
                raise ValueError("Input sequence is too short for the specified Markov order.")

            model = defaultdict(lambda: defaultdict(int))
            for i in range(seq_len - k):
                prefix = sequence[i:(i + k)]
                next_char = sequence[i + k]
                model[prefix][next_char] += 1

            markov_model = {}
            for prefix, suffix_counts in model.items():
                total = sum(suffix_counts.values())
                chars, probs = zip(*[(c, cnt / total) for c, cnt in suffix_counts.items()])
                markov_model[prefix] = (list(chars), list(probs))

            start = random.choice(list(markov_model.keys()))
            result = list(start)
            # Generate sequence
            for _ in range(seq_len - k):
                prefix = ''.join(result[-k:])
                if prefix in markov_model:
                    chars, probs = markov_model[prefix]
                    next_char = random.choices(chars, weights=probs, k=1)[0]
                else:
                    # fallback: random choice from input sequence
                    next_char = random.choice(sequence)
                result.append(next_char)

            return ''.join(result)

        for i, rec in enumerate(tqdm(records)):
            random.seed(int(random_seeds[i]))

            ID = rec.id
            seq = str(rec.seq)
            line_num = math.ceil(len(seq) / 60)

            if include_origin:
                out_file.write(f">{ID}\n")
                for i in range(line_num):
                    out_file.write(seq[60 * i: 60 * (i + 1)] + "\n")
                    
            for s in range(1, num_samples + 1):
                mkv_seq = protein_markovgen(seq, markov_order)
                assert len(mkv_seq) == len(seq)
                out_file.write(f">{ID}_mkv{s}\n")
                for i in range(line_num):
                    out_file.write(f"{mkv_seq[60*i:60*(i+1)]}\n")
    out_file.close()            




def fasta_genshufpartly(fasta_url, output_url, percentage=0.2, seed=None):

    out_records = []

    for record in SeqIO.parse(fasta_url, "fasta"):
        seq_str = str(record.seq)
        L = len(seq_str)

        k = int(round(L * percentage))
        if k <= 1:
            out_seq = seq_str
        else:
            rec_seed = f"{seed}|{record.id}" if seed is not None else None
            rng = random.Random(rec_seed)

            idx = list(range(L))
            chosen = rng.sample(idx, k)
            chosen.sort()

            aa = [seq_str[i] for i in chosen]
            rng.shuffle(aa)

            seq_list = list(seq_str)
            for i, new_aa in zip(chosen, aa):
                seq_list[i] = new_aa
            out_seq = "".join(seq_list)

        out_records.append(
            SeqRecord(
                Seq(out_seq),
                id=record.id + "_shuf",
                description="",
            )
        )

    SeqIO.write(out_records, output_url, "fasta")
    print(f"Saved partially shuffled sequences to {output_url}")



def fasta_copy(fasta_url, output_url):

    out_records = []

    for record in SeqIO.parse(fasta_url, "fasta"):
        seq = record.seq
        id = record.id + "_copy"
        out_records.append(
            SeqRecord(
                Seq(seq),
                id=id,
                description="",
            )
        )

    SeqIO.write(out_records, output_url, "fasta")
    print(f"Saved copied sequences to {output_url}")



if __name__ == "__main__":
    device = torch.device("cuda:2")

    ## Data reading
    file_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    base_name = 'astral_s500'
    input_fasta = os.path.join(file_dir, f'{base_name}.fa')

    # shuffle
    num_shuffles = 1
    output_fasta = os.path.join(file_dir, f'{base_name}_shuf.fa')
    multi_output_fasta = os.path.join(file_dir, f'{base_name}_shuf_multiN{num_shuffles}.fa')
    fasta_genshuf(input_fasta, output_fasta, include_origin=False, seed=123)
    print(f"Saved shuffled sequences to {output_fasta}")
    # multi_shuffle_onefile(input_fasta, multi_output_fasta, num_shuffles=num_shuffles, include_origin=False)
    # print(f"Saved shuffled sequences to {multi_output_fasta}")

    # # markov 1
    # num_samples = 100
    # mkv_order = 1
    # output_fasta_mar = os.path.join(file_dir, f'{base_name}_mkv.fa')
    # multi_output_fasta_mar = os.path.join(file_dir, f'{base_name}_mkv_multiN{num_samples}.fa')

    # fasta_genmarkov(input_fasta, output_fasta_mar, mkv_order, include_origin=False)
    # multi_markov_onefile(input_fasta, multi_output_fasta_mar, markov_order=2, num_samples=num_samples, include_origin=False)

    # # shuffle partly
    # output_fasta = os.path.join(file_dir, f'{base_name}_shufpartly.fa')
    # fasta_genshufpartly(input_fasta, output_fasta, percentage=0.2, seed=123)

    # # reverse 
    # output_fasta = os.path.join(file_dir, f'{base_name}_rev.fa')
    # fasta_genreverse(input_fasta, output_fasta)
    # print(f"Save to {output_fasta}")

    # # copy
    # output_fasta = os.path.join(file_dir, f'{base_name}_copy.fa')
    # fasta_copy(input_fasta, output_fasta)
    # print(f"Save to {output_fasta}")