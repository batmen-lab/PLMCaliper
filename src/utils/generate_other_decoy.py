import argparse
import math
import os
import random
from collections import defaultdict

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm


def protein_markovgen(sequence: str, k: int, on_short: str = "raise") -> str:
    """Resample a sequence from an order-k Markov chain fitted on itself."""
    seq_len = len(sequence)
    if seq_len < k + 1:
        if on_short == "shuffle":
            seq_list = list(sequence)
            random.shuffle(seq_list)
            return ''.join(seq_list)
        raise ValueError("Input sequence is too short for the specified Markov order.")

    model = defaultdict(lambda: defaultdict(int))
    for i in range(seq_len - k):
        prefix = sequence[i:(i + k)]
        next_char = sequence[i + k]
        model[prefix][next_char] += 1

    markov_model = {}
    for prefix, suffix_counts in model.items():
        total = sum(suffix_counts.values())
        chars, probs = [], []
        for char, count in suffix_counts.items():
            chars.append(char)
            probs.append(count * 1.0 / total)
        markov_model[prefix] = (chars, probs)

    start = random.choice(list(markov_model.keys()))
    result = list(start)
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


def partly_shuffle(seq, n=1, alpha=0.2):
    """Shuffle a random window covering `alpha` of the sequence, `n` times."""
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


def _write_wrapped(out_file, seq, width=60):
    for i in range(math.ceil(len(seq) / width)):
        out_file.write("{}\n".format(seq[width * i: width * (i + 1)]))


def generate_duplicate_decoy(fasta_path, output_path):
    """seq + seq, id suffix `_dup`."""
    records = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = record.seq
        record.seq = seq + seq
        record.id = record.id + "_dup"
        record.description = ""
        records.append(record)

    SeqIO.write(records, output_path, "fasta")


def generate_shuf_decoy(fasta_path, output_path, seed=0):
    """`extended_shuf`: seq + shuffle(seq), id suffix `_shuf`. The paper default."""
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
    """seq + reverse(seq), id suffix `_rev`."""
    records = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq)
        record.seq = Seq(seq + seq[::-1])
        record.id = record.id + "_rev"
        record.description = ""
        records.append(record)

    SeqIO.write(records, output_path, "fasta")


def generate_extend_then_shuffle_decoy(fasta_path, output_path, seed=0):
    """shuffle(seq + seq) -- the whole doubled sequence is shuffled. id suffix `_shuf`."""
    random.seed(seed)
    records = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq)
        seq_list = list(seq + seq)
        random.shuffle(seq_list)

        record.seq = Seq("".join(seq_list))
        record.id = record.id + "_shuf"
        record.description = ""
        records.append(record)

    SeqIO.write(records, output_path, "fasta")


def generate_extended_markov_decoy(fasta_path, output_path, markov_order=1, seed=0):
    """`extended_mkv{k}`: seq + markov_k(seq), id suffix `_mkv`. PLMCaliper's default (k=2)."""
    random.seed(seed)

    final_records = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper()
        mkv_seq = protein_markovgen(seq, markov_order, on_short="shuffle")

        final_records.append(SeqRecord(
            Seq(seq + mkv_seq),
            id=record.id + "_mkv",
            description="",
        ))

    SeqIO.write(final_records, output_path, "fasta")
    print(f"[OK] Successfully generated A+A_mkv database. Total records: {len(final_records)}")


def generate_partly_shuffle_decoy(fasta_path, output_path, n=1, alpha=0.2, seed=0):
    """seq + partly_shuffle(seq), id suffix `_shufpartly`."""
    random.seed(seed)
    records = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq)
        record.seq = Seq(seq + partly_shuffle(seq, n=n, alpha=alpha))
        record.id = record.id + "_shufpartly"
        record.description = ""
        records.append(record)

    SeqIO.write(records, output_path, "fasta")


def generate_shufpartly_only_decoy(fasta_path, output_path, n=2, alpha=0.2, seed=123):
    """partly_shuffle(seq) without the original prefix, id suffix `_shufpartly`."""
    random.seed(seed)
    records = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        decoy_seq = partly_shuffle(record.seq, n=n, alpha=alpha)
        records.append(SeqRecord(Seq(decoy_seq), id=record.id + "_shufpartly", description=""))

    SeqIO.write(records, output_path, "fasta")
    print(f"[OK] Saved the decoys to {output_path}")


def fasta_genshuf(fasta_url, output_url, include_origin=False, seed=None):
    """shuffle(seq), id suffix `_shuf`."""
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))

    rng = np.random.default_rng(seed=seed)
    for record in tqdm(records):
        ID = record.id
        seq = record.seq

        seq_list = list(seq)
        rng.shuffle(seq_list)
        shuf_seq = "".join(seq_list)
        assert len(seq) == len(shuf_seq)

        if include_origin:
            out_file.write(">{}\n".format(ID))
            _write_wrapped(out_file, seq)

        out_file.write(">{}_shuf\n".format(ID))
        _write_wrapped(out_file, shuf_seq)

    out_file.close()


def fasta_genreverse(fasta_url, output_url, include_origin=False):
    """reverse(seq), id suffix `_rev`."""
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))

    for record in tqdm(records):
        ID = record.id
        seq = record.seq

        rev_seq = str(seq)[::-1]
        assert len(seq) == len(rev_seq)

        if include_origin:
            out_file.write(">{}\n".format(ID))
            _write_wrapped(out_file, seq)

        out_file.write(">{}_rev\n".format(ID))
        _write_wrapped(out_file, rev_seq)

    out_file.close()


def fasta_genmarkov(fasta_url, output_url, markov_order, include_origin=False, seed=None):
    """markov_k(seq), id suffix `_mkv{k}`. NOTE: `seed` has never had any effect --
    the generator draws from the global `random`. Kept so existing outputs reproduce."""
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))

    for record in tqdm(records):
        ID = record.id
        seq = record.seq

        mkv_seq = protein_markovgen(str(seq), markov_order, on_short="raise")
        assert len(seq) == len(mkv_seq)

        if include_origin:
            out_file.write(">{}\n".format(ID))
            _write_wrapped(out_file, seq)

        out_file.write(">{}_mkv{}\n".format(ID, markov_order))
        _write_wrapped(out_file, mkv_seq)

    out_file.close()


def fasta_genshufpartly(fasta_url, output_url, percentage=0.2, seed=None):
    """Shuffle residues at a random `percentage` of positions (scattered, not a window)."""
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

            chosen = sorted(rng.sample(list(range(L)), k))
            aa = [seq_str[i] for i in chosen]
            rng.shuffle(aa)

            seq_list = list(seq_str)
            for i, new_aa in zip(chosen, aa):
                seq_list[i] = new_aa
            out_seq = "".join(seq_list)

        out_records.append(SeqRecord(Seq(out_seq), id=record.id + "_shuf", description=""))

    SeqIO.write(out_records, output_url, "fasta")
    print(f"Saved partially shuffled sequences to {output_url}")


def fasta_copy(fasta_url, output_url):
    """Verbatim copy with id suffix `_copy` -- the positive control decoy."""
    out_records = []

    for record in SeqIO.parse(fasta_url, "fasta"):
        out_records.append(SeqRecord(Seq(record.seq), id=record.id + "_copy", description=""))

    SeqIO.write(out_records, output_url, "fasta")
    print(f"Saved copied sequences to {output_url}")


def multi_shuffle_onefile(fasta_url, output_url, num_shuffles=5, include_origin=False):
    """`_shuf1` .. `_shuf{N}` per sequence."""
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))

    np.random.seed(0)
    random_seeds = np.random.randint(0, 1000000, len(records))

    for i, record in enumerate(tqdm(records)):
        rng = np.random.default_rng(random_seeds[i])
        ID = record.id
        seq = record.seq

        if include_origin:
            out_file.write(">{}\n".format(ID))
            _write_wrapped(out_file, seq)

        for s in range(1, num_shuffles + 1):
            seq_list = list(seq)
            rng.shuffle(seq_list)
            shuf_seq = "".join(seq_list)
            out_file.write(f">{ID}_shuf{s}\n")
            _write_wrapped(out_file, shuf_seq)
            assert len(seq) == len(shuf_seq)

    out_file.close()


def multi_markov_onefile(fasta_url, output_url, markov_order=2, num_samples=5,
                         include_origin=False, seed=None):
    """`_mkv1` .. `_mkv{N}` per sequence."""
    assert os.path.exists(fasta_url)
    records = list(SeqIO.parse(fasta_url, "fasta"))

    np.random.seed(0)
    random_seeds = np.random.randint(0, 1000000, len(records))

    with open(output_url, "w") as out_file:
        for i, rec in enumerate(tqdm(records)):
            random.seed(int(random_seeds[i]))

            ID = rec.id
            seq = str(rec.seq)

            if include_origin:
                out_file.write(f">{ID}\n")
                _write_wrapped(out_file, seq)

            for s in range(1, num_samples + 1):
                mkv_seq = protein_markovgen(seq, markov_order, on_short="raise")
                assert len(mkv_seq) == len(seq)
                out_file.write(f">{ID}_mkv{s}\n")
                _write_wrapped(out_file, mkv_seq)


# name -> (builder, default output suffix)
DECOY_BUILDERS = {
    "extended_shuf":     (lambda fa, out, a: generate_shuf_decoy(fa, out, a.seed), "_extended_shuf"),
    "extended_mkv":      (lambda fa, out, a: generate_extended_markov_decoy(fa, out, a.markov_order, a.seed),
                          "_extended_mkv"),
    "extended_rev":      (lambda fa, out, a: generate_reverse_decoy(fa, out), "_extended_rev"),
    "extended_dup":      (lambda fa, out, a: generate_duplicate_decoy(fa, out), "_extended_dup"),
    "extend_then_shuf":  (lambda fa, out, a: generate_extend_then_shuffle_decoy(fa, out, a.seed),
                          "_extended_then_shuf"),
    "extended_shufpartly": (lambda fa, out, a: generate_partly_shuffle_decoy(fa, out, a.n, a.alpha, a.seed),
                            "_extended_shufpartly"),
    "shuf":              (lambda fa, out, a: fasta_genshuf(fa, out, seed=a.seed), "_shuf"),
    "rev":               (lambda fa, out, a: fasta_genreverse(fa, out), "_rev"),
    "mkv":               (lambda fa, out, a: fasta_genmarkov(fa, out, a.markov_order), "_mkv"),
    "shufpartly":        (lambda fa, out, a: generate_shufpartly_only_decoy(fa, out, a.n, a.alpha, a.seed),
                          "_shufpartly"),
    "shufpartly_pos":    (lambda fa, out, a: fasta_genshufpartly(fa, out, a.alpha, a.seed), "_shufpartly_pos"),
    "copy":              (lambda fa, out, a: fasta_copy(fa, out), "_copy"),
    "multi_shuf":        (lambda fa, out, a: multi_shuffle_onefile(fa, out, a.num_reps), "_shuf_multiN"),
    "multi_mkv":         (lambda fa, out, a: multi_markov_onefile(fa, out, a.markov_order, a.num_reps), "_mkv_multiN"),
}


def main():
    ap = argparse.ArgumentParser(description="Build decoy sequences from a FASTA.")
    ap.add_argument("--fasta", required=True, help="input FASTA (e.g. data/astral.fa)")
    ap.add_argument("--decoy-type", default="extended_shuf", choices=sorted(DECOY_BUILDERS),
                    help="which decoy to build (default: extended_shuf)")
    ap.add_argument("--out", default=None, help="output FASTA (default: <fasta stem><suffix>.fa)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--markov-order", type=int, default=2, help="k for the Markov variants")
    ap.add_argument("--n", type=int, default=2, help="window-shuffle repeats (shufpartly)")
    ap.add_argument("--alpha", type=float, default=0.2, help="fraction shuffled (shufpartly)")
    ap.add_argument("--num-reps", type=int, default=5, help="decoys per sequence (multi_* types)")
    args = ap.parse_args()

    builder, suffix = DECOY_BUILDERS[args.decoy_type]
    if args.decoy_type in ("extended_mkv", "mkv", "multi_mkv"):
        suffix = f"{suffix}{args.markov_order}"
    out = args.out or f"{os.path.splitext(args.fasta)[0]}{suffix}.fa"

    builder(args.fasta, out, args)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

