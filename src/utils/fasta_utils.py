import os, sys, argparse, math, logging, subprocess, random
from collections import defaultdict
sys.path.append('..')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
from Bio import SeqIO
from tqdm import tqdm

# from pyvolve import *

def fasta_to_upper(fasta_url, output_url):
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))
    for record in tqdm(records):
        ID = record.id
        seq = record.seq.upper()
        desc = record.description
        # print('ID={}\tdesc={}'.format(ID, desc))

        out_file.write(">{}\n".format(desc))
        line_num = math.ceil(len(seq) / 60)
        for i in range(line_num):
            out_file.write("{}\n".format(seq[60 * i: 60 * (i + 1)]))

    out_file.close()

def fasta_filter_by_keyword(fasta_url, output_url, keyword):
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))
    for record in tqdm(records):
        ID = record.id
        seq = record.seq.upper()
        if keyword in ID: continue

        out_file.write(">{}\n".format(ID))
        line_num = math.ceil(len(seq) / 60)
        for i in range(line_num):
            out_file.write("{}\n".format(seq[60 * i: 60 * (i + 1)]))

    out_file.close()

def fasta_to_tsv(fasta_url, output_url):
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))
    for record in tqdm(records):
        ID = record.id
        seq = record.seq.upper()
        # desc = record.description
        # print('ID={}\tdesc={}'.format(ID, desc))
        out_file.write("{}\t{}\n".format(ID, seq))

    out_file.close()

def fasta_genrev(fasta_url, output_url, include_origin=False):
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))
    for record in tqdm(records):
        ID = record.id
        seq = record.seq
        desc = record.description
        # print('ID={}\tdesc={}'.format(ID, desc))

        seq_list = list(seq)
        rev_seq = "".join(seq_list[::-1])
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

def fasta_genshuf(fasta_url, output_url, include_origin=False):
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))

    rng = np.random.default_rng(0)
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

def fasta_genmarkov(fasta_url, output_url, markov_order, include_origin=False):
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))

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


def fasta_trunclen(fasta_url, output_url, ratio, replicate=1):
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))
    percent = int(ratio * 100)

    rng = np.random.default_rng(0)
    for record in tqdm(records):
        ID = record.id
        seq = record.seq.upper()
        desc = record.description

        seqlen = len(seq)
        reduced_seqlen = int(seqlen * ratio)
        start_indices = np.random.randint(0, (seqlen - reduced_seqlen - 1), replicate)
    
        for idx in range(replicate):
            start_idx = start_indices[idx]
            reduced_seq = seq[start_idx: (start_idx+reduced_seqlen)]
            out_file.write(">{}_trunc_{}pct_rep{}\n".format(ID, percent, idx))

            line_num = math.ceil(len(reduced_seq) / 60)
            for i in range(line_num):
                out_file.write("{}\n".format(reduced_seq[60 * i: 60 * (i + 1)]))

    out_file.close()



def fasta_doublen(fasta_url, output_url, shuffle=True):
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))

    rng = np.random.default_rng(0)
    for record in tqdm(records):
        ID = record.id
        seq = record.seq.upper()
        desc = record.description

        seq_list = list(seq)
        if shuffle: rng.shuffle(seq_list)
        append_seq = "".join(seq_list)
        concat_seq = seq + append_seq

        out_file.write(">{}_doub_shuf{}\n".format(ID, int(shuffle)))
        line_num = math.ceil(len(concat_seq) / 60)
        for i in range(line_num):
            out_file.write("{}\n".format(concat_seq[60 * i: 60 * (i + 1)]))

    out_file.close()


def fasta_mutation(fasta_url, output_url, model='WAG'):
    import pyvolve
    tree = pyvolve.read_tree(tree="(((t1:0.36,t2:0.45):0.001,t3:0.77):0.44,(t5:0.77,t4:0.41):0.89);", scale_tree=2.0)
    ### root > t1 > t2 > t3 > t4 > t5
    assert model in ['JTT', 'WAG', 'LG', 'DAYHOFF', 'MTMAM', 'MTREV24']
    m = pyvolve.Model(model)

    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))

    for record in tqdm(records):
        ID = record.id
        seq = record.seq.upper()
        desc = record.description

        if 'B' in seq: seq = seq.replace('B', '')
        if 'J' in seq: seq = seq.replace('J', '')
        if 'Z' in seq: seq = seq.replace('Z', '')
        if 'X' in seq: seq = seq.replace('X', '')
        if 'U' in seq: seq = seq.replace('U', '')
        if 'O' in seq: seq = seq.replace('O', '')

        p = pyvolve.Partition(root_sequence=str(seq), models=m)
        evolve = pyvolve.Evolver(partitions=p, tree=tree)
        evolve(ratefile=False, infofile=False, seqfile=False)
        seqdict = evolve.get_sequences(anc=True)

        for key in seqdict:
            if 'internal' in key or 'root' in key: continue

            sim_seq = seqdict[key]
            out_file.write(">{}_mut_{}\n".format(ID, key))
            line_num = math.ceil(len(sim_seq) / 60)
            for i in range(line_num):
                out_file.write("{}\n".format(sim_seq[60 * i: 60 * (i + 1)]))

    out_file.close()


def get_ID_Seq_info(fasta_url):
    assert os.path.exists(fasta_url)
    pid_seq_map = {}
    records = list(SeqIO.parse(fasta_url, "fasta"))
    for record in tqdm(records):
        ID = record.id
        seq = record.seq
        assert ID not in pid_seq_map
        pid_seq_map[ID] = seq
    return pid_seq_map

def fasta_gen_batch(fasta_url, output_dir, batch_size):
    assert os.path.exists(fasta_url)
    if os.path.exists(output_dir): subprocess.run(["rm", "-r", output_dir])
    os.mkdir(output_dir)

    seq_cnt = 0
    curr_batch_idx = 0
    out_file = open(os.path.join(output_dir, os.path.basename(fasta_url).replace('.fa','_batch{}_{}.fa'.format(batch_size, curr_batch_idx))), "w")

    records = list(SeqIO.parse(fasta_url, "fasta"))
    for record in tqdm(records):
        ID = record.id
        seq = record.seq
        # desc = record.description
        seq_cnt += 1

        if (seq_cnt % batch_size) == 0:
            if out_file != None: out_file.close()
            curr_batch_idx += 1
            out_file = open(os.path.join(output_dir, os.path.basename(fasta_url).replace('.fa','_batch{}_{}.fa'.format(batch_size, curr_batch_idx))), "w")

        out_file.write(">{}\n".format(ID))
        line_num = math.ceil(len(seq) / 60)
        for i in range(line_num):
            out_file.write("{}\n".format(seq[60 * i: 60 * (i + 1)]))

    if out_file != None: out_file.close()

def fasta_gen_shuf_batch(fasta_url, output_dir, batch_size, shuf_size):
    assert os.path.exists(fasta_url)
    if os.path.exists(output_dir): subprocess.run(["rm", "-r", output_dir])
    os.mkdir(output_dir)

    seq_cnt = 0
    curr_batch_idx = 0
    out_file = open(os.path.join(output_dir, os.path.basename(fasta_url).replace('.fa','_batch{}_{}.fa'.format(batch_size, curr_batch_idx))), "w")

    rng = np.random.default_rng(0)
    records = list(SeqIO.parse(fasta_url, "fasta"))
    for record in tqdm(records):
        ID = record.id
        seq = record.seq
        desc = record.description
        seq_cnt += 1

        if (seq_cnt % batch_size) == 0:
            if out_file != None: out_file.close()
            curr_batch_idx += 1
            out_file = open(os.path.join(output_dir, os.path.basename(fasta_url).replace('.fa','_batch{}_{}.fa'.format(batch_size, curr_batch_idx))), "w")

        for shuf_idx in range(shuf_size):
            seq_list = list(seq)
            rng.shuffle(seq_list)
            shuf_seq = "".join(seq_list)
            assert len(seq) == len(shuf_seq)

            out_file.write(">{}_shuf{}\n".format(ID, shuf_idx))
            line_num = math.ceil(len(shuf_seq) / 60)
            for i in range(line_num):
                out_file.write("{}\n".format(shuf_seq[60 * i: 60 * (i + 1)]))

    if out_file != None: out_file.close()


def main(args):
    assert os.path.exists(args.fasta_url)
    fa_url_list = []
    fa_url_list.append(args.fasta_url)

    ### generate rev sequence (Sanity check 1)
    output_url = args.fasta_url.replace('.fa', '_rev.fa')
    fa_url_list.append(output_url)
    # fasta_genrev(args.fasta_url, output_url)
    # fasta_to_tsv(output_url, output_url.replace('.fa', '.tsv'))

    ### generate shuf sequence (Sanity check 1)
    output_url = args.fasta_url.replace('.fa', '_shuf.fa')
    fa_url_list.append(output_url)
    # fasta_genshuf(args.fasta_url, output_url)
    # fasta_to_tsv(output_url, output_url.replace('.fa', '.tsv'))

    ## generate markov sequence (Sanity check 1)
    for markov_order in [1, 2]:
        output_url = args.fasta_url.replace('.fa', '_mkv{}.fa'.format(markov_order))
        fa_url_list.append(output_url)
    #     fasta_genmarkov(args.fasta_url, output_url, markov_order)
    #     fasta_to_tsv(output_url, output_url.replace('.fa', '.tsv'))

    ### generate sequence with redundancy (Sanity check 2)
    output_url = args.fasta_url.replace('.fa', '_doubshuf.fa')
    fa_url_list.append(output_url)
    # fasta_doublen(args.fasta_url, output_url, shuffle=True)
    # fasta_to_tsv(output_url, output_url.replace('.fa', '.tsv'))
    #
    output_url = args.fasta_url.replace('.fa', '_doubself.fa')
    fa_url_list.append(output_url)
    # fasta_doublen(args.fasta_url, output_url, shuffle=False)
    # fasta_to_tsv(output_url, output_url.replace('.fa', '.tsv'))

    ### generate mutant sequences (Sanity check 3)
    output_url = args.fasta_url.replace('.fa', '_mutantWAG.fa')
    fa_url_list.append(output_url)
    # fasta_mutation(args.fasta_url, output_url, model='WAG')
    # fasta_to_tsv(output_url, output_url.replace('.fa', '.tsv'))
    # 
    output_url = args.fasta_url.replace('.fa', '_mutantJTT.fa')
    fa_url_list.append(output_url)
    # fasta_mutation(args.fasta_url, output_url, model='JTT')
    # fasta_to_tsv(output_url, output_url.replace('.fa', '.tsv'))
    
    ### generate truncated sequences (Sanity check 6)
    output_url = args.fasta_url.replace('.fa', '_truncqrt.fa')
    fa_url_list.append(output_url)
    # fasta_trunclen(args.fasta_url, output_url, ratio=0.25, replicate=1)
    # fasta_to_tsv(output_url, output_url.replace('.fa', '.tsv'))
    
    output_url = args.fasta_url.replace('.fa', '_trunchalf.fa')
    fa_url_list.append(output_url)
    # fasta_trunclen(args.fasta_url, output_url, ratio=0.5, replicate=1)
    # fasta_to_tsv(output_url, output_url.replace('.fa', '.tsv'))
    
    

    for fa_url in fa_url_list:
        output_dir = os.path.dirname(fa_url)
        output_dir = os.path.join(output_dir, os.path.basename(fa_url).replace('.fa','_batch'))
        print('fa_url={}'.format(fa_url))
        fasta_gen_batch(fa_url, output_dir, batch_size=500)

        # output_dir = os.path.dirname(fa_url)
        # output_dir = os.path.join(output_dir, os.path.basename(fa_url).replace('.fa','_shuf_batch'))
        # fasta_gen_shuf_batch(fa_url, output_dir, batch_size=500, shuf_size=50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--fasta_url', type=str, help='fasta_url')
    # parser.add_argument('--output_url', type=str, help='output_url')
    # parser.add_argument('--output_dir', type=str, help='output_dir')
    main(parser.parse_args())

### command
# python utils/fasta_utils.py --fasta_url ../data/astral.fa
# python utils/fasta_utils.py --fasta_url ../data/astral.fa --output_url ../data/astral_rev.fa --output_dir ../data/astral_rev_batch
