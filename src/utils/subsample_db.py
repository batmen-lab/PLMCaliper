import argparse
from Bio import SeqIO
from tqdm import tqdm
from collections import defaultdict
import random
import numpy as np
import os

def set_randseed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    
def get_seq_label_map(fasta_url):
    records = list(SeqIO.parse(fasta_url, "fasta"))

    fa_id_label_map = {}
    for record in tqdm(records):
        fa_id = record.id
        desc = record.description
        scop_label = desc.split(' ')[1]
        arr = scop_label.split('.')
        assert len(arr) == 4

        scop_fold = '.'.join(arr[:2])
        scop_supf = '.'.join(arr[:3])
        scop_fam = '.'.join(arr[:4])

        assert fa_id not in fa_id_label_map
        fa_id_label_map[fa_id] = (scop_fold, scop_supf, scop_fam)

    return fa_id_label_map

    
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


def subsample_db(db_path, num_seq):
    fasta_url = "../data/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
    fa_id_label_map = get_seq_label_map(fasta_url)
    supf_map = defaultdict(list)

    # Build superfamily → sequence ID mapping
    for fa_id, (_, supf, _) in fa_id_label_map.items():
        supf_map[supf].append(fa_id)
    
    selected_ids = set()
    supf_keys = list(supf_map.keys())
    random.shuffle(supf_keys)

    for supf in supf_keys:
        ids = supf_map[supf]
        selected_ids.update(ids)
        if len(selected_ids) >= num_seq:
            break

    print(f"Selected {len(selected_ids)} sequences from {len(supf_keys)} superfamilies.")
    # Save to output FASTA
    output_path = db_path.replace(".fa", f"_s{num_seq}.fa")
    with open(output_path, 'w') as out_f:
        for record in SeqIO.parse(db_path, "fasta"):
            if record.id in selected_ids:
                SeqIO.write(record, out_f, "fasta")

    fasta_to_tsv(output_path, output_path.replace('.fa', '.tsv'))
    print(f"Saved subsampled FASTA to: {output_path}")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Subsample a database to a specified number of sequences.")
    parser.add_argument("--db", help="Path to the database file.")
    parser.add_argument("--num_seq", type=int, help="Number of sequences to subsample.")
    
    args = parser.parse_args()
    
    set_randseed(seed=0)
    
    subsample_db(args.db, args.num_seq)
    
    
# Command:
# python utils/subsample_db.py --db ../data/astral.fa --num_seq 1000