import os
import argparse
import numpy as np
from Bio import SeqIO
from tqdm import tqdm

def recover_dplm(dplm_url, output_url):
    assert os.path.exists(dplm_url)
    
    records = list(SeqIO.parse(dplm_url, "fasta"))
    dplm = {}
    for record in tqdm(records):
        ID = record.id
        dplm[ID] = record.seq.upper()
        
    out_file = open(output_url, "w")
    for key in parse_astral_key():
        if key in dplm:
            out_file.write(">{}\n{}\n".format(key, dplm[key]))
        else:
            print(f"Warning: {key} not found in DPLM data.")
    out_file.close()


def parse_astral_key(fasta_url="../data/astral.fa"):
    astral_key = []
    for record in SeqIO.parse(fasta_url, "fasta"):
        seq_id = record.id + "_dplm"
        astral_key.append(seq_id)
    return astral_key
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FASTA to TSV format.")
    parser.add_argument("--fasta_url", type=str, help="Path to the input FASTA file.")
    parser.add_argument("--output_url", type=str, help="Path to the output TSV file.")
    args = parser.parse_args()

    recover_dplm(args.fasta_url, args.output_url)