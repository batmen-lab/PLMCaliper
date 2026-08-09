import os
import argparse
import numpy as np
from Bio import SeqIO
from tqdm import tqdm



def fasta_to_tsv(fasta_url, output_url):
    assert os.path.exists(fasta_url), fasta_url
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))
    for record in tqdm(records):
        ID = record.id
        seq = record.seq.upper()
        # desc = record.description
        # print('ID={}\tdesc={}'.format(ID, desc))
        out_file.write("{}\t{}\n".format(ID, seq))

    out_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FASTA to TSV format.")
    parser.add_argument("--fasta_url", type=str, help="Path to the input FASTA file.")
    parser.add_argument("--output_url", type=str, help="Path to the output TSV file.")
    args = parser.parse_args()
    
    fasta_to_tsv(args.fasta_url, args.output_url)


# python utils/fasta_to_tsv.py --fasta_url ../data/astra_dplm.fa --output_url ../data/astra_dplm.tsv