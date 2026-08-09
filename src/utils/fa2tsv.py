from Bio import SeqIO
import pandas as pd
import numpy as np



def fa2tsv(fa_path, tsv_path):
    with open(tsv_path, "w") as fout:
        for record in SeqIO.parse(fa_path, "fasta"):
            seq_id = record.id
            seq = str(record.seq)
            fout.write(f"{seq_id}\t{seq}\n")
    
    print(f"[OK] Saved to {tsv_path}")



if __name__ == "__main__":
    for method in ["extended_shuf"]:
        DATA_DIR = "/home/yfnyang/2025_yifan_plmfool/data"
        seq_name = "class_all"
        fa_dir = f"{DATA_DIR}/{seq_name}_{method}.fa"
        tsv_dir = f"{DATA_DIR}/{seq_name}_{method}.tsv"
        
        fa2tsv(fa_dir, tsv_dir)