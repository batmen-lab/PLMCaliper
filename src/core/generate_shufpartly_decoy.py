import os
import pandas as pd
from Bio import SeqIO
import random
from Bio.SeqRecord import SeqRecord

def partly_shuffle(seq, n=None, alpha=None):

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
    



if __name__ == "__main__":
    seed = 123
    random.seed(seed)
    n = 2
    alpha = 20
    SEQ_NAME = "astral"

    org_fa = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", f"{SEQ_NAME}.fa")

    records = []
    for record in SeqIO.parse(org_fa, "fasta"):
        decoy_seq = partly_shuffle(record.seq, n=n, alpha=alpha*0.01)
        decoy_id = record.id + "_shufpartly"
        records.append(SeqRecord(decoy_seq, id=decoy_id, description=""))

    output_file_fa = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", f"{SEQ_NAME}_shufpartly_{int(n*alpha)}.fa")
    SeqIO.write(records, output_file_fa, "fasta")
    print(f"[OK] Saved the decoys to {output_file_fa}")

