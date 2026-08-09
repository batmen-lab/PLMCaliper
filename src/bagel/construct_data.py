import os
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import defaultdict
import pandas as pd



def load_fasta_to_dict_by_id(fasta_file, class_label):
    seq_dict = {}
    for rec in SeqIO.parse(fasta_file, "fasta"):
        seq_id = rec.id 
        seq_dict[seq_id] = class_label
    return seq_dict





def load_fasta(fa_file):
    seqs = []
    with open(fa_file, "r") as f:
        lines = f.read().split("\n")

    header = None
    seq = []

    for line in lines:
        if line.startswith(">"):
            if header:
                seqs.append((header, "".join(seq)))
            header = line.strip()
            seq = []
        else:
            seq.append(line.strip())

    if header:
        seqs.append((header, "".join(seq)))

    return seqs





def write_fasta(filename, entries):
    records = []

    for header, seq in entries:
        if header.startswith(">"):
            header = header[1:]

        record = SeqRecord(
            Seq(seq),
            id=header,
            description=""
        )
        records.append(record)

    SeqIO.write(records, filename, "fasta")
    print(f"Saved to {filename}")




def dedupe_headers(entries):
    seen = defaultdict(int)
    new_entries = []

    for header, seq in entries:
        clean_header = header[1:] if header.startswith(">") else header

        seen[clean_header] += 1

        if seen[clean_header] == 1:
            new_header = clean_header
        else:
            new_header = f"{clean_header}_dup{seen[clean_header]-1}"

        new_entries.append((new_header, seq))

    return new_entries





if __name__ == "__main__":

    # ==============================================================
    # 1. read bacteriocin class database
    # ==============================================================

    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
    bagel1 = os.path.join(DATA_PATH, "bagel4_class1bacteriocin_db.fasta")
    bagel2 = os.path.join(DATA_PATH, "bagel4_class2bacteriocin_db.fasta")
    bagel3 = os.path.join(DATA_PATH, "bagel4_class3bacteriocin_db.fasta")
    db_I = load_fasta_to_dict_by_id(bagel1, "ClassI")
    db_II = load_fasta_to_dict_by_id(bagel2, "ClassII")
    db_III = load_fasta_to_dict_by_id(bagel3, "ClassIII")

    print(" Class I:", len(db_I))
    print(" Class II:", len(db_II))
    print(" Class III:", len(db_III))

    # merge into one master dict
    master_db = {}
    master_db.update(db_I)
    master_db.update(db_II)
    master_db.update(db_III)

    df_map = pd.DataFrame(
        [{"ids": k, "class": v} for k, v in master_db.items()]
    )

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "bagel_class.txt")
    df_map.to_csv(out_path, sep="\t", index=False)
    print(f"[OK] Saved to {out_path}")

    # ============================================================== 
    # 2. concate bagel.fa
    # ==============================================================

    records_list = []
    records_1 = []
    records_2 = []
    records_3 = []
    for record in SeqIO.parse(bagel1, "fasta"):
        records_list.append(record)
        if "putative" not in record.id:
            records_1.append(record)
    for record in SeqIO.parse(bagel2, "fasta"):
        records_list.append(record)
        if "putative" not in record.id:
            records_2.append(record)
    for record in SeqIO.parse(bagel3, "fasta"):
        records_list.append(record)
        if "putative" not in record.id:
            records_3.append(record)


    records_putative = []
    records_non_putative = []
    for record in records_list:
        if "putative" in record.id:
            records_putative.append(record)
        else:
            records_non_putative.append(record)

    print(f"[INFO] Putative: {len(records_putative)}, Non-putative: {len(records_non_putative)}")
    print(f"[INFO] Class 1: {len(records_1)}, Class 2: {len(records_2)}, Class 3: {len(records_3)}")

    out_fa_putative = os.path.join(DATA_PATH, "putative.fa")
    SeqIO.write(records_putative, out_fa_putative, "fasta")
    print(f"[INFO] Saved to {out_fa_putative}")

    out_fa_non_putative = os.path.join(DATA_PATH, "class_all.fa")
    out_fa = os.path.join(DATA_PATH, "class_all.fa")
    SeqIO.write(records_non_putative, out_fa, "fasta")
    print(f"[INFO] Saved to {out_fa}")

    out_fa_1 = os.path.join(DATA_PATH, "class_1.fa")
    SeqIO.write(records_1, out_fa_1, "fasta")
    print(f"[INFO] Saved to {out_fa_1}")

    out_fa_2 = os.path.join(DATA_PATH, "class_2.fa")
    SeqIO.write(records_2, out_fa_2, "fasta")
    print(f"[INFO] Saved to {out_fa_2}")

    out_fa_3 = os.path.join(DATA_PATH, "class_3.fa")
    SeqIO.write(records_3, out_fa_3, "fasta")
    print(f"[INFO] Saved to {out_fa_3}")


    
