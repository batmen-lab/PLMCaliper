from Bio import SeqIO
import os, argparse

def split_fasta(fasta_path, output_dir=None, batch_size=500):
    assert os.path.exists(fasta_path), f"File not found: {fasta_path}"

    output_prefix = os.path.basename(fasta_path).split('.fa')[0]
    save_dir = output_dir if output_dir else os.path.join(os.path.dirname(fasta_path), f"{output_prefix}_batch")
    os.makedirs(save_dir, exist_ok=True)
    
    records = list(SeqIO.parse(fasta_path, "fasta"))
    total = len(records)
    num_batches = (total + batch_size - 1) // batch_size  # ceiling division

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, total)
        batch_records = records[start:end]
        output_file = os.path.join(save_dir, f"{output_prefix}_batch{batch_size}_{i}.fa")
        with open(output_file, "w") as out_handle:
            SeqIO.write(batch_records, out_handle, "fasta")
        print(f"Wrote batch {i}: {output_file} ({len(batch_records)} records)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split fasta to batches.")
    parser.add_argument("--fasta_path", type=str, help="Path to the input FASTA file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for the split files.")
    args = parser.parse_args()

    split_fasta(args.fasta_path, output_dir=args.output_dir)

# Command to run:
# python utils/split_fasta_batches.py --fasta_path ../data/astral_dplm.fa --output_dir ../data/split