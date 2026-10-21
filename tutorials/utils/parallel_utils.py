from Bio import SeqIO
import os, argparse

DEFAULT_BATCH_SIZE = 500


def batch_filename(fasta_path, batch_size, index):
    prefix = os.path.basename(fasta_path).split('.fa')[0]
    return f"{prefix}_batch{batch_size}_{index}.fa"


def split_fasta(fasta_path, output_dir=None, batch_size=DEFAULT_BATCH_SIZE):
    """Shard a query FASTA into batches that can be searched in parallel and resumed."""
    assert os.path.exists(fasta_path), f"File not found: {fasta_path}"

    output_prefix = os.path.basename(fasta_path).split('.fa')[0]
    save_dir = output_dir if output_dir else os.path.join(os.path.dirname(fasta_path), f"{output_prefix}_batch")
    os.makedirs(save_dir, exist_ok=True)

    records = list(SeqIO.parse(fasta_path, "fasta"))
    total = len(records)
    num_batches = (total + batch_size - 1) // batch_size

    written = []
    for i in range(num_batches):
        batch_records = records[i * batch_size: min((i + 1) * batch_size, total)]
        output_file = os.path.join(save_dir, batch_filename(fasta_path, batch_size, i))
        with open(output_file, "w") as out_handle:
            SeqIO.write(batch_records, out_handle, "fasta")
        print(f"Wrote batch {i}: {output_file} ({len(batch_records)} records)")
        written.append(output_file)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shard a query FASTA into parallel batches.")
    parser.add_argument("--fasta_path", type=str, help="Path to the input FASTA file.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for the split files.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Sequences per batch.")
    args = parser.parse_args()

    split_fasta(args.fasta_path, output_dir=args.output_dir, batch_size=args.batch_size)
