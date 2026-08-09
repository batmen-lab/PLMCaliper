import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

from Bio import SeqIO
from tqdm import tqdm


# ---------------------------------------------------------------------------
# SCOP label helpers
# ---------------------------------------------------------------------------

def parse_scop_label(description: str):
    parts = description.split()
    if len(parts) < 2:
        return None
    scop_str = parts[1]
    arr = scop_str.split(".")
    if len(arr) != 4:
        return None
    fold = ".".join(arr[:2])
    supf = ".".join(arr[:3])
    fam  = ".".join(arr[:4])
    return fold, supf, fam


def build_label_map(ref_fa: str) -> dict:
    label_map = {}
    for rec in SeqIO.parse(ref_fa, "fasta"):
        labels = parse_scop_label(rec.description)
        if labels is not None:
            label_map[rec.id] = labels
    return label_map


# ---------------------------------------------------------------------------
# Core filter
# ---------------------------------------------------------------------------

def compute_kept_ids(label_map: dict) -> set:
    # Group IDs by each level
    by_fold  = defaultdict(set)   # fold  -> {ids}
    by_supf  = defaultdict(set)   # supf  -> {ids}
    by_fam   = defaultdict(set)   # fam   -> {ids}

    for sid, (fold, supf, fam) in label_map.items():
        by_fold[fold].add(sid)
        by_supf[supf].add(sid)
        by_fam[fam].add(sid)

    kept = set()
    for sid, (fold, supf, fam) in tqdm(label_map.items(), desc="Filtering"):
        # 1. family-level homolog: same fam, different id
        if len(by_fam[fam]) < 2:
            continue

        # 2. superfamily-level homolog: same supf, different fam
        # i.e. some member of by_supf[supf] belongs to a different family
        has_supf_homolog = any(
            label_map[t][2] != fam
            for t in by_supf[supf]
        )
        if not has_supf_homolog:
            continue

        # 3. fold-level homolog: same fold, different supf
        has_fold_homolog = any(
            label_map[t][1] != supf
            for t in by_fold[fold]
        )
        if not has_fold_homolog:
            continue

        kept.add(sid)

    return kept


# ---------------------------------------------------------------------------
# FASTA / TSV I/O
# ---------------------------------------------------------------------------

DECOY_SUFFIXES = [
    "_extended_shuf",
    "_shuf",
    "_rev",
    "_mkv1",
    "_mkv2",
    "_dplm",
    "_PGmsk25pct",
    "_adp",
    "_copy",
    "_dup",
]


def get_origin_id(seq_id: str, decoy_suffix: str | None = None) -> str:
    if decoy_suffix is not None:
        if seq_id.endswith(decoy_suffix):
            return seq_id[: -len(decoy_suffix)]
        return seq_id

    for suf in sorted(DECOY_SUFFIXES, key=len, reverse=True):
        if seq_id.endswith(suf):
            return seq_id[: -len(suf)]
    return seq_id


def filter_fasta(in_fa: str, out_fa: str, kept_ids: set,
                 decoy_suffix: str | None = None) -> int:
    written = 0
    with open(out_fa, "w") as fh:
        for rec in tqdm(SeqIO.parse(in_fa, "fasta"), desc=f"  filtering {Path(in_fa).name}"):
            origin = get_origin_id(rec.id, decoy_suffix)
            if origin in kept_ids:
                SeqIO.write(rec, fh, "fasta")
                written += 1
    return written


def filter_tsv(in_tsv: str, out_tsv: str, kept_ids: set,
               decoy_suffix: str | None = None) -> int:
    written = 0
    with open(in_tsv) as fin, open(out_tsv, "w") as fout:
        for line in tqdm(fin, desc=f"  filtering {Path(in_tsv).name}"):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            seq_id = parts[0]
            origin = get_origin_id(seq_id, decoy_suffix)
            if origin in kept_ids:
                fout.write(line + "\n")
                written += 1
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref-fa", required=True,
                   help="Reference FASTA with SCOP labels (e.g. astral.fa); used to build the label map.")
    p.add_argument("--in-fa",  required=True,
                   help="Input FASTA to filter (may be the same as --ref-fa or a decoy .fa).")
    p.add_argument("--out-fa", required=True,
                   help="Output filtered FASTA.")
    p.add_argument("--in-tsv",  default=None,
                   help="Optional input TSV (id<TAB>seq) to filter.")
    p.add_argument("--out-tsv", default=None,
                   help="Output filtered TSV (required if --in-tsv is given).")
    p.add_argument("--decoy-suffix", default=None,
                   help="Explicit decoy suffix to strip from IDs (e.g. '_shuf'). "
                        "If omitted, auto-detected from DECOY_SUFFIXES list.")
    p.add_argument("--kept-ids-out", default=None,
                   help="Optional path to write the kept ID list (one per line).")
    args = p.parse_args()

    print(f"[filter_foldseek] Building label map from: {args.ref_fa}")
    label_map = build_label_map(args.ref_fa)
    print(f"  {len(label_map)} sequences with valid SCOP labels")

    print("[filter_foldseek] Computing kept IDs ...")
    kept = compute_kept_ids(label_map)
    print(f"  {len(kept)} / {len(label_map)} queries pass the three-level filter")

    if args.kept_ids_out:
        Path(args.kept_ids_out).write_text("\n".join(sorted(kept)) + "\n")
        print(f"  Kept IDs written to: {args.kept_ids_out}")

    print(f"[filter_foldseek] Filtering FASTA: {args.in_fa} -> {args.out_fa}")
    n_fa = filter_fasta(args.in_fa, args.out_fa, kept, args.decoy_suffix)
    print(f"  Written: {n_fa} sequences")

    if args.in_tsv:
        if not args.out_tsv:
            p.error("--out-tsv is required when --in-tsv is provided")
        print(f"[filter_foldseek] Filtering TSV: {args.in_tsv} -> {args.out_tsv}")
        n_tsv = filter_tsv(args.in_tsv, args.out_tsv, kept, args.decoy_suffix)
        print(f"  Written: {n_tsv} rows")

    print("[filter_foldseek] Done.")


if __name__ == "__main__":
    main()
