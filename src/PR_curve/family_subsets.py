from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

_here = os.path.dirname(os.path.realpath(__file__))
_utils_dir = os.path.normpath(os.path.join(_here, "..", "utils"))
# repo root: .../src/PR_curve/family_subsets.py -> ../..
_ROOT = os.path.normpath(os.path.join(_here, "..", ".."))
if _utils_dir not in sys.path:
    sys.path.insert(0, _utils_dir)

from parser import (
    check_muti_level_homolog,
    get_origin_prot_id,
    get_seq_label_map,
)

LabelTuple = Tuple[str, str, str]
FamilyMap = Dict[str, List[str]]

DEFAULT_FASTA = os.path.join(
    _ROOT, "data", "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
)


def get_homo_type(
    seq_id_a: str,
    seq_id_b: str,
    fa_id_label_map: Dict[str, LabelTuple],
) -> int:
    origin_a = get_origin_prot_id(seq_id_a)
    origin_b = get_origin_prot_id(seq_id_b)
    if origin_a not in fa_id_label_map:
        raise KeyError(f"Sequence not found in label map: {seq_id_a} ({origin_a})")
    if origin_b not in fa_id_label_map:
        raise KeyError(f"Sequence not found in label map: {seq_id_b} ({origin_b})")
    return check_muti_level_homolog(
        fa_id_label_map[origin_a],
        fa_id_label_map[origin_b],
    )


def get_family_label(
    seq_id: str,
    fa_id_label_map: Dict[str, LabelTuple],
) -> str:
    origin_id = get_origin_prot_id(seq_id)
    if origin_id not in fa_id_label_map:
        raise KeyError(f"Sequence not found in label map: {seq_id} ({origin_id})")
    return fa_id_label_map[origin_id][2]


def build_family_map(
    fa_id_label_map: Dict[str, LabelTuple],
) -> FamilyMap:
    family_map: FamilyMap = defaultdict(list)
    for seq_id, (_, _, scop_fam) in fa_id_label_map.items():
        family_map[scop_fam].append(seq_id)
    return dict(family_map)


def get_family_groups(
    fasta_url: str = DEFAULT_FASTA,
    min_size: int = 2,
) -> FamilyMap:
    fa_id_label_map = get_seq_label_map(fasta_url)
    family_map = build_family_map(fa_id_label_map)
    return {
        fam: sorted(seq_ids)
        for fam, seq_ids in family_map.items()
        if len(seq_ids) >= min_size
    }


def get_sequences_by_family(
    family_id: str,
    fasta_url: str = DEFAULT_FASTA,
) -> List[str]:
    fa_id_label_map = get_seq_label_map(fasta_url)
    family_map = build_family_map(fa_id_label_map)
    if family_id not in family_map:
        raise KeyError(f"Family not found: {family_id}")
    return sorted(family_map[family_id])


def find_family_subsets_in_pool(
    seq_ids: Iterable[str],
    fa_id_label_map: Dict[str, LabelTuple],
    min_size: int = 2,
) -> FamilyMap:
    pool_by_family: FamilyMap = defaultdict(list)
    for seq_id in seq_ids:
        origin_id = get_origin_prot_id(seq_id)
        if origin_id not in fa_id_label_map:
            continue
        scop_fam = fa_id_label_map[origin_id][2]
        pool_by_family[scop_fam].append(seq_id)

    return {
        fam: sorted(ids)
        for fam, ids in pool_by_family.items()
        if len(ids) >= min_size
    }


def is_pairwise_same_family(
    seq_ids: Sequence[str],
    fa_id_label_map: Dict[str, LabelTuple],
) -> bool:
    if len(seq_ids) <= 1:
        return True
    for seq_a, seq_b in combinations(seq_ids, 2):
        if get_homo_type(seq_a, seq_b, fa_id_label_map) != 2:
            return False
    return True


def filter_families_by_size(
    family_map: FamilyMap,
    min_size: int = 2,
    max_size: int | None = None,
) -> FamilyMap:
    filtered: FamilyMap = {}
    for fam, seq_ids in family_map.items():
        n = len(seq_ids)
        if n < min_size:
            continue
        if max_size is not None and n > max_size:
            continue
        filtered[fam] = seq_ids
    return filtered


def find_families_near_size(
    family_map: FamilyMap,
    target_size: int,
    tolerance: int = 0,
) -> List[Tuple[str, int]]:
    matches: List[Tuple[str, int]] = []
    for fam, seq_ids in family_map.items():
        n = len(seq_ids)
        if abs(n - target_size) <= tolerance:
            matches.append((fam, n))
    matches.sort(key=lambda item: (abs(item[1] - target_size), item[0]))
    return matches


def sample_families(
    family_map: FamilyMap,
    n_families: int,
    seed: int = 0,
) -> FamilyMap:
    import random

    rng = random.Random(seed)
    families = sorted(family_map.keys())
    if n_families >= len(families):
        return family_map
    chosen = sorted(rng.sample(families, n_families))
    return {fam: family_map[fam] for fam in chosen}


def write_query_list(seq_ids: Sequence[str], output_path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as fp:
        fp.write("\n".join(seq_ids))
        fp.write("\n")


def summarize_family_map(family_map: FamilyMap) -> dict:
    sizes = [len(v) for v in family_map.values()]
    return {
        "num_families": len(family_map),
        "num_sequences": sum(sizes),
        "min_family_size": min(sizes) if sizes else 0,
        "max_family_size": max(sizes) if sizes else 0,
        "mean_family_size": (sum(sizes) / len(sizes)) if sizes else 0.0,
    }


def write_family_map_json(family_map: FamilyMap, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(family_map, fp, indent=2, sort_keys=True)


def main(args: argparse.Namespace) -> None:
    family_map = get_family_groups(args.fasta_url, min_size=args.min_size)
    family_map = filter_families_by_size(
        family_map,
        min_size=args.min_size,
        max_size=args.max_size,
    )
    if args.n_families is not None:
        family_map = sample_families(family_map, args.n_families, seed=args.seed)

    stats = summarize_family_map(family_map)
    print(json.dumps(stats, indent=2))

    if args.show_examples > 0:
        print("\nExample families:")
        for fam in sorted(family_map.keys())[: args.show_examples]:
            seq_ids = family_map[fam]
            print(f"  {fam}\t(n={len(seq_ids)})\t{', '.join(seq_ids[:5])}")
            if len(seq_ids) > 5:
                print(f"    ... and {len(seq_ids) - 5} more")

    if args.output_json:
        write_family_map_json(family_map, args.output_json)
        print(f"\nSaved family map to: {args.output_json}")


if __name__ == "__main__":
    # --- Part 1: search families by query count ---
    RUN_SEARCH_NEAR_SIZE = False
    TARGET_QUERY_COUNT = 20
    SIZE_TOLERANCE = 2

    # --- Part 2: export all queries for one family ---
    RUN_EXPORT_FAMILY = True
    SELECTED_FAMILY = "b.47.1.2"
    DATA_DIR = os.path.join(_ROOT, "data")
    OUTPUT_QUERYLIST = os.path.join(
        DATA_DIR,
        f"querylist_{SELECTED_FAMILY}.txt",
    )

    if not RUN_SEARCH_NEAR_SIZE and not RUN_EXPORT_FAMILY:
        raise SystemExit("Enable RUN_SEARCH_NEAR_SIZE and/or RUN_EXPORT_FAMILY.")

    family_map = get_family_groups(DEFAULT_FASTA, min_size=2)

    if RUN_SEARCH_NEAR_SIZE:
        lo = TARGET_QUERY_COUNT - SIZE_TOLERANCE
        hi = TARGET_QUERY_COUNT + SIZE_TOLERANCE
        print("=" * 60)
        print(
            f"Families with ~{TARGET_QUERY_COUNT} queries "
            f"(n in [{lo}, {hi}]):"
        )
        near_families = find_families_near_size(
            family_map,
            TARGET_QUERY_COUNT,
            SIZE_TOLERANCE,
        )
        for fam, n in near_families:
            print(f"  {fam}\t(n={n})")
        print(f"Total: {len(near_families)} families")
        print("=" * 60)

    if RUN_EXPORT_FAMILY:
        if SELECTED_FAMILY not in family_map:
            selected_seqs = get_sequences_by_family(SELECTED_FAMILY, DEFAULT_FASTA)
        else:
            selected_seqs = family_map[SELECTED_FAMILY]

        fa_id_label_map = get_seq_label_map(DEFAULT_FASTA)
        assert is_pairwise_same_family(selected_seqs, fa_id_label_map)
        write_query_list(selected_seqs, OUTPUT_QUERYLIST)

        print(f"Family: {SELECTED_FAMILY}")
        print(f"Number of sequences: {len(selected_seqs)}")
        print("Sequences:")
        for seq_id in selected_seqs:
            print(f"  {seq_id}")
        print(
            f"\nPairwise homo_type == 2 verified for all "
            f"{len(selected_seqs)} sequences."
        )
        print(f"Saved query list to: {OUTPUT_QUERYLIST}")
