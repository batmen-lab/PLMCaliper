#!/usr/bin/env python3
import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pysam.libcfaidx import FastxFile
from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
from transformers import T5EncoderModel, T5Tokenizer

TMVEC_SRC = Path(__file__).resolve().parents[5] / "libs/tm-vec-master"
# Same ProtTrans model the astral tmvec DBs were built with (run_tmvec_single.sh).
PROTRANS_MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"
# ProtT5 self-attention is O(L^2) in memory; UniRef50 has titin-scale sequences
# (tens of thousands of residues) that OOM a 48 GB GPU. Cap length like ESM/ProtT5
# do (~1022) -- the astral queries are short domains, so this only clips the rare
# very long UR50 targets. Configurable via --max-len.
DEFAULT_MAX_LEN = 1022
EMB_DIM = 512


def load_models(device):
    tokenizer = T5Tokenizer.from_pretrained(PROTRANS_MODEL, do_lower_case=False)
    # ProtT5-XL is a 3B model; this is the *half*-precision checkpoint, so load it in
    # fp16 (2x faster + half the memory -> bigger batches). The 512-d TM-Vec head
    # stays fp32; _embed_batch casts the ProtT5 output back to fp32 for it.
    model = T5EncoderModel.from_pretrained(PROTRANS_MODEL, torch_dtype=torch.float16).to(device).eval()
    cfg = trans_basic_block_Config.from_json(
        str(TMVEC_SRC / "model/tm_vec_cath_model_params.json"))
    model_deep = trans_basic_block.load_from_checkpoint(
        str(TMVEC_SRC / "model/tm_vec_cath_model.ckpt"),
        config=cfg, map_location=device).to(device).eval()
    return tokenizer, model, model_deep


def read_chunk(fa: Path):
    heads, seqs = [], []
    with FastxFile(str(fa)) as fh:
        for r in fh:
            heads.append(r.name)
            seqs.append(r.sequence.replace("*", ""))
    return heads, seqs


def _featurize_batch(seqs, model, tokenizer, device):
    sp = [" ".join(list(s)) for s in seqs]
    sp = [re.sub(r"[UZOB]", "X", x) for x in sp]
    enc = tokenizer.batch_encode_plus(sp, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(enc["input_ids"], device=device)
    attn = torch.tensor(enc["attention_mask"], device=device)
    with torch.no_grad():
        h = model(input_ids=input_ids, attention_mask=attn).last_hidden_state
    return h, attn


def _embed_batch(h, attn, model_deep, device):
    lens = attn.sum(dim=1).long() - 1               # residue count per seq (drop EOS)
    lmax = int(lens.max().item())
    feat = h[:, :lmax, :].float()                   # fp16 ProtT5 -> fp32 for TM-Vec head
    idx = torch.arange(lmax, device=device).unsqueeze(0)
    pad = idx >= lens.unsqueeze(1)                   # True on EOS + padded positions
    with torch.no_grad():
        out = model_deep(feat, src_mask=None, src_key_padding_mask=pad)
    return out.detach().cpu().numpy()


def encode_chunk(seqs, model_deep, model, tokenizer, device, max_len,
                 max_batch_tokens=24576, max_batch=192, progress_every=3000):
    n = len(seqs)
    out = np.zeros((n, EMB_DIM), dtype=np.float32)
    capped = [s[:max_len] for s in seqs]
    n_clip = sum(1 for s in seqs if len(s) > max_len)
    n_fail = 0
    order = sorted(range(n), key=lambda i: len(capped[i]))

    t0 = time.time()
    last_log = 0
    i = 0
    while i < n:
        j, lmax = i, len(capped[order[i]]) + 1
        while j < n and (j - i) < max_batch:
            L = max(lmax, len(capped[order[j]]) + 1)
            if (j - i + 1) * L > max_batch_tokens and j > i:
                break
            lmax = L
            j += 1
        idx = order[i:j]
        try:
            h, attn = _featurize_batch([capped[k] for k in idx], model, tokenizer, device)
            emb = _embed_batch(h, attn, model_deep, device)
            for bi, k in enumerate(idx):
                out[k] = emb[bi]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for k in idx:                            # per-seq fallback for this batch
                try:
                    h, attn = _featurize_batch([capped[k]], model, tokenizer, device)
                    out[k] = _embed_batch(h, attn, model_deep, device)[0]
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    n_fail += 1
        i = j
        if i - last_log >= progress_every:
            dt = time.time() - t0
            print(f"    ...{i}/{n} ({i/max(dt,1e-9):.1f} seq/s, {dt:.0f}s)", flush=True)
            last_log = i
    return out, n_clip, n_fail


def save_atomic(npy_path: Path, emb: np.ndarray) -> None:
    tmp = npy_path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, emb)
    os.replace(tmp, npy_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-dir", required=True, help="dir of chunk_*.fasta")
    ap.add_argument("--out-dir", required=True, help="dir for per-chunk .npy/.ids.txt")
    ap.add_argument("--worker-id", type=int, required=True)
    ap.add_argument("--num-workers", type=int, required=True)
    ap.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("[fatal] CUDA not available")
    device = torch.device("cuda:0")  # the only visible GPU (pinned by env)

    chunks = sorted(Path(args.chunk_dir).glob("chunk_*.fasta"))
    mine = [c for i, c in enumerate(chunks) if i % args.num_workers == args.worker_id]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in mine if not ((out / f"{c.stem}.npy").exists()
                                    and (out / f"{c.stem}.ids.txt").exists())]
    print(f"[w{args.worker_id}] {len(mine)} assigned / {len(chunks)} total; "
          f"{len(todo)} to do", flush=True)
    if not todo:
        print(f"[w{args.worker_id}] nothing to do", flush=True)
        return

    tokenizer, model, model_deep = load_models(device)
    print(f"[w{args.worker_id}] models loaded", flush=True)

    for ch in mine:
        npy = out / f"{ch.stem}.npy"
        ids = out / f"{ch.stem}.ids.txt"
        if npy.exists() and ids.exists():
            continue
        heads, seqs = read_chunk(ch)
        t0 = time.time()
        emb, n_clip, n_fail = encode_chunk(seqs, model_deep, model, tokenizer,
                                           device, args.max_len)
        emb = np.ascontiguousarray(emb, dtype=np.float16)   # [n, 512]
        save_atomic(npy, emb)
        ids.write_text("\n".join(heads) + "\n")
        dt = time.time() - t0
        print(f"[w{args.worker_id}] {ch.stem} n={len(seqs)} {emb.shape} "
              f"{dt:.0f}s ({len(seqs)/max(dt,1e-9):.1f} seq/s) "
              f"clip>{args.max_len}={n_clip} fail={n_fail}", flush=True)

    print(f"[w{args.worker_id}] done", flush=True)


if __name__ == "__main__":
    main()
