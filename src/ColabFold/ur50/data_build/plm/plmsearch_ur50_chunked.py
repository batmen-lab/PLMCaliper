import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "libs/PLMSearch-main/plmsearch"))
from plmsearch_util.model import plmsearch  # noqa: E402


def load_embeddings(path: str):
    with open(path, "rb") as h:
        d = pickle.load(h)
    ids = list(d.keys())
    mat = torch.stack([d[k] for k in ids]).float()
    del d
    return ids, mat


def load_target_matrix_frugal(path: str):
    with open(path, "rb") as h:
        d = pickle.load(h)
    ids = list(d.keys())
    n = len(ids)
    dim = int(d[ids[0]].numel())
    mat = torch.empty((n, dim), dtype=torch.float16)
    for i, k in enumerate(ids):
        mat[i] = d.pop(k).to(torch.float16)
    del d
    return ids, mat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-iqe", "--query_emb", required=True)
    ap.add_argument("-ite", "--target_emb", required=True)
    ap.add_argument("-smp", "--model_path", required=True)
    ap.add_argument("-osr", "--out", required=True)
    ap.add_argument("-k", "--top_k", type=int, default=1_000_000)
    ap.add_argument("--device", default="0")
    ap.add_argument("--target-batch", type=int, default=1_000_000,
                    help="targets moved to GPU per batch")
    args = ap.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    print(f"[load] query embeddings {args.query_emb}", flush=True)
    q_ids, q_mat = load_embeddings(args.query_emb)
    q_mat = q_mat.to(device)                      # [Q, D], Q ~ 83
    Q, D = q_mat.shape
    print(f"[load] {Q} queries, dim {D}", flush=True)

    # SigNet projection: precompute W q for every query once.
    model = plmsearch(embed_dim=D)
    model.load_pretrained(args.model_path)
    model.eval().to(device)
    with torch.no_grad():
        q_proj = model.forward(q_mat)             # [Q, D]
        q_norm = q_mat / q_mat.norm(dim=1, keepdim=True).clamp_min(1e-8)

    if args.target_emb.endswith(".f16"):
        # memmap-backed fp16 matrix (from pkl_to_memmap.py): never loads all 99G
        # into RAM -- each batch slice pages in on demand.
        stem = args.target_emb[:-4]
        with open(stem + ".shape.txt") as fh:
            N, Dt = (int(x) for x in fh.read().split())
        print(f"[load] target memmap {args.target_emb} ({N:,} x {Dt}, fp16 on disk)", flush=True)
        t_ids = [ln.rstrip("\n") for ln in open(stem + ".ids.txt")]
        t_np = np.memmap(args.target_emb, dtype=np.float16, mode="r", shape=(N, Dt))
        t_mat = torch.from_numpy(t_np)                # memmap-backed fp16 tensor
    else:
        print(f"[load] target embeddings {args.target_emb} (~200G fp16, be patient) ...", flush=True)
        t_ids, t_mat = load_target_matrix_frugal(args.target_emb)   # CPU [N, D] fp16
    N = t_mat.shape[0]
    print(f"[load] {N:,} targets loaded in {time.time()-t0:.0f}s", flush=True)

    K = min(args.top_k, N)
    # running top-K per query, kept on GPU
    top_scores = torch.full((Q, K), -1.0, device=device)
    top_idx = torch.zeros((Q, K), dtype=torch.long, device=device)

    bs = args.target_batch
    with torch.no_grad():
        for start in range(0, N, bs):
            end = min(start + bs, N)
            tb = t_mat[start:end].to(device, non_blocking=True).float()   # [B, D] fp16->fp32 on GPU
            tb_norm = tb / tb.norm(dim=1, keepdim=True).clamp_min(1e-8)
            cos = q_norm @ tb_norm.t()                            # [Q, B]
            signet = torch.sigmoid(q_proj @ tb.t())               # [Q, B]
            score = torch.where(cos > 0.995, cos, cos * signet)   # [Q, B]

            # merge this batch into the running top-K
            gidx = torch.arange(start, end, device=device).expand(Q, -1)
            cat_s = torch.cat([top_scores, score], dim=1)
            cat_i = torch.cat([top_idx, gidx], dim=1)
            sel = cat_s.topk(K, dim=1)
            top_scores = sel.values
            top_idx = torch.gather(cat_i, 1, sel.indices)

            if (start // bs) % 5 == 0:
                print(f"[search] {end:,}/{N:,} targets "
                      f"({time.time()-t0:.0f}s)", flush=True)

    print(f"[write] top-{K} per query -> {args.out}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    top_scores_c = top_scores.cpu()
    top_idx_c = top_idx.cpu()
    with open(args.out, "w") as f:
        for qi, qid in enumerate(q_ids):
            s_row = top_scores_c[qi]
            i_row = top_idx_c[qi]
            order = torch.argsort(s_row, descending=True)
            qid_tok = qid.strip().split()[0]
            for j in order.tolist():
                sc = float(s_row[j])
                if sc < 0:
                    continue
                tid_tok = t_ids[int(i_row[j])].strip().split()[0]
                f.write(f"{qid_tok}\t{tid_tok}\t{round(sc, 4)}\n")
    print(f"[done] total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
