#!/usr/bin/env python3
import argparse
import os
import time

import numpy as np
import torch


def load_query(emb_path: str, ids_path: str):
    emb = np.load(emb_path).astype(np.float32)          # [Q, 512]
    ids = [ln.rstrip("\n") for ln in open(ids_path) if ln.strip()]
    if len(ids) != emb.shape[0]:
        raise ValueError(f"query ids {len(ids)} != emb rows {emb.shape[0]}")
    return ids, torch.from_numpy(emb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query-emb", required=True)
    ap.add_argument("--query-ids", required=True)
    ap.add_argument("--target-emb", required=True, help="ur50_embedding.f16 memmap")
    ap.add_argument("--out", required=True)
    ap.add_argument("-k", "--top-k", type=int, default=1_000_000)
    ap.add_argument("--device", default="0")
    ap.add_argument("--target-batch", type=int, default=2_000_000,
                    help="targets moved to GPU per batch")
    args = ap.parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    q_ids, q_mat = load_query(args.query_emb, args.query_ids)
    q_mat = q_mat.to(device)
    Q, D = q_mat.shape
    q_norm = q_mat / q_mat.norm(dim=1, keepdim=True).clamp_min(1e-8)
    print(f"[load] {Q} queries, dim {D}", flush=True)

    stem = args.target_emb[:-4] if args.target_emb.endswith(".f16") else args.target_emb
    with open(stem + ".shape.txt") as fh:
        N, Dt = (int(x) for x in fh.read().split())
    if Dt != D:
        raise ValueError(f"target dim {Dt} != query dim {D}")
    print(f"[load] target memmap {args.target_emb} ({N:,} x {Dt}, fp16 on disk)", flush=True)
    t_ids = [ln.rstrip("\n") for ln in open(stem + ".ids.txt")]
    if len(t_ids) != N:
        raise ValueError(f"target ids {len(t_ids)} != N {N}")
    t_np = np.memmap(args.target_emb, dtype=np.float16, mode="r", shape=(N, Dt))
    t_mat = torch.from_numpy(t_np)

    K = min(args.top_k, N)
    top_scores = torch.full((Q, K), -2.0, device=device)   # cosine in [-1, 1]
    top_idx = torch.zeros((Q, K), dtype=torch.long, device=device)

    bs = args.target_batch
    with torch.no_grad():
        for start in range(0, N, bs):
            end = min(start + bs, N)
            tb = t_mat[start:end].to(device, non_blocking=True).float()   # [B, D]
            tb_norm = tb / tb.norm(dim=1, keepdim=True).clamp_min(1e-8)
            score = q_norm @ tb_norm.t()                                  # [Q, B] cosine
            gidx = torch.arange(start, end, device=device).expand(Q, -1)
            cat_s = torch.cat([top_scores, score], dim=1)
            cat_i = torch.cat([top_idx, gidx], dim=1)
            sel = cat_s.topk(K, dim=1)
            top_scores = sel.values
            top_idx = torch.gather(cat_i, 1, sel.indices)
            if (start // bs) % 5 == 0:
                print(f"[search] {end:,}/{N:,} targets ({time.time()-t0:.0f}s)", flush=True)

    print(f"[write] top-{K} per query -> {args.out}", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    top_scores_c = top_scores.cpu()
    top_idx_c = top_idx.cpu()
    with open(args.out, "w") as f:
        f.write("qid\ttid\tscore\thomo_type\trank\n")
        for qi, qid in enumerate(q_ids):
            s_row = top_scores_c[qi]
            i_row = top_idx_c[qi]
            order = torch.argsort(s_row, descending=True)
            qid_tok = qid.strip().split()[0]
            rank = 0
            for j in order.tolist():
                sc = float(s_row[j])
                if sc < -1.5:       # unfilled slot (fewer than K targets)
                    continue
                rank += 1
                tid_tok = t_ids[int(i_row[j])].strip().split()[0]
                f.write(f"{qid_tok}\t{tid_tok}\t{round(sc, 4)}\t-1\t{rank}\n")
    print(f"[done] total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
