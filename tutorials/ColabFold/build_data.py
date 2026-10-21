import argparse
import glob
import math
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"

DATASETS = ("astral", "ur50")
DECOY_METHOD = "extended_mkv2"     # seq + order-2 Markov resample
DECOY_SUFFIX = "_mkv"              # id suffix the decoy generator adds to each query
MARKOV_ORDER = 2                   # the "2" in extended_mkv2
# DHR reports distances; subtract from this to get larger-is-better, the same
# constant src/utils/process_searching_score.py uses.
DHR_SCORE_CEILING = 150.0

# the two tables step 2 consumes; step 1 is what puts them there
RAW_REAL = {"astral": DATA_DIR / "result_{m}_astral_astral.txt",
            "ur50": DATA_DIR / "search_data" / "result_{m}_astral4f_ur50.txt"}
RAW_DECOY = {"astral": DATA_DIR / "_hf_decoy" / "raw" / "scores" / "astral_main"
                       / f"result_{{m}}_astral_{DECOY_METHOD}_astral.txt",
             "ur50": DATA_DIR / "search_data"
                     / f"result_{{m}}_astral4f_{DECOY_METHOD}_ur50.txt"}

UR50_FASTA = DATA_DIR / "uniref50" / "uniref50.fasta"
UR50_TSV = DATA_DIR / "uniref50" / "uniref50.tsv"
TMVEC_CHUNKS = DATA_DIR / "uniref50" / "tmvec_chunks"
TMVEC_DB = DATA_DIR / "db" / "db_ur50_tmvec"
PLM_CHUNKS = DATA_DIR / "uniref50" / "plm_chunks"
PLM_DB = DATA_DIR / "db" / "db_ur50_plm"
UR50_RESULTS = ROOT / "data" / "ColabFold" / "ur50_exp"


def raw_real(dataset: str, method: str) -> Path:
    return Path(str(RAW_REAL[dataset]).format(m=method))


def raw_decoy(dataset: str, method: str) -> Path:
    return Path(str(RAW_DECOY[dataset]).format(m=method))

TMVEC_EMB_DIM = 512
# ProtT5 self-attention is O(L^2) in memory and UniRef50 has titin-scale sequences,
# so cap length the way ESM/ProtT5 do. ASTRAL queries are short domains.
TMVEC_MAX_LEN = 1022
PROTRANS_MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"


def read_fasta(path: Path) -> dict:
    seqs, name, buf = {}, None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                seqs[name] = "".join(buf)
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def read_query_ids(path: Path) -> list[str]:
    ids, seen = [], set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        qid = line.split("\t")[0].strip()
        if qid and qid not in seen:
            ids.append(qid)
            seen.add(qid)
    return ids


def read_fasta_ids(path: Path) -> list[str]:
    ids, seen = [], set()
    for line in Path(path).read_text().splitlines():
        if not line.startswith(">"):
            continue
        qid = line[1:].split()[0].strip()
        if qid and qid not in seen:
            ids.append(qid)
            seen.add(qid)
    return ids


def read_astral4f_query_ids() -> list[str]:
    for path, reader in (
        (DATA_DIR / "astral4f_query.tsv", read_query_ids),
        (DATA_DIR / "astral4f_query.fa", read_fasta_ids),
        (DATA_DIR / "querylist_4families.txt", read_query_ids),
    ):
        if path.exists() and path.stat().st_size:
            return reader(path)
    raise SystemExit("[step1] missing astral4f query ids: expected "
                     "data/astral4f_query.tsv or data/astral4f_query.fa")


def env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default)
    if not v:
        raise SystemExit(f"[step1] {name} unset -- run `bash src/setup.sh` then\n                          `source src/PLM_searching_cmds/.env`")
    return v


def check(dataset: str, method: str) -> bool:
    ok = True
    for label, p in (("real ", raw_real(dataset, method)),
                     ("decoy", raw_decoy(dataset, method))):
        exists = p.exists() and p.stat().st_size > 0
        size = f"{p.stat().st_size / 1e9:.1f} GB" if exists else "-"
        print(f"  {label}  {'OK  ' if exists else 'MISSING'}  {size:>8}  {p}")
        ok &= exists
    return ok


# --------------------------------------------------------------------------
# fasta sharding, shared by the tmvec and plm encoders
# --------------------------------------------------------------------------
def split_fasta(fasta: Path, chunk_dir: Path, seqs_per_chunk: int) -> list[Path]:
    """Split into chunk_NNNN.fasta; idempotent via a .split_done marker."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    marker = chunk_dir / ".split_done"
    if marker.exists():
        return sorted(chunk_dir.glob("chunk_*.fasta"))
    if not (fasta.exists() and fasta.stat().st_size):
        raise SystemExit(f"[step1] {fasta} missing")

    for old in chunk_dir.glob("chunk_*.fasta"):
        old.unlink()
    print(f"[split] {fasta} -> {chunk_dir} ({seqs_per_chunk} seqs/chunk)", flush=True)
    n, cur, out = 0, -1, None
    with open(fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                idx = n // seqs_per_chunk
                if idx != cur:
                    if out is not None:
                        out.close()
                    cur, out = idx, open(chunk_dir / f"chunk_{idx:04d}.fasta", "w")
                n += 1
            if out is not None:
                out.write(line)
    if out is not None:
        out.close()
    marker.touch()
    chunks = sorted(chunk_dir.glob("chunk_*.fasta"))
    print(f"[split] done: {len(chunks)} chunks, {n:,} sequences", flush=True)
    return chunks


def launch_per_gpu(cmd_for, gpus: list[str]) -> int:
    """Run one worker per GPU and wait; `cmd_for(worker_id, gpu)` builds each command."""
    procs = []
    for wid, gpu in enumerate(gpus):
        cmd = cmd_for(wid, gpu)
        print(f"[launch] worker {wid} on GPU {gpu}", flush=True)
        procs.append(subprocess.Popen([str(c) for c in cmd], cwd=ROOT,
                                      env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}))
    rc = 0
    for p in procs:
        rc |= p.wait()
    return rc


# --------------------------------------------------------------------------
# TM-Vec
# --------------------------------------------------------------------------
def _tmvec_models(device):
    from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
    from transformers import T5EncoderModel, T5Tokenizer
    import torch

    src = ROOT / "libs" / "tm-vec-master"
    tokenizer = T5Tokenizer.from_pretrained(PROTRANS_MODEL, do_lower_case=False)
    # the *half*-precision ProtT5-XL checkpoint: 2x faster, half the memory. The
    # 512-d TM-Vec head stays fp32; _tmvec_embed casts back for it.
    model = T5EncoderModel.from_pretrained(
        PROTRANS_MODEL, torch_dtype=torch.float16).to(device).eval()
    cfg = trans_basic_block_Config.from_json(
        str(src / "model/tm_vec_cath_model_params.json"))
    deep = trans_basic_block.load_from_checkpoint(
        str(src / "model/tm_vec_cath_model.ckpt"),
        config=cfg, map_location=device).to(device).eval()
    return tokenizer, model, deep


def _tmvec_featurize(seqs, model, tokenizer, device):
    import torch
    sp = [re.sub(r"[UZOB]", "X", " ".join(list(s))) for s in seqs]
    enc = tokenizer.batch_encode_plus(sp, add_special_tokens=True, padding="longest")
    ids = torch.tensor(enc["input_ids"], device=device)
    attn = torch.tensor(enc["attention_mask"], device=device)
    with torch.no_grad():
        h = model(input_ids=ids, attention_mask=attn).last_hidden_state
    return h, attn


def _tmvec_embed(h, attn, deep, device):
    import torch
    lens = attn.sum(dim=1).long() - 1                # residues per seq, drop EOS
    lmax = int(lens.max().item())
    feat = h[:, :lmax, :].float()
    idx = torch.arange(lmax, device=device).unsqueeze(0)
    pad = idx >= lens.unsqueeze(1)
    with torch.no_grad():
        out = deep(feat, src_mask=None, src_key_padding_mask=pad)
    return out.detach().cpu().numpy()


def tmvec_encode_seqs(seqs, deep, model, tokenizer, device, max_len,
                      max_batch_tokens=24576, max_batch=192, progress_every=3000):
    """Length-sorted dynamic batching with a per-sequence OOM fallback."""
    import torch
    n = len(seqs)
    out = np.zeros((n, TMVEC_EMB_DIM), dtype=np.float32)
    capped = [s[:max_len] for s in seqs]
    n_clip = sum(1 for s in seqs if len(s) > max_len)
    n_fail = 0
    order = sorted(range(n), key=lambda i: len(capped[i]))

    t0, last_log, i = time.time(), 0, 0
    while i < n:
        j, lmax = i, len(capped[order[i]]) + 1
        while j < n and (j - i) < max_batch:
            L = max(lmax, len(capped[order[j]]) + 1)
            if (j - i + 1) * L > max_batch_tokens and j > i:
                break
            lmax, j = L, j + 1
        idx = order[i:j]
        try:
            h, attn = _tmvec_featurize([capped[k] for k in idx], model, tokenizer, device)
            emb = _tmvec_embed(h, attn, deep, device)
            for bi, k in enumerate(idx):
                out[k] = emb[bi]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for k in idx:                            # per-seq retry for this batch
                try:
                    h, attn = _tmvec_featurize([capped[k]], model, tokenizer, device)
                    out[k] = _tmvec_embed(h, attn, deep, device)[0]
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    n_fail += 1
        i = j
        if i - last_log >= progress_every:
            dt = time.time() - t0
            print(f"    ...{i}/{n} ({i / max(dt, 1e-9):.1f} seq/s, {dt:.0f}s)", flush=True)
            last_log = i
    return out, n_clip, n_fail


def tmvec_encode_worker(chunk_dir: Path, out_dir: Path, worker_id: int,
                        num_workers: int, max_len: int) -> None:
    import torch
    from pysam.libcfaidx import FastxFile
    if not torch.cuda.is_available():
        raise SystemExit("[step1] CUDA not available")
    device = torch.device("cuda:0")                  # pinned by CUDA_VISIBLE_DEVICES

    chunks = sorted(Path(chunk_dir).glob("chunk_*.fasta"))
    mine = [c for i, c in enumerate(chunks) if i % num_workers == worker_id]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [c for c in mine if not ((out_dir / f"{c.stem}.npy").exists()
                                    and (out_dir / f"{c.stem}.ids.txt").exists())]
    print(f"[w{worker_id}] {len(mine)} assigned / {len(chunks)} total; "
          f"{len(todo)} to do", flush=True)
    if not todo:
        return

    tokenizer, model, deep = _tmvec_models(device)
    for ch in todo:
        heads, seqs = [], []
        with FastxFile(str(ch)) as fh:
            for r in fh:
                heads.append(r.name)
                seqs.append(r.sequence.replace("*", ""))
        t0 = time.time()
        emb, n_clip, n_fail = tmvec_encode_seqs(seqs, deep, model, tokenizer,
                                                device, max_len)
        emb = np.ascontiguousarray(emb, dtype=np.float16)
        tmp = (out_dir / f"{ch.stem}.npy").with_suffix(".npy.tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, emb)
        os.replace(tmp, out_dir / f"{ch.stem}.npy")
        (out_dir / f"{ch.stem}.ids.txt").write_text("\n".join(heads) + "\n")
        dt = time.time() - t0
        print(f"[w{worker_id}] {ch.stem} n={len(seqs)} {emb.shape} {dt:.0f}s "
              f"clip>{max_len}={n_clip} fail={n_fail}", flush=True)


def tmvec_encode_db(gpus: list[str], seqs_per_chunk: int) -> int:
    split_fasta(UR50_FASTA, TMVEC_CHUNKS, seqs_per_chunk)
    out_dir = TMVEC_DB / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    py = env("TMVEC_PYTHON_PATH")
    n = len(gpus)
    return launch_per_gpu(
        lambda wid, gpu: [py, __file__, "tmvec-encode-worker",
                          "--chunk-dir", TMVEC_CHUNKS, "--out-dir", out_dir,
                          "--worker-id", wid, "--num-workers", n], gpus)


def tmvec_merge_db(allow_incomplete: bool, delete_chunks: bool) -> None:
    chunk_dir, split_dir = TMVEC_DB / "chunks", TMVEC_CHUNKS
    prefix = TMVEC_DB / "ur50_embedding"
    expected = sorted(p.stem for p in split_dir.glob("chunk_*.fasta"))
    npys = {p.stem: p for p in chunk_dir.glob("chunk_*.npy")}
    if not expected:
        raise SystemExit(f"[step1] no chunk_*.fasta in {split_dir}")
    missing = [s for s in expected if s not in npys]
    if missing and not allow_incomplete:
        raise SystemExit(f"[step1] {len(missing)}/{len(expected)} chunks not encoded "
                         f"(e.g. {missing[:3]}); pass --allow-incomplete to merge anyway")
    order = [s for s in expected if s in npys]

    N = 0
    for stem in order:
        a = np.load(npys[stem], mmap_mode="r")
        if a.ndim != 2 or a.shape[1] != TMVEC_EMB_DIM:
            raise SystemExit(f"[step1] {stem}: bad shape {a.shape}")
        N += a.shape[0]
    print(f"[merge] {len(order)} chunks, N={N:,} x {TMVEC_EMB_DIM} "
          f"({N * TMVEC_EMB_DIM * 2 / 1e9:.1f} GB)", flush=True)

    out_f16 = Path(f"{prefix}.f16")
    out_ids = Path(f"{prefix}.ids.txt")
    out_f16.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(out_f16, dtype=np.float16, mode="w+", shape=(N, TMVEC_EMB_DIM))
    row = 0
    with open(out_ids, "w") as fids:
        for i, stem in enumerate(order):
            a = np.load(npys[stem]).astype(np.float16, copy=False)
            mm[row:row + a.shape[0]] = a
            row += a.shape[0]
            ids = [x for x in (chunk_dir / f"{stem}.ids.txt").read_text().split("\n") if x]
            if len(ids) != a.shape[0]:
                raise SystemExit(f"[step1] {stem}: {len(ids)} ids vs {a.shape[0]} rows")
            fids.write("\n".join(ids) + "\n")
            if i % 20 == 0:
                print(f"  merged {i + 1}/{len(order)}, {row:,} rows", flush=True)
    mm.flush()
    del mm
    Path(f"{prefix}.shape.txt").write_text(f"{N} {TMVEC_EMB_DIM}\n")
    print(f"[done] {out_f16} ({N:,} x {TMVEC_EMB_DIM})", flush=True)

    if delete_chunks:
        n_ids = sum(1 for _ in open(out_ids))
        expect = N * TMVEC_EMB_DIM * 2
        if out_f16.stat().st_size != expect or n_ids != N:
            raise SystemExit("[step1] refusing to delete chunks: merge looks incomplete")
        freed = 0
        for stem in order:
            for p in (npys[stem], chunk_dir / f"{stem}.ids.txt"):
                if p.exists():
                    freed += p.stat().st_size
                    p.unlink()
        print(f"[cleanup] freed {freed / 1e9:.1f} GB", flush=True)


def tmvec_encode_query(query_tsv: Path, out_prefix: str, max_len: int) -> None:
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("[step1] CUDA not available")
    device = torch.device("cuda:0")
    ids, seqs = [], []
    for line in open(query_tsv):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            ids.append(parts[0])
            seqs.append(parts[1].replace("*", ""))
    print(f"[query] {len(ids)} sequences from {query_tsv}", flush=True)
    tokenizer, model, deep = _tmvec_models(device)
    emb, n_clip, n_fail = tmvec_encode_seqs(seqs, deep, model, tokenizer, device, max_len)
    emb = np.ascontiguousarray(emb, dtype=np.float32)      # keep the query at f32
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{out_prefix}.npy", emb)
    Path(f"{out_prefix}.ids.txt").write_text("\n".join(ids) + "\n")
    print(f"[done] {emb.shape} -> {out_prefix}.npy (clip={n_clip} fail={n_fail})", flush=True)


def _topk_stream(q_norm, t_mat, N, K, device, batch, score_fn, t0):
    """Running top-K over a memmap-backed target matrix, one GPU batch at a time."""
    import torch
    Q = q_norm.shape[0]
    top_s = torch.full((Q, K), -2.0, device=device)
    top_i = torch.zeros((Q, K), dtype=torch.long, device=device)
    with torch.no_grad():
        for start in range(0, N, batch):
            end = min(start + batch, N)
            tb = t_mat[start:end].to(device, non_blocking=True).float()
            score = score_fn(tb)
            gidx = torch.arange(start, end, device=device).expand(Q, -1)
            sel = torch.cat([top_s, score], dim=1).topk(K, dim=1)
            top_s = sel.values
            top_i = torch.gather(torch.cat([top_i, gidx], dim=1), 1, sel.indices)
            if (start // batch) % 5 == 0:
                print(f"[search] {end:,}/{N:,} ({time.time() - t0:.0f}s)", flush=True)
    return top_s.cpu(), top_i.cpu()


def _load_memmap_targets(target_emb: str):
    stem = target_emb[:-4] if target_emb.endswith(".f16") else target_emb
    with open(stem + ".shape.txt") as fh:
        N, D = (int(x) for x in fh.read().split())
    ids = [ln.rstrip("\n") for ln in open(stem + ".ids.txt")]
    if len(ids) != N:
        raise SystemExit(f"[step1] target ids {len(ids)} != N {N}")
    return ids, np.memmap(target_emb, dtype=np.float16, mode="r", shape=(N, D)), N, D


def tmvec_search(query_emb, query_ids, target_emb, out, top_k, device_id, batch) -> None:
    import torch
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    emb = np.load(query_emb).astype(np.float32)
    q_ids = [ln.rstrip("\n") for ln in open(query_ids) if ln.strip()]
    if len(q_ids) != emb.shape[0]:
        raise SystemExit(f"[step1] query ids {len(q_ids)} != rows {emb.shape[0]}")
    q = torch.from_numpy(emb).to(device)
    q_norm = q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)
    print(f"[load] {q.shape[0]} queries, dim {q.shape[1]}", flush=True)

    t_ids, t_np, N, D = _load_memmap_targets(target_emb)
    if D != q.shape[1]:
        raise SystemExit(f"[step1] target dim {D} != query dim {q.shape[1]}")
    t_mat = torch.from_numpy(t_np)
    K = min(top_k, N)

    def score_fn(tb):
        return q_norm @ (tb / tb.norm(dim=1, keepdim=True).clamp_min(1e-8)).t()

    top_s, top_i = _topk_stream(q_norm, t_mat, N, K, device, batch, score_fn, t0)
    _write_hits(out, q_ids, t_ids, top_s, top_i, header=True, min_score=-1.5)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


def _write_hits(out, q_ids, t_ids, top_s, top_i, *, header: bool, min_score: float):
    import torch
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        if header:
            f.write("qid\ttid\tscore\thomo_type\trank\n")
        for qi, qid in enumerate(q_ids):
            s_row, i_row = top_s[qi], top_i[qi]
            rank = 0
            for j in torch.argsort(s_row, descending=True).tolist():
                sc = float(s_row[j])
                if sc < min_score:                    # unfilled slot
                    continue
                rank += 1
                tid = t_ids[int(i_row[j])].strip().split()[0]
                if header:
                    f.write(f"{qid.strip().split()[0]}\t{tid}\t{round(sc, 4)}\t-1\t{rank}\n")
                else:
                    f.write(f"{qid.strip().split()[0]}\t{tid}\t{round(sc, 4)}\n")


# --------------------------------------------------------------------------
# PLMsearch
# --------------------------------------------------------------------------
def plm_encode_db(gpu: str, seqs_per_chunk: int) -> None:
    chunks = split_fasta(UR50_FASTA, PLM_CHUNKS, seqs_per_chunk)
    out_dir = PLM_DB / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(env("PLMSEARCH_SRC_DIR"))
    esm = src / "plmsearch_data/model/esm/esm1b_t33_650M_UR50S.pt"
    gen = src / "plmsearch/embedding_generate.py"
    if not gen.exists():
        gen = src / "plmsearch/embedding_generate_esm1b.py"
    py = env("PLMSEARCH_PYTHON_PATH")
    if not esm.exists():
        raise SystemExit(f"[step1] ESM checkpoint missing: {esm}")
    if not gen.exists():
        raise SystemExit(f"[step1] PLMSearch embedding script missing under {src / 'plmsearch'}")

    done = 0
    for ch in chunks:
        out = out_dir / f"{ch.stem}.pkl"
        if out.exists() and out.stat().st_size:
            done += 1
            continue
        tmp = out.with_suffix(".pkl.tmp")
        print(f"  encoding {ch.stem} -> {out.name}", flush=True)
        rc = subprocess.run([py, str(gen), "-emp", str(esm), "-f", str(ch), "-e", str(tmp)],
                            cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}).returncode
        if rc != 0:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"[step1] {ch.stem} failed (rc={rc}); re-run to resume")
        os.replace(tmp, out)
        done += 1
    print(f"[ALL DONE] {done}/{len(chunks)} chunks -> {out_dir}", flush=True)


def _free_gb(path) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / (1024 ** 3)


def plm_merge_db(n_groups: int = 4) -> None:
    """Staged merge: chunks -> groups -> one pkl -> fp16 memmap, freeing as it goes."""
    chunk_dir, group_dir = PLM_DB / "chunks", PLM_DB / "groups"
    out_pkl = PLM_DB / "ur50_embedding.pkl"
    group_dir.mkdir(parents=True, exist_ok=True)

    chunks = sorted(glob.glob(str(chunk_dir / "chunk_*.pkl")))
    if chunks:
        per = math.ceil(len(chunks) / n_groups)
        print(f"[stage1] {len(chunks)} chunks -> {n_groups} groups (~{per} each)")
        for g in range(n_groups):
            gpath = group_dir / f"group_{g}.pkl"
            sub = chunks[g * per:(g + 1) * per]
            if not sub:
                continue
            if gpath.exists():
                for f in sub:
                    Path(f).unlink(missing_ok=True)
                continue
            merged = {}
            for f in sub:
                with open(f, "rb") as h:
                    merged.update(pickle.load(h))
            tmp = str(gpath) + ".tmp"
            with open(tmp, "wb") as h:
                pickle.dump(merged, h, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, gpath)
            for f in sub:
                Path(f).unlink(missing_ok=True)
            print(f"[stage1] group_{g}: {len(merged):,} emb "
                  f"(free={_free_gb(group_dir):.0f}G)", flush=True)

    if not out_pkl.exists():
        groups = sorted(glob.glob(str(group_dir / "group_*.pkl")))
        if not groups:
            raise SystemExit("[step1] no group pkls and no output -- nothing to merge")
        merged = {}
        for f in groups:
            with open(f, "rb") as h:
                merged.update(pickle.load(h))
            Path(f).unlink()
            print(f"[stage2] {Path(f).name} -> {len(merged):,} emb "
                  f"(free={_free_gb(group_dir):.0f}G)", flush=True)
        tmp = str(out_pkl) + ".tmp"
        with open(tmp, "wb") as h:
            pickle.dump(merged, h, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, out_pkl)
        print(f"[stage2] -> {out_pkl} ({len(merged):,} embeddings)", flush=True)

    plm_pkl_to_memmap(out_pkl)


def plm_pkl_to_memmap(src: Path) -> None:
    """Stream the embedding pkl straight to an fp16 memmap; never materialise the dict."""
    import torch
    stem = src.with_suffix("")
    raw_path, ids_path = Path(f"{stem}.f16"), Path(f"{stem}.ids.txt")
    if raw_path.exists():
        print(f"[memmap] {raw_path} already present; skipping")
        return

    class _Sink:
        def __init__(self):
            self.raw = open(raw_path, "wb", buffering=1 << 20)
            self.ids = open(ids_path, "w", buffering=1 << 20)
            self.n, self.dim, self.t0 = 0, None, time.time()

        def __setitem__(self, key, val):
            arr = val.detach().to(torch.float16).contiguous().numpy()
            if self.dim is None:
                self.dim = int(arr.shape[-1])
            self.raw.write(arr.tobytes())
            self.ids.write(f"{key}\n")
            self.n += 1
            if self.n % 2_000_000 == 0:
                print(f"  [stream] {self.n:,} ({time.time() - self.t0:.0f}s)", flush=True)

        def __len__(self):
            return self.n

        def close(self):
            self.raw.close()
            self.ids.close()

    class _StreamUnpickler(pickle._Unpickler):
        """Hands the top-level dict to the sink so each item is flushed and dropped."""

        def __init__(self, file, sink):
            super().__init__(file)
            self._sink, self._used = sink, False

        def load_empty_dictionary(self):
            if not self._used:
                self._used = True
                self.append(self._sink)
            else:
                self.append({})
        dispatch = dict(pickle._Unpickler.dispatch)
        dispatch[pickle.EMPTY_DICT[0]] = load_empty_dictionary

    sink = _Sink()
    print(f"[memmap] {src} -> {raw_path}", flush=True)
    with open(src, "rb") as fh:
        _StreamUnpickler(fh, sink).load()
    sink.close()
    Path(f"{stem}.shape.txt").write_text(f"{sink.n} {sink.dim}\n")
    got, expect = raw_path.stat().st_size, sink.n * sink.dim * 2
    print(f"[memmap] N={sink.n:,} D={sink.dim} "
          f"({'OK' if got == expect else 'SIZE MISMATCH'})", flush=True)


def plm_search(query_emb, target_emb, model_path, out, top_k, device_id, batch) -> None:
    import torch
    sys.path.insert(0, str(ROOT / "libs/PLMSearch-main/plmsearch"))
    from plmsearch_util.model import plmsearch

    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    with open(query_emb, "rb") as h:
        d = pickle.load(h)
    q_ids = list(d.keys())
    q = torch.stack([d[k] for k in q_ids]).float().to(device)
    del d
    Q, D = q.shape
    print(f"[load] {Q} queries, dim {D}", flush=True)

    model = plmsearch(embed_dim=D)
    model.load_pretrained(model_path)
    model.eval().to(device)
    with torch.no_grad():
        q_proj = model.forward(q)
        q_norm = q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)

    t_ids, t_np, N, _ = _load_memmap_targets(str(target_emb))
    t_mat = torch.from_numpy(t_np)
    K = min(top_k, N)

    def score_fn(tb):
        tb_norm = tb / tb.norm(dim=1, keepdim=True).clamp_min(1e-8)
        cos = q_norm @ tb_norm.t()
        signet = torch.sigmoid(q_proj @ tb.t())
        return torch.where(cos > 0.995, cos, cos * signet)

    top_s, top_i = _topk_stream(q_norm, t_mat, N, K, device, batch, score_fn, t0)
    _write_hits(out, q_ids, t_ids, top_s, top_i, header=False, min_score=0.0)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------
# DHR
# --------------------------------------------------------------------------
def dhr_aggregate(shard_tsv_glob: str, ebd_base: Path, out: Path, dim: int) -> None:
    """One faiss IndexFlatL2 over every shard's embeddings, in shard order."""
    import faiss
    import pandas as pd
    import torch
    from pyarrow import csv as pacsv

    def load_vec(pt_path: Path):
        obj = torch.load(str(pt_path), map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (list, tuple)):
            return obj[0] if len(obj) == 1 else torch.cat(list(obj), dim=0)
        if isinstance(obj, dict):
            tensors = [v for v in obj.values() if isinstance(v, torch.Tensor)]
            if not tensors:
                raise TypeError(f"no tensors in {pt_path}")
            return tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
        raise TypeError(f"unsupported type {type(obj)} in {pt_path}")

    tsvs = sorted(glob.glob(shard_tsv_glob))
    if not tsvs:
        raise SystemExit(f"[step1] no shard TSVs match {shard_tsv_glob}")
    print(f"[agg] {len(tsvs)} shards")

    index = faiss.IndexFlatL2(dim)
    parts = []
    for tsv in tsvs:
        name = Path(tsv).stem
        cands = sorted(glob.glob(str(Path(ebd_base) / name / "ebd" / "*" / "*.pt")))
        if not cands:
            raise FileNotFoundError(f"no .pt under {Path(ebd_base) / name}/ebd/")
        vec = load_vec(Path(cands[0]))
        df = pacsv.read_csv(
            tsv, read_options=pacsv.ReadOptions(column_names=["id", "sequence"]),
            parse_options=pacsv.ParseOptions(delimiter="\t")).to_pandas()
        if vec.shape[0] != len(df):
            raise SystemExit(f"[step1] {name}: {vec.shape[0]} embeddings vs {len(df)} seqs")
        index.add(vec.cpu().numpy())
        parts.append(df)
        print(f"  {name}: {vec.shape[0]:,} vecs", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_pickle(str(out / "df-ebd.pkl"))
    faiss.write_index(index, str(out / "index-ebd.index"))
    print(f"[OK] index size {index.ntotal:,} -> {out}/index-ebd.index")


def dhr_build_db(gpus: list[str], n_shards: int) -> None:
    src = Path(env("DHR_SRC_DIR"))
    py = env("DHR_PYTHON_PATH")
    shard_dir = DATA_DIR / "uniref50" / "shards"
    db_out = Path(os.environ.get("DHR_DB_OUT", src / "data" / "db_ur50"))
    ckpt = src / "dhr2_ckpt"
    if not (UR50_TSV.exists() and UR50_TSV.stat().st_size):
        raise SystemExit(f"[step1] {UR50_TSV} missing")

    shard_dir.mkdir(parents=True, exist_ok=True)
    have = sorted(shard_dir.glob("shard_*.tsv"))
    if len(have) != n_shards:
        print(f"[dhr] splitting into {n_shards} shards (had {len(have)})", flush=True)
        for p in have:
            p.unlink()
        subprocess.run(["split", "-n", f"l/{n_shards}", "-d", "-a", "2",
                        "--additional-suffix=.tsv", str(UR50_TSV),
                        str(shard_dir / "shard_")], check=True)
    shards = sorted(shard_dir.glob("shard_*.tsv"))

    def pt_of(shard: Path) -> Path:
        return db_out / shard.stem / "ebd" / "0" / "predictions.pt"

    todo = [s for s in shards if not (pt_of(s).exists() and pt_of(s).stat().st_size)]
    print(f"[dhr] {len(shards) - len(todo)}/{len(shards)} done; {len(todo)} to encode",
          flush=True)

    i = 0
    while i < len(todo):
        procs = []
        for gpu in gpus:
            if i >= len(todo):
                break
            shard = todo[i]
            outdir = db_out / shard.stem
            subprocess.run(["rm", "-rf", str(outdir / "ebd"), str(outdir / ".hydra")])
            outdir.mkdir(parents=True, exist_ok=True)
            log = open(outdir / "encode.log", "w")
            procs.append(subprocess.Popen(
                [py, "./do_embedding.py", f"trainer.ur90_path={ROOT / shard}",
                 f"model.ckpt_path={ckpt}", "trainer.gpus=[0]",
                 f"trainer.devices='{gpu}'", f"hydra.run.dir={outdir}"],
                cwd=src, stdout=log, stderr=subprocess.STDOUT))
            print(f"  GPU {gpu} <- {shard.name}", flush=True)
            i += 1
        for p in procs:
            p.wait()

    miss = [s.name for s in shards if not (pt_of(s).exists() and pt_of(s).stat().st_size)]
    if miss:
        raise SystemExit(f"[step1] {len(miss)} shards incomplete; re-run to resume")
    dhr_aggregate(str(shard_dir / "shard_*.tsv"), db_out, db_out / "agg", dim=480)


def dhr_search(top_n: int) -> None:
    src = Path(env("DHR_SRC_DIR"))
    py = env("DHR_PYTHON_PATH")
    agg = src / "data" / "db_ur50" / "agg"
    out_dir = DATA_DIR / "uniref50" / "dhr_search_astral4f"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (agg / "index-ebd.index").exists():
        raise SystemExit(f"[step1] DHR UR50 index missing: {agg}")

    q_tsv, d_tsv = _write_astral4f_query_tsvs()
    for qtsv, tag in ((q_tsv, "astral4f"), (d_tsv, f"astral4f_{DECOY_METHOD}")):
        dst = out_dir / f"{tag}.txt"
        if dst.exists() and dst.stat().st_size:
            print(f"  [skip] {tag} exists")
            continue
        subprocess.run([py, "./do_retrieval.py", "-i", str(ROOT / qtsv), "-d", str(agg),
                        "-o", str(dst), "-n", str(top_n)], cwd=src, check=True)
    # DHR returns distances; flip to a larger-is-better similarity
    for tag, dst in (("astral4f", raw_real("ur50", "dhr_postprocess")),
                     (f"astral4f_{DECOY_METHOD}",
                      raw_decoy("ur50", "dhr_postprocess"))):
        _dhr_to_score_table(out_dir / f"{tag}.txt", dst)


def _load_plmcaliper():
    """The released PLM-Caliper implementation, imported by path (as calibration.py does)."""
    import importlib.util
    path = ROOT / "src" / "PLMCaliper" / "PLMCaliper.py"
    if not path.exists():
        raise SystemExit(f"[step1] {path} is missing")
    spec = importlib.util.spec_from_file_location("PLMCaliper_core", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_astral4f_query_tsvs() -> tuple[Path, Path]:
    """The 4-family query set and its decoy, as the two-column TSV DHR takes."""
    import random
    random.seed(43)
    # DECOY_METHOD is extended_mkv2, so the decoy is seq + an order-2 Markov
    # resample of itself. Reuse PLMCaliper's own generator instead of open-coding
    # one here, so this path cannot drift from the design the calibration assumes.
    protein_markovgen = _load_plmcaliper().protein_markovgen
    seqs = read_fasta(DATA_DIR / "astral.fa")
    # astral.fa wins; astral4f_query.fa only fills ids it does not carry, so a
    # subset astral.fa still resolves every query while full sets are unchanged.
    qfa = DATA_DIR / "astral4f_query.fa"
    if qfa.exists() and qfa.stat().st_size:
        for k, v in read_fasta(qfa).items():
            seqs.setdefault(k, v)
    qids = read_astral4f_query_ids()
    q_tsv = DATA_DIR / "astral4f_query.tsv"
    d_tsv = DATA_DIR / f"astral4f_{DECOY_METHOD}.tsv"
    q_rows, d_rows = [], []
    for qid in qids:
        if qid not in seqs:
            continue
        s = seqs[qid]
        q_rows.append(f"{qid}\t{s}\n")
        d_rows.append(f"{qid}{DECOY_SUFFIX}\t{s}{protein_markovgen(s, MARKOV_ORDER)}\n")
    # q_tsv doubles as an id source for read_astral4f_query_ids(), so build the
    # rows first and never truncate it on a run that resolved nothing.
    if not q_rows:
        raise SystemExit(
            f"[step1] none of the {len(qids)} astral4f query ids have a sequence in "
            f"data/astral.fa or data/{qfa.name}; not writing empty query TSVs")
    q_tsv.write_text("".join(q_rows))
    d_tsv.write_text("".join(d_rows))
    print(f"  built {len(q_rows)} query + decoy rows", flush=True)
    return q_tsv, d_tsv


def _dhr_to_score_table(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    last, rank, seen = None, 0, set()
    invalid_faiss_distance = np.finfo(np.float32).max / 2
    skipped_invalid = skipped_duplicate = 0
    with open(src) as fh, open(dst, "w") as out:
        out.write("qid\ttid\tscore\thomo_type\trank\n")
        for line in fh:
            p = line.rstrip("\n").split(",")
            if len(p) < 3:
                continue
            if p[0] != last:
                last, rank, seen = p[0], 0, set()
            try:
                distance = float(p[2])
            except ValueError:
                skipped_invalid += 1
                continue
            if not math.isfinite(distance) or distance >= invalid_faiss_distance:
                skipped_invalid += 1
                continue
            if p[1] in seen:
                skipped_duplicate += 1
                continue
            seen.add(p[1])
            rank += 1
            out.write(f"{p[0]}\t{p[1]}\t{DHR_SCORE_CEILING - distance}\t-1\t{rank}\n")
    print(f"  wrote {dst}", flush=True)
    if skipped_invalid or skipped_duplicate:
        print(f"  skipped {skipped_invalid:,} invalid and "
              f"{skipped_duplicate:,} duplicate DHR hit(s)", flush=True)


# --------------------------------------------------------------------------
# MSA-side inputs (plain numpy/pandas)
# --------------------------------------------------------------------------
def build_cand_seqs(method: str, out: Path | None = None) -> None:
    """The subset of UR50 sequences any query retrieved, so step 3 need not rescan."""
    noisy = raw_real("ur50", method)
    out = out or (UR50_RESULTS / method / "iter1" / "parallel" / "cand_seqs.tsv")
    print(f"[1/2] reading candidate tids from {noisy.name} ...", flush=True)
    tids = set()
    with open(noisy) as fh:
        fh.readline()
        for line in fh:
            i = line.find("\t")
            j = line.find("\t", i + 1)
            if i > 0 and j > i:
                tids.add(line[i + 1:j])
    print(f"      {len(tids):,} unique tids", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[2/2] streaming {UR50_TSV.name} -> {out} ...", flush=True)
    n = found = 0
    with open(UR50_TSV) as fh, open(out, "w") as fo:
        for line in fh:
            n += 1
            i = line.find("\t")
            if i > 0 and line[:i] in tids:
                fo.write(line if line.endswith("\n") else line + "\n")
                found += 1
            if n % 5_000_000 == 0:
                print(f"      scanned {n:,}, found {found:,}", flush=True)
    print(f"[DONE] {found:,}/{len(tids):,} -> {out}")


def build_top400k(method: str, n_top: int) -> None:
    """One fasta per query holding its top-N retrieved targets: the top400k baseline."""
    from collections import defaultdict
    outdir = DATA_DIR / "ur50_top400k" / method
    outdir.mkdir(parents=True, exist_ok=True)
    rpath = raw_real("ur50", method)
    if not rpath.exists():
        raise SystemExit(f"[step1] retrieval file missing: {rpath}")

    print(f"[1/3] reading top-{n_top} tids/query from {rpath.name} ...", flush=True)
    want, per_q = defaultdict(list), defaultdict(int)
    with open(rpath) as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            if int(p[4]) <= n_top:
                want[p[1]].append(p[0])
                per_q[p[0]] += 1
    print(f"      {len(per_q)} queries, {len(want):,} unique tids", flush=True)

    handles = {}
    for qid in per_q:
        fp = outdir / f"{qid}.fa"
        if fp.exists() and fp.stat().st_size:            # resume
            continue
        handles[qid] = open(fp, "w")
    print(f"[2/3] writing {len(handles)} fastas "
          f"({len(per_q) - len(handles)} already done)", flush=True)
    if not handles:
        return

    print(f"[3/3] streaming {UR50_TSV.name} ...", flush=True)
    written, n_lines = defaultdict(int), 0
    with open(UR50_TSV) as fh:
        for line in fh:
            n_lines += 1
            i = line.find("\t")
            if i <= 0:
                continue
            qs = want.get(line[:i])
            if not qs:
                continue
            rec = f">{line[:i]}\n{line[i + 1:].rstrip()}\n"
            for qid in qs:
                h = handles.get(qid)
                if h is not None:
                    h.write(rec)
                    written[qid] += 1
            if n_lines % 5_000_000 == 0:
                print(f"      scanned {n_lines:,} ...", flush=True)
    for h in handles.values():
        h.close()
    print(f"[DONE] {len(handles)} fastas -> {outdir}")


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def drive_db(dataset: str, method: str, gpus: str) -> None:
    if dataset == "astral":
        print("[step1] astral searches ASTRAL40 against itself; no index to build")
        return
    gpu_list = gpus.split(",")
    if method == "tmvec":
        _self(env("TMVEC_PYTHON_PATH"), "tmvec-encode-db", "--gpus", gpus)
        _self(sys.executable, "tmvec-merge-db")
    elif method == "plm":
        _self(env("PLMSEARCH_PYTHON_PATH"), "plm-encode-db", "--gpu", gpu_list[0])
        _self(env("PLMSEARCH_PYTHON_PATH"), "plm-merge-db")
    elif method == "dhr_postprocess":
        _self(env("DHR_PYTHON_PATH"), "dhr-build-db", "--gpus", gpus)
    else:
        raise SystemExit(f"[step1] unknown method {method}")


def drive_search(dataset: str, method: str, gpu: str) -> None:
    if dataset == "astral":
        raise SystemExit(
            "[step1] the ASTRAL40 self-search tables are inputs to this experiment, "
            "not products of it.\n"
            "        Build them with src/PLM_searching_cmds/run_search.sh "
            "(real and decoy query).")
    if method == "tmvec":
        tdb = TMVEC_DB
        for name, tag in ((f"astral4f_query", "astral4f"),
                          (f"astral4f_{DECOY_METHOD}", f"astral4f_{DECOY_METHOD}")):
            emb = tdb / f"query_{tag}"
            _self(env("TMVEC_PYTHON_PATH"), "tmvec-encode-query",
                  "--query-tsv", DATA_DIR / f"{name}.tsv", "--out-prefix", emb)
            dst = (raw_real(dataset, method) if tag == "astral4f"
                   else raw_decoy(dataset, method))
            _self(env("TMVEC_PYTHON_PATH"), "tmvec-search",
                  "--query-emb", f"{emb}.npy", "--query-ids", f"{emb}.ids.txt",
                  "--target-emb", tdb / "ur50_embedding.f16", "--out", dst,
                  "--device", gpu)
    elif method == "plm":
        model = ROOT / "libs/PLMSearch-main/plmsearch_data/model/plmsearch.sav"
        for pkl, dst in ((PLM_DB / "query_astral4f_emb.pkl", raw_real(dataset, method)),
                         (PLM_DB / f"query_astral4f_{DECOY_METHOD}_emb.pkl",
                          raw_decoy(dataset, method))):
            _self(env("PLMSEARCH_PYTHON_PATH"), "plm-search",
                  "--query-emb", pkl, "--target-emb", PLM_DB / "ur50_embedding.f16",
                  "--model-path", model, "--out", dst, "--device", gpu)
    elif method == "dhr_postprocess":
        _self(env("DHR_PYTHON_PATH"), "dhr-search")
    else:
        raise SystemExit(f"[step1] unknown method {method}")


def _self(python_exe: str, stage: str, *args) -> None:
    cmd = [python_exe, __file__, stage, *[str(a) for a in args]]
    print("[step1] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 1: build the search database and the raw score tables step 2 reads.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=[
        "check", "db", "search", "split-fasta",
        "tmvec-encode-db", "tmvec-encode-worker", "tmvec-merge-db",
        "tmvec-encode-query", "tmvec-search",
        "plm-encode-db", "plm-merge-db", "plm-search",
        "dhr-build-db", "dhr-aggregate", "dhr-search",
        "cand-seqs", "top400k"])
    p.add_argument("--dataset", choices=sorted(DATASETS), default="ur50")
    p.add_argument("--method", default="dhr_postprocess")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--gpu", default="0")
    p.add_argument("--seqs-per-chunk", type=int, default=None,
                   help="default: 50k for tmvec, 500k for plm")
    p.add_argument("--n-shards", type=int, default=60, help="dhr encode shards")
    p.add_argument("--top-k", type=int, default=1_000_000)
    p.add_argument("--target-batch", type=int, default=2_000_000)
    p.add_argument("--max-len", type=int, default=TMVEC_MAX_LEN)
    p.add_argument("--n-top", type=int, default=400_000, help="top400k baseline size")
    # stage-local paths
    p.add_argument("--chunk-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--query-tsv", type=Path, default=None)
    p.add_argument("--query-emb", type=Path, default=None)
    p.add_argument("--query-ids", type=Path, default=None)
    p.add_argument("--target-emb", type=Path, default=None)
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--out-prefix", default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--device", default="0")
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--delete-chunks-after", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    s = a.stage

    if s == "check":
        raise SystemExit(0 if check(a.dataset, a.method) else 1)
    if s == "db":
        drive_db(a.dataset, a.method, a.gpus)
    elif s == "search":
        drive_search(a.dataset, a.method, a.gpu)
    elif s == "split-fasta":
        split_fasta(UR50_FASTA, a.chunk_dir or TMVEC_CHUNKS, a.seqs_per_chunk or 50_000)
    elif s == "tmvec-encode-db":
        raise SystemExit(tmvec_encode_db(a.gpus.split(","), a.seqs_per_chunk or 50_000))
    elif s == "tmvec-encode-worker":
        tmvec_encode_worker(a.chunk_dir or TMVEC_CHUNKS, a.out_dir or TMVEC_DB / "chunks",
                            a.worker_id, a.num_workers, a.max_len)
    elif s == "tmvec-merge-db":
        tmvec_merge_db(a.allow_incomplete, a.delete_chunks_after)
    elif s == "tmvec-encode-query":
        tmvec_encode_query(a.query_tsv, a.out_prefix, a.max_len)
    elif s == "tmvec-search":
        tmvec_search(a.query_emb, a.query_ids, str(a.target_emb), a.out,
                     a.top_k, a.device, a.target_batch)
    elif s == "plm-encode-db":
        plm_encode_db(a.gpu, a.seqs_per_chunk or 500_000)
    elif s == "plm-merge-db":
        plm_merge_db()
    elif s == "plm-search":
        plm_search(a.query_emb, a.target_emb, str(a.model_path), a.out,
                   a.top_k, a.device, min(a.target_batch, 1_000_000))
    elif s == "dhr-build-db":
        dhr_build_db(a.gpus.split(","), a.n_shards)
    elif s == "dhr-aggregate":
        src = Path(env("DHR_SRC_DIR")) / "data" / "db_ur50"
        dhr_aggregate(str(DATA_DIR / "uniref50" / "shards" / "shard_*.tsv"),
                      src, src / "agg", dim=480)
    elif s == "dhr-search":
        dhr_search(a.top_k)
    elif s == "cand-seqs":
        build_cand_seqs(a.method, a.out)
    elif s == "top400k":
        build_top400k(a.method, a.n_top)

    if s in ("db", "search"):
        print("\n[step1] inputs for step 2:")
        check(a.dataset, a.method)


if __name__ == "__main__":
    main()
