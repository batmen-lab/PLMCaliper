"""Search query sequences against UniRef50 with TM-Vec, PLMsearch or DHR.

    source tutorials/PLM_searching_cmds/.env
    python tutorials/PLM_searching_cmds/search_ur50.py -m <tmvec|plm|dhr> -q <query.fa> -o <score table>

The first search of a method embeds all of data/uniref50/uniref50.fasta with it, one worker per GPU
in --gpus, into data/db/db_ur50_<method>/; later searches reuse that index, and an interrupted build
resumes where it stopped. The score table holds the top --top-k targets of every query as
`qid tid score homo_type rank`, a higher score meaning more similar.

Each method runs in its own Python environment, named in .env; this script reruns itself there
through --stage, so the Python that calls it only needs numpy.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UR50_FASTA = ROOT / "data" / "uniref50" / "uniref50.fasta"
CHUNKS = ROOT / "data" / "uniref50" / "chunks"       # FASTA chunks that TM-Vec and PLMsearch embed
DHR_SHARDS = ROOT / "data" / "uniref50" / "shards"   # id<TAB>seq shards that DHR embeds
DB = {m: ROOT / "data" / "db" / f"db_ur50_{m}" for m in ("tmvec", "plm", "dhr")}
PYTHON = {"tmvec": "TMVEC_PYTHON_PATH", "plm": "PLMSEARCH_PYTHON_PATH", "dhr": "DHR_PYTHON_PATH"}

CHUNK_SEQS = 50_000
DHR_SHARD_SEQS = 650_000    # 60 shards of UniRef50
DHR_DIM = 480
# DHR reports distances; subtract from this to get larger-is-better, the same
# constant tutorials/utils/process_searching_score.py uses.
DHR_SCORE_CEILING = 150.0
# ProtT5 self-attention is O(L^2) in memory and UniRef50 has titin-scale sequences,
# so cap length the way ESM/ProtT5 do. ASTRAL queries are short domains.
TMVEC_MAX_LEN = 1022
PROTRANS_MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"


def env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"[search_ur50] {name} unset -- run `source tutorials/PLM_searching_cmds/.env` first")
    return v


def iter_fasta(path):
    """(id, sequence) of every record; the id is the first word of the header."""
    name, buf = None, []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf)
                name, buf = line[1:].split(None, 1)[0], []
            else:
                buf.append(line.strip())
    if name is not None:
        yield name, "".join(buf)


# --------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------
def split_ur50(out_dir: Path, per_part: int, tsv: bool) -> list[Path]:
    """UniRef50 in parts of per_part sequences, as FASTA or as id<TAB>seq; split once."""
    suffix, prefix = (".tsv", "shard") if tsv else (".fasta", "chunk")
    if not (out_dir / ".split_done").exists():
        if not UR50_FASTA.is_file():
            raise SystemExit(f"[search_ur50] {UR50_FASTA} missing")
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob(f"{prefix}_*{suffix}"):
            old.unlink()
        print(f"[split] {UR50_FASTA} -> {out_dir} ({per_part:,} seqs each)", flush=True)
        out = None
        for i, (name, seq) in enumerate(iter_fasta(UR50_FASTA)):
            if i % per_part == 0:
                if out:
                    out.close()
                out = open(out_dir / f"{prefix}_{i // per_part:04d}{suffix}", "w")
            out.write(f"{name}\t{seq}\n" if tsv else f">{name}\n{seq}\n")
        out.close()
        (out_dir / ".split_done").touch()
    return sorted(out_dir.glob(f"{prefix}_*{suffix}"))


def dhr_embedding(shard: Path) -> Path:
    return DB["dhr"] / shard.stem / "ebd" / "0" / "predictions.pt"


def is_embedded(method: str, part: Path) -> bool:
    done = dhr_embedding(part) if method == "dhr" else DB[method] / f"{part.stem}.ids.txt"
    return done.exists() and done.stat().st_size > 0


def build_index(method: str, gpus: list[str]) -> None:
    """Embed every UniRef50 sequence with one method, one worker per GPU; a finished index is reused."""
    db = DB[method]
    done = db / "agg" / "index-ebd.index" if method == "dhr" else db / ".done"
    if done.exists():
        print(f"[index] {method}: reusing {db}", flush=True)
        return
    if method == "dhr":
        parts = split_ur50(DHR_SHARDS, DHR_SHARD_SEQS, tsv=True)
    else:
        parts = split_ur50(CHUNKS, CHUNK_SEQS, tsv=False)
    db.mkdir(parents=True, exist_ok=True)
    print(f"[index] {method}: {sum(is_embedded(method, p) for p in parts)}/{len(parts)} parts embedded; "
          f"workers on GPUs {','.join(gpus)}", flush=True)
    workers = [subprocess.Popen([env(PYTHON[method]), __file__, "-m", method, "--stage", "encode",
                                 "--worker-id", str(i), "--num-workers", str(len(gpus))],
                                cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu})
               for i, gpu in enumerate(gpus)]
    for w in workers:
        w.wait()
    missing = [p.name for p in parts if not is_embedded(method, p)]
    if missing:
        raise SystemExit(f"[search_ur50] {len(missing)}/{len(parts)} parts not embedded "
                         f"(e.g. {missing[:3]}); re-run to resume")
    if method == "dhr":
        subprocess.run([env(PYTHON["dhr"]), __file__, "-m", "dhr", "--stage", "dhr-aggregate"], cwd=ROOT, check=True)
    else:
        done.touch()
    print(f"[index] {method}: done -> {db}", flush=True)


def encode(method: str, worker_id: int, num_workers: int) -> None:
    """One worker's share of the UniRef50 parts, run in the method's environment on its one GPU."""
    db = DB[method]
    parts = sorted(DHR_SHARDS.glob("shard_*.tsv")) if method == "dhr" else sorted(CHUNKS.glob("chunk_*.fasta"))
    todo = [p for p in parts[worker_id::num_workers] if not is_embedded(method, p)]
    print(f"[w{worker_id}] {len(todo)} parts to embed", flush=True)
    if not todo:
        return
    if method == "dhr":
        src = Path(env("DHR_SRC_DIR"))
        for shard in todo:
            out = db / shard.stem
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir(parents=True)
            t0 = time.time()
            # do_embedding.py sets CUDA_VISIBLE_DEVICES from trainer.devices itself
            with open(out / "encode.log", "w") as log:
                subprocess.run([sys.executable, "./do_embedding.py", f"trainer.ur90_path={shard}",
                                f"model.ckpt_path={src / 'dhr2_ckpt'}", "trainer.gpus=[0]",
                                f"trainer.devices='{os.environ['CUDA_VISIBLE_DEVICES']}'", f"hydra.run.dir={out}"],
                               cwd=src, stdout=log, stderr=subprocess.STDOUT)
            print(f"[w{worker_id}] {shard.stem} {'ok' if is_embedded('dhr', shard) else 'FAILED'} "
                  f"{time.time() - t0:.0f}s", flush=True)
        return

    embed = tmvec_embedder() if method == "tmvec" else plm_embedder()
    for chunk in todo:
        t0 = time.time()
        ids, seqs = zip(*iter_fasta(chunk))
        emb = embed(list(seqs)).astype(np.float16)
        with open(db / f"{chunk.stem}.npy.tmp", "wb") as f:
            np.save(f, emb)
        os.replace(db / f"{chunk.stem}.npy.tmp", db / f"{chunk.stem}.npy")
        (db / f"{chunk.stem}.ids.txt").write_text("\n".join(ids) + "\n")
        print(f"[w{worker_id}] {chunk.stem} {emb.shape} {time.time() - t0:.0f}s", flush=True)


def dhr_aggregate() -> None:
    """One faiss IndexFlatL2 over every shard's embeddings, in shard order."""
    import faiss
    import pandas as pd
    import torch
    from pyarrow import csv as pacsv

    def load_vec(pt_path: Path):
        obj = torch.load(str(pt_path), map_location="cpu")
        if isinstance(obj, (list, tuple)):
            obj = obj[0] if len(obj) == 1 else torch.cat(list(obj), dim=0)
        elif isinstance(obj, dict):
            tensors = [v for v in obj.values() if isinstance(v, torch.Tensor)]
            obj = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
        return obj

    agg = DB["dhr"] / "agg"
    index, tables = faiss.IndexFlatL2(DHR_DIM), []
    for shard in sorted(DHR_SHARDS.glob("shard_*.tsv")):
        vec = load_vec(dhr_embedding(shard))
        df = pacsv.read_csv(shard, read_options=pacsv.ReadOptions(column_names=["id", "sequence"]),
                            parse_options=pacsv.ParseOptions(delimiter="\t")).to_pandas()
        if vec.shape[0] != len(df):
            raise SystemExit(f"[search_ur50] {shard.stem}: {vec.shape[0]} embeddings vs {len(df)} seqs")
        index.add(vec.cpu().numpy())
        tables.append(df)
        print(f"  {shard.stem}: {vec.shape[0]:,} vecs", flush=True)
    agg.mkdir(parents=True, exist_ok=True)
    pd.concat(tables, ignore_index=True).to_pickle(str(agg / "df-ebd.pkl"))
    faiss.write_index(index, str(agg / "index-ebd.index.tmp"))   # the index file marks the build done
    os.replace(agg / "index-ebd.index.tmp", agg / "index-ebd.index")
    print(f"[OK] index size {index.ntotal:,} -> {agg}/index-ebd.index", flush=True)


# --------------------------------------------------------------------------
# embedders (run in the method's environment)
# --------------------------------------------------------------------------
def tmvec_embedder():
    """sequences -> 512-d TM-Vec embeddings (fp32), from half-precision ProtT5-XL features."""
    import torch
    from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
    from transformers import T5EncoderModel, T5Tokenizer

    src, device = Path(env("TMVEC_SRC_DIR")), torch.device("cuda")
    tokenizer = T5Tokenizer.from_pretrained(PROTRANS_MODEL, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROTRANS_MODEL, torch_dtype=torch.float16).to(device).eval()
    cfg = trans_basic_block_Config.from_json(str(src / "model/tm_vec_cath_model_params.json"))
    deep = trans_basic_block.load_from_checkpoint(str(src / "model/tm_vec_cath_model.ckpt"),
                                                  config=cfg, map_location=device).to(device).eval()

    def embed_batch(seqs):
        enc = tokenizer.batch_encode_plus([re.sub(r"[UZOB]", "X", " ".join(s)) for s in seqs],
                                          add_special_tokens=True, padding="longest")
        ids = torch.tensor(enc["input_ids"], device=device)
        attn = torch.tensor(enc["attention_mask"], device=device)
        with torch.no_grad():
            h = model(input_ids=ids, attention_mask=attn).last_hidden_state
            lens = attn.sum(dim=1).long() - 1            # residues per seq, drop EOS
            lmax = int(lens.max().item())
            pad = torch.arange(lmax, device=device).unsqueeze(0) >= lens.unsqueeze(1)
            return deep(h[:, :lmax, :].float(), src_mask=None, src_key_padding_mask=pad).cpu().numpy()

    def embed(seqs, max_batch_tokens=24576, max_batch=192):
        # length-sorted dynamic batching; a batch that runs out of memory is retried one sequence at a time
        seqs = [s.replace("*", "")[:TMVEC_MAX_LEN] for s in seqs]
        out = np.zeros((len(seqs), 512), dtype=np.float32)
        order = sorted(range(len(seqs)), key=lambda k: len(seqs[k]))
        i, n_fail = 0, 0
        while i < len(order):
            j, lmax = i, len(seqs[order[i]]) + 1
            while j < len(order) and (j - i) < max_batch:
                L = max(lmax, len(seqs[order[j]]) + 1)
                if (j - i + 1) * L > max_batch_tokens and j > i:
                    break
                lmax, j = L, j + 1
            idx = order[i:j]
            try:
                out[idx] = embed_batch([seqs[k] for k in idx])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                for k in idx:
                    try:
                        out[k] = embed_batch([seqs[k]])[0]
                    except torch.cuda.OutOfMemoryError:  # left as a zero vector
                        torch.cuda.empty_cache()
                        n_fail += 1
            i = j
        if n_fail:
            print(f"  [tmvec] {n_fail} sequences ran out of GPU memory; left as zero vectors", flush=True)
        return out

    return embed


def plm_embedder():
    """sequences -> 1280-d mean ESM-1b embeddings (fp32), batched as PLMsearch's embedding_generate_esm1b.py."""
    import torch
    from esm import FastaBatchedDataset, pretrained

    esm = Path(env("PLMSEARCH_SRC_DIR")) / "plmsearch_data/model/esm/esm1b_t33_650M_UR50S.pt"
    model, alphabet = pretrained.load_model_and_alphabet(str(esm))
    model = model.eval().cuda()
    convert = alphabet.get_batch_converter(1022)

    def embed(seqs):
        ds = FastaBatchedDataset(list(range(len(seqs))), seqs)
        out = np.zeros((len(seqs), 1280), dtype=np.float32)
        with torch.no_grad():
            for batch in ds.get_batch_indices(16384, extra_toks_per_seq=1):
                labels, strs, toks = convert([ds[k] for k in batch])
                rep = model(toks.cuda(non_blocking=True), repr_layers=[33], return_contacts=False)["representations"][33]
                for i, label in enumerate(labels):
                    out[label] = rep[i, 1:len(strs[i]) + 1].mean(0).cpu().numpy()
        return out

    return embed


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------
def search_embeddings(method: str, query: Path, out: Path, top_k: int) -> None:
    """TM-Vec / PLMsearch: embed the queries, then keep each one's top_k targets over the index chunks."""
    import torch
    t0 = time.time()
    q_ids, q_seqs = zip(*iter_fasta(query))
    embed = tmvec_embedder() if method == "tmvec" else plm_embedder()
    q = torch.from_numpy(embed(list(q_seqs))).cuda()
    del embed
    torch.cuda.empty_cache()
    q_norm = q / q.norm(dim=1, keepdim=True).clamp_min(1e-8)

    if method == "tmvec":                                # cosine similarity
        def score_fn(tb):
            return q_norm @ (tb / tb.norm(dim=1, keepdim=True).clamp_min(1e-8)).t()
        min_score = -1.5
    else:                                                # PLMsearch: cosine, reweighted by its SS-predictor
        src = Path(env("PLMSEARCH_SRC_DIR"))
        sys.path.insert(0, str(src / "plmsearch"))
        from plmsearch_util.model import plmsearch
        model = plmsearch(embed_dim=q.shape[1])
        model.load_pretrained(str(src / "plmsearch_data/model/plmsearch.sav"))
        model.eval().cuda()
        with torch.no_grad():
            q_proj = model.forward(q)

        def score_fn(tb):
            cos = q_norm @ (tb / tb.norm(dim=1, keepdim=True).clamp_min(1e-8)).t()
            return torch.where(cos > 0.995, cos, cos * torch.sigmoid(q_proj @ tb.t()))
        min_score = 0.0

    # running top-k over the fp16 index, one chunk at a time; unfilled slots keep score -2
    top_s = torch.full((len(q_ids), top_k), -2.0, device="cuda")
    top_i = torch.zeros((len(q_ids), top_k), dtype=torch.long, device="cuda")
    t_ids = []
    chunks = sorted(DB[method].glob("chunk_*.npy"))
    with torch.no_grad():
        for n, chunk in enumerate(chunks):
            tb = torch.from_numpy(np.load(chunk)).cuda().float()
            gidx = torch.arange(len(t_ids), len(t_ids) + len(tb), device="cuda").expand(len(q_ids), -1)
            t_ids += chunk.with_suffix(".ids.txt").read_text().split()
            sel = torch.cat([top_s, score_fn(tb)], dim=1).topk(top_k, dim=1)
            top_s, top_i = sel.values, torch.gather(torch.cat([top_i, gidx], dim=1), 1, sel.indices)
            if n % 100 == 0:
                print(f"[search] {len(t_ids):,} targets ({time.time() - t0:.0f}s)", flush=True)

    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w") as f:
        f.write("qid\ttid\tscore\thomo_type\trank\n")
        for qid, s_row, i_row in zip(q_ids, top_s.cpu().tolist(), top_i.cpu().tolist()):
            for rank, (sc, ti) in enumerate(zip(s_row, i_row), start=1):
                if sc < min_score:                       # sorted, so the rest are unfilled too
                    break
                f.write(f"{qid}\t{t_ids[ti]}\t{round(sc, 4)}\t-1\t{rank}\n")
    os.replace(tmp, out)
    print(f"[done] {len(q_ids)} queries x {len(t_ids):,} targets, {time.time() - t0:.0f}s", flush=True)


def search_dhr(query: Path, out: Path, top_k: int, gpu: str) -> None:
    """DHR: retrieve with its faiss index, then flip the distances into larger-is-better scores."""
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        q_tsv, hits = Path(tmp) / "query.tsv", Path(tmp) / "hits.csv"
        q_tsv.write_text("".join(f"{name}\t{seq}\n" for name, seq in iter_fasta(query)))
        subprocess.run([env("DHR_PYTHON_PATH"), "./do_retrieval.py", "-i", str(q_tsv), "-d", str(DB["dhr"] / "agg"),
                        "-o", str(hits), "-n", str(top_k)],
                       cwd=env("DHR_SRC_DIR"), env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}, check=True)

        last, rank, seen, skipped = None, 0, set(), 0
        invalid_faiss_distance = np.finfo(np.float32).max / 2
        with open(hits) as fh, open(Path(tmp) / "table.txt", "w") as f:
            f.write("qid\ttid\tscore\thomo_type\trank\n")
            for line in fh:                              # qid,tid,distance, nearest first
                p = line.rstrip("\n").split(",")
                if len(p) < 3:
                    continue
                if p[0] != last:
                    last, rank, seen = p[0], 0, set()
                try:
                    distance = float(p[2])
                except ValueError:
                    distance = np.inf
                if not np.isfinite(distance) or distance >= invalid_faiss_distance or p[1] in seen:
                    skipped += 1
                    continue
                seen.add(p[1])
                rank += 1
                f.write(f"{p[0]}\t{p[1]}\t{DHR_SCORE_CEILING - distance}\t-1\t{rank}\n")
        os.replace(Path(tmp) / "table.txt", out)
    if skipped:
        print(f"  skipped {skipped:,} invalid or repeated DHR hit(s)", flush=True)


# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--method", required=True, choices=sorted(PYTHON))
    p.add_argument("-q", "--query", type=Path, help="query FASTA")
    p.add_argument("-o", "--out", type=Path, help="score table to write")
    p.add_argument("--gpus", default="0,1,2,3", help="GPUs that build the index, one worker each")
    p.add_argument("--gpu", default="0", help="GPU that runs the search")
    p.add_argument("--top-k", type=int, default=1_000_000, help="targets kept per query")
    # internal: the steps this script runs in each method's own environment
    p.add_argument("--stage", choices=["encode", "search", "dhr-aggregate"], help=argparse.SUPPRESS)
    p.add_argument("--worker-id", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--num-workers", type=int, default=1, help=argparse.SUPPRESS)
    a = p.parse_args()

    if a.stage == "encode":
        encode(a.method, a.worker_id, a.num_workers)
    elif a.stage == "search":
        search_embeddings(a.method, a.query, a.out, a.top_k)
    elif a.stage == "dhr-aggregate":
        dhr_aggregate()
    else:
        if not (a.query and a.out):
            p.error("-q and -o are required")
        query, out = a.query.resolve(), a.out.resolve()
        build_index(a.method, a.gpus.split(","))
        out.parent.mkdir(parents=True, exist_ok=True)
        if a.method == "dhr":
            search_dhr(query, out, a.top_k, a.gpu)
        else:
            subprocess.run([env(PYTHON[a.method]), __file__, "-m", a.method, "--stage", "search",
                            "-q", str(query), "-o", str(out), "--top-k", str(a.top_k)],
                           cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": a.gpu}, check=True)
        print(f"[search_ur50] {a.method}: {query.name} -> {out}", flush=True)


if __name__ == "__main__":
    main()
