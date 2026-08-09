#!/usr/bin/env python3
import glob
import math
import os
import pickle
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 5)))
CHUNK_DIR = os.path.join(ROOT, "data/db/db_ur50_plm/chunks")
GROUP_DIR = os.path.join(ROOT, "data/db/db_ur50_plm/groups")
OUT = os.path.join(ROOT, "data/db/db_ur50_plm/ur50_embedding.pkl")
N_GROUPS = 4


def _free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / (1024 ** 3)


def stage1():
    os.makedirs(GROUP_DIR, exist_ok=True)
    chunks = sorted(glob.glob(os.path.join(CHUNK_DIR, "chunk_*.pkl")))
    if not chunks:
        print("[stage1] no chunks left -> already staged into groups; skipping")
        return
    per = math.ceil(len(chunks) / N_GROUPS)
    print(f"[stage1] {len(chunks)} chunks -> {N_GROUPS} groups (~{per} chunks each)")
    for g in range(N_GROUPS):
        gpath = os.path.join(GROUP_DIR, f"group_{g}.pkl")
        sub = chunks[g * per:(g + 1) * per]
        if not sub:
            continue
        if os.path.exists(gpath):
            print(f"[stage1] group_{g}.pkl exists -> deleting its source chunks & skipping")
            for f in sub:
                if os.path.exists(f):
                    os.remove(f)
            continue
        merged = {}
        for f in sub:
            with open(f, "rb") as h:
                merged.update(pickle.load(h))
        tmp = gpath + ".tmp"
        with open(tmp, "wb") as h:
            pickle.dump(merged, h, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, gpath)
        print(f"[stage1] group_{g}: {len(sub)} chunks -> {gpath} "
              f"({len(merged)} emb, free={_free_gb(GROUP_DIR):.0f}G)")
        for f in sub:
            os.remove(f)
        print(f"[stage1] freed {len(sub)} chunks (free={_free_gb(GROUP_DIR):.0f}G)")


def stage2():
    if os.path.exists(OUT):
        print(f"[stage2] {OUT} already exists ({_free_gb(GROUP_DIR):.0f}G free); done")
        return
    groups = sorted(glob.glob(os.path.join(GROUP_DIR, "group_*.pkl")))
    if not groups:
        sys.exit("[stage2] no group pkls found and no output -> nothing to do")
    print(f"[stage2] merging {len(groups)} groups -> {OUT}")
    merged = {}
    for f in groups:
        with open(f, "rb") as h:
            merged.update(pickle.load(h))
        os.remove(f)  
        print(f"[stage2] loaded+freed {os.path.basename(f)} "
              f"(total {len(merged)} emb, free={_free_gb(GROUP_DIR):.0f}G)")
    tmp = OUT + ".tmp"
    with open(tmp, "wb") as h:
        pickle.dump(merged, h, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, OUT)
    print(f"[stage2] DONE -> {OUT} ({len(merged)} embeddings, free={_free_gb(OUT):.0f}G)")


if __name__ == "__main__":
    print(f"[start] free={_free_gb(ROOT):.0f}G")
    stage1()
    stage2()
    print("[all done]")
