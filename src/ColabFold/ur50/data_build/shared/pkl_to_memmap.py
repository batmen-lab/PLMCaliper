#!/usr/bin/env python3
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch


class _StreamSink:
    """A dict-like object the unpickler writes (id -> tensor) into; each item is
    flushed to disk as fp16 and dropped instead of being retained."""

    def __init__(self, raw_path: Path, ids_path: Path):
        self.raw = open(raw_path, "wb", buffering=1024 * 1024)
        self.ids = open(ids_path, "w", buffering=1024 * 1024)
        self.n = 0
        self.dim = None
        self.t0 = time.time()

    def __setitem__(self, key, val):
        arr = val.detach().to(torch.float16).contiguous().numpy()
        if self.dim is None:
            self.dim = int(arr.shape[-1])
        self.raw.write(arr.tobytes())
        self.ids.write(f"{key}\n")
        self.n += 1
        if self.n % 2_000_000 == 0:
            print(f"  [stream] {self.n:,} embeddings ({time.time()-self.t0:.0f}s)", flush=True)

    # the unpickler may call these on the top dict; make it dict-compatible enough
    def __len__(self):
        return self.n

    def close(self):
        self.raw.close()
        self.ids.close()


class _StreamUnpickler(pickle._Unpickler):
    """Pure-python unpickler whose OUTERMOST empty-dict is replaced by a sink, so
    top-level (id -> tensor) items stream to disk. Nested dicts (none in this
    data) would also hit the sink, which is fine for a flat {id: tensor} map."""

    def __init__(self, file, sink):
        super().__init__(file)
        self._sink = sink
        self._used = False

    def load_empty_dictionary(self):
        if not self._used:
            self._used = True
            self.append(self._sink)
        else:
            self.append({})
    dispatch = dict(pickle._Unpickler.dispatch)
    dispatch[pickle.EMPTY_DICT[0]] = load_empty_dictionary


def main() -> None:
    src = Path(sys.argv[1])
    stem = src.with_suffix("")  # drop .pkl
    raw_path = Path(f"{stem}.f16")
    ids_path = Path(f"{stem}.ids.txt")
    shape_path = Path(f"{stem}.shape.txt")

    if raw_path.exists():
        sys.exit(f"[abort] {raw_path} already exists; remove it to re-run")

    sink = _StreamSink(raw_path, ids_path)
    print(f"[stream] {src} -> {raw_path} (+ .ids.txt)", flush=True)
    with open(src, "rb") as fh:
        _StreamUnpickler(fh, sink).load()
    sink.close()

    shape_path.write_text(f"{sink.n} {sink.dim}\n")
    expect = sink.n * sink.dim * 2
    got = raw_path.stat().st_size
    print(f"[done] N={sink.n:,} D={sink.dim} -> {raw_path}", flush=True)
    print(f"[check] raw size {got:,} bytes, expected {expect:,} "
          f"({'OK' if got == expect else 'MISMATCH'})", flush=True)


if __name__ == "__main__":
    main()
