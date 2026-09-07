import argparse
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
DEFAULT_OUT_PARENT = BASE_DIR / "results" / "bagel"
DEFAULT_SEARCH_METHOD = "plm"
DEFAULT_DECOY_METHOD = "extended_shuf"


@dataclass(frozen=True)
class AlignmentResult:
    pairs: list[tuple[int, int]]
    score: float
    query_start: int
    query_end: int
    target_start: int
    target_end: int


@dataclass(frozen=True)
class PairToPlot:
    qid: str
    tid: str
    rank: int | None = None
    score_real: float | None = None
    score_decoy: float | None = None
    q: float | None = None
    rep_id: int | None = None
    n_rep_support: int | None = None


@dataclass(frozen=True)
class HeatmapComputation:
    heat: np.ndarray
    aln: AlignmentResult | None
    label: str
    source: str


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    current_id: str | None = None

    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_id = line[1:].split()[0]
                records[current_id] = []
            elif current_id is not None:
                records[current_id].append(line)

    return {seq_id: "".join(parts).upper() for seq_id, parts in records.items()}


def read_id_list(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line.split("\t", 1)[0])
    return ids


def sanitize_filename(text: str, max_len: int = 90) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("_")
    return text[:max_len] if len(text) > max_len else text


def append_extension(path: Path, extension: str) -> Path:
    extension = extension if extension.startswith(".") else f".{extension}"
    return path.parent / f"{path.name}{extension}"


def default_single_hits_file(search_method: str, decoy_method: str, q: float) -> Path:
    return DATA_DIR / f"putative_discovery_hits_{search_method}_{decoy_method}_q{q:.2f}.tsv"


def default_qscan_hits_file(search_method: str, decoy_method: str) -> Path:
    return DATA_DIR / f"putative_discovery_hits_scan_{search_method}_{decoy_method}.tsv"


def default_qscan_summary_file(search_method: str, decoy_method: str) -> Path:
    return DATA_DIR / f"putative_discovery_scan_{search_method}_{decoy_method}.tsv"


def legacy_single_hits_file(decoy_method: str, q: float) -> Path:
    return DATA_DIR / f"putative_discovery_hits_{decoy_method}_q{q:.2f}.tsv"


def legacy_qscan_hits_file(decoy_method: str) -> Path:
    return DATA_DIR / f"putative_discovery_hits_scan_{decoy_method}.tsv"


def legacy_qscan_summary_file(decoy_method: str) -> Path:
    return DATA_DIR / f"putative_discovery_scan_{decoy_method}.tsv"


def resolve_default_table(primary: Path, legacy: Path | None, label: str) -> Path:
    if primary.exists():
        return primary
    if legacy is not None and legacy.exists():
        print(f"[WARN] Using legacy {label} without search-method in filename: {legacy}")
        return legacy
    return primary


def sequence_ticks(length: int, max_ticks: int = 9) -> list[int]:
    if length <= 0:
        return []
    if length == 1:
        return [0]
    ticks = np.linspace(0, length - 1, num=min(max_ticks, length))
    ticks = sorted({int(round(x)) for x in ticks})
    if ticks[0] != 0:
        ticks.insert(0, 0)
    if ticks[-1] != length - 1:
        ticks.append(length - 1)
    return ticks


def parse_bagel_label(seq_id: str) -> tuple[str, str, str]:
    left, _, right = seq_id.partition(";")
    parts = left.split(".")
    class_code = parts[1] if len(parts) >= 2 else "?"
    class_label = f"Class {class_code}" if class_code in {"1", "2", "3"} else "Class ?"
    display_name = right if right else seq_id
    return class_code, class_label, display_name


def short_axis_label(seq_id: str, *, prefix: str = "") -> str:
    _, class_label, display_name = parse_bagel_label(seq_id)
    if prefix:
        return f"{prefix}\n{seq_id}"
    return f"{class_label}, {display_name}"


def limit_rows(df: pd.DataFrame, top_n: int | None) -> pd.DataFrame:
    if top_n is None or top_n <= 0:
        return df
    return df.head(top_n)


def load_hits_for_qid(path: Path, qid: str, top_n: int | None) -> list[PairToPlot]:
    if not path.exists():
        raise FileNotFoundError(f"Missing hits table: {path}")

    df = pd.read_csv(path, sep="\t")
    required = {"qid", "tid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df[df["qid"].astype(str) == qid].copy()
    if df.empty:
        raise ValueError(f"No rows for qid={qid!r} in {path}")

    agg_spec = {}
    if "score_real" in df.columns:
        agg_spec["score_real"] = ("score_real", "mean")
    if "score_decoy" in df.columns:
        agg_spec["score_decoy"] = ("score_decoy", "mean")
    if "hit_rank" in df.columns:
        agg_spec["hit_rank"] = ("hit_rank", "min")
    if "q" in df.columns:
        agg_spec["q"] = ("q", "first")
    if "rep_id" in df.columns:
        agg_spec["rep_id"] = ("rep_id", "first")

    if agg_spec:
        pooled = df.groupby("tid", as_index=False).agg(**agg_spec)
    else:
        pooled = df.drop_duplicates("tid").copy()
        pooled["hit_rank"] = np.arange(1, len(pooled) + 1)

    if "hit_rank" in pooled.columns:
        pooled = pooled.sort_values(["hit_rank", "score_real"] if "score_real" in pooled.columns else ["hit_rank"],
                                    ascending=[True, False] if "score_real" in pooled.columns else [True])
    elif "score_real" in pooled.columns:
        pooled = pooled.sort_values("score_real", ascending=False)

    out: list[PairToPlot] = []
    for i, row in enumerate(limit_rows(pooled, top_n).itertuples(index=False), start=1):
        row_dict = row._asdict()
        out.append(
            PairToPlot(
                qid=qid,
                tid=str(row_dict["tid"]),
                rank=i,
                score_real=float(row_dict["score_real"]) if "score_real" in row_dict else None,
                score_decoy=float(row_dict["score_decoy"]) if "score_decoy" in row_dict else None,
                q=float(row_dict["q"]) if "q" in row_dict else None,
                rep_id=int(row_dict["rep_id"]) if "rep_id" in row_dict and not pd.isna(row_dict["rep_id"]) else None,
                n_rep_support=int(row_dict["n_rep_support"])
                if "n_rep_support" in row_dict and not pd.isna(row_dict["n_rep_support"])
                else None,
            )
        )
    return out


def q_label(q: float) -> str:
    return f"q_{q:.2f}"


def q_is_selected(q: float, q_levels: list[float] | None) -> bool:
    if q_levels is None:
        return True
    return any(np.isclose(q, q_ref) for q_ref in q_levels)


def load_supported_qs(summary_path: Path, qid: str) -> set[float]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing q-scan summary table: {summary_path}")
    df = pd.read_csv(summary_path, sep="\t")
    required = {"qid", "q", "discovery_status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{summary_path} is missing columns: {sorted(missing)}")
    df = df[
        (df["qid"].astype(str) == qid)
        & (df["discovery_status"].astype(str) == "supported")
    ].copy()
    return set(float(q) for q in df["q"].dropna().unique())


def load_qscan_hits_for_qid(
    path: Path,
    qid: str,
    *,
    top_n: int | None,
    q_levels: list[float] | None,
    rank_by: str,
    new_targets_only: bool,
    summary_path: Path | None,
    require_supported: bool,
) -> dict[float, list[PairToPlot]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing q-scan hits table: {path}")

    df = pd.read_csv(path, sep="\t")
    required = {"qid", "tid", "q"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df[df["qid"].astype(str) == qid].copy()
    if df.empty:
        raise ValueError(f"No q-scan hit rows for qid={qid!r} in {path}")

    df["q"] = pd.to_numeric(df["q"], errors="coerce")
    df = df[df["q"].notna()]
    if q_levels is not None:
        df = df[df["q"].apply(lambda q: q_is_selected(float(q), q_levels))]

    if require_supported:
        if summary_path is None:
            raise ValueError("--require-supported needs --q-summary-file")
        supported_qs = load_supported_qs(summary_path, qid)
        df = df[df["q"].apply(lambda q: q_is_selected(float(q), list(supported_qs)))]

    if df.empty:
        return {}

    agg_spec = {"n_rep_support": ("rep_id", "nunique") if "rep_id" in df.columns else ("tid", "size")}
    if "score_real" in df.columns:
        agg_spec["score_real"] = ("score_real", "mean")
        agg_spec["score_real_max"] = ("score_real", "max")
    if "score_decoy" in df.columns:
        agg_spec["score_decoy"] = ("score_decoy", "mean")
    if "hit_rank" in df.columns:
        agg_spec["hit_rank"] = ("hit_rank", "min")

    pooled = df.groupby(["q", "tid"], as_index=False).agg(**agg_spec)

    if rank_by == "support":
        sort_cols = ["n_rep_support"]
        ascending = [False]
        if "score_real" in pooled.columns:
            sort_cols.append("score_real")
            ascending.append(False)
        if "hit_rank" in pooled.columns:
            sort_cols.append("hit_rank")
            ascending.append(True)
    elif rank_by == "score":
        sort_cols = ["score_real"] if "score_real" in pooled.columns else ["n_rep_support"]
        ascending = [False]
    else:
        sort_cols = ["hit_rank"] if "hit_rank" in pooled.columns else ["score_real"]
        ascending = [True] if "hit_rank" in pooled.columns else [False]
        if "score_real" in pooled.columns and "score_real" not in sort_cols:
            sort_cols.append("score_real")
            ascending.append(False)

    out: dict[float, list[PairToPlot]] = {}
    seen_tids: set[str] = set()
    for q in sorted(float(x) for x in pooled["q"].unique()):
        df_q = pooled[np.isclose(pooled["q"], q)].copy()
        df_q = df_q.sort_values(sort_cols, ascending=ascending)
        if new_targets_only:
            df_q = df_q[~df_q["tid"].astype(str).isin(seen_tids)].copy()
        df_q = limit_rows(df_q, top_n)

        pairs: list[PairToPlot] = []
        for rank, row in enumerate(df_q.itertuples(index=False), start=1):
            row_dict = row._asdict()
            tid = str(row_dict["tid"])
            pairs.append(
                PairToPlot(
                    qid=qid,
                    tid=tid,
                    rank=rank,
                    score_real=float(row_dict["score_real"]) if "score_real" in row_dict else None,
                    score_decoy=float(row_dict["score_decoy"]) if "score_decoy" in row_dict else None,
                    q=q,
                    n_rep_support=int(row_dict["n_rep_support"])
                    if "n_rep_support" in row_dict and not pd.isna(row_dict["n_rep_support"])
                    else None,
                )
            )
        out[q] = pairs
        if new_targets_only:
            seen_tids.update(str(tid) for tid in df_q["tid"])
    if q_levels is not None:
        for q in sorted(q_levels):
            out.setdefault(float(q), [])
    return out


def write_selected_targets_table(path: Path, pairs_by_q: dict[float, list[PairToPlot]]) -> None:
    rows = []
    for q, pairs in sorted(pairs_by_q.items()):
        if not pairs:
            rows.append(
                {
                    "q": q,
                    "q_folder": q_label(q),
                    "rank": None,
                    "qid": None,
                    "tid": None,
                    "score_real_mean": None,
                    "score_decoy_mean": None,
                    "n_rep_support": 0,
                    "status": "no_selected_target",
                }
            )
            continue
        for pair in pairs:
            rows.append(
                {
                    "q": q,
                    "q_folder": q_label(q),
                    "rank": pair.rank,
                    "qid": pair.qid,
                    "tid": pair.tid,
                    "score_real_mean": pair.score_real,
                    "score_decoy_mean": pair.score_decoy,
                    "n_rep_support": pair.n_rep_support,
                    "status": "selected",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    print(f"[OK] selected target summary -> {path}")


def write_q_hits_table(path: Path, q: float, pairs: list[PairToPlot]) -> None:
    rows = []
    if not pairs:
        rows.append(
            {
                "q": q,
                "q_folder": q_label(q),
                "rank": None,
                "qid": None,
                "tid": None,
                "score_real_mean": None,
                "score_decoy_mean": None,
                "n_rep_support": 0,
                "status": "no_selected_target",
            }
        )
    else:
        for pair in pairs:
            rows.append(
                {
                    "q": q,
                    "q_folder": q_label(q),
                    "rank": pair.rank,
                    "qid": pair.qid,
                    "tid": pair.tid,
                    "score_real_mean": pair.score_real,
                    "score_decoy_mean": pair.score_decoy,
                    "n_rep_support": pair.n_rep_support,
                    "status": "selected",
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    print(f"[OK] q hits -> {path}")


def load_blosum62():
    try:
        from Bio.Align import substitution_matrices

        return substitution_matrices.load("BLOSUM62")
    except Exception:
        return None


AA_GROUPS = [
    set("AVLIM"),
    set("FYW"),
    set("STNQ"),
    set("KRH"),
    set("DE"),
    set("CGP"),
]


def substitution_score(a: str, b: str, matrix) -> float:
    if matrix is not None:
        try:
            return float(matrix[a, b])
        except Exception:
            pass
    if a == b:
        return 5.0
    if any(a in group and b in group for group in AA_GROUPS):
        return 1.0
    return -3.0


def aligned_pair_weight(a: str, b: str, matrix) -> float:
    if a == b:
        return 1.0
    score = substitution_score(a, b, matrix)
    if score > 0:
        return 0.82
    if score == 0:
        return 0.62
    return 0.42


def pairwise_align(
    query_seq: str,
    target_seq: str,
    *,
    mode: str,
    gap: float,
    matrix,
) -> AlignmentResult:
    n = len(query_seq)
    m = len(target_seq)
    if n == 0 or m == 0:
        return AlignmentResult([], 0.0, 0, 0, 0, 0)

    dp = np.zeros((n + 1, m + 1), dtype=np.float32)
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)
    # 1=diag, 2=up(query residue to gap), 3=left(target residue to gap)

    if mode == "global":
        for i in range(1, n + 1):
            dp[i, 0] = dp[i - 1, 0] + gap
            trace[i, 0] = 2
        for j in range(1, m + 1):
            dp[0, j] = dp[0, j - 1] + gap
            trace[0, j] = 3

    best_score = -np.inf
    best_pos = (n, m)

    for i in range(1, n + 1):
        qi = query_seq[i - 1]
        for j in range(1, m + 1):
            tj = target_seq[j - 1]
            diag = dp[i - 1, j - 1] + substitution_score(qi, tj, matrix)
            up = dp[i - 1, j] + gap
            left = dp[i, j - 1] + gap

            if mode == "local":
                candidates = (0.0, diag, up, left)
                choice = int(np.argmax(candidates))
                dp[i, j] = candidates[choice]
                trace[i, j] = choice
                if dp[i, j] > best_score:
                    best_score = float(dp[i, j])
                    best_pos = (i, j)
            else:
                candidates = (diag, up, left)
                choice0 = int(np.argmax(candidates))
                dp[i, j] = candidates[choice0]
                trace[i, j] = choice0 + 1

    if mode == "global":
        best_score = float(dp[n, m])
        best_pos = (n, m)

    i, j = best_pos
    q_end, t_end = i, j
    pairs_rev: list[tuple[int, int]] = []

    while i > 0 or j > 0:
        step = int(trace[i, j])
        if mode == "local" and (step == 0 or dp[i, j] <= 0):
            break
        if step == 1:
            pairs_rev.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif step == 2:
            i -= 1
        elif step == 3:
            j -= 1
        else:
            break

    pairs = list(reversed(pairs_rev))
    q_start = pairs[0][0] if pairs else i
    t_start = pairs[0][1] if pairs else j
    return AlignmentResult(pairs, best_score, q_start, q_end, t_start, t_end)


def build_proxy_matrix(
    query_seq: str,
    target_seq: str,
    aln: AlignmentResult,
    *,
    sigma: float,
    matrix,
) -> np.ndarray:
    heat = np.zeros((len(query_seq), len(target_seq)), dtype=np.float32)
    if not aln.pairs:
        return heat

    radius = max(0, int(math.ceil(3.0 * sigma)))
    for i, j in aln.pairs:
        weight = aligned_pair_weight(query_seq[i], target_seq[j], matrix)
        if sigma <= 0:
            heat[i, j] = max(heat[i, j], weight)
            continue
        for ii in range(max(0, i - radius), min(len(query_seq), i + radius + 1)):
            for jj in range(max(0, j - radius), min(len(target_seq), j + radius + 1)):
                dist2 = (ii - i) ** 2 + (jj - j) ** 2
                value = weight * math.exp(-dist2 / (2.0 * sigma * sigma))
                if value > heat[ii, jj]:
                    heat[ii, jj] = value

    max_val = float(np.nanmax(heat)) if heat.size else 0.0
    if max_val > 0:
        heat = heat / max_val
    return heat


def load_matrix(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        matrix = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        if "matrix" in data:
            matrix = data["matrix"]
        else:
            first_key = list(data.keys())[0]
            matrix = data[first_key]
    else:
        matrix = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {matrix.shape} from {path}")
    return matrix


def alignment_from_traceback_states(
    states: list[tuple[int, int, int]],
    *,
    query_len: int,
    target_len: int,
) -> AlignmentResult:
    match_state = 1
    pairs = [
        (int(i), int(j))
        for i, j, state in states
        if int(state) == match_state and 0 <= int(i) < query_len and 0 <= int(j) < target_len
    ]
    if not pairs:
        return AlignmentResult([], 0.0, 0, 0, 0, 0)
    return AlignmentResult(
        pairs=pairs,
        score=float(len(pairs)),
        query_start=pairs[0][0],
        query_end=pairs[-1][0] + 1,
        target_start=pairs[0][1],
        target_end=pairs[-1][1] + 1,
    )


class DeepBlastRunner:
    def __init__(
        self,
        *,
        model_path: Path,
        protrans_model: str,
        alignment_mode: str,
        device: str,
    ) -> None:
        self.model_path = model_path
        self.protrans_model = protrans_model
        self.alignment_mode = alignment_mode
        self.device_name = device
        self.cache: dict[tuple[str, str], tuple[np.ndarray, AlignmentResult | None]] = {}
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Missing DeepBLAST checkpoint: {self.model_path}")

        try:
            import torch
            from deepblast.trainer import DeepBLAST
            from transformers import T5EncoderModel, T5Tokenizer
        except Exception as exc:
            raise RuntimeError(
                "DeepBLAST backend requires the tmvec/deepblast Python environment. "
                "Try running with $TMVEC_PYTHON_PATH."
            ) from exc

        if self.device_name == "auto":
            torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        elif self.device_name in {"gpu", "cuda"}:
            if not torch.cuda.is_available():
                raise RuntimeError("--deepblast-device requested GPU, but CUDA is not available")
            torch_device = torch.device("cuda:0")
        else:
            torch_device = torch.device("cpu")

        decoder_device = "gpu" if torch_device.type == "cuda" else "cpu"
        print(
            "[INFO] Loading DeepBLAST "
            f"checkpoint={self.model_path}, protrans={self.protrans_model}, device={torch_device}"
        )
        tokenizer = T5Tokenizer.from_pretrained(
            self.protrans_model,
            do_lower_case=False,
            legacy=True,
            local_files_only=True,
        )
        lm = T5EncoderModel.from_pretrained(
            self.protrans_model,
            local_files_only=True,
        )
        lm = lm.to(torch_device).eval()

        model = DeepBLAST(
            layers=8,
            alignment_mode=self.alignment_mode,
            dropout=0.5,
            device=decoder_device,
        )
        state = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(state)
        model.tokenizer = tokenizer
        model.aligner.lm = lm
        model = model.to(torch_device).eval()

        self.torch = torch
        self.get_sequence = __import__("deepblast.dataset.utils", fromlist=["get_sequence"]).get_sequence
        self.pack_sequences = __import__("deepblast.dataset.utils", fromlist=["pack_sequences"]).pack_sequences
        self.model = model
        self.tokenizer = tokenizer
        self.torch_device = torch_device

    def align_pair(
        self,
        query_seq: str,
        target_seq: str,
        *,
        cache_key: tuple[str, str] | None = None,
    ) -> tuple[np.ndarray, AlignmentResult | None]:
        key = cache_key or (query_seq, target_seq)
        if key in self.cache:
            return self.cache[key]

        torch = self.torch
        x_code = self.get_sequence(query_seq, self.tokenizer)[0].to(self.torch_device)
        y_code = self.get_sequence(target_seq, self.tokenizer)[0].to(self.torch_device)
        seq, order = self.pack_sequences([x_code], [y_code])
        seq = seq.to(self.torch_device)

        with torch.enable_grad():
            aln_tensor, _, _ = self.model(seq, order)
        matrix = aln_tensor.squeeze(0).detach().cpu().numpy()
        matrix = matrix[: len(query_seq), : len(target_seq)].astype(np.float32, copy=False)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=0.0)
        matrix = np.clip(matrix, 0.0, 1.0)

        hard_alignment: AlignmentResult | None = None
        try:
            states = self.model.aligner.ddp.traceback(aln_tensor.squeeze(0).detach())
            hard_alignment = alignment_from_traceback_states(
                states,
                query_len=len(query_seq),
                target_len=len(target_seq),
            )
        except Exception as exc:
            print(f"[WARN] DeepBLAST traceback failed; heatmap matrix will still be plotted: {exc}")

        self.cache[key] = (matrix, hard_alignment)
        return matrix, hard_alignment


def crop_to_alignment(
    heat: np.ndarray,
    aln: AlignmentResult | None,
    *,
    pad: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if aln is None or not aln.pairs:
        return heat, (0, heat.shape[0], 0, heat.shape[1])

    q_positions = [p[0] for p in aln.pairs]
    t_positions = [p[1] for p in aln.pairs]
    q0 = max(0, min(q_positions) - pad)
    q1 = min(heat.shape[0], max(q_positions) + pad + 1)
    t0 = max(0, min(t_positions) - pad)
    t1 = min(heat.shape[1], max(t_positions) + pad + 1)
    return heat[q0:q1, t0:t1], (q0, q1, t0, t1)


def write_pair_tsv(
    path: Path,
    pair: PairToPlot,
    query_seq: str,
    target_seq: str,
    aln: AlignmentResult,
    matrix,
    *,
    probability_matrix: np.ndarray | None = None,
    source: str = "proxy",
) -> None:
    rows = []
    for i, j in aln.pairs:
        rows.append(
            {
                "qid": pair.qid,
                "tid": pair.tid,
                "query_pos0": i,
                "target_pos0": j,
                "query_pos1": i + 1,
                "target_pos1": j + 1,
                "query_aa": query_seq[i],
                "target_aa": target_seq[j],
                "substitution_score": substitution_score(query_seq[i], target_seq[j], matrix),
                "heatmap_weight": aligned_pair_weight(query_seq[i], target_seq[j], matrix),
                "alignment_probability": float(probability_matrix[i, j])
                if probability_matrix is not None
                and 0 <= i < probability_matrix.shape[0]
                and 0 <= j < probability_matrix.shape[1]
                else None,
                "heatmap_source": source,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


INDEX_COLUMNS = [
    "qid", "tid", "q", "rank", "rep_id", "n_rep_support",
    "score_real", "score_decoy", "source", "label",
    "qlen", "tlen", "q0", "q1", "t0", "t1",
    "heat_npy", "path_npy", "out_prefix",
]


def write_index(output_dir: Path, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.csv"
    pd.DataFrame(rows, columns=INDEX_COLUMNS).to_csv(index_path, index=False)
    print(f"[OK] wrote {len(rows)} heatmap(s) to {output_dir}")
    print(f"[OK] index -> {index_path}")
    print(f"     Next: python plot_alignment_heatmap.py --index {index_path}")


def build_pairs(args: argparse.Namespace) -> list[PairToPlot]:
    if not args.qid:
        raise ValueError("--qid is required")
    if args.tid:
        return [PairToPlot(qid=args.qid, tid=tid, rank=i) for i, tid in enumerate(args.tid, start=1)]
    return load_hits_for_qid(args.hits_file, args.qid, args.top_n)


def compute_pair(
    pair: PairToPlot,
    *,
    output_dir: Path,
    query_records: dict[str, str],
    target_records: dict[str, str],
    blosum62,
    deepblast_runner: DeepBlastRunner | None,
    args: argparse.Namespace,
) -> dict | None:
    if pair.qid not in query_records:
        raise KeyError(f"qid not found in {args.query_fasta}: {pair.qid}")
    if pair.tid not in target_records:
        raise KeyError(f"tid not found in {args.target_fasta}: {pair.tid}")

    query_seq = query_records[pair.qid]
    target_seq = target_records[pair.tid]
    aln: AlignmentResult | None = None

    matrix_source = "external_matrix"
    if args.matrix is not None:
        heat = load_matrix(args.matrix)
        label = "Residue alignment probability"
        if heat.shape != (len(query_seq), len(target_seq)):
            print(
                "[WARN] matrix shape does not match sequence lengths: "
                f"matrix={heat.shape}, seqs={(len(query_seq), len(target_seq))}"
            )
        crop_bounds = (0, heat.shape[0], 0, heat.shape[1])
    elif deepblast_runner is not None:
        heat, aln = deepblast_runner.align_pair(
            query_seq,
            target_seq,
            cache_key=(pair.qid, pair.tid),
        )
        label = "Residue alignment probability"
        matrix_source = "deepblast"
        crop_bounds = (0, heat.shape[0], 0, heat.shape[1])
    else:
        raise RuntimeError(
            f"DeepBLAST runner is required but failed to load. "
            f"Run with the tmvec or bagel_heatmap conda environment."
        )

    q_name = sanitize_filename(pair.qid)
    t_name = sanitize_filename(pair.tid)
    rank = f"nn{pair.rank:02d}" if pair.rank is not None else "pair"
    if output_dir.name.startswith("q_"):
        out_prefix = output_dir / f"{rank}__{t_name}"
    else:
        out_prefix = output_dir / f"{q_name}__{rank}__{t_name}"

    # --- persist the plotting data (matrix + alignment path) ---
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    heat_npy = append_extension(out_prefix, "npy")
    np.save(heat_npy, heat.astype(np.float32, copy=False))

    path_npy: Path | None = None
    if aln is not None and aln.pairs:
        path_npy = append_extension(out_prefix, "path.npy")
        np.save(path_npy, np.asarray(aln.pairs, dtype=np.int32))

    if not args.no_pair_tsv and aln is not None:
        write_pair_tsv(
            append_extension(out_prefix, "aligned_pairs.tsv"),
            pair,
            query_seq,
            target_seq,
            aln,
            blosum62,
            probability_matrix=heat,
            source=matrix_source,
        )

    root = args.output_dir
    q0, q1, t0, t1 = crop_bounds
    return {
        "qid": pair.qid,
        "tid": pair.tid,
        "q": pair.q,
        "rank": pair.rank,
        "rep_id": pair.rep_id,
        "n_rep_support": pair.n_rep_support,
        "score_real": pair.score_real,
        "score_decoy": pair.score_decoy,
        "source": matrix_source,
        "label": label,
        "qlen": len(query_seq),
        "tlen": len(target_seq),
        "q0": q0, "q1": q1, "t0": t0, "t1": t1,
        "heat_npy": str(heat_npy.relative_to(root)),
        "path_npy": str(path_npy.relative_to(root)) if path_npy is not None else "",
        "out_prefix": str(out_prefix.relative_to(root)),
    }


def compute_qscan_for_query(
    qid: str,
    *,
    query_records: dict[str, str],
    target_records: dict[str, str],
    blosum62,
    deepblast_runner: DeepBlastRunner | None,
    args: argparse.Namespace,
    index_rows: list[dict],
) -> bool:
    try:
        pairs_by_q = load_qscan_hits_for_qid(
            args.q_scan_hits_file,
            qid,
            top_n=args.top_n,
            q_levels=args.q_levels,
            rank_by=args.rank_by,
            new_targets_only=args.new_targets_only,
            summary_path=args.q_summary_file,
            require_supported=args.require_supported,
        )
    except ValueError as exc:
        print(f"[WARN] {qid}: {exc}")
        return False

    if not pairs_by_q or not any(pairs for pairs in pairs_by_q.values()):
        print(f"[INFO] {qid}: no valid hits at any q-level, skipping")
        return False

    q_dir_name = sanitize_filename(qid)
    query_out_dir = args.output_dir / q_dir_name
    for q, pairs in sorted(pairs_by_q.items()):
        if not pairs:
            continue
        q_out_dir = query_out_dir / q_label(q)
        for pair in pairs:
            row = compute_pair(
                pair,
                output_dir=q_out_dir,
                query_records=query_records,
                target_records=target_records,
                blosum62=blosum62,
                deepblast_runner=deepblast_runner,
                args=args,
            )
            if row is not None:
                index_rows.append(row)
    return True


def resolve_query_ids(args: argparse.Namespace, query_records: dict[str, str]) -> list[str]:
    if args.qid_list is not None:
        qids = read_id_list(args.qid_list)
    elif args.all_putative:
        qids = list(query_records.keys())
    elif args.qid is not None:
        qids = [args.qid]
    else:
        raise ValueError("Provide --qid for one query, or --all-putative / --qid-list for batch mode.")

    if args.max_queries is not None:
        qids = qids[: args.max_queries]
    return qids


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

@dataclass(frozen=True)
class PairInfo:
    qid: str
    tid: str
    rank: int | None = None
    q: float | None = None


@dataclass(frozen=True)
class AlignmentPath:
    pairs: list[tuple[int, int]] = field(default_factory=list)


def plot_heatmap(
    heat: np.ndarray,
    *,
    pair: PairInfo,
    out_prefix: Path,
    formats: list[str],
    dpi: int,
    label: str,
    aln: AlignmentPath | None,
    crop_bounds: tuple[int, int, int, int],
    draw_path: bool,
) -> None:
    q0, q1, t0, t1 = crop_bounds
    fig_w = max(4.8, min(8.2, 2.5 + 0.09 * heat.shape[1]))
    fig_h = max(4.6, min(8.2, 2.4 + 0.09 * heat.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        heat,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xlim(-0.5, heat.shape[1] - 0.5)
    ax.set_ylim(heat.shape[0] - 0.5, -0.5)
    xticks = sequence_ticks(heat.shape[1])
    yticks = sequence_ticks(heat.shape[0])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([str(int(x + t0)) if x >= 0 else "" for x in xticks])
    ax.set_yticklabels([str(int(y + q0)) if y >= 0 else "" for y in yticks])

    if draw_path and aln is not None and aln.pairs:
        xs = [j - t0 for _, j in aln.pairs if t0 <= j < t1]
        ys = [i - q0 for i, _ in aln.pairs if q0 <= i < q1]
        if len(xs) == len(ys) and xs:
            ax.plot(xs, ys, color="white", linewidth=0.8, alpha=0.85)

    rank_label = f"Near neighbor {pair.rank}" if pair.rank is not None else "BAGEL pair"
    if pair.q is not None:
        rank_label = f"Target FDR threshold = {pair.q:.2f} | {rank_label}"
    ax.set_title(rank_label, fontsize=14)
    ax.set_ylabel(short_axis_label(pair.qid, prefix="Putative bacteriocin"), fontsize=13)
    ax.set_xlabel(short_axis_label(pair.tid), fontsize=13)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, rotation=270, labelpad=18, fontsize=13)

    fig.tight_layout()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_path = append_extension(out_prefix, fmt)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"[OK] heatmap -> {out_path}")
    plt.close(fig)


def _opt_int(value) -> int | None:
    return None if pd.isna(value) else int(value)


def _opt_float(value) -> float | None:
    return None if pd.isna(value) else float(value)


def render_from_index(index_path: Path, *, formats: list[str], dpi: int, draw_path: bool) -> int:
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing heatmap index: {index_path}\n"
            f"Run: python compute_alignment_heatmap.py ..."
        )
    root = index_path.parent
    df = pd.read_csv(index_path)
    if df.empty:
        print(f"[INFO] {index_path} has no rows; nothing to plot")
        return 0

    n = 0
    for row in df.itertuples(index=False):
        heat = np.load(root / row.heat_npy)

        aln: AlignmentPath | None = None
        path_npy = getattr(row, "path_npy", "")
        if isinstance(path_npy, str) and path_npy:
            arr = np.load(root / path_npy)
            aln = AlignmentPath(pairs=[(int(i), int(j)) for i, j in arr])

        pair = PairInfo(
            qid=str(row.qid),
            tid=str(row.tid),
            rank=_opt_int(row.rank),
            q=_opt_float(row.q),
        )
        crop_bounds = (int(row.q0), int(row.q1), int(row.t0), int(row.t1))
        label = row.label if isinstance(row.label, str) else "Residue alignment probability"

        plot_heatmap(
            heat,
            pair=pair,
            out_prefix=root / row.out_prefix,
            formats=formats,
            dpi=dpi,
            label=label,
            aln=aln,
            crop_bounds=crop_bounds,
            draw_path=draw_path,
        )
        n += 1
    return n


DISCOVERY_DATA = os.path.join(BASE_DIR, "data")
DISCOVERY_OUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(DISCOVERY_OUT_DIR, exist_ok=True)

SEARCH_METHOD = "plm"
QUERY_NAME = "putative"
TARGET_NAME = "class_all"
DECOY_METHOD = "extended_shuf"
WEIGHT_METHOD = "AdaptiveBell"
DECOY_SUFFIX = "_shuf"

# A query is "supported" if at least this many noisy reps produce >=1 accepted hit.
MIN_REPS_WITH_HITS = 1

Q_SINGLE = 0.20
Q_LEVELS = np.round(np.arange(0.10, 0.70, 0.10), 2)


def remove_decoy_suffix(qid, decoy_suffix):
    if qid.endswith(decoy_suffix):
        return qid[: -len(decoy_suffix)]
    return qid


def make_pair_table_keep_tid(path_real, path_decoy, decoy_suffix):
    df_real = pd.read_csv(path_real, sep="\t")
    df_decoy = pd.read_csv(path_decoy, sep="\t")

    df_real = df_real[["qid", "tid", "homo_type", "score", "rep_id"]].copy()
    df_real = df_real.rename(columns={
        "score": "score_real",
        "homo_type": "homo_type_real",
    })

    df_decoy = df_decoy[["qid", "tid", "score", "rep_id"]].copy()
    df_decoy["qid_orig"] = df_decoy["qid"].apply(
        lambda x: remove_decoy_suffix(x, decoy_suffix)
    )
    df_decoy = df_decoy.rename(columns={"score": "score_decoy"})

    df_merge = df_real.merge(
        df_decoy[["qid_orig", "tid", "score_decoy", "rep_id"]],
        left_on=["qid", "tid", "rep_id"],
        right_on=["qid_orig", "tid", "rep_id"],
        how="inner",
    )
    df_merge = df_merge.drop(columns=["qid_orig"], errors="ignore")
    df_merge["score_real"] = df_merge["score_real"].astype(np.float32)
    df_merge["score_decoy"] = df_merge["score_decoy"].astype(np.float32)
    if df_merge["homo_type_real"].dtype != object:
        df_merge["homo_type_real"] = df_merge["homo_type_real"].astype(np.int8)
    df_merge["rep_id"] = pd.to_numeric(df_merge["rep_id"], downcast="integer")
    return df_merge


def _get_class_from_name(name):
    if not isinstance(name, str):
        return "Unknown"
    left = name.split(";", 1)[0].strip()
    parts = left.split(".")
    if len(parts) < 2:
        return "Unknown"
    return {"1": "1", "2": "2", "3": "3"}.get(parts[1], "Unknown")


def _find_threshold(scores_t, scores_d, q):
    """Lowest score threshold T with eFDP(T) <= q."""
    n_t = len(scores_t)
    if n_t == 0:
        return None

    cand_t = np.sort(np.unique(np.concatenate([scores_t, scores_d])))
    scores_t_sorted = np.sort(scores_t)
    scores_d_sorted = np.sort(scores_d)

    target_count = n_t - np.searchsorted(scores_t_sorted, cand_t, side="left")
    decoy_count = n_d - np.searchsorted(scores_d_sorted, cand_t, side="left") if (n_d := len(scores_d)) else np.zeros_like(cand_t)
    est_fdp = (decoy_count + 1.0) / np.maximum(target_count, 1.0)

    valid = np.where(est_fdp <= q)[0]
    if len(valid) == 0:
        return None

    best_idx = valid[np.argmin(cand_t[valid])]
    return {
        "T_q": float(cand_t[best_idx]),
        "eFDP_at_T": float(est_fdp[best_idx]),
        "n_pass": int(target_count[best_idx]),
    }


def select_accepted_hits(df_qr, q):
    """Return metadata + DataFrame of accepted hits for one (qid, rep_id)."""
    scores_t = df_qr["score_real"].values
    scores_d = df_qr["score_decoy"].values
    thr = _find_threshold(scores_t, scores_d, q)
    if thr is None:
        return thr, pd.DataFrame()

    pass_mask = scores_t >= thr["T_q"]
    df_pass = df_qr.loc[pass_mask, [
        "tid", "score_real", "score_decoy", "homo_type_real",
    ]].copy()
    df_pass = df_pass.sort_values("score_real", ascending=False).reset_index(drop=True)
    df_pass["hit_rank"] = np.arange(1, len(df_pass) + 1)
    thr["n_pass"] = len(df_pass)
    return thr, df_pass


def run_discovery_for_q(df_merge, q, min_reps_with_hits=MIN_REPS_WITH_HITS):
    """Discovery summary + long-form accepted hits at one q."""
    summary_rows = []
    hit_rows = []

    grouped = df_merge.groupby("qid", sort=False, observed=True)
    for qid, df_q in tqdm(grouped, desc=f"Discovery @ q={q:.2f}"):
        rep_ids = sorted(df_q["rep_id"].unique().tolist())
        per_rep_n_pass = []
        per_rep_Tq = []
        per_rep_efdp = []

        for rep_id in rep_ids:
            df_qr = df_q[df_q["rep_id"] == rep_id]
            thr, df_pass = select_accepted_hits(df_qr, q)
            if thr is None:
                per_rep_n_pass.append(0)
                per_rep_Tq.append(np.nan)
                per_rep_efdp.append(np.nan)
                continue

            per_rep_n_pass.append(thr["n_pass"])
            per_rep_Tq.append(thr["T_q"])
            per_rep_efdp.append(thr["eFDP_at_T"])

            if len(df_pass) > 0:
                df_pass = df_pass.copy()
                df_pass["qid"] = qid
                df_pass["q"] = q
                df_pass["rep_id"] = rep_id
                hit_rows.append(df_pass)

        n_reps_with_hits = sum(1 for n in per_rep_n_pass if n > 0)
        n_accepted_mean = float(np.mean(per_rep_n_pass))
        supported = n_reps_with_hits >= min_reps_with_hits

        summary_rows.append({
            "qid": qid,
            "q": q,
            "discovery_status": "supported" if supported else "no_supported_hit",
            "n_reps_with_hits": n_reps_with_hits,
            "n_reps_total": len(rep_ids),
            "n_accepted_mean": n_accepted_mean,
            "n_accepted_max": int(max(per_rep_n_pass) if per_rep_n_pass else 0),
            "T_q_mean": float(np.nanmean(per_rep_Tq)),
            "eFDP_at_T_mean": float(np.nanmean(per_rep_efdp)),
            "bagel_predicted_class": _get_class_from_name(qid),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_hits = pd.concat(hit_rows, ignore_index=True) if hit_rows else pd.DataFrame()
    return df_summary, df_hits


def run_discovery_q_scan(df_merge, q_levels, min_reps_with_hits=MIN_REPS_WITH_HITS):
    summaries, hits = [], []
    for q in q_levels:
        df_s, df_h = run_discovery_for_q(
            df_merge, q=q, min_reps_with_hits=min_reps_with_hits
        )
        summaries.append(df_s)
        if len(df_h) > 0:
            hits.append(df_h)
    return pd.concat(summaries, ignore_index=True), (
        pd.concat(hits, ignore_index=True) if hits else pd.DataFrame()
    )


def plot_discovery_q_scan(df_scan, out_path):
    q_levels = sorted(df_scan["q"].unique())
    n_supported, n_not = [], []

    for q in q_levels:
        df_q = df_scan[np.isclose(df_scan["q"], q)]
        n_supported.append(int((df_q["discovery_status"] == "supported").sum()))
        n_not.append(int((df_q["discovery_status"] == "no_supported_hit").sum()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(q_levels, n_supported, "-o", color="steelblue", label="supported")
    axes[0].plot(q_levels, n_not, "-o", color="gray", label="no_supported_hit")
    axes[0].set_xlabel("q (target FDR)")
    axes[0].set_ylabel("# putative")
    axes[0].set_title("Discovery coverage vs q")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.3)

    mean_hits = []
    for q in q_levels:
        df_q = df_scan[
            np.isclose(df_scan["q"], q)
            & (df_scan["discovery_status"] == "supported")
        ]
        mean_hits.append(float(df_q["n_accepted_mean"].mean()) if len(df_q) else np.nan)

    axes[1].plot(q_levels, mean_hits, "-o", color="darkgreen")
    axes[1].set_xlabel("q (target FDR)")
    axes[1].set_ylabel("Mean # accepted hits (supported only)")
    axes[1].set_title("Discovery yield vs q")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Discovery q-scan plot -> {out_path}")


def write_table(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return path


def method_aware_name(prefix, search_method, decoy_method, suffix):
    joiner = "" if suffix.startswith(".") else "_"
    return f"{prefix}_{search_method}_{decoy_method}{joiner}{suffix}"


def _discover_args(argv):
    parser = argparse.ArgumentParser(
        description="FDR-controlled putative BAGEL homolog discovery."
    )
    parser.add_argument(
        "--search-method",
        default=SEARCH_METHOD,
        help="Scoring/search method prefix in result_<method>_* files, e.g. plm, tmvec, dhr_postprocess.",
    )
    parser.add_argument("--query-name", default=QUERY_NAME)
    parser.add_argument("--target-name", default=TARGET_NAME)
    parser.add_argument("--decoy-method", default=DECOY_METHOD)
    parser.add_argument("--weight-method", default=WEIGHT_METHOD)
    parser.add_argument("--decoy-suffix", default=DECOY_SUFFIX)
    parser.add_argument("--target-fdr", "--q-single", dest="q_single",
                        type=float, default=Q_SINGLE,
                        help="FDR level of the single-level discovery table "
                             "(--q-single is kept as an alias)")
    parser.add_argument(
        "--q-levels",
        nargs="+",
        type=float,
        default=Q_LEVELS.tolist(),
        help="q levels for the scan.",
    )
    parser.add_argument("--min-reps-with-hits", type=int, default=MIN_REPS_WITH_HITS)
    parser.add_argument("--base-dir", default=BASE_DIR)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--write-legacy",
        action="store_true",
        help="Also write the old method-less output filenames.",
    )
    return parser.parse_args(argv)


def run_discover(argv):
    args = _discover_args(argv)
    data_dir = args.data_dir or os.path.join(args.base_dir, "data")
    out_dir = args.out_dir or os.path.join(args.base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    path_real = os.path.join(
        data_dir,
        f"result_{args.search_method}_{args.query_name}_target_noisy_{args.target_name}.txt",
    )
    path_decoy = os.path.join(
        data_dir,
        f"result_{args.search_method}_{args.query_name}_{args.decoy_method}"
        f"_calibrated_gam_{args.weight_method}_{args.target_name}.txt",
    )
    if not os.path.exists(path_real):
        raise FileNotFoundError(f"Missing real score table: {path_real}")
    if not os.path.exists(path_decoy):
        raise FileNotFoundError(f"Missing decoy score table: {path_decoy}")

    print("[INFO] Loading paired score table...")
    print(f"        search_method: {args.search_method}")
    print(f"        real:  {path_real}")
    print(f"        decoy: {path_decoy}")
    df_merge = make_pair_table_keep_tid(path_real, path_decoy, args.decoy_suffix)
    print(f"        rows: {len(df_merge)}")

    print(f"\n=== Discovery @ q = {args.q_single} ===")
    df_single, df_hits_single = run_discovery_for_q(
        df_merge,
        q=args.q_single,
        min_reps_with_hits=args.min_reps_with_hits,
    )

    out_single = os.path.join(
        data_dir,
        method_aware_name(
            "putative_discovery",
            args.search_method,
            args.decoy_method,
            f"q{args.q_single:.2f}.tsv",
        ),
    )
    out_hits_single = os.path.join(
        data_dir,
        method_aware_name(
            "putative_discovery_hits",
            args.search_method,
            args.decoy_method,
            f"q{args.q_single:.2f}.tsv",
        ),
    )
    write_table(df_single, out_single)
    write_table(df_hits_single, out_hits_single)
    print(f"[OK] Discovery summary -> {out_single}")
    print(f"[OK] Accepted hits (long) -> {out_hits_single}")
    if args.write_legacy:
        legacy_single = os.path.join(
            data_dir, f"putative_discovery_{args.decoy_method}_q{args.q_single:.2f}.tsv"
        )
        legacy_hits_single = os.path.join(
            data_dir, f"putative_discovery_hits_{args.decoy_method}_q{args.q_single:.2f}.tsv"
        )
        write_table(df_single, legacy_single)
        write_table(df_hits_single, legacy_hits_single)
        print(f"[OK] Legacy discovery summary -> {legacy_single}")
        print(f"[OK] Legacy accepted hits (long) -> {legacy_hits_single}")

    n_sup = (df_single["discovery_status"] == "supported").sum()
    print(f"\n[INFO] supported: {n_sup} / {len(df_single)}")
    print(df_single["discovery_status"].value_counts())

    q_levels = np.round(np.array(args.q_levels, dtype=float), 2)
    print(f"\n=== Discovery q-scan: {q_levels.tolist()} ===")
    df_scan, df_hits_scan = run_discovery_q_scan(
        df_merge,
        q_levels,
        min_reps_with_hits=args.min_reps_with_hits,
    )

    out_scan = os.path.join(
        data_dir,
        method_aware_name(
            "putative_discovery_scan",
            args.search_method,
            args.decoy_method,
            ".tsv",
        ),
    )
    out_hits_scan = os.path.join(
        data_dir,
        method_aware_name(
            "putative_discovery_hits_scan",
            args.search_method,
            args.decoy_method,
            ".tsv",
        ),
    )
    write_table(df_scan, out_scan)
    write_table(df_hits_scan, out_hits_scan)
    print(f"[OK] Q-scan summary -> {out_scan}")
    print(f"[OK] Q-scan accepted hits -> {out_hits_scan}")
    if args.write_legacy:
        legacy_scan = os.path.join(data_dir, f"putative_discovery_scan_{args.decoy_method}.tsv")
        legacy_hits_scan = os.path.join(
            data_dir, f"putative_discovery_hits_scan_{args.decoy_method}.tsv"
        )
        write_table(df_scan, legacy_scan)
        write_table(df_hits_scan, legacy_hits_scan)
        print(f"[OK] Legacy Q-scan summary -> {legacy_scan}")
        print(f"[OK] Legacy Q-scan accepted hits -> {legacy_hits_scan}")

    out_png = os.path.join(
        out_dir,
        f"putative_discovery_scan_{args.search_method}_{args.decoy_method}.png",
    )
    plot_discovery_q_scan(df_scan, out_png)

    print("\n[DONE] Discovery complete.")


def _plot_args(argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render BAGEL residue-alignment heatmaps from a precomputed index.csv + npy matrices.",
    )
    parser.add_argument(
        "--search-method", default=DEFAULT_SEARCH_METHOD,
        help="Used only to resolve the default index directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory holding index.csv + matrices. "
             "Default results/bagel/alignment_heatmaps_all_hits_<search-method>.",
    )
    parser.add_argument(
        "--index", type=Path, default=None,
        help="Explicit index.csv path (overrides --output-dir).",
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=("png", "pdf", "svg"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--draw-path", action="store_true")
    return parser.parse_args(argv)


def run_heatmap_plot(argv) -> None:
    args = _plot_args(argv)
    if args.index is not None:
        index_path = args.index
    else:
        out_dir = args.output_dir or (
            DEFAULT_OUT_PARENT / f"alignment_heatmaps_all_hits_{sanitize_filename(args.search_method)}"
        )
        index_path = out_dir / "index.csv"

    print(f"[INFO] index: {index_path}")
    n = render_from_index(index_path, formats=args.formats, dpi=args.dpi, draw_path=args.draw_path)
    print(f"\n[DONE] rendered {n} heatmap(s)")


def run_heatmap_data(argv) -> None:
    parser = argparse.ArgumentParser(
        description="Plot BAGEL residue-alignment heatmaps for putative/class_all pairs."
    )
    parser.add_argument(
        "--search-method",
        default=DEFAULT_SEARCH_METHOD,
        help="Search/scoring method whose FDR-controlled hits are plotted, e.g. plm, tmvec, dhr_postprocess.",
    )
    parser.add_argument(
        "--decoy-method",
        default=DEFAULT_DECOY_METHOD,
        help="Decoy method suffix used in the discovery hit tables.",
    )
    parser.add_argument(
        "--single-q",
        type=float,
        default=0.20,
        help="q value used to resolve the default non-q-scan hit table.",
    )
    parser.add_argument("--qid", default=None, help="BAGEL putative query id")
    parser.add_argument(
        "--all-putative",
        action="store_true",
        help="Run every query id in --query-fasta.",
    )
    parser.add_argument(
        "--qid-list",
        type=Path,
        default=None,
        help="Optional newline-delimited query id list for batch mode.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit the number of query ids processed, useful for dry runs.",
    )
    parser.add_argument("--tid", action="append", help="Target id. Repeat to plot multiple targets.")
    parser.add_argument("--top-n", type=int, default=3, help="Top accepted hits to plot when --tid is omitted.")
    parser.add_argument(
        "--all-hits",
        action="store_true",
        help="Plot every accepted hit per q threshold; ignores --top-n.",
    )
    parser.add_argument(
        "--hits-file",
        type=Path,
        default=None,
        help="Accepted-hit table used to choose top targets. Default is method-aware.",
    )
    parser.add_argument(
        "--q-scan",
        action="store_true",
        help="Use a q-scan accepted-hit table and save heatmaps under one folder per q.",
    )
    parser.add_argument(
        "--q-scan-hits-file",
        type=Path,
        default=None,
        help="Long-form accepted-hit table from the discover q-scan. Default is method-aware.",
    )
    parser.add_argument(
        "--q-summary-file",
        type=Path,
        default=None,
        help="Q-scan summary table used by --require-supported. Default is method-aware.",
    )
    parser.add_argument(
        "--q-levels",
        nargs="+",
        type=float,
        default=None,
        help="Optional q levels to plot. Default: all q values in the q-scan hit table.",
    )
    parser.add_argument(
        "--rank-by",
        choices=("rank", "score", "support"),
        default="rank",
        help="How to choose top targets per q after pooling reps.",
    )
    parser.add_argument(
        "--new-targets-only",
        action="store_true",
        help="For relaxed q levels, plot only targets not selected at stricter q levels; q=0.20 excludes q=0.10 hits.",
    )
    parser.add_argument(
        "--require-supported",
        action="store_true",
        help="Only plot q folders where discovery marks this query as supported.",
    )
    parser.add_argument("--query-fasta", type=Path, default=DATA_DIR / "putative.fa")
    parser.add_argument("--target-fasta", type=Path, default=DATA_DIR / "class_all.fa")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=None,
        help="Optional residue alignment probability matrix (.npy/.npz/.tsv/.csv). Use with one pair.",
    )
    parser.add_argument(
        "--heatmap-source",
        choices=("deepblast",),
        default="deepblast",
        help="How to build heatmaps when --matrix is not supplied (only deepblast is supported).",
    )
    parser.add_argument(
        "--deepblast-model",
        type=Path,
        default=BASE_DIR / "libs" / "tm-vec-master" / "model" / "deepblast-v3.ckpt",
        help="DeepBLAST checkpoint used for residue alignment probabilities.",
    )
    parser.add_argument(
        "--protrans-model",
        default="Rostlab/prot_t5_xl_uniref50",
        help="Local HuggingFace model id/path for ProtT5. Loaded with local_files_only=True.",
    )
    parser.add_argument(
        "--deepblast-alignment-mode",
        choices=("needleman-wunsch", "smith-waterman"),
        default="needleman-wunsch",
    )
    parser.add_argument(
        "--deepblast-device",
        choices=("auto", "cpu", "gpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--mode", choices=("local", "global"), default="local")
    parser.add_argument("--gap", type=float, default=-8.0)
    parser.add_argument("--sigma", type=float, default=1.5, help="Gaussian smoothing for proxy heatmap.")
    parser.add_argument(
        "--crop-to-alignment",
        action="store_true",
        help="Deprecated compatibility flag; heatmaps are now always plotted at full sequence length.",
    )
    parser.add_argument("--crop-pad", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (matrices + index.csv). Default results/bagel/alignment_heatmaps_all_hits_<search-method>.",
    )
    parser.add_argument("--no-pair-tsv", action="store_true")
    args = parser.parse_args(argv)
    if args.all_hits:
        args.top_n = None
    if args.hits_file is None:
        args.hits_file = resolve_default_table(
            default_single_hits_file(args.search_method, args.decoy_method, args.single_q),
            legacy_single_hits_file(args.decoy_method, args.single_q)
            if args.search_method == DEFAULT_SEARCH_METHOD
            else None,
            "single-q hit table",
        )
    if args.q_scan_hits_file is None:
        args.q_scan_hits_file = resolve_default_table(
            default_qscan_hits_file(args.search_method, args.decoy_method),
            legacy_qscan_hits_file(args.decoy_method) if args.search_method == DEFAULT_SEARCH_METHOD else None,
            "q-scan hit table",
        )
    if args.q_summary_file is None:
        args.q_summary_file = resolve_default_table(
            default_qscan_summary_file(args.search_method, args.decoy_method),
            legacy_qscan_summary_file(args.decoy_method) if args.search_method == DEFAULT_SEARCH_METHOD else None,
            "q-scan summary table",
        )
    if args.output_dir is None:
        suffix = sanitize_filename(args.search_method)
        args.output_dir = DEFAULT_OUT_PARENT / f"alignment_heatmaps_all_hits_{suffix}"

    if args.q_scan and args.tid:
        raise ValueError("--q-scan chooses targets from --q-scan-hits-file; omit --tid")
    if args.q_scan and args.matrix is not None:
        raise ValueError("--matrix is for one explicit pair; q-scan mode expects proxy heatmaps")
    if (args.all_putative or args.qid_list is not None) and args.tid:
        raise ValueError("Batch mode chooses targets from hit tables; omit --tid")
    if (args.all_putative or args.qid_list is not None) and args.matrix is not None:
        raise ValueError("Batch mode cannot use one explicit --matrix")

    print(f"[INFO] search_method={args.search_method}, decoy_method={args.decoy_method}")
    if args.q_scan:
        print(f"[INFO] q-scan hits: {args.q_scan_hits_file}")
        print(f"[INFO] q-scan summary: {args.q_summary_file}")
    else:
        print(f"[INFO] hits: {args.hits_file}")
    print(f"[INFO] output_dir: {args.output_dir}")

    query_records = read_fasta(args.query_fasta)
    target_records = read_fasta(args.target_fasta)
    blosum62 = load_blosum62()
    if blosum62 is None:
        print("[WARN] Biopython BLOSUM62 not available; using a simple amino-acid group score.")

    query_ids = resolve_query_ids(args, query_records)
    if len(query_ids) > 1:
        print(f"[INFO] Running {len(query_ids)} putative queries")

    deepblast_runner: DeepBlastRunner | None = None
    if args.matrix is None:
        deepblast_runner = DeepBlastRunner(
            model_path=args.deepblast_model,
            protrans_model=args.protrans_model,
            alignment_mode=args.deepblast_alignment_mode,
            device=args.deepblast_device,
        )

    index_rows: list[dict] = []

    if args.q_scan:
        n_done = 0
        for qid in query_ids:
            print(f"\n=== q-scan compute: {qid} ===")
            n_done += int(
                compute_qscan_for_query(
                    qid,
                    query_records=query_records,
                    target_records=target_records,
                    blosum62=blosum62,
                    deepblast_runner=deepblast_runner,
                    args=args,
                    index_rows=index_rows,
                )
            )
        write_index(args.output_dir, index_rows)
        print(f"\n[DONE] q-scan compute completed for {n_done}/{len(query_ids)} queries")
        return

    if len(query_ids) != 1:
        raise ValueError("Non-q-scan batch mode is not supported yet; use --q-scan for all putatives.")
    args.qid = query_ids[0]
    pairs = build_pairs(args)
    if args.matrix is not None and len(pairs) != 1:
        raise ValueError("--matrix can only be used when exactly one pair is plotted")
    for pair in pairs:
        row = compute_pair(
            pair,
            output_dir=args.output_dir,
            query_records=query_records,
            target_records=target_records,
            blosum62=blosum62,
            deepblast_runner=deepblast_runner,
            args=args,
        )
        if row is not None:
            index_rows.append(row)
    write_index(args.output_dir, index_rows)


STAGES = {
    "discover": run_discover,
    "heatmap-data": run_heatmap_data,
    "heatmap-plot": run_heatmap_plot,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        print(__doc__)
        print("stages: " + " | ".join(STAGES))
        raise SystemExit(2)
    STAGES[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
