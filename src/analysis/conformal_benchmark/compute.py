import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
CONFORMAL_LIB = PROJECT_ROOT / "libs" / "conformal-protein-retrieval"

DEFAULT_SCORE_DIR = PROJECT_ROOT / "data"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "conformal_cache"
# one folder per (n_calib, split level); CSVs go in <combo>/data, plots in <combo>/figures
DEFAULT_OUT_DIR = PROJECT_ROOT / "results_conformal"

# raw search-result tables: query = real ASTRAL40, database = ASTRAL40
SCORE_FILES = {
    "blastp_postprocessed": "_hf_blastp/raw/scores/search_data/"
                            "result_blastp_postprocessed_astral_astral.txt",
    "plm": "result_plm_astral_astral.txt",
    "tmvec": "result_tmvec_astral_astral.txt",
    "dhr_postprocess": "result_dhr_postprocess_astral_astral.txt",
}

SEARCH_METHOD_DISPLAY = {"plm": "PLMsearch", "tmvec": "TMvec",
                         "dhr_postprocess": "DHR", "blastp_postprocessed": "BLASTp"}

# PLM-Caliper's canonical decoy per search method: extended-Markov for the PLMs,
# plain Markov for BLASTp, where a local alignment would recover the query half.
CALIPER_DECOY = {"blastp_postprocessed": "mkv2", "plm": "extended_mkv2",
                 "tmvec": "extended_mkv2", "dhr_postprocess": "extended_mkv2"}

HOMO_LEVELS = (1, 2)
NONHOMO_LEVEL = -1
UNCERTAIN_LEVEL = 0

UNCERTAIN_SENTINEL = -1e-6

DEFAULT_Q_LEVELS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 ... 0.95

DEFAULT_FASTA = PROJECT_ROOT / "data" / "astral.fa"

# SCOP sccs is class.fold.superfamily.family, e.g. a.1.1.1; a split level keeps a
# whole group on one side of the calibration/test boundary
SPLIT_LEVELS = {
    "random": None,
    "fold": 2,
    "superfamily": 3,
    "family": 4,
}


# --------------------------------------------------------------------------
# conformal code, imported unmodified
# --------------------------------------------------------------------------
def load_protein_conformal():
    if not CONFORMAL_LIB.exists():
        raise FileNotFoundError(
            f"{CONFORMAL_LIB} missing -- run:\n"
            f"  git clone https://github.com/ronboger/conformal-protein-retrieval.git "
            f"{CONFORMAL_LIB}"
        )

    # Only the symbols util.py imports at module scope, and nothing else: a stub that
    # answers every attribute (dunders included) makes matplotlib's unit-converter
    # probing recurse forever if plotting later happens in the same process.
    STUBBED = {
        "Bio": (),
        "Bio.Align": ("PairwiseAligner",),
        "transformers": ("AutoModelForMaskedLM", "AutoTokenizer"),
        # scipy's array-API dispatch probes torch.Tensor as soon as torch looks
        # importable, so the stub has to answer -- with a class nothing is a
        # subclass of, or numpy arrays would be dispatched as torch tensors.
        "torch": ("Tensor",),
    }

    class _NotATensor:
        """Sentinel type: issubclass(anything, _NotATensor) is False."""

    class _Stub(types.ModuleType):
        def __init__(self, name, allowed):
            super().__init__(name)
            self._allowed = set(allowed)

        def __getattr__(self, name):
            if name == "Tensor":
                return _NotATensor
            if name in self._allowed:
                return object
            raise AttributeError(f"{self.__name__} is a stub; {name!r} is not available")

    for name, allowed in STUBBED.items():
        sys.modules.setdefault(name, _Stub(name, allowed))

    util_path = CONFORMAL_LIB / "protein_conformal" / "util.py"
    spec = importlib.util.spec_from_file_location("protein_conformal_util", util_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# prepare: raw TSV -> dense matrices
# --------------------------------------------------------------------------
def resolve_score_path(search_method: str, score_dir: Path) -> Path:
    if search_method not in SCORE_FILES:
        raise KeyError(f"unknown search method {search_method!r}; "
                       f"known: {sorted(SCORE_FILES)}")
    return score_dir / SCORE_FILES[search_method]


def cache_paths(cache_dir: Path, search_method: str) -> dict[str, Path]:
    return {
        "sims": cache_dir / f"{search_method}_sims.npy",
        "homo": cache_dir / f"{search_method}_homo.npy",
        "ids": cache_dir / f"{search_method}_ids.txt",
    }


def prepare_matrices(search_method: str, score_dir: Path, cache_dir: Path,
                     chunk_queries: int = 200, force: bool = False) -> None:
    paths = cache_paths(cache_dir, search_method)
    if not force and all(p.exists() for p in paths.values()):
        print(f"[skip] {search_method}: cache already present in {cache_dir}")
        return

    src = resolve_score_path(search_method, score_dir)
    if not src.exists():
        raise FileNotFoundError(f"missing score table: {src}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cols = ["qid", "tid", "score", "homo_type"]
    head = pd.read_csv(src, sep="\t", usecols=cols, nrows=1_000_000)
    block = int((head["qid"] == head["qid"].iloc[0]).sum())
    ids = head["tid"].iloc[:block].tolist()
    if len(set(ids)) != block:
        raise ValueError(f"{src.name}: first query block has duplicate tids")
    n = block
    print(f"[prepare] {search_method}: {n} targets per query, source {src}")

    id_index = pd.Index(ids)
    sims = np.zeros((n, n), dtype=np.float32)
    homo = np.zeros((n, n), dtype=np.int8)
    seen: dict[str, int] = {}

    reader = pd.read_csv(src, sep="\t", usecols=cols, chunksize=n * chunk_queries,
                         dtype={"score": np.float32, "homo_type": np.int8})
    leftover = None
    for chunk in reader:
        if leftover is not None:
            chunk = pd.concat([leftover, chunk], ignore_index=True)
            leftover = None
        # keep only whole query blocks; carry the tail to the next chunk
        tail = len(chunk) % n
        if tail:
            leftover = chunk.iloc[len(chunk) - tail:]
            chunk = chunk.iloc[: len(chunk) - tail]
        if chunk.empty:
            continue

        col = id_index.get_indexer(chunk["tid"].to_numpy())
        if (col < 0).any():
            raise ValueError(f"{src.name}: tid outside the first query block")
        qids = chunk["qid"].to_numpy()[::n]
        score = chunk["score"].to_numpy()
        htype = chunk["homo_type"].to_numpy()

        for b, qid in enumerate(qids):
            if qid in seen:
                raise ValueError(f"{src.name}: query {qid} appears in two blocks; "
                                 "rows must be grouped by qid")
            row = seen.setdefault(qid, len(seen))
            sl = slice(b * n, (b + 1) * n)
            sims[row, col[sl]] = score[sl]
            homo[row, col[sl]] = htype[sl]
        print(f"  ... {len(seen)}/{n} queries", end="\r", flush=True)

    if leftover is not None and not leftover.empty:
        raise ValueError(f"{src.name}: trailing partial query block ({len(leftover)} rows)")
    if len(seen) != n:
        raise ValueError(f"{src.name}: got {len(seen)} queries, expected {n}")

    # rows are in file order; reorder them to match the column (target) order so
    # sims[i, j] is always score(query i, target j) over one shared id list
    order = np.array([seen[i] for i in ids])
    sims = sims[order]
    homo = homo[order]

    np.save(paths["sims"], sims)
    np.save(paths["homo"], homo)
    paths["ids"].write_text("\n".join(ids) + "\n")
    print(f"\n[prepare] {search_method}: wrote {paths['sims'].name} "
          f"({sims.nbytes / 1e9:.2f} GB) and {paths['homo'].name}")



# --------------------------------------------------------------------------
# calibration / test splits
# --------------------------------------------------------------------------
def load_sccs(ids, fasta_path: Path) -> np.ndarray:
    lookup = {}
    with open(fasta_path) as fh:
        for line in fh:
            if line.startswith(">"):
                parts = line[1:].split()
                if len(parts) >= 2:
                    lookup[parts[0]] = parts[1]
    missing = [i for i in ids if i not in lookup]
    if missing:
        raise KeyError(f"{len(missing)} ids absent from {fasta_path} "
                       f"(first: {missing[:3]})")
    return np.array([lookup[i] for i in ids])


def split_groups(ids, level: str, fasta_path: Path) -> np.ndarray:
    if level not in SPLIT_LEVELS:
        raise ValueError(f"unknown split level {level!r}; known: {sorted(SPLIT_LEVELS)}")
    depth = SPLIT_LEVELS[level]
    if depth is None:                       # random: every query is its own group
        return np.arange(len(ids))
    sccs = load_sccs(ids, fasta_path)
    keys = np.array([".".join(s.split(".")[:depth]) for s in sccs])
    _, codes = np.unique(keys, return_inverse=True)
    return codes


def group_members(groups: np.ndarray) -> dict:
    """group id -> member row indices, built in one pass instead of per trial."""
    order = np.argsort(groups, kind="stable")
    keys = groups[order]
    edges = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1], True])
    return {keys[edges[k]]: order[edges[k]:edges[k + 1]]
            for k in range(len(edges) - 1)}


def make_split(rng, groups: np.ndarray, n_calib: int, *, overshoot: float = 0.5,
               members: dict | None = None):
    n = len(groups)
    order = rng.permutation(np.unique(groups))
    if members is None:
        members = group_members(groups)

    cap = n_calib * (1.0 + overshoot)
    taken, total = [], 0
    for g in order:
        idx = members[g]
        if total >= n_calib:
            break
        if taken and total + len(idx) > cap:
            continue
        taken.append(idx)
        total += len(idx)

    cal = np.concatenate(taken)
    mask = np.zeros(n, dtype=bool)
    mask[cal] = True
    return cal, np.flatnonzero(~mask)



# --------------------------------------------------------------------------
# run: split -> calibrate -> evaluate
# --------------------------------------------------------------------------
def _load_matrices(search_method: str, cache_dir: Path, exclude_uncertain: bool
                   ) -> tuple[np.ndarray, np.ndarray]:
    paths = cache_paths(cache_dir, search_method)
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing cache {missing}; run the `prepare` subcommand")

    sims = np.load(paths["sims"])
    homo = np.load(paths["homo"])

    lo, hi = float(sims.min()), float(sims.max())
    if hi <= lo:
        raise ValueError(f"{search_method}: degenerate score range [{lo}, {hi}]")
    sims -= lo
    sims /= (hi - lo)

    labels = np.isin(homo, HOMO_LEVELS).astype(np.int8)
    if exclude_uncertain:
        sims[homo == UNCERTAIN_LEVEL] = UNCERTAIN_SENTINEL
    return sims, labels


def sorted_structures(sims: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sorted_all = np.sort(sims, axis=1)
    width = max(int(labels.sum(axis=1).max()), 1)
    homo_padded = np.full((sims.shape[0], width), -np.inf, dtype=np.float32)
    for i in range(sims.shape[0]):
        v = np.sort(sims[i][labels[i] == 1])
        if v.size:
            homo_padded[i, -v.size:] = v
    return sorted_all, homo_padded


def risk_std_curve(rows, lambdas, sorted_all, homo_padded):
    n_target = sorted_all.shape[1]
    width = homo_padded.shape[1]
    fdp = np.empty((len(rows), len(lambdas)), dtype=np.float64)
    for r, i in enumerate(rows):
        total = n_target - np.searchsorted(sorted_all[i], lambdas, side="left")
        pos = width - np.searchsorted(homo_padded[i], lambdas, side="left")
        fdp[r] = (total - pos) / np.maximum(total, 1)
    return fdp.mean(axis=0), fdp.std(axis=0)


def lambda_grid(sims_cal: np.ndarray, n_lambda: int, mode: str) -> np.ndarray:
    if mode == "linear":
        return np.linspace(sims_cal.min(), sims_cal.max(), n_lambda)
    if mode == "quantile":
        probs = np.linspace(0.0, 1.0, n_lambda)
        grid = np.quantile(sims_cal, probs, method="lower")
        return np.unique(grid)
    raise ValueError(f"unknown lambda grid mode {mode!r}")


def thresholds_for_alphas(sims_cal, labels_cal, rows, alphas, *, pc, delta: float,
                          n_lambda: int, grid_mode: str, sorted_all, homo_padded):
    n = len(labels_cal)
    lambdas = lambda_grid(sims_cal, n_lambda, grid_mode)
    risks, stds = risk_std_curve(rows, lambdas, sorted_all, homo_padded)
    stds = np.maximum(stds, 1e-6)

    out = {}
    for alpha in alphas:
        pvals = pc.clt_p_value(risks, stds, n, alpha)
        below = pvals <= delta
        # smallest lambda such that every larger lambda also passes
        satisfies = np.cumprod(below[::-1])[::-1].astype(bool)
        idx = int(np.argmax(satisfies))
        lhat = min(float(lambdas[idx]), 1.0)
        out[alpha] = (lhat, float(risks[idx]), bool(satisfies[idx]))
    return out


def _verify_fast_path(pc, sims, labels, sorted_all, homo_padded, rows, lambdas) -> None:
    probe = lambdas[:: max(len(lambdas) // 12, 1)]
    fast_r, fast_s = risk_std_curve(rows, probe, sorted_all, homo_padded)
    slow_r = np.array([pc.risk(sims[rows], labels[rows], lam) for lam in probe])
    slow_s = np.array([pc.std_loss(sims[rows], labels[rows], lam) for lam in probe])
    if not (np.allclose(fast_r, slow_r) and np.allclose(fast_s, slow_s)):
        raise AssertionError("fast lambda sweep disagrees with protein_conformal.util")
    print(f"[verify] lambda sweep matches util.risk / util.std_loss "
          f"at {len(probe)} grid points")


# --------------------------------------------------------------------------
# per-query results: the counts every FDR convention can be derived from
# --------------------------------------------------------------------------
def per_query_counts(sims_rows, homo_rows, threshold):
    """(n_tp, n_fp, n_selected) per query at one score threshold."""
    sel = sims_rows >= threshold
    n_selected = sel.sum(axis=1)
    n_tp = (sel & np.isin(homo_rows, HOMO_LEVELS)).sum(axis=1)
    n_fp = (sel & (homo_rows == NONHOMO_LEVEL)).sum(axis=1)
    return (n_tp.astype(np.int32), n_fp.astype(np.int32),
            n_selected.astype(np.int32))


def default_tag(search_method: str, n_calib: int, grid_mode: str, n_lambda: int,
                keep_uncertain: bool = False) -> str:
    """Filename tag; plot.py rebuilds it from the same arguments."""
    return (f"{search_method}_astral_ncal{n_calib}_{grid_mode}{n_lambda}"
            + ("_withuncertain" if keep_uncertain else ""))


def default_out_dir(n_calib: int, split_level: str, n_trials: int) -> Path:
    """One results folder per (n_calib, split level, trial count)."""
    return DEFAULT_OUT_DIR / f"ncal{n_calib}_{split_level}_t{n_trials}"


def perquery_path(data_dir: Path, tag: str, prefix: str = "conformal") -> Path:
    return data_dir / f"{prefix}_perquery_{tag}.npz"


def save_perquery(path: Path, *, n_tp, n_fp, n_selected, is_test, total_homo,
                  qids, q_levels, split_level, n_calib_target) -> None:
    """Counts for every (trial, query, q); rows are ALL queries, `is_test` selects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, n_tp=n_tp, n_fp=n_fp, n_selected=n_selected, is_test=is_test,
        total_homo=np.asarray(total_homo, dtype=np.int32),
        qids=np.asarray(qids, dtype=object), q_levels=np.asarray(q_levels, float),
        split_level=str(split_level), n_calib_target=int(n_calib_target))
    print(f"[perquery] {path.name} "
          f"({n_tp.shape[0]} trials x {n_tp.shape[1]} queries x {n_tp.shape[2]} q)")


def load_perquery(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def fdp_per_query(d: dict, *, convention: str = "plmcaliper"):
    """Per-query FDP, shape (trials, queries, q).

    plmcaliper  fp / (tp + fp), NaN when the query selected nothing annotated
    risk        (selected - tp) / max(selected, 1), 0 for an empty selection
    no_empties  same numerator as `risk`, NaN for an empty selection
    """
    n_tp = d["n_tp"].astype(float)
    n_fp = d["n_fp"].astype(float)
    n_sel = d["n_selected"].astype(float)
    if convention == "plmcaliper":
        ann = n_tp + n_fp
        return np.where(ann > 0, n_fp / np.maximum(ann, 1), np.nan)
    false_disc = n_sel - n_tp                      # their (1 - labels) * selected
    if convention == "risk":
        return false_disc / np.maximum(n_sel, 1)
    if convention == "no_empties":
        return np.where(n_sel > 0, false_disc / np.maximum(n_sel, 1), np.nan)
    raise ValueError(f"unknown FDP convention {convention!r}")


def power_per_query(d: dict):
    total = np.maximum(d["total_homo"].astype(float), 1)[None, :, None]
    return d["n_tp"].astype(float) / total


def mask_test(values, d: dict):
    """Blank out queries that were in the calibration half of their trial."""
    return np.where(d["is_test"][:, :, None], values, np.nan)


# --------------------------------------------------------------------------
# run: split -> lambda-hat per q -> per-query counts on the held-out queries
# --------------------------------------------------------------------------
def run(search_method: str, cache_dir: Path, out_dir: Path, *, q_levels,
        n_calib: int, n_trials: int, delta: float, n_lambda: int, grid_mode: str,
        seed: int, exclude_uncertain: bool, tag: str, verify: bool,
        split_level: str = "random", fasta_path: Path = DEFAULT_FASTA) -> pd.DataFrame:
    pc = load_protein_conformal()
    sims, labels = _load_matrices(search_method, cache_dir, exclude_uncertain)
    homo = np.load(cache_paths(cache_dir, search_method)["homo"])
    n_query = sims.shape[0]
    if n_calib >= n_query:
        raise ValueError(f"--n-calib {n_calib} must be < {n_query} queries")

    ids = cache_paths(cache_dir, search_method)["ids"].read_text().split()
    groups = split_groups(ids, split_level, fasta_path)
    members = group_members(groups)

    print(f"[run] {search_method}: {n_query} queries, n_calib~{n_calib}, "
          f"split={split_level} ({len(np.unique(groups))} groups), "
          f"n_trials={n_trials}, delta={delta}, N={n_lambda}, grid={grid_mode}, "
          f"exclude_uncertain={exclude_uncertain}")
    print("[run] sorting per-query scores ...", flush=True)
    sorted_all, homo_padded = sorted_structures(sims, labels)

    if verify:
        probe_rows = np.random.default_rng(seed).choice(n_query, size=200, replace=False)
        _verify_fast_path(pc, sims, labels, sorted_all, homo_padded, probe_rows,
                          lambda_grid(sims[probe_rows], n_lambda, grid_mode))

    n_q = len(q_levels)
    pq = {k: np.zeros((n_trials, n_query, n_q), dtype=np.int32)
          for k in ("n_tp", "n_fp", "n_selected")}
    is_test = np.zeros((n_trials, n_query), dtype=bool)
    total_homo = np.isin(homo, HOMO_LEVELS).sum(axis=1).astype(np.int32)

    rng = np.random.default_rng(seed)
    rows = []
    for trial in range(n_trials):
        cal, test = make_split(rng, groups, n_calib, members=members)
        is_test[trial, test] = True
        sims_test, labels_test = sims[test], labels[test]

        found = thresholds_for_alphas(sims[cal], labels[cal], cal, q_levels, pc=pc,
                                      delta=delta, n_lambda=n_lambda,
                                      grid_mode=grid_mode, sorted_all=sorted_all,
                                      homo_padded=homo_padded)
        # several alphas usually land on the same grid point; evaluate each once
        evaluated: dict[float, tuple] = {}
        for j, q in enumerate(q_levels):
            lhat, fdr_cal, satisfied = found[q]
            if lhat not in evaluated:
                evaluated[lhat] = (
                    per_query_counts(sims_test, homo[test], lhat),
                    float(pc.risk(sims_test, labels_test, lhat)),
                    float(pc.risk_no_empties(sims_test, labels_test, lhat)),
                    float(pc.calculate_true_positives(sims_test, labels_test, lhat)),
                    float(np.mean((sims_test >= lhat).sum(axis=1))),
                )
            counts, real_fdr, real_fdr_nonempty, power, mean_hits = evaluated[lhat]
            for key, col in zip(("n_tp", "n_fp", "n_selected"), counts):
                pq[key][trial, test, j] = col
            rows.append({
                "trial": trial, "q": q, "lambda": lhat, "fdr_cal": fdr_cal,
                "satisfied": satisfied, "real_fdr": real_fdr,
                "real_fdr_nonempty": real_fdr_nonempty, "power": power,
                "mean_hits": mean_hits, "split_level": split_level,
                "n_calib_target": n_calib, "n_calib": len(cal), "n_test": len(test),
            })
        print(f"  trial {trial + 1}/{n_trials} done", end="\r", flush=True)

    df = pd.DataFrame(rows)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    trials_path = data_dir / f"conformal_trials_{tag}.csv"
    df.to_csv(trials_path, index=False)

    agg = (df.groupby("q")
             .agg(mean_real_fdr=("real_fdr", "mean"),
                  sd_real_fdr=("real_fdr", "std"),
                  mean_real_fdr_nonempty=("real_fdr_nonempty", "mean"),
                  mean_power=("power", "mean"),
                  sd_power=("power", "std"),
                  mean_lambda=("lambda", "mean"),
                  sd_lambda=("lambda", "std"),
                  mean_fdr_cal=("fdr_cal", "mean"),
                  mean_hits=("mean_hits", "mean"),
                  mean_n_calib=("n_calib", "mean"),
                  sd_n_calib=("n_calib", "std"),
                  n_trials=("trial", "count"),
                  n_satisfied=("satisfied", "sum"))
             .reset_index())
    agg_path = data_dir / f"conformal_agg_{tag}.csv"
    agg.to_csv(agg_path, index=False)

    pq_path = perquery_path(data_dir, tag)
    save_perquery(pq_path, n_tp=pq["n_tp"], n_fp=pq["n_fp"],
                  n_selected=pq["n_selected"], is_test=is_test,
                  total_homo=total_homo, qids=ids, q_levels=q_levels,
                  split_level=split_level, n_calib_target=n_calib)
    _assert_perquery_matches(pq_path, df, q_levels)

    print(f"\n[run] wrote {trials_path.name}, {agg_path.name} and {pq_path.name} "
          f"in {data_dir}")
    print(agg[["q", "mean_real_fdr", "mean_power", "mean_lambda"]].to_string(index=False))
    return agg


def _assert_perquery_matches(pq_path: Path, df: pd.DataFrame, q_levels) -> None:
    """The stored counts must reproduce the aggregates the CSV reports."""
    d = load_perquery(pq_path)
    risk = mask_test(fdp_per_query(d, convention="risk"), d)
    power = mask_test(power_per_query(d), d)
    for _, r in df.iterrows():
        j = int(np.argmin(np.abs(np.asarray(q_levels) - r["q"])))
        t = int(r["trial"])
        for name, got in (("real_fdr", np.nanmean(risk[t, :, j])),
                          ("power", np.nanmean(power[t, :, j]))):
            if not np.isclose(got, r[name], rtol=1e-9, atol=1e-12):
                raise AssertionError(
                    f"per-query counts disagree with {name} at trial {t}, q={r['q']}: "
                    f"{got!r} vs {r[name]!r}")
    print("[check] per-query counts reproduce real_fdr and power exactly")


# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["prepare", "run", "all"])
    p.add_argument("--search-method", default="blastp_postprocessed",
                   choices=sorted(SCORE_FILES))
    p.add_argument("--q-levels", nargs="+", type=float, default=DEFAULT_Q_LEVELS,
                   help="target FDR levels (default 0.05 ... 0.95)")
    p.add_argument("--n-calib", type=int, default=1000,
                   help="calibration queries per trial (rest are the test set)")
    p.add_argument("--n-trials", type=int, default=5,
                   help="calibration/test splits; the split-to-split spread is "
                        "reported as a 95%% CI, so raise this when that CI matters")
    p.add_argument("--delta", type=float, default=0.5, help="LTT p-value limit")
    p.add_argument("--n-lambda", type=int, default=5000,
                   help="size of their lambda grid (their SCOPe default is 5000)")
    p.add_argument("--lambda-grid", default="linear", choices=["linear", "quantile"],
                   help="canonical setting, used for every reported result: "
                        "linear = their linspace(min, max, N). quantile is kept as "
                        "a diagnostic only -- method='lower' can only return score "
                        "values that occur, so tied BLASTp bit scores collapse the "
                        "5000 probabilities to 52 usable points")
    p.add_argument("--split-level", default="random", choices=sorted(SPLIT_LEVELS),
                   help="what a calibration/test split may not straddle. random = "
                        "per-query (their protocol); family/superfamily/fold keep "
                        "whole SCOP groups on one side, so a test query has no "
                        "calibration relative at that level. fold is the strictest "
                        "and the one that matches how homology is labelled here")
    p.add_argument("--fasta", type=Path, default=DEFAULT_FASTA,
                   help="ASTRAL fasta supplying the SCOP sccs for the split levels")
    p.add_argument("--seed", type=int, default=0,
                   help="draws the splits; pass the same value to the PLM-Caliper "
                        "side to put both methods on an identical set of splits")
    p.add_argument("--keep-uncertain", action="store_true",
                   help="count same-fold/different-superfamily pairs as false "
                        "discoveries instead of excluding them (their protocol, "
                        "not the paper's)")
    p.add_argument("--verify", action="store_true",
                   help="assert the multi-alpha sweep reproduces their get_thresh_FDR")
    p.add_argument("--score-dir", type=Path, default=DEFAULT_SCORE_DIR)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--tag", default=None, help="output filename tag")
    p.add_argument("--force", action="store_true", help="rebuild the matrix cache")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tag = args.tag or default_tag(args.search_method, args.n_calib,
                                  args.lambda_grid, args.n_lambda,
                                  args.keep_uncertain)
    if args.out_dir == DEFAULT_OUT_DIR:
        args.out_dir = default_out_dir(args.n_calib, args.split_level, args.n_trials)

    if args.stage in ("prepare", "all"):
        prepare_matrices(args.search_method, args.score_dir, args.cache_dir,
                         force=args.force)
    if args.stage in ("run", "all"):
        run(args.search_method, args.cache_dir, args.out_dir,
            q_levels=args.q_levels, n_calib=args.n_calib, n_trials=args.n_trials,
            delta=args.delta, n_lambda=args.n_lambda, grid_mode=args.lambda_grid,
            seed=args.seed, exclude_uncertain=not args.keep_uncertain, tag=tag,
            verify=args.verify, split_level=args.split_level, fasta_path=args.fasta)


if __name__ == "__main__":
    main()
