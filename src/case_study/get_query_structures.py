import shutil
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR))

from config import CASE_STUDY_DIR, PDBSTYLE_DIR, pdbstyle_path  # noqa: E402


def copy_query_structures(
    query_ids: list[str],
    output_dir: Path,
    pdbstyle_dir: Path | None = None,
) -> list[tuple[str, Path]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: list[tuple[str, Path]] = []
    missing: list[str] = []

    for qid in query_ids:
        qid = qid.strip()
        if not qid:
            continue
        src = pdbstyle_path(qid, pdbstyle_dir=pdbstyle_dir)
        if not src.is_file():
            missing.append(qid)
            continue
        dst = output_dir / src.name
        shutil.copy2(src, dst)
        copied.append((qid, dst))
        print(f"copied {qid}: {src} -> {dst}")

    if missing:
        raise FileNotFoundError(
            "Missing pdbstyle structures for: " + ", ".join(missing)
        )
    return copied


if __name__ == "__main__":
    QUERY_IDS = [
        "d3tg7a1", "d3ku3a1",
        "d5awwy_", "d2a65a1",
    ]
    OUTPUT_DIR = CASE_STUDY_DIR / "structure"
    PDBSTYLE_ROOT = PDBSTYLE_DIR  

    copy_query_structures(QUERY_IDS, OUTPUT_DIR, pdbstyle_dir=PDBSTYLE_ROOT)
