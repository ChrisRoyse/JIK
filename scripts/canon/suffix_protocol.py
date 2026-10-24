"""Strict interpretation of the pinned suffix tool's collect protocol."""
from pathlib import Path
import numpy as np
from artifact_seals import refuse

EMPTY = "S 0\nMerging.\nNo duplicates found! Either the dataset is duplicate-free or something went wrong.\n"


def normalize_collect(sa):
    raw = sa / "collect.txt"
    # The large nonempty output stays streamed; only inspect the bounded header.
    with raw.open("rb") as f:
        header = f.read(512)
    if header.decode("utf-8", errors="replace") == EMPTY and raw.stat().st_size == len(header):
        dups = sorted((sa / "cache").glob("dups_*"))
        sizes = sorted((sa / "cache").glob("sizes_*"))
        if (not dups or len(dups) != len(sizes)
                or {p.name.removeprefix("dups_") for p in dups} != {p.name.removeprefix("sizes_") for p in sizes}
                or any(p.stat().st_size for p in dups + sizes)):
            refuse("MNLM_SA_EMPTY_UNATTESTED", "collect said zero but producer cache is missing/nonempty")
        (sa / "ranges.txt").write_bytes(b"out\n")
        return {"zero_duplicates": True, "empty_cache_pairs": len(dups)}
    import shutil
    shutil.copyfile(raw, sa / "ranges.txt")
    return {"zero_duplicates": False}


def read_ranges(path: Path) -> np.ndarray:
    import pandas as pd
    skip, found = 0, False
    with path.open("r", encoding="utf-8", errors="strict") as f:
        for line in f:
            skip += 1
            if line.strip() == "out":
                found = True
                break
            if skip > 16:
                break
    if not found:
        refuse("MNLM_SA_RANGES_INVALID", f"{path}: missing collect output marker")
    try:
        df = pd.read_csv(path, sep=" ", header=None, skiprows=skip, dtype=np.int64, engine="c")
    except pd.errors.EmptyDataError:
        return np.empty((0, 2), dtype=np.int64)
    except (ValueError, pd.errors.ParserError) as exc:
        refuse("MNLM_SA_RANGES_INVALID", str(exc))
    if df.shape[1] != 2:
        refuse("MNLM_SA_RANGES_INVALID", "expected two integer columns")
    return df.to_numpy()
