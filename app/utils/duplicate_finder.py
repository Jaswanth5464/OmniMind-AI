"""
Duplicate Finder — lightweight, zero extra libraries.

Layer 1: SHA-256 streaming hash     → exact duplicates (all file types)
Layer 2: TF-IDF cosine similarity   → near-duplicates (text & image captions)
         Uses cached content/summary already stored in the index JSON.
         sklearn (already in requirements) is the only dependency.
"""

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ── Layer 1: Exact match via SHA-256 ─────────────────────────────────────────

def _sha256(path: Path, chunk: int = 65_536) -> str:
    """Stream-hash a file in 64 KB chunks — constant ~64 KB RAM regardless of file size."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while buf := f.read(chunk):
                h.update(buf)
        return h.hexdigest()
    except Exception:
        return ""


def find_exact_duplicates(paths: List[str]) -> List[List[str]]:
    """
    Group files by SHA-256. Returns only groups with 2+ files.
    Filename-independent: same content = same hash regardless of name.
    """
    bucket: Dict[str, List[str]] = defaultdict(list)
    for p in paths:
        h = _sha256(Path(p))
        if h:
            bucket[h].append(p)
    return [grp for grp in bucket.values() if len(grp) >= 2]


# ── Layer 2: Near-duplicate via TF-IDF ───────────────────────────────────────

def find_near_duplicates(
    rows: List[Dict[str, Any]],
    threshold: float = 0.92,
) -> List[List[str]]:
    """
    Find near-duplicate files using TF-IDF cosine similarity on the
    cached content + summary already stored in the index JSON.

    Only the first chunk (chunk_id == 0) per file is used → fast & low RAM.
    sklearn is already in requirements.txt — no new library needed.
    """
    # Collect one row per unique file path (first chunk only)
    seen: set = set()
    file_rows: List[Dict[str, Any]] = []
    for r in rows:
        p = r.get("path", "")
        if r.get("chunk_id", 0) == 0 and p and p not in seen:
            seen.add(p)
            file_rows.append(r)

    if len(file_rows) < 2:
        return []

    paths_list = [r["path"] for r in file_rows]
    # Use summary + first 400 chars of content as the text fingerprint
    texts = [
        (r.get("summary") or "") + " " + (r.get("content") or "")[:400]
        for r in file_rows
    ]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        vec = TfidfVectorizer(max_features=512, sublinear_tf=True)
        mat = vec.fit_transform(texts)          # sparse matrix — minimal RAM
        sim = cosine_similarity(mat)            # n×n float32 array
    except Exception:
        return []

    n = len(file_rows)
    visited: set = set()
    groups: List[List[str]] = []

    for i in range(n):
        if i in visited:
            continue
        group = [paths_list[i]]
        for j in range(i + 1, n):
            if j not in visited and sim[i, j] >= threshold:
                group.append(paths_list[j])
                visited.add(j)
        if len(group) >= 2:
            visited.add(i)
            groups.append(group)

    return groups


# ── Combined entry point ──────────────────────────────────────────────────────

def run_duplicate_scan(
    rows: List[Dict[str, Any]],
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Returns:
        exact_groups  — list of path-groups that are byte-identical
        near_groups   — list of path-groups that are content-similar
                        (already excluding files found in exact_groups)
    """
    all_paths = list({r["path"] for r in rows if r.get("path")})

    exact_groups = find_exact_duplicates(all_paths)

    # Exclude exact duplicates from near-duplicate scan (avoid double-reporting)
    exact_flat = {p for grp in exact_groups for p in grp}
    near_rows  = [r for r in rows if r.get("path") not in exact_flat]
    near_groups = find_near_duplicates(near_rows)

    return exact_groups, near_groups
