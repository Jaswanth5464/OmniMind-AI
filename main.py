"""
Local AI RAG Search
Split from original monolithic beta_new3.py
Behavior: IDENTICAL
"""

import os
import sys
import json
import argparse
from pathlib import Path

from app.core.device import log_device
from app.core.models import load_embeddings
from app.indexing.builder import build_index
from app.indexing.delta import DeltaIndexer
from app.chat.cli import chat
from app.chat.gui import launch_gui
from app.config import SUPPORTED_TEXT, SUPPORTED_IMG
from app.utils.duplicate_finder import run_duplicate_scan

# Start MENU
def run_menu(folder_args: list[str]):
    paths = [Path(f).expanduser() for f in folder_args]
    
    for p in paths:
        if not p.exists():
            print(f"❌ Folder does not exist: {p}")
            sys.exit(1)

    # Use first folder name or "MasterIndex" if multiple
    if len(paths) == 1:
        name = paths[0].name
    else:
        name = "MasterIndex"

    embeddings = load_embeddings()

    print(f"\nRunning on {len(paths)} folders: {[str(p) for p in paths]}")
    print(f"Index name: '{name}'\n")

    print("Choose mode:")
    print("1. Index")
    print("2. Chat (CLI)")
    print("3. Watchdog")
    print("4. Chat (GUI)")
    print("5. Find Duplicates")
    print("q. Quit")

    choice = input("> ").strip().lower()

    if choice in {"q", "quit", "exit"}:
        print("->Exiting.")
        sys.exit(0)


    # 1. Full Index Build
    if choice == "1":
        print(f"-> Building index for '{name}' …")
        build_index([str(p) for p in paths], name)
        sys.exit(0)


    # 2. CLI Chat
    elif choice == "2":
        db_dir = Path("vector_dbs") / f"{name}_bge_db"
        idx_json = Path("file_indexes", f"{name}.json")

        if not db_dir.exists() or not idx_json.exists():
            print("Index not found. Run Index mode first.")
            sys.exit(1)

        print("-> Starting CLI chat")
        chat(name, embeddings_obj=embeddings, folders=[str(p) for p in paths])
        sys.exit(0)

   
    # 3. Watchdog/Delta Sync
    elif choice == "3":
        db_dir = Path("vector_dbs") / f"{name}_bge_db"
        idx_json = Path("file_indexes", f"{name}.json")

        if not db_dir.exists() or not idx_json.exists():
            print("No index found. Run Index first.")
            sys.exit(1)

        rows = json.loads(idx_json.read_text())
        
        # For simplicity, we sync from the first folder or combined
        # (In a production app we'd loop over all paths, but we'll start with this)
        handler = DeltaIndexer(str(paths[0]), name, embeddings, db_dir, rows)

        curr = {}
        for folder in paths:
            for r, _, fns in os.walk(folder):
                for fn in fns:
                    ext = Path(fn).suffix.lower().lstrip(".")
                    if ext in SUPPORTED_TEXT or ext in SUPPORTED_IMG:
                        p = Path(r) / fn
                        try:
                            stat = p.stat()
                            curr[str(p)] = (stat.st_size, int(stat.st_mtime))
                        except Exception:
                            curr[str(p)] = (0, 0)

        idx_map = {
            r["path"]: (r.get("size", 0), r.get("mtime", 0))
            for r in rows
            if r.get("chunk_id", 0) == 0
        }

        to_add = set(curr) - set(idx_map)
        to_delete = set(idx_map) - set(curr)
        to_mod = {p for p in (set(curr) & set(idx_map)) if curr[p] != idx_map[p]}

        for p in sorted(to_delete):
            print(f"-> Removing {p}")
            handler._remove_path(Path(p))

        for p in sorted(to_mod):
            print(f"-> Re-indexing {p}")
            handler._remove_path(Path(p))
            handler._index_path(Path(p))

        for p in sorted(to_add):
            print(f"-> Adding {p}")
            handler._index_path(Path(p))

        print("✅ Delta sync complete.")
        sys.exit(0)

    
    # 4. GUI Chat
    elif choice == "4":
        db_dir = Path("vector_dbs") / f"{name}_bge_db"
        idx_json = Path("file_indexes", f"{name}.json")

        if not db_dir.exists() or not idx_json.exists():
            print("Index not found. Run Index mode first.")
            sys.exit(1)

        print("-> Launching GUI")
        launch_gui(name, folders=[str(p) for p in paths])
        sys.exit(0)

    # 5. Find Duplicates
    elif choice == "5":
        idx_json = Path("file_indexes", f"{name}.json")
        if not idx_json.exists():
            print("Index not found. Run Index mode first.")
            sys.exit(1)

        print("\n🔍 Scanning for duplicates…")
        rows = json.loads(idx_json.read_text())
        exact_groups, near_groups = run_duplicate_scan(rows)
        total = len(exact_groups) + len(near_groups)

        if not total:
            print("✅ No duplicates found!")
            sys.exit(0)

        print(f"\nFound {total} duplicate group(s):\n")

        if exact_groups:
            print("🔴 EXACT DUPLICATES (byte-identical, safe to delete extra copies):")
            for i, grp in enumerate(exact_groups, 1):
                print(f"  Group {i} ({len(grp)} files):")
                for fp in grp:
                    p = Path(fp)
                    try:
                        kb = p.stat().st_size // 1024
                        print(f"    • {p.name}  ({kb} KB)  →  {fp}")
                    except Exception:
                        print(f"    • {fp}")
            print()

        if near_groups:
            print("🟡 NEAR-DUPLICATES (similar content, review before deleting):")
            for i, grp in enumerate(near_groups, 1):
                print(f"  Group {i} ({len(grp)} files):")
                for fp in grp:
                    p = Path(fp)
                    try:
                        kb = p.stat().st_size // 1024
                        print(f"    • {p.name}  ({kb} KB)  →  {fp}")
                    except Exception:
                        print(f"    • {fp}")
            print()

        sys.exit(0)

    else:
        print("❌ Invalid choice.")
        sys.exit(1)


if __name__ == "__main__":
    from typing import List
    log_device()

    parser = argparse.ArgumentParser(
        description="Local AI RAG Search (Index / Chat / Watchdog / GUI)"
    )
    parser.add_argument(
        "folders",
        nargs="*",
        help="One or more folders to operate on (default: pre-configured path)",
    )

    args = parser.parse_args()
    
    # Use default path if none provided
    folders = args.folders
    if not folders:
        folders = [r"C:\Users\kanam\Downloads\OFFLINE AI FILE ASSISTANT\documents"]
        
    run_menu(folders)
