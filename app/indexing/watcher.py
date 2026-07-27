"""
LiveWatcher — background thread that monitors folders and auto-indexes
new / modified / deleted files into the in-memory FAISS vectordb
while the user is actively chatting.
"""

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app.config import SUPPORTED_IMG, SUPPORTED_TEXT
from app.core.file_extractors import extract_text
from app.core.image_caption import caption_images
from app.core.text_utils import chunk_text, clean_words, summarize


# ── helpers ────────────────────────────────────────────────────────────────────

def _sync_faiss(vectordb, new_store) -> None:
    """Copy FAISS internal state from new_store into the live vectordb object."""
    vectordb.index               = new_store.index
    vectordb.docstore            = new_store.docstore
    vectordb.index_to_docstore_id = new_store.index_to_docstore_id


def _build_docs_for_path(path: Path) -> List[Dict[str, Any]]:
    """Extract text / caption for a file and return a list of row dicts."""
    ext = path.suffix.lower().lstrip(".")
    rows: List[Dict[str, Any]] = []

    try:
        size  = path.stat().st_size
        mtime = int(path.stat().st_mtime)
    except Exception:
        size = mtime = 0

    if ext in SUPPORTED_TEXT:
        raw = extract_text(path)[:30_000]
        for cid, chunk in enumerate(chunk_text(raw)):
            rows.append({
                "path":     str(path),
                "type":     ext,
                "chunk_id": cid,
                "content":  chunk,
                "summary":  summarize(chunk),
                "tags":     clean_words(chunk) or clean_words(path.stem),
                "size":     size,
                "mtime":    mtime,
            })
    elif ext in SUPPORTED_IMG:
        cap = caption_images([path])[0]
        rows.append({
            "path":     str(path),
            "type":     ext,
            "chunk_id": 0,
            "content":  cap,
            "summary":  cap,
            "tags":     clean_words(cap),
            "size":     size,
            "mtime":    mtime,
        })

    return rows


def _rows_to_docs(rows: List[Dict[str, Any]]) -> List[Document]:
    return [
        Document(
            page_content=r["content"],
            metadata={k: r[k] for k in r if k != "content"},
        )
        for r in rows
    ]


# ── event handler ──────────────────────────────────────────────────────────────

class _WatchHandler(FileSystemEventHandler):
    """Debounced handler that queues file events and processes them every second."""

    DEBOUNCE = 2.0   # seconds — wait this long after last event before indexing

    def __init__(self, watcher: "LiveWatcher"):
        super().__init__()
        self._watcher = watcher
        self._pending: Dict[str, tuple] = {}   # path -> (timestamp, action)

    # ── watchdog callbacks ────────────────────────────────────────────────────

    def on_created(self, event):
        if not event.is_directory:
            self._queue(event.src_path, "add")

    def on_modified(self, event):
        if not event.is_directory:
            self._queue(event.src_path, "add")

    def on_deleted(self, event):
        if not event.is_directory:
            self._queue(event.src_path, "remove")

    def on_moved(self, event):
        if not event.is_directory:
            self._queue(event.src_path,  "remove")
            self._queue(event.dest_path, "add")

    # ── internal ──────────────────────────────────────────────────────────────

    def _queue(self, path: str, action: str):
        self._pending[path] = (time.time(), action)

    def flush(self):
        """Called every second from the flush thread to process debounced events."""
        now  = time.time()
        done = [
            (p, act) for p, (ts, act) in list(self._pending.items())
            if now - ts >= self.DEBOUNCE
        ]
        for path, action in done:
            del self._pending[path]
            p   = Path(path)
            ext = p.suffix.lower().lstrip(".")
            if ext not in SUPPORTED_TEXT and ext not in SUPPORTED_IMG:
                continue
            self._watcher._process(p, action)


# ── public API ─────────────────────────────────────────────────────────────────

class LiveWatcher:
    """
    Start a background watchdog that keeps the in-memory FAISS vectordb
    up-to-date as files are added / changed / deleted while the user chats.

    Usage::

        watcher = LiveWatcher(folders, name, embeddings, db_dir, rows)
        watcher.start(vectordb)          # non-blocking
        ...                              # user chats normally
        watcher.stop()                   # on app exit
    """

    def __init__(
        self,
        folders:    List[str],
        name:       str,
        embeddings,
        db_dir:     Path,
        rows:       List[Dict[str, Any]],
    ):
        self._folders    = [str(f) for f in folders]
        self._name       = name
        self._embeddings = embeddings
        self._db_dir     = Path(db_dir)
        self._rows       = list(rows)          # master row list (mutable)
        self._vectordb   = None                # set in .start()
        self._lock       = threading.Lock()
        self._running    = False
        self._observers: List[Observer] = []
        self._handler:   _WatchHandler  = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self, vectordb) -> None:
        """Start watching in the background. Non-blocking."""
        self._vectordb = vectordb
        self._handler  = _WatchHandler(self)
        self._running  = True

        for folder in self._folders:
            obs = Observer()
            obs.schedule(self._handler, folder, recursive=True)
            obs.daemon = True
            obs.start()
            self._observers.append(obs)

        t = threading.Thread(target=self._flush_loop, daemon=True, name="LiveWatcher-flush")
        t.start()

        # Start a background catch-up scan to index changes made while app was closed
        t_catch = threading.Thread(target=self._catch_up_scan, daemon=True, name="LiveWatcher-catchup")
        t_catch.start()

        print(f"👁️  Live watcher active on {len(self._folders)} folder(s). New files will be indexed automatically.")

    def stop(self) -> None:
        self._running = False
        for obs in self._observers:
            try:
                obs.stop()
                obs.join(timeout=2)
            except Exception:
                pass
        print("👁️  Live watcher stopped.")

    # ── internal ──────────────────────────────────────────────────────────────

    def _catch_up_scan(self):
        """Scan folders on start and sync with index rows without blocking GUI."""
        from app.config import SUPPORTED_IMG, SUPPORTED_TEXT
        import os

        # 1. Gather current state of disk
        curr: Dict[str, tuple] = {}
        for folder in self._folders:
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

        # 2. Compare with internal rows
        with self._lock:
            idx_map = {
                r["path"]: (r.get("size", 0), r.get("mtime", 0))
                for r in self._rows
                if r.get("chunk_id", 0) == 0
            }

        to_add = set(curr) - set(idx_map)
        to_delete = set(idx_map) - set(curr)
        to_mod = {p for p in (set(curr) & set(idx_map)) if curr[p] != idx_map[p]}

        if not to_add and not to_delete and not to_mod:
            return

        print(f"🔄  Catch-up scan: {len(to_add)} added, {len(to_mod)} modified, {len(to_delete)} removed.")

        for p in sorted(to_delete):
            self._process(Path(p), "remove")
        for p in sorted(to_mod):
            self._process(Path(p), "remove")
            self._process(Path(p), "add")
        for p in sorted(to_add):
            self._process(Path(p), "add")

    def _flush_loop(self):
        while self._running:
            if self._handler:
                self._handler.flush()
            time.sleep(1)

    def _process(self, path: Path, action: str):
        with self._lock:
            if action == "add" and path.exists():
                self._do_add(path)
            elif action == "remove":
                self._do_remove(path)

    def _do_add(self, path: Path):
        print(f"\n🔄  Auto-indexing new/changed file: {path.name} …")
        new_rows = _build_docs_for_path(path)
        if not new_rows:
            return

        # Remove stale entries for this path, then append fresh ones
        self._rows = [r for r in self._rows if r["path"] != str(path)] + new_rows

        # Add directly to in-memory vectordb
        docs = _rows_to_docs(new_rows)
        self._vectordb.add_documents(docs)

        self._persist()
        print(f"✅  Auto-indexed: {path.name}")

    def _do_remove(self, path: Path):
        print(f"\n🗑️   Removing from index: {path.name} …")
        self._rows = [r for r in self._rows if r["path"] != str(path)]

        # Rebuild FAISS from remaining rows (no native delete in FAISS)
        from langchain_community.vectorstores import FAISS
        docs = _rows_to_docs(self._rows)
        if docs:
            new_store = FAISS.from_documents(docs, self._embeddings)
        else:
            new_store = FAISS.from_documents(
                [Document(page_content=" ", metadata={})], self._embeddings
            )

        _sync_faiss(self._vectordb, new_store)
        self._persist()
        print(f"✅  Removed from index: {path.name}")

    def _persist(self):
        """Save updated rows JSON and FAISS store to disk."""
        Path("file_indexes").mkdir(exist_ok=True)
        Path("file_indexes", f"{self._name}.json").write_text(
            json.dumps(self._rows, indent=2)
        )
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._vectordb.save_local(str(self._db_dir))
