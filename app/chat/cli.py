import os
import sys
import json
import re
import subprocess
import time
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS

from app.rag.llm import init_llm
from app.rag.answer import generate_answer_with_llm
from app.utils.misc import looks_like_gibberish
from app.core.models import load_embeddings
from app.indexing.watcher import LiveWatcher

#watcher
def chat(name: str, embeddings_obj=None, folders: List[str] = None):
    db_dir = Path("vector_dbs") / f"{name}_bge_db"
    if not db_dir.exists():
        sys.exit(f"Index '{name}' not found. Run index mode first.")

    age_hours = (time.time() - db_dir.stat().st_mtime) / 3600
    if age_hours > 12:
        print(f"DB is {age_hours:.1f}h old – re-index if files changed.")

    embeddings = embeddings_obj or load_embeddings()
    vectordb = FAISS.load_local(
        str(db_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    llm = init_llm()
    rows = json.loads(Path("file_indexes", f"{name}.json").read_text())

    # ── Live file watcher ────────────────────────────────────────────────────
    watcher = None
    if folders:
        db_dir = Path("vector_dbs") / f"{name}_bge_db"
        watcher = LiveWatcher(
            folders=folders,
            name=name,
            embeddings=embeddings,
            db_dir=db_dir,
            rows=rows,
        )
        watcher.start(vectordb)
    # ────────────────────────────────────────────────────────────────────────

    last_paths: List[str] = []

    print("Ask me anything (type 'exit' to quit)")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q or q.lower() in {"exit", "quit"}:
            if watcher:
                watcher.stop()
            break

        m = re.match(
            r"open\s+(?:the\s+file\s+)?(?:which\s+has|that\s+contains)?\s*(.+)",
            q,
            re.I,
        )
        if m:
            term = m.group(1).strip()
            hits = vectordb.similarity_search_with_score(term, k=5)
            hits = [(d, s) for d, s in hits if s >= 0.3]

            if not hits:
                print("No file found.")
                continue

            path = hits[0][0].metadata.get("path")
            print(f"Opening {path}")
            try:
                if sys.platform.startswith("win"):
                    os.startfile(path)
                elif sys.platform == "darwin":
                    subprocess.call(["open", path])
                else:
                    subprocess.call(["xdg-open", path])
            except Exception as e:
                print("Could not open:", e)
            continue

        
        hits = vectordb.similarity_search_with_score(q, k=10)
        hits = [(d, s) for d, s in hits if s >= 0.3]

        if not hits:
            print("No relevant docs.")
            continue

        last_paths = [d.metadata.get("path") for d, _ in hits]

        if llm:
            try:
                ans = generate_answer_with_llm(llm, q, hits[:6])
                if looks_like_gibberish(ans):
                    raise RuntimeError
                print("\n🧠", ans)
            except Exception:
                print("\nTop results:")
                for d, s in hits[:6]:
                    print("•", d.metadata.get("path"), f"(score {s:.2f})")
        else:
            for d, s in hits[:6]:
                print("•", d.metadata.get("path"), f"(score {s:.2f})")
