import json
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import SUPPORTED_TEXT, SUPPORTED_IMG
from app.core.file_extractors import extract_text
from app.core.text_utils import chunk_text, summarize, clean_words
from app.core.image_caption import caption_images

# Smart Index Updater (To Update the Existing Index efficiently by Updating the modified files Index only)
class DeltaIndexer:

    def __init__(
        self,
        folder: str,
        name: str,
        embeddings=None,
        db_dir: Path = None,
        rows: List[Dict[str, Any]] = None,
    ):
        self.folder = folder
        self.name = name
        self.embeddings = embeddings
        self.db_dir = db_dir
        self.rows = rows or []

        
        try:
            self.store = FAISS.load_local(
                str(db_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            docs = [
                Document(
                    page_content=r["content"],
                    metadata={k: r[k] for k in r if k != "content"},
                )
                for r in self.rows
            ]
            self.store = (
                FAISS.from_documents(docs, embeddings)
                if docs
                else FAISS.from_documents([], embeddings)
            )


    # Index / Re-index single path
    def _index_path(self, path: Path):
        ext = path.suffix.lower().lstrip(".")
        if ext not in SUPPORTED_TEXT and ext not in SUPPORTED_IMG:
            return

        new = []

        try:
            size = path.stat().st_size
            mtime = int(path.stat().st_mtime)
        except Exception:
            size = 0
            mtime = 0

        if ext in SUPPORTED_TEXT:
            raw = extract_text(path)[:30000]
            for cid, chunk in enumerate(chunk_text(raw)):
                new.append(
                    {
                        "path": str(path),
                        "type": ext,
                        "chunk_id": cid,
                        "content": chunk,
                        "summary": summarize(chunk),
                        "tags": clean_words(chunk) or clean_words(path.stem),
                        "size": size,
                        "mtime": mtime,
                    }
                )
        else:
            cap = caption_images([path])[0]
            new.append(
                {
                    "path": str(path),
                    "type": ext,
                    "chunk_id": 0,
                    "content": cap,
                    "summary": cap,
                    "tags": clean_words(cap),
                    "size": size,
                    "mtime": mtime,
                }
            )

        docs = [
            Document(
                page_content=r["content"],
                metadata={k: r[k] for k in r if k != "content"},
            )
            for r in new
        ]

        if docs:
            self.store.add_documents(docs)

       
        self.rows = [r for r in self.rows if r["path"] != str(path)] + new

        Path("file_indexes").mkdir(exist_ok=True)
        Path("file_indexes", f"{self.name}.json").write_text(
            json.dumps(self.rows, indent=2)
        )

        out = Path(self.db_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.store.save_local(str(out))


    def _remove_path(self, path: Path):
        p = str(path)

        self.rows = [r for r in self.rows if r["path"] != p]

        docs = [
            Document(
                page_content=r["content"],
                metadata={k: r[k] for k in r if k != "content"},
            )
            for r in self.rows
        ]

        self.store = (
            FAISS.from_documents(docs, self.embeddings)
            if docs
            else FAISS.from_documents([], self.embeddings)
        )

        Path("file_indexes").mkdir(exist_ok=True)
        Path("file_indexes", f"{self.name}.json").write_text(
            json.dumps(self.rows, indent=2)
        )

        out = Path(self.db_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.store.save_local(str(out))
