import os
import json
from pathlib import Path
from typing import List, Dict, Union

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import (
    SUPPORTED_TEXT,
    SUPPORTED_IMG,
    JUNK_EXTS,
)
from app.core.file_extractors import extract_text
from app.core.text_utils import chunk_text, summarize, clean_words
from app.core.image_caption import caption_images
from app.core.models import load_embeddings

#Index Builder Pipeline Module
def build_index(folders: Union[str, List[str]], name: str):
    if isinstance(folders, str):
        folders = [folders]
    
    rows: List[Dict] = []
    img_files: List[Path] = []

    for folder_path in folders:
        folder = Path(folder_path)
        if not folder.exists():
            print(f"⚠️ Warning: Folder not found, skipping: {folder}")
            continue

        for root, _, files in os.walk(folder):
            for fn in files:
                ext = Path(fn).suffix.lower().lstrip(".")
                if ext in JUNK_EXTS or (
                    ext not in SUPPORTED_TEXT and ext not in SUPPORTED_IMG
                ):
                    continue

                full = Path(root) / fn
                print(f"-> Indexing: {full.name}")
                
                try:
                    size = full.stat().st_size
                    mtime = int(full.stat().st_mtime)
                except Exception:
                    size = 0
                    mtime = 0

                entry_base = {
                    "path": str(full),
                    "type": ext,
                    "tags": [],
                    "size": size,
                    "mtime": mtime,
                }

                if ext in SUPPORTED_TEXT:
                    raw = extract_text(full)[:30000]
                    if not raw.strip():
                        continue

                    for chunk_id, chunk in enumerate(chunk_text(raw)):
                        rows.append(
                            {
                                **entry_base,
                                "summary": summarize(chunk),
                                "content": chunk,
                                "tags": clean_words(chunk) or clean_words(fn),
                                "chunk_id": chunk_id,
                            }
                        )
                else:
                    img_files.append(full)
                    rows.append(
                        {
                            **entry_base,
                            "tags": ["__IMG__"],
                            "chunk_id": 0,
                            "content": "",
                        }
                    )

    
    # Caption images
    if img_files:
        print(f"📸 Captioning {len(img_files)} images …")
        caps = caption_images(img_files)
        img_rows = [r for r in rows if r["type"] in SUPPORTED_IMG]

        for r, cap in zip(img_rows, caps):
            r["summary"] = cap
            r["content"] = cap
            r["tags"] = clean_words(cap) or clean_words(Path(r["path"]).stem)

    if not rows:
        raise RuntimeError("No supported files found.")

    docs = [
        Document(
            page_content=r["content"],
            metadata={
                k: r[k]
                for k in (
                    "path",
                    "type",
                    "tags",
                    "summary",
                    "chunk_id",
                    "size",
                    "mtime",
                )
            },
        )
        for r in rows
    ]

    print("Intialize - Embedding & writing FAISS")
    embeddings = load_embeddings()
    store = FAISS.from_documents(docs, embeddings)

    out_dir = Path("vector_dbs") / f"{name}_bge_db"
    out_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out_dir))

    Path("file_indexes").mkdir(exist_ok=True)
    Path("file_indexes", f"{name}.json").write_text(
        json.dumps(rows, indent=2)
    )

    print(f"✅ Index built with {len(rows)} docs → {out_dir}")
