import re
import torch
from typing import List

from app.config import *
from app.core.device import DEVICE
from app.core.models import load_summarizer

WORD_RE = re.compile(r"\b[a-z]{3,}\b")

sum_tok, sum_mod = load_summarizer()


def clean_words(text: str) -> List[str]:
    uniq: List[str] = []
    for w in WORD_RE.findall((text or "").lower()):
        if w not in uniq:
            uniq.append(w)
        if len(uniq) >= TAG_MAX:
            break
    return uniq


def chunk_text(text: str) -> List[str]:
    if not text:
        return []
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [text]
    return [
        " ".join(words[i:i+CHUNK_WORDS])
        for i in range(0, len(words), CHUNK_WORDS-CHUNK_OVERLAP)
    ]


@torch.inference_mode()
def summarize(text: str, max_len: int = 160) -> str:
    if not text:
        return ""
    if len(text.split()) < 60:
        return text[:400]

    inp = sum_tok(
        text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)

    out = sum_mod.generate(
        **inp,
        max_length=max_len,
        min_length=40
    )

    return sum_tok.decode(out[0], skip_special_tokens=True)
