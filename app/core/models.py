import torch
from packaging.version import parse as version_parse
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM,
)
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import *
from app.core.device import DEVICE


def load_caption_model():
    caption_proc = BlipProcessor.from_pretrained(CAPTION_MODEL, local_files_only=True)
    try:
        caption_model = BlipForConditionalGeneration.from_pretrained(
            CAPTION_MODEL,
            torch_dtype=torch.float16 if DEVICE.type=="cuda" else torch.float32,
            use_safetensors=True,
            local_files_only=True,
        ).to(DEVICE).eval()
    except (FileNotFoundError, ValueError, Exception):
        caption_model = BlipForConditionalGeneration.from_pretrained(
            CAPTION_MODEL,
            torch_dtype=torch.float16 if DEVICE.type=="cuda" else torch.float32,
            use_safetensors=False,
            local_files_only=True,
        ).to(DEVICE).eval()

    return caption_proc, caption_model


def load_summarizer():
    sum_tok = AutoTokenizer.from_pretrained(SUMMARIZER, local_files_only=True)
    sum_mod = AutoModelForSeq2SeqLM.from_pretrained(
        SUMMARIZER,
        torch_dtype=torch.float16 if DEVICE.type=="cuda" else torch.float32,
        local_files_only=True,
    ).to(DEVICE).eval()

    return sum_tok, sum_mod

def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": DEVICE, "local_files_only": True},
    )
