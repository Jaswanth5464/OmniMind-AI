from pathlib import Path
from typing import List
from PIL import Image
import torch

from app.core.device import DEVICE
from app.core.models import load_caption_model

caption_proc, caption_model = load_caption_model()


@torch.inference_mode()
def caption_images(paths: List[Path], batch: int = 6) -> List[str]:
    caps: List[str] = []

    for i in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i+batch]]
        inp = caption_proc(
            images=imgs,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)

        out = caption_model.generate(**inp, max_new_tokens=20)
        caps += caption_proc.batch_decode(out, skip_special_tokens=True)

    return [c.lower() for c in caps]
