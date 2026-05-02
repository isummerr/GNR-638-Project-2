from __future__ import annotations
from pathlib import Path
from PIL import Image, UnidentifiedImageError

def open_image(path: Path | None) -> Image.Image | None:
    if path is None:
        return None
    try:
        with Image.open(path) as im:
            return im.convert('RGB')
    except (UnidentifiedImageError, OSError, ValueError):
        return None
