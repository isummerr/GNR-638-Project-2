from __future__ import annotations
from pathlib import Path
from typing import Iterator, NamedTuple
import pandas as pd
IMAGE_FOLDER_CANDIDATES = ('image', 'images', 'Images', 'input', 'inputs')

class Row(NamedTuple):
    image_name: str
    image_id: str
    image_path: Path | None

def resolve_image_dir(parent_dir: Path) -> Path | None:
    for name in IMAGE_FOLDER_CANDIDATES:
        candidate = parent_dir / name
        if candidate.is_dir():
            return candidate
    return None

def resolve_image_path(image_dir: Path | None, image_name: str) -> Path | None:
    if image_dir is None:
        return None
    name = image_name.strip()
    literal = image_dir / name
    if literal.is_file():
        return literal
    with_png = image_dir / f'{name}.png'
    if with_png.is_file():
        return with_png
    return None

def load_test_csv(parent_dir: Path) -> Iterator[Row]:
    parent_dir = Path(parent_dir).resolve()
    csv_path = parent_dir / 'test.csv'
    df = pd.read_csv(csv_path, dtype=str)
    if 'image_name' not in df.columns:
        raise ValueError(f"test.csv missing required column 'image_name': {list(df.columns)}")
    if 'image_id' not in df.columns:
        df['image_id'] = df['image_name']
    image_dir = resolve_image_dir(parent_dir)
    for _, r in df.iterrows():
        name = str(r['image_name']).strip()
        rid = str(r['image_id']).strip()
        path = resolve_image_path(image_dir, name)
        yield Row(image_name=name, image_id=rid, image_path=path)
