from __future__ import annotations
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
QWEN_REPO = 'Qwen/Qwen2.5-VL-7B-Instruct-AWQ'
QWEN_REVISION = 'main'
PROJECT_ROOT = Path(__file__).parent.resolve()
MODEL_DIR = PROJECT_ROOT / 'models' / 'Qwen2.5-VL-7B-Instruct-AWQ'
SKIP_PATTERNS = ['*.md', '*.txt', 'README*', '*.png', '*.jpg', '*.jpeg', 'LICENSE*', '.gitattributes']

def main() -> int:
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {QWEN_REPO}@{QWEN_REVISION} -> {MODEL_DIR}')
    print('This is a one-time setup step; internet is REQUIRED.')
    snapshot_download(repo_id=QWEN_REPO, revision=QWEN_REVISION, local_dir=str(MODEL_DIR), local_dir_use_symlinks=False, ignore_patterns=SKIP_PATTERNS, etag_timeout=30, max_workers=4)
    required = ['config.json', 'tokenizer_config.json', 'preprocessor_config.json']
    missing = [f for f in required if not (MODEL_DIR / f).exists()]
    if missing:
        print(f'ERROR: missing required files after download: {missing}', file=sys.stderr)
        return 1
    shards = list(MODEL_DIR.glob('*.safetensors'))
    if not shards:
        print(f'ERROR: no .safetensors shards found in {MODEL_DIR}', file=sys.stderr)
        return 1
    total_gb = sum((f.stat().st_size for f in MODEL_DIR.rglob('*'))) / 1000000000.0
    print(f'OK: {len(shards)} safetensors shards, total {total_gb:.1f} GB')
    return 0
if __name__ == '__main__':
    sys.exit(main())
