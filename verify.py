from __future__ import annotations
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ.setdefault('VLLM_DO_NOT_TRACK', '1')
os.environ.setdefault('VLLM_LOGGING_LEVEL', 'WARNING')
import csv
import sys
from pathlib import Path
EXPECTED_COLUMNS = ('id', 'image_name', 'option')
ALLOWED = {'1', '2', '3', '4', '5'}

def verify_csv(submission_path: Path, test_csv_path: Path) -> tuple[bool, str]:
    if not submission_path.exists():
        return (False, f'missing submission.csv at {submission_path}')
    raw = submission_path.read_bytes()
    if raw.startswith(b'\xef\xbb\xbf'):
        return (False, 'BOM present')
    if b'\r\n' in raw:
        return (False, 'CRLF line endings detected')
    if not raw.endswith(b'\n'):
        return (False, 'missing trailing newline')
    with submission_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            return (False, f'columns {reader.fieldnames!r} != {EXPECTED_COLUMNS!r}')
        sub_rows = list(reader)
    for i, r in enumerate(sub_rows, start=1):
        if r['id'] != r['image_name']:
            return (False, f"row {i}: id={r['id']!r} != image_name={r['image_name']!r}")
        if r['option'] not in ALLOWED:
            return (False, f"row {i}: option={r['option']!r} not in {ALLOWED}")
    with test_csv_path.open(newline='', encoding='utf-8') as f:
        test_reader = csv.DictReader(f)
        test_n = sum((1 for _ in test_reader))
    if len(sub_rows) != test_n:
        return (False, f'submission row count {len(sub_rows)} != test.csv {test_n}')
    return (True, f'OK ({len(sub_rows)} rows)')

def main() -> int:
    if len(sys.argv) != 2:
        print('usage: python verify.py <parent_dir>', file=sys.stderr)
        return 2
    parent_dir = Path(sys.argv[1]).resolve()
    test_csv = parent_dir / 'test.csv'
    submission = parent_dir / 'submission.csv'
    if not test_csv.exists():
        print(f'error: missing {test_csv}', file=sys.stderr)
        return 2
    sys.path.insert(0, str(Path(__file__).parent))
    from src.solver.pipeline import run
    run(parent_dir)
    ok, msg = verify_csv(submission, test_csv)
    print(msg)
    return 0 if ok else 1
if __name__ == '__main__':
    sys.exit(main())
