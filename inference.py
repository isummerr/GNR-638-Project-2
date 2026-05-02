from __future__ import annotations
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ.setdefault('VLLM_DO_NOT_TRACK', '1')
os.environ.setdefault('VLLM_LOGGING_LEVEL', 'WARNING')
import argparse
import sys
from pathlib import Path

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='inference.py', description='Run the offline VLM pipeline on a test directory and write submission.csv to the project directory (cwd).')
    p.add_argument('--test_dir', required=True, type=str, help='Absolute path to the directory containing test.csv and images/.')
    return p

def main(argv: list[str] | None=None) -> int:
    args = _build_parser().parse_args(argv)
    test_dir = Path(args.test_dir).resolve()
    if not test_dir.exists():
        print(f'error: --test_dir does not exist: {test_dir}', file=sys.stderr)
        return 2
    if not (test_dir / 'test.csv').exists():
        print(f"error: missing test.csv: {test_dir / 'test.csv'}", file=sys.stderr)
        return 2
    from src.solver.pipeline import run
    output_dir = Path.cwd().resolve()
    out = run(test_dir, output_dir=output_dir)
    print(f'OK: {out}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
