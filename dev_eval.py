from __future__ import annotations
import os
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import argparse
import csv
import sys
import time
from pathlib import Path

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('submission', type=str, help='path to submission.csv')
    p.add_argument('labels', type=str, help='path to labels.csv (image_name,gold_option)')
    args = p.parse_args()
    t0 = time.perf_counter()
    with Path(args.submission).open(newline='', encoding='utf-8') as f:
        sub = list(csv.DictReader(f))
    with Path(args.labels).open(newline='', encoding='utf-8') as f:
        gold = {r['image_name']: int(r['gold_option']) for r in csv.DictReader(f)}
    correct = wrong = abstain = halluc = 0
    score = 0.0
    for r in sub:
        opt_str = r['option']
        try:
            opt = int(opt_str)
        except ValueError:
            halluc += 1
            score += -1.0
            continue
        if opt not in {1, 2, 3, 4, 5}:
            halluc += 1
            score += -1.0
            continue
        if opt == 5:
            abstain += 1
            score += 0.0
            continue
        g = gold.get(r['image_name'])
        if g is None:
            wrong += 1
            score += -0.25
            continue
        if opt == g:
            correct += 1
            score += 1.0
        else:
            wrong += 1
            score += -0.25
    n = len(sub)
    abst_rate = abstain / n * 100.0 if n else 0.0
    elapsed = time.perf_counter() - t0
    print(f'dev_rubric_score=+{score:.2f} (correct={correct}, wrong={wrong}, abstain={abstain}, hallucinated={halluc}=0); abstention_rate={abst_rate:.1f}%; wall_clock={elapsed:.2f}s')
    return 0 if halluc == 0 else 2
if __name__ == '__main__':
    sys.exit(main())
