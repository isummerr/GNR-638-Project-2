from __future__ import annotations
import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable
TAU_GRID: tuple[float, ...] = tuple((round(0.1 + 0.025 * i, 3) for i in range(25)))
ABSTENTION_WINDOW: tuple[float, float] = (5.0, 60.0)
ABSTAIN: int = 5
DEFAULT_TAU_RE = re.compile('^DEFAULT_TAU\\s*=\\s*[\\d.]+', re.MULTILINE)
CONFIDENCE_PATH = Path(__file__).resolve().parent / 'src' / 'solver' / 'confidence.py'

def gate(opt: int, margin: float, tau: float) -> int:
    if opt == ABSTAIN:
        return ABSTAIN
    if opt not in (1, 2, 3, 4):
        return ABSTAIN
    if margin >= tau:
        return opt
    return ABSTAIN

def score_one_tau(rows: list[tuple[str, int, float]], golds: dict[str, int], tau: float) -> dict:
    correct = wrong = abstain = halluc = 0
    score = 0.0
    for name, raw_opt, margin in rows:
        gated = gate(raw_opt, margin, tau)
        if gated == ABSTAIN:
            abstain += 1
            score += 0.0
            continue
        if gated not in (1, 2, 3, 4):
            halluc += 1
            score += -1.0
            continue
        g = golds.get(name)
        if g is None or gated != g:
            wrong += 1
            score += -0.25
        else:
            correct += 1
            score += 1.0
    n = len(rows)
    abst_rate = abstain / n * 100.0 if n else 0.0
    return {'tau': tau, 'score': round(score, 4), 'correct': correct, 'wrong': wrong, 'abstain': abstain, 'halluc': halluc, 'abstention_rate': round(abst_rate, 2)}

def select(results: list[dict]) -> tuple[dict, bool]:
    in_window = [r for r in results if ABSTENTION_WINDOW[0] <= r['abstention_rate'] <= ABSTENTION_WINDOW[1]]
    if in_window:
        chosen = max(in_window, key=lambda r: (r['score'], r['tau']))
        return (chosen, False)

    def distance(r: dict) -> float:
        a = r['abstention_rate']
        return max(0.0, ABSTENTION_WINDOW[0] - a, a - ABSTENTION_WINDOW[1])
    chosen = min(results, key=lambda r: (distance(r), -r['score']))
    return (chosen, True)

def _update_default_tau(path: Path, chosen_tau: float) -> int:
    text = path.read_text(encoding='utf-8')
    new_text, n = DEFAULT_TAU_RE.subn(f'DEFAULT_TAU = {chosen_tau}', text, count=1)
    if n == 1:
        path.write_text(new_text, encoding='utf-8')
    return n

def _read_margins(path: Path) -> list[tuple[str, int, float]]:
    with path.open(newline='', encoding='utf-8') as f:
        return [(r['image_name'], int(r['raw_option']), float(r['margin'])) for r in csv.DictReader(f)]

def _read_labels(path: Path) -> dict[str, int]:
    with path.open(newline='', encoding='utf-8') as f:
        return {r['image_name']: int(r['gold_option']) for r in csv.DictReader(f)}

def _write_tau_sweep_csv(out_path: Path, results: list[dict], chosen_tau: float, out_of_window: bool) -> None:
    with out_path.open(mode='w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
        w.writerow(['tau', 'score', 'correct', 'wrong', 'abstain', 'halluc', 'abstention_rate', 'out_of_window'])
        for r in results:
            flag = 'true' if out_of_window and r['tau'] == chosen_tau else 'false'
            w.writerow([r['tau'], r['score'], r['correct'], r['wrong'], r['abstain'], r['halluc'], r['abstention_rate'], flag])

def main(argv: list[str] | None=None) -> int:
    p = argparse.ArgumentParser(description='τ-sweep over dev_margins.csv + labels.csv; pure-Python rubric.')
    p.add_argument('--margins', required=True, type=str, help='path to dev_margins.csv (image_name,raw_option,margin)')
    p.add_argument('--labels', required=True, type=str, help='path to labels.csv (image_name,gold_option)')
    p.add_argument('--out-csv', default='tau_sweep.csv', type=str, help='output path for the per-threshold report (default: ./tau_sweep.csv)')
    p.add_argument('--update-tau', action='store_true', help='after writing tau_sweep.csv, edit src/solver/confidence.py to set DEFAULT_TAU = <chosen_tau> in place.')
    args = p.parse_args(argv)
    margins_path = Path(args.margins)
    labels_path = Path(args.labels)
    if not margins_path.exists():
        print(f'error: margins file missing: {margins_path}', file=sys.stderr)
        return 2
    if not labels_path.exists():
        print(f'error: labels file missing: {labels_path}', file=sys.stderr)
        return 2
    rows = _read_margins(margins_path)
    golds = _read_labels(labels_path)
    results = [score_one_tau(rows, golds, tau) for tau in TAU_GRID]
    chosen, out_of_window = select(results)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_tau_sweep_csv(out_path, results, chosen['tau'], out_of_window)
    if args.update_tau:
        n = _update_default_tau(CONFIDENCE_PATH, chosen['tau'])
        if n != 1:
            print(f'error: could not update DEFAULT_TAU in {CONFIDENCE_PATH} (regex matched {n} times — expected exactly 1)', file=sys.stderr)
            return 2
        print(f"updated DEFAULT_TAU = {chosen['tau']} in {CONFIDENCE_PATH}")
    flag = ' (out_of_window)' if out_of_window else ''
    print(f"chosen tau={chosen['tau']} score={chosen['score']} abst={chosen['abstention_rate']:.1f}%{flag}")
    return 0
if __name__ == '__main__':
    sys.exit(main())
