from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
PARENT = Path(__file__).resolve().parent

def render_mcq(question: str, options: list[str], out_path: Path) -> None:
    fig = plt.figure(figsize=(10.24, 7.68), dpi=100)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
    ax.axis('off')
    body = 'Question:\n' + question + '\n\n'
    for i, opt in enumerate(options, start=1):
        body += f'{i}. {opt}\n'
    ax.text(0.0, 1.0, body, va='top', ha='left', wrap=True, fontsize=14, family='DejaVu Sans')
    fig.savefig(out_path, format='png', bbox_inches='tight')
    plt.close(fig)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=15)
    p.add_argument('--out', type=str, default=str(PARENT))
    args = p.parse_args()
    ds = load_dataset('cais/mmlu', 'machine_learning', split='test')
    rows: list[tuple[str, int]] = []
    out_dir = Path(args.out) / 'images'
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(args.count, len(ds))
    for i, row in enumerate(ds.select(range(n))):
        name = f'mmlu_ml_{i:03d}.png'
        gold_idx_0_based = int(row['answer'])
        gold_option = gold_idx_0_based + 1
        render_mcq(row['question'], list(row['choices']), out_dir / name)
        rows.append((name, gold_option))
    labels_path = Path(args.out) / 'labels.csv'
    file_exists = labels_path.exists() and labels_path.stat().st_size > 0
    mode = 'a' if file_exists else 'w'
    with labels_path.open(mode=mode, encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            w.writerow(['image_name', 'gold_option'])
        for name, gold in rows:
            w.writerow([name, gold])
    print(f'OK: {len(rows)} questions rendered to {out_dir}; labels at {labels_path}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
