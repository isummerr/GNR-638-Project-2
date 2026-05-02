from __future__ import annotations
import csv
from pathlib import Path
CALIBRATION_COLUMNS: tuple[str, ...] = ('image_name', 'raw_option', 'margin')

def write_calibration_row(path: Path, image_name: str, raw_option: int, margin: float) -> None:
    with Path(path).open(mode='a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([str(image_name), int(raw_option), f'{float(margin):.6f}'])

class CalibrationWriter:

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / 'dev_margins.csv'
        with self.path.open(mode='w', encoding='utf-8', newline='') as f:
            csv.writer(f, lineterminator='\n', quoting=csv.QUOTE_MINIMAL).writerow(CALIBRATION_COLUMNS)

    def append(self, image_name: str, raw_option: int, margin: float) -> None:
        write_calibration_row(self.path, image_name, raw_option, margin)
