from __future__ import annotations
import csv
import os
from pathlib import Path
from typing import Iterable, Sequence
from .output_validator import ABSTAIN, assert_valid, coerce_option
SUBMISSION_COLUMNS: tuple[str, ...] = ('id', 'image_name', 'option')

class SubmissionWriter:

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.final_path = self.output_dir / 'submission.csv'
        self.partial_path = self.output_dir / 'submission.partial.csv'
        if self.partial_path.exists():
            self.partial_path.unlink()

    def prefill(self, image_names: Sequence[str]) -> None:
        with self.final_path.open(mode='w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(SUBMISSION_COLUMNS)
            for name in image_names:
                row = self._row(name, ABSTAIN)
                writer.writerow(row)
        with self.partial_path.open(mode='w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(SUBMISSION_COLUMNS)

    def append(self, image_name: str, option_raw) -> int:
        option = coerce_option(option_raw)
        assert_valid(option, image_name=image_name)
        row = self._row(image_name, option)
        with self.partial_path.open(mode='a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        return option

    def finalize(self) -> None:
        os.replace(self.partial_path, self.final_path)

    def _row(self, image_name: str, option: int) -> list:
        if SUBMISSION_COLUMNS == ('id', 'image_name', 'option'):
            return [image_name, image_name, int(option)]
        if SUBMISSION_COLUMNS == ('image_name', 'option'):
            return [image_name, int(option)]
        raise AssertionError(f'unsupported SUBMISSION_COLUMNS: {SUBMISSION_COLUMNS!r}')
