from __future__ import annotations
import gc
import random
import re
import time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from .calibration_writer import CalibrationWriter
from .confidence import ConfidenceGate
from .data_loader import Row, load_test_csv
from .preprocess import open_image
from .reasoner import Reasoner
from .submission_writer import SubmissionWriter
CLEANUP_INTERVAL = 10
ABSTAIN = 5
DIGIT_RE = re.compile('[1-5]')
PER_IMAGE_TIMEOUT_S = 60.0
RUN_BUDGET_LIMIT_S = 50 * 60.0
FINALIZE_BUFFER_S = 60.0
EWMA_DECAY = 0.7

def _set_seeds() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower():
        return True
    return False

def run(test_dir: str | Path, *, tau: float | None=None, output_dir: str | Path | None=None, calibrate: bool=False) -> Path:
    test_dir = Path(test_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir is not None else Path.cwd().resolve()
    rows: list[Row] = list(load_test_csv(test_dir))
    image_names = [r.image_name for r in rows]
    n = len(rows)
    writer = SubmissionWriter(output_dir)
    writer.prefill(image_names)
    cal_writer: CalibrationWriter | None = CalibrationWriter(output_dir) if calibrate else None
    _set_seeds()
    t_load_start = time.time()
    reasoner = Reasoner()
    t_load = time.time() - t_load_start
    print(f'[load] model loaded in {t_load:.1f}s')
    gate = ConfidenceGate(tau=tau)
    pipeline_start = time.perf_counter()
    ewma_latency: float | None = None
    for i, row in enumerate(rows):
        t_solve_start = time.perf_counter()
        image = open_image(row.image_path)
        option_int: int = ABSTAIN
        margin: float = 0.0
        for attempt in (1, 2):
            try:
                option_int, margin = reasoner.solve(image, image_name=row.image_name)
                break
            except BaseException as exc:
                import traceback
                print(f'[{row.image_name}] EXCEPTION attempt={attempt} type={type(exc).__name__} msg={exc}', flush=True)
                traceback.print_exc()
                if not _is_oom(exc):
                    option_int, margin = (ABSTAIN, 0.0)
                    break
                torch.cuda.empty_cache()
                gc.collect()
                if attempt == 2:
                    option_int, margin = (ABSTAIN, 0.0)
        elapsed_to_solve = time.perf_counter() - t_solve_start
        if elapsed_to_solve > PER_IMAGE_TIMEOUT_S:
            print(f'[timeout] image={row.image_name} solve elapsed={elapsed_to_solve:.1f}s — abstain')
            option_int, margin = (ABSTAIN, 0.0)
        if cal_writer is not None:
            cal_writer.append(row.image_name, option_int, margin)
            gated = option_int
        else:
            gated = gate.gate(option_int, margin)
        raw = str(gated).strip()
        if not DIGIT_RE.fullmatch(raw):
            raw = '5'
        validated = writer.append(row.image_name, raw)
        elapsed_total = time.perf_counter() - t_solve_start
        print(f'[{i + 1}/{n}] image_name={row.image_name} duration={elapsed_total:.2f}s margin={margin:.2f} option={validated}')
        if (i + 1) % CLEANUP_INTERVAL == 0:
            torch.cuda.empty_cache()
            gc.collect()
        if ewma_latency is None:
            ewma_latency = elapsed_total
        else:
            ewma_latency = EWMA_DECAY * elapsed_total + (1 - EWMA_DECAY) * ewma_latency
        if i < 2:
            continue
        time_so_far = time.perf_counter() - pipeline_start
        remaining_rows = n - (i + 1)
        projected_total = time_so_far + ewma_latency * remaining_rows + FINALIZE_BUFFER_S
        if projected_total > RUN_BUDGET_LIMIT_S:
            print(f'[watchdog] projected_total={projected_total:.0f}s exceeds {RUN_BUDGET_LIMIT_S:.0f}s — mass-abstain on remaining {remaining_rows} rows')
            for j in range(i + 1, n):
                writer.append(rows[j].image_name, 5)
            break
    writer.finalize()
    return writer.final_path
