from __future__ import annotations
import math
import os
import re
from pathlib import Path
from typing import Any
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / 'models' / 'Qwen2.5-VL-7B-Instruct-AWQ'
COT_PROMPT = 'This image shows a multiple choice question about deep learning with four options labeled 1, 2, 3, and 4. Read the question carefully, briefly reason about which option is correct, and conclude with the chosen option number.'
ANSWER_PROMPT = 'Based on your reasoning above, output only the digit (1, 2, 3, or 4) of the correct option.'
_DIGIT_TOKEN_IDS: dict[int, int] | None = None


class Reasoner:

    def __init__(self, max_tokens: int = 128) -> None:
        self.max_tokens = max_tokens
        self.llm = self._load_model()
        self._warmup()

    def _load_model(self) -> LLM:
        if not MODEL_DIR.exists():
            raise FileNotFoundError(f'Model dir not found: {MODEL_DIR}. Run `python download_weights.py` during environment setup.')
        gpu_mem_util = float(os.environ.get('VLLM_GPU_MEM_UTIL', '0.5'))
        return LLM(model=str(MODEL_DIR), trust_remote_code=True, quantization='awq_marlin', dtype='bfloat16', max_model_len=16384, gpu_memory_utilization=gpu_mem_util, limit_mm_per_prompt={'image': 1}, mm_processor_kwargs={'max_pixels': 5120 * 28 * 28}, enforce_eager=True, disable_log_stats=True)

    def _warmup(self) -> None:
        dummy = Image.new('RGB', (32, 32), color='white')
        guided = GuidedDecodingParams(choice=['1', '2', '3', '4'])
        sp = SamplingParams(max_tokens=1, temperature=0.0, seed=0, logprobs=5, guided_decoding=guided)
        _ = self.llm.chat(messages=[{'role': 'user', 'content': [{'type': 'image_pil', 'image_pil': dummy}, {'type': 'text', 'text': '1'}]}], sampling_params=sp)

    def _digit_token_ids(self) -> dict[int, int]:
        global _DIGIT_TOKEN_IDS
        if _DIGIT_TOKEN_IDS is not None:
            return _DIGIT_TOKEN_IDS
        tok = self.llm.get_tokenizer()
        ids: dict[int, int] = {}
        for d in (1, 2, 3, 4):
            encoded = tok.encode(str(d), add_special_tokens=False)
            assert len(encoded) == 1, f'Qwen tokenizer expected single token for {d!r}, got {encoded!r}'
            ids[d] = encoded[0]
        _DIGIT_TOKEN_IDS = ids
        return ids

    def solve(self, image: Image.Image | None, image_name: str = '?') -> tuple[int, float]:
        if image is None:
            print(f'[{image_name}] image=None, abstaining', flush=True)
            return (5, 0.0)
        print(f'[{image_name}] solve start, image_size={image.size}', flush=True)
        try:
            cot_sp = SamplingParams(max_tokens=self.max_tokens, temperature=0.0, seed=0)
            cot_messages = [{'role': 'user', 'content': [{'type': 'image_pil', 'image_pil': image}, {'type': 'text', 'text': COT_PROMPT}]}]
            print(f'[{image_name}] pass1 (CoT) calling llm.chat', flush=True)
            cot_outputs = self.llm.chat(messages=cot_messages, sampling_params=cot_sp)
            reasoning = cot_outputs[0].outputs[0].text.strip()
            print(f'[{image_name}] pass1 reasoning_len={len(reasoning)} preview={reasoning[:120]!r}', flush=True)
            guided = GuidedDecodingParams(choice=['1', '2', '3', '4'])
            ans_sp = SamplingParams(max_tokens=1, temperature=0.0, seed=0, logprobs=5, guided_decoding=guided)
            ans_messages = cot_messages + [{'role': 'assistant', 'content': reasoning}, {'role': 'user', 'content': ANSWER_PROMPT}]
            print(f'[{image_name}] pass2 (answer) calling llm.chat', flush=True)
            outputs = self.llm.chat(messages=ans_messages, sampling_params=ans_sp)
            print(f'[{image_name}] llm.chat returned, parsing', flush=True)
        except BaseException as exc:
            import traceback
            print(f'[{image_name}] solve INNER EXCEPTION type={type(exc).__name__} msg={exc}', flush=True)
            traceback.print_exc()
            raise
        completion = outputs[0].outputs[0]
        raw_text = completion.text.strip()
        digits = re.findall(r'[1-4]', raw_text)
        option_int = int(digits[0]) if digits else 5
        margin = self._compute_margin(completion)
        print(f'[{image_name}] raw={raw_text!r} option={option_int} margin={margin:.3f}', flush=True)
        return (option_int, margin)

    def _compute_margin(self, completion: Any) -> float:
        if not completion.logprobs:
            return 0.0
        digit_ids = self._digit_token_ids()
        answer_slot = completion.logprobs[0]
        raw_lps: list[float] = []
        for d in (1, 2, 3, 4):
            entry = answer_slot.get(digit_ids[d])
            raw_lps.append(entry.logprob if entry is not None else float('-inf'))
        finite = [x for x in raw_lps if math.isfinite(x)]
        if not finite:
            return 0.0
        max_lp = max(finite)
        exps = [math.exp(x - max_lp) if math.isfinite(x) else 0.0 for x in raw_lps]
        denom = sum(exps)
        if denom == 0.0:
            return 0.0
        probs = sorted([e / denom for e in exps], reverse=True)
        return float(probs[0] - probs[1])
