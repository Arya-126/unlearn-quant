"""GGUF k-quant path via llama.cpp — the true edge-deployment format.

Steps:
  1. convert the unlearned HF model to a f16 GGUF (llama.cpp convert_hf_to_gguf.py)
  2. quantize to a k-quant (Q4_K_M / Q4_0 / Q3_K_M) with the llama-quantize binary
  3. score via llama-cpp-python, exposing the same Scorer contract as HFScorer

Set LLAMACPP_DIR to a built llama.cpp checkout (containing convert_hf_to_gguf.py and
build/bin/llama-quantize). This is the heaviest dependency; the dispatcher isolates it so
a failure here doesn't block the GPTQ/AWQ/bnb conditions.
"""
from __future__ import annotations

import os
import subprocess
import sys

from ..data.tofu import build_prompt


def _llamacpp_dir() -> str:
    d = os.environ.get("LLAMACPP_DIR")
    if not d or not os.path.isdir(d):
        raise RuntimeError("Set LLAMACPP_DIR to a built llama.cpp checkout to use the GGUF path.")
    return d


def _quantize_bin(d: str) -> str:
    for cand in ("build/bin/llama-quantize", "build/bin/quantize", "llama-quantize"):
        p = os.path.join(d, cand)
        if os.path.exists(p):
            return p
    raise RuntimeError("llama-quantize binary not found under LLAMACPP_DIR (build llama.cpp first).")


def _convert_to_f16(model_dir: str, out_gguf: str, llamacpp: str):
    script = os.path.join(llamacpp, "convert_hf_to_gguf.py")
    subprocess.run(
        [sys.executable, script, model_dir, "--outfile", out_gguf, "--outtype", "f16"],
        check=True,
    )


def _quantize(f16_gguf: str, out_gguf: str, ftype: str, llamacpp: str):
    subprocess.run([_quantize_bin(llamacpp), f16_gguf, out_gguf, ftype], check=True)


def quantize_and_scorer(model_dir, spec, work_dir):
    llamacpp = _llamacpp_dir()
    os.makedirs(work_dir, exist_ok=True)
    ftype = spec["ftype"]

    f16 = os.path.join(work_dir, "model-f16.gguf")
    if not os.path.exists(f16):
        _convert_to_f16(model_dir, f16, llamacpp)

    out_gguf = os.path.join(work_dir, f"model-{ftype}.gguf")
    if not os.path.exists(out_gguf):
        _quantize(f16, out_gguf, ftype, llamacpp)

    return GGUFScorer(out_gguf)


class GGUFScorer:
    """Scorer backed by a quantized GGUF model loaded through llama-cpp-python."""

    def __init__(self, gguf_path: str, n_ctx: int = 512):
        from llama_cpp import Llama

        self.llm = Llama(model_path=gguf_path, n_ctx=n_ctx, logits_all=True, verbose=False)

    def answer_logprob(self, question: str, answer: str):
        prompt = build_prompt(question)
        full = prompt + answer
        res = self.llm.create_completion(full, max_tokens=0, echo=True, logprobs=0, temperature=0.0)
        lp = res["choices"][0]["logprobs"]
        offsets = lp["text_offset"]
        token_logprobs = lp["token_logprobs"]
        prompt_len = len(prompt)
        total, n = 0.0, 0
        for off, tlp in zip(offsets, token_logprobs):
            if tlp is None:
                continue
            if off >= prompt_len:          # token belongs to the answer
                total += float(tlp)
                n += 1
        return total, n

    def generate(self, question: str, max_new_tokens: int = 64) -> str:
        prompt = build_prompt(question)
        res = self.llm.create_completion(prompt, max_tokens=max_new_tokens, temperature=0.0)
        return res["choices"][0]["text"].strip()
