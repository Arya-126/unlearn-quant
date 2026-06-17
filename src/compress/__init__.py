"""Compression dispatcher.

Given an unlearned model dir and a spec from configs/compress.yaml, return a Scorer.
Each backend is imported lazily so a missing/broken dependency (e.g. llama.cpp) only
disables its own conditions instead of crashing the whole sweep.
"""
from __future__ import annotations

from typing import List, Optional

from ..data.tofu import QAExample
from ..eval.scorers import HFScorer


def load_compressed(
    model_dir: str,
    spec: dict,
    device: str = "cuda",
    calib: Optional[List[QAExample]] = None,
    work_dir: str = "./.quant",
    prompt_fn=None,
    fp16_device_map=None,
):
    """Return a Scorer for ``model_dir`` compressed according to ``spec``.

    spec['backend'] in {none, bnb, gptq, awq, gguf}. ``prompt_fn`` overrides the prompt
    template (needed for Llama-2 chat models in the 7B reproduction).
    """
    backend = spec.get("backend", "none")

    if backend in ("none", None):
        from .bnb import load_fp16
        model, tok = load_fp16(model_dir, device, device_map=fp16_device_map)
        return HFScorer(model, tok, device, prompt_fn=prompt_fn)

    if backend == "bnb":
        from .bnb import load_bnb
        model, tok = load_bnb(model_dir, spec, device)
        return HFScorer(model, tok, device, prompt_fn=prompt_fn)

    if backend == "gptq":
        from .gptq import quantize_and_load
        model, tok = quantize_and_load(model_dir, spec, calib, work_dir, device)
        return HFScorer(model, tok, device, prompt_fn=prompt_fn)

    if backend == "awq":
        from .awq import quantize_and_load
        model, tok = quantize_and_load(model_dir, spec, calib, work_dir, device)
        return HFScorer(model, tok, device, prompt_fn=prompt_fn)

    if backend == "gguf":
        from .gguf import quantize_and_scorer
        return quantize_and_scorer(model_dir, spec, work_dir)

    raise ValueError(f"unknown backend {backend}")
