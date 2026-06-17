"""AWQ activation-aware quantization (4-bit), calibrated on retain text.

AWQ protects the most salient weight channels using activation statistics, so it can
preserve different weights than RTN/GPTQ. Whether unlearning recovery survives AWQ is
one of the open questions this harness answers.

Quantize with autoawq, save, then reload via transformers (which supports AWQ inference).
"""
from __future__ import annotations

import os
from typing import List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.tofu import QAExample, build_full


def quantize_and_load(model_dir, spec, calib, work_dir, device="cuda"):
    from awq import AutoAWQForCausalLM

    bits = spec.get("bits", 4)
    group_size = spec.get("group_size", 128)
    out_dir = os.path.join(work_dir, f"awq_{bits}bit")
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if not os.path.exists(os.path.join(out_dir, "config.json")):
        model = AutoAWQForCausalLM.from_pretrained(model_dir)
        quant_config = {"w_bit": bits, "q_group_size": group_size, "zero_point": True, "version": "GEMM"}
        calib_texts = [build_full(ex.question, ex.answer) for ex in (calib or [])[: spec.get("n_samples", 512)]]
        if len(calib_texts) < 16:
            raise RuntimeError("AWQ needs calibration text from the retain split (got <16).")
        model.quantize(tok, quant_config=quant_config, calib_data=calib_texts)
        model.save_quantized(out_dir)
        tok.save_pretrained(out_dir)

    model = AutoModelForCausalLM.from_pretrained(out_dir, device_map={"": 0})
    return model, tok
