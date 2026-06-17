"""GPTQ quantization (4-bit / 3-bit), calibrated on retain text.

Uses gptqmodel (the maintained successor to auto-gptq). We quantize to a work dir, then
reload through transformers AutoModelForCausalLM so the rest of the harness sees an
ordinary HF model (logits + generate) via HFScorer.

GPTQ is the realistic edge format prior unlearning work did *not* test, so this is core
to the contribution. API surface moves between gptqmodel versions; keep the call minimal.
"""
from __future__ import annotations

import os
from typing import List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.tofu import QAExample, build_full


def _calib_texts(calib: Optional[List[QAExample]], n: int) -> List[str]:
    calib = calib or []
    return [build_full(ex.question, ex.answer) for ex in calib[:n]]


def quantize_and_load(model_dir, spec, calib, work_dir, device="cuda"):
    from gptqmodel import GPTQModel, QuantizeConfig

    bits = spec.get("bits", 4)
    group_size = spec.get("group_size", 128)
    out_dir = os.path.join(work_dir, f"gptq_{bits}bit")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(os.path.join(out_dir, "config.json")):
        qcfg = QuantizeConfig(bits=bits, group_size=group_size)
        model = GPTQModel.load(model_dir, qcfg, trust_remote_code=True)
        texts = _calib_texts(calib, spec.get("n_samples", 512))
        if len(texts) < 16:
            raise RuntimeError("GPTQ needs calibration text from the retain split (got <16).")
        model.quantize(texts)
        model.save(out_dir)
        # tokenizer copied alongside for downstream loading
        AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True).save_pretrained(out_dir)

    model = AutoModelForCausalLM.from_pretrained(out_dir, device_map={"": 0}, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(out_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok
