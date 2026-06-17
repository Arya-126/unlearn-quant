"""bitsandbytes round-to-nearest quantization (int8 / NF4 4-bit) + fp16 reference.

This is the path the ICLR 2025 paper used; it anchors our replication.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _tok(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_fp16(model_dir, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
    return model, _tok(model_dir)


def load_bnb(model_dir, spec, device="cuda"):
    from transformers import BitsAndBytesConfig

    bits = spec.get("bits", 4)
    if bits == 8:
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    elif bits == 4:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4" if spec.get("nf4", True) else "fp4",
            bnb_4bit_use_double_quant=spec.get("double_quant", True),
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        raise ValueError(f"bnb supports 4 or 8 bits, got {bits}")

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, quantization_config=bnb_cfg, device_map={"": 0}, trust_remote_code=True
    )
    return model, _tok(model_dir)
