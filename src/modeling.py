"""Shared model/config loading.

The `locuslab/tofu_ft_phi-1.5` config predates transformers' native PhiConfig and is
missing `pad_token_id`. Native `PhiModel.__init__` (transformers >=4.45) reads
`config.pad_token_id` unconditionally for the embedding padding_idx, so loading raises
AttributeError. We patch the config (pad_token_id <- eos_token_id) before instantiating.
Centralized here so every backend (trainer, eval, bnb, distill) loads consistently.
"""
from __future__ import annotations

from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM


def load_config(name: str, cache_dir: Optional[str] = None):
    cfg = AutoConfig.from_pretrained(name, cache_dir=cache_dir)
    if getattr(cfg, "pad_token_id", None) is None:
        eos = getattr(cfg, "eos_token_id", None)
        cfg.pad_token_id = eos if isinstance(eos, int) else 0
    return cfg


def load_causal_lm(name: str, cache_dir: Optional[str] = None, **kwargs):
    """AutoModelForCausalLM.from_pretrained with the pad_token_id patch applied."""
    cfg = load_config(name, cache_dir)
    return AutoModelForCausalLM.from_pretrained(name, config=cfg, cache_dir=cache_dir, **kwargs)
