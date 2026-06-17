"""Shared model/config loading.

The `locuslab/tofu_ft_phi-1.5` config predates transformers' native PhiConfig and is
missing `pad_token_id`. Native `PhiModel.__init__` (transformers >=4.45) reads
`config.pad_token_id` unconditionally for the embedding padding_idx, so loading raises
AttributeError. We patch the config (pad_token_id <- eos_token_id) before instantiating.
Centralized here so every backend (trainer, eval, bnb, distill) loads consistently.
"""
from __future__ import annotations

from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# The locuslab/tofu_ft_phi-1.5 repo ships a broken GPT2Tokenizer that encodes text to
# ZERO tokens under current transformers. Its weights are plain Phi-1.5 (identical vocab),
# so we fall back to the base tokenizer when the bundled one is broken.
_FALLBACK_TOKENIZER = "microsoft/phi-1_5"
_PROBE = "Question: test?\nAnswer: ok"


def load_tokenizer(name: str, cache_dir: Optional[str] = None):
    """Load a tokenizer, falling back to microsoft/phi-1_5 if it encodes to empty."""
    tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    try:
        n = len(tok(_PROBE)["input_ids"])
    except Exception:
        n = 0
    if n == 0:
        print(f"[modeling] tokenizer at {name} encodes empty; falling back to {_FALLBACK_TOKENIZER}")
        tok = AutoTokenizer.from_pretrained(_FALLBACK_TOKENIZER, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


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
