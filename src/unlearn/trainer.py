"""T4-safe unlearning trainer.

A manual loop (not HF Trainer) because the losses use paired forget/retain batches and,
for NPO, a frozen reference model. Memory tricks to fit Phi-1.5 full-parameter FT on a
16GB T4: bf16 weights, gradient checkpointing, 8-bit Adam (bitsandbytes), grad accum.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from ..data.tofu import ForgetRetainDataset, PairCollator, QAExample
from . import methods


@dataclass
class UnlearnConfig:
    method_type: str           # grad_ascent | grad_diff | npo
    lr: float = 1e-5
    epochs: int = 5
    batch_size: int = 4
    grad_accum: int = 4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    beta: float = 0.1          # NPO
    retain_weight: float = 1.0
    use_8bit_adam: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
    max_length: int = 256


def _load_model(name, dtype, cache_dir, gradient_checkpointing):
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, cache_dir=cache_dir, trust_remote_code=True
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model


def _make_optimizer(model, lr, weight_decay, use_8bit_adam):
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=weight_decay)
        except Exception as e:  # noqa: BLE001 - fall back gracefully on non-CUDA / no-bnb
            print(f"[trainer] 8-bit Adam unavailable ({e}); using fp32 AdamW.")
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def unlearn(
    model_name: str,
    forget: list[QAExample],
    retain: list[QAExample],
    cfg: UnlearnConfig,
    out_dir: str,
    dtype=torch.bfloat16,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
):
    """Run unlearning and save the resulting model + tokenizer to ``out_dir``."""
    torch.manual_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model(model_name, dtype, cache_dir, cfg.gradient_checkpointing).to(device)
    model.train()

    ref_model = None
    if methods.needs_reference(cfg.method_type):
        ref_model = _load_model(model_name, dtype, cache_dir, False).to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    ds = ForgetRetainDataset(forget, retain, tokenizer, cfg.max_length, cfg.seed)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=PairCollator(tokenizer))

    steps_per_epoch = max(1, len(dl) // cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    opt = _make_optimizer(model, cfg.lr, cfg.weight_decay, cfg.use_8bit_adam)
    sched = get_cosine_schedule_with_warmup(opt, int(cfg.warmup_ratio * total_steps), total_steps)

    step = 0
    for epoch in range(cfg.epochs):
        for i, pair in enumerate(dl):
            f = _move(pair["forget"], device)
            r = _move(pair["retain"], device)

            f_logits = model(input_ids=f["input_ids"], attention_mask=f["attention_mask"]).logits

            if cfg.method_type == "grad_ascent":
                loss = methods.ga_loss(f_logits, f["labels"])
            elif cfg.method_type == "grad_diff":
                r_logits = model(input_ids=r["input_ids"], attention_mask=r["attention_mask"]).logits
                loss = methods.grad_diff_loss(f_logits, f["labels"], r_logits, r["labels"], cfg.retain_weight)
            elif cfg.method_type == "npo":
                with torch.no_grad():
                    ref_logits = ref_model(input_ids=f["input_ids"], attention_mask=f["attention_mask"]).logits
                    ref_lp = methods.sequence_logprob(ref_logits, f["labels"])
                r_logits = model(input_ids=r["input_ids"], attention_mask=r["attention_mask"]).logits
                loss = methods.npo_loss(
                    f_logits, f["labels"], ref_lp, r_logits, r["labels"], cfg.beta, cfg.retain_weight
                )
            else:
                raise ValueError(f"unknown method_type {cfg.method_type}")

            (loss / cfg.grad_accum).backward()

            if (i + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f}")

    os.makedirs(out_dir, exist_ok=True)
    model.config.use_cache = True
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[trainer] saved unlearned model -> {out_dir}")
    return out_dir
