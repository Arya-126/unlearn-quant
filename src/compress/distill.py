"""Distillation arm: distill an unlearned teacher into a fresh student on retain data.

Follows the "distillation robustifies unlearning" idea — a student initialized from the
base architecture learns only the teacher's expressed behavior on the retain set, so
latent forget knowledge should not transfer. The novel question here is whether that
robustness *survives int4*: after distilling, we run the student through the same
quantization sweep and measure recovery.

Loss = T^2 * KL(softmax(student/T) || softmax(teacher/T)) over answer tokens
       (+ optional CE on retain gold answers).
"""
from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from ..data.tofu import IGNORE_INDEX, QAExample, TofuCollator, tokenize_qa


class _RetainDS(Dataset):
    def __init__(self, examples: List[QAExample], tokenizer, max_length=256):
        self.ex = examples
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        return tokenize_qa(self.ex[i], self.tok, self.max_length)


def distill(
    teacher_dir: str,
    student_init: str,
    retain: List[QAExample],
    out_dir: str,
    temperature: float = 2.0,
    alpha_kd: float = 1.0,
    alpha_ce: float = 0.0,
    epochs: int = 3,
    lr: float = 1e-4,
    batch_size: int = 4,
    grad_accum: int = 4,
    dtype=torch.bfloat16,
    cache_dir: Optional[str] = None,
    device: str = "cuda",
):
    from ..modeling import load_causal_lm, load_tokenizer

    tok = load_tokenizer(teacher_dir, cache_dir=cache_dir)

    teacher = load_causal_lm(teacher_dir, torch_dtype=dtype).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = load_causal_lm(student_init, cache_dir=cache_dir, torch_dtype=dtype).to(device)
    student.gradient_checkpointing_enable()
    student.config.use_cache = False

    dl = DataLoader(
        _RetainDS(retain, tok), batch_size=batch_size, shuffle=True, collate_fn=TofuCollator(tok)
    )
    total_steps = max(1, len(dl) // grad_accum) * epochs
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(student.parameters(), lr=lr)
    except Exception:
        opt = torch.optim.AdamW(student.parameters(), lr=lr)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * total_steps), total_steps)

    T = temperature
    step = 0
    for epoch in range(epochs):
        for i, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                t_logits = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            s_out = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            s_logits = s_out.logits

            mask = (batch["labels"] != IGNORE_INDEX)[..., 1:].contiguous()      # answer tokens
            s_shift = s_logits[..., :-1, :].contiguous()
            t_shift = t_logits[..., :-1, :].contiguous()

            kd = F.kl_div(
                F.log_softmax(s_shift / T, dim=-1),
                F.softmax(t_shift / T, dim=-1),
                reduction="none",
            ).sum(-1)
            kd = (kd * mask).sum() / mask.sum().clamp(min=1) * (T * T)

            loss = alpha_kd * kd
            if alpha_ce > 0:
                ce = F.cross_entropy(
                    s_shift.view(-1, s_shift.size(-1)),
                    batch["labels"][..., 1:].contiguous().view(-1),
                    ignore_index=IGNORE_INDEX,
                )
                loss = loss + alpha_ce * ce

            (loss / grad_accum).backward()
            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(f"[distill] epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f}")

    os.makedirs(out_dir, exist_ok=True)
    student.config.use_cache = True
    student.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[distill] saved student -> {out_dir}")
    return out_dir
