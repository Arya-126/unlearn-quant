"""Uniform scoring interface so every compression backend is evaluated identically.

A Scorer exposes two methods:
  - answer_logprob(question, answer) -> (sum_logprob: float, n_tokens: int)
  - generate(question, max_new_tokens) -> str

HFScorer covers fp16, bitsandbytes, GPTQ and AWQ models (all load as AutoModelForCausalLM).
GGUF models use a separate scorer in src/compress/gguf.py that satisfies the same contract.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..data.tofu import build_prompt


class HFScorer:
    def __init__(self, model, tokenizer, device="cuda", max_length=256):
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.max_length = max_length
        self.model.eval()

    @torch.no_grad()
    def answer_logprob(self, question: str, answer: str):
        prompt = build_prompt(question)
        full = prompt + answer
        enc = self.tok(full, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        prompt_len = len(self.tok(prompt, add_special_tokens=True)["input_ids"])

        logits = self.model(**enc).logits
        shift_logits = logits[0, :-1, :]
        shift_labels = enc["input_ids"][0, 1:]
        logp = F.log_softmax(shift_logits.float(), dim=-1)
        tok_logp = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        ans_logp = tok_logp[prompt_len - 1:]      # tokens predicting the answer
        return float(ans_logp.sum().item()), int(ans_logp.numel())

    @torch.no_grad()
    def generate(self, question: str, max_new_tokens: int = 64) -> str:
        prompt = build_prompt(question)
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
        )
        gen = out[0, enc["input_ids"].size(1):]
        return self.tok.decode(gen, skip_special_tokens=True).strip()
