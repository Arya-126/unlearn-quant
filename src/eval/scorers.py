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
        """Deterministic greedy decode via forward+KV cache.

        Avoids transformers' generate() input-prep path (a 4.57 regression throws
        IndexError on cache_position for some models). Equivalent for our needs.
        """
        enc = self.tok(build_prompt(question), return_tensors="pt").to(self.device)
        attn = enc["attention_mask"]
        cur = enc["input_ids"]
        eos = self.tok.eos_token_id
        past = None
        out_ids = []
        for _ in range(max_new_tokens):
            out = self.model(input_ids=cur, attention_mask=attn, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax().item())
            if eos is not None and nxt == eos:
                break
            out_ids.append(nxt)
            cur = torch.tensor([[nxt]], device=self.device)
            attn = torch.cat([attn, attn.new_ones((attn.size(0), 1))], dim=1)
        return self.tok.decode(out_ids, skip_special_tokens=True).strip()
