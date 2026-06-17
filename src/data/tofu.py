"""TOFU data loading + prompt formatting.

TOFU (`locuslab/TOFU`) is a QA benchmark over 200 fictitious authors. Each row has
`question` and `answer`. The `*_perturbed` configs additionally carry
`paraphrased_answer` and `perturbed_answer` (a list of plausible-but-wrong answers),
which the Truth Ratio metric needs.

Phi-1.5 is a *base* (non-chat) model, so we use the plain TOFU QA template used by the
official repo for phi: ``Question: {q}\nAnswer: {a}``. Only the answer tokens contribute
to the loss / probability (the prompt is masked with -100).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from datasets import load_dataset

# TOFU phi prompt tags (match locuslab/tofu phi config).
Q_START = "Question: "
Q_END = "\n"
A_TAG = "Answer: "

IGNORE_INDEX = -100


def build_prompt(question: str) -> str:
    return f"{Q_START}{question}{Q_END}{A_TAG}"


def build_full(question: str, answer: str) -> str:
    return build_prompt(question) + answer


@dataclass
class QAExample:
    question: str
    answer: str
    paraphrased_answer: Optional[str] = None
    perturbed_answers: List[str] = field(default_factory=list)


def load_split(
    split: str,
    hf_dataset: str = "locuslab/TOFU",
    cache_dir: Optional[str] = None,
    perturbed: bool = False,
) -> List[QAExample]:
    """Load one TOFU config into a list of QAExample.

    If ``perturbed`` is True we load the ``{split}_perturbed`` config (used for Truth
    Ratio); some splits such as ``real_authors`` / ``world_facts`` are already perturbed
    and are loaded directly.
    """
    name = split
    if perturbed and not split.endswith("_perturbed") and split not in ("real_authors", "world_facts"):
        name = f"{split}_perturbed"
    ds = load_dataset(hf_dataset, name, split="train", cache_dir=cache_dir)

    out: List[QAExample] = []
    for row in ds:
        pert = row.get("perturbed_answer", [])
        if isinstance(pert, str):
            pert = [pert]
        out.append(
            QAExample(
                question=row["question"],
                answer=row["answer"],
                paraphrased_answer=row.get("paraphrased_answer"),
                perturbed_answers=list(pert),
            )
        )
    return out


def tokenize_qa(
    example: QAExample,
    tokenizer,
    max_length: int = 256,
    answer: Optional[str] = None,
):
    """Tokenize one QA pair, masking prompt tokens so loss is computed on the answer only.

    Returns a dict of 1-D tensors: input_ids, attention_mask, labels.
    ``answer`` overrides example.answer (used to score paraphrased / perturbed answers).
    """
    ans = example.answer if answer is None else answer
    prompt = build_prompt(example.question)
    full = prompt + ans + (tokenizer.eos_token or "")

    enc = tokenizer(full, max_length=max_length, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"][0]
    attn = enc["attention_mask"][0]

    # Length of the prompt portion (no special tokens added so offsets line up).
    prompt_len = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    labels = input_ids.clone()
    labels[:prompt_len] = IGNORE_INDEX

    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


class TofuCollator:
    """Pad a batch of {input_ids, attention_mask, labels} to the longest sequence."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, batch):
        maxlen = max(x["input_ids"].size(0) for x in batch)

        def pad(t, value):
            if t.size(0) == maxlen:
                return t
            return torch.cat([t, t.new_full((maxlen - t.size(0),), value)])

        input_ids = torch.stack([pad(x["input_ids"], self.pad_id) for x in batch])
        attn = torch.stack([pad(x["attention_mask"], 0) for x in batch])
        labels = torch.stack([pad(x["labels"], IGNORE_INDEX) for x in batch])
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


class ForgetRetainDataset(torch.utils.data.Dataset):
    """Pairs a forget example with a randomly sampled retain example per index.

    GradDiff / NPO consume both arms: they ascend (or apply preference loss) on the
    forget answer while descending on the retain answer. GA ignores the retain arm.
    """

    def __init__(self, forget: List[QAExample], retain: List[QAExample], tokenizer, max_length=256, seed=42):
        self.forget = forget
        self.retain = retain
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.forget)

    def __getitem__(self, idx):
        f = tokenize_qa(self.forget[idx], self.tokenizer, self.max_length)
        r_idx = int(torch.randint(0, len(self.retain), (1,), generator=self.rng).item())
        r = tokenize_qa(self.retain[r_idx], self.tokenizer, self.max_length)
        return {"forget": f, "retain": r}


class PairCollator:
    """Collate ForgetRetainDataset items into batched forget/retain sub-batches."""

    def __init__(self, tokenizer):
        self.base = TofuCollator(tokenizer)

    def __call__(self, batch):
        return {
            "forget": self.base([b["forget"] for b in batch]),
            "retain": self.base([b["retain"] for b in batch]),
        }
