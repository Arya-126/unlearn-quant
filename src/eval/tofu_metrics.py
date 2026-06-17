"""TOFU metrics computed on top of a Scorer.

- ROUGE-L recall: greedy generation vs ground-truth answer. On the *forget* set, high
  ROUGE = the model still produces the right answer = knowledge present.
- Answer probability: length-normalized P(answer | question).
- Truth Ratio (TOFU): mean length-normalized prob of perturbed (wrong) answers divided by
  that of the paraphrased correct answer. Higher on forget = model prefers wrong answers.
- Forget Quality (TOFU): KS-test p-value between forget and retain truth-ratio distributions.
  p near 1 => forget set is statistically indistinguishable from retain => well unlearned.
- Model Utility: aggregate of prob + ROUGE on retain / real_authors / world_facts.

We track a single ``forget_score`` (ROUGE-L recall by default) for the recovery ratio.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
from rouge_score import rouge_scorer
from scipy.stats import ks_2samp

from ..data.tofu import QAExample

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _norm_prob(scorer, question, answer):
    """Length-normalized probability exp(mean token logprob)."""
    s, n = scorer.answer_logprob(question, answer)
    if n == 0:
        return 0.0
    return math.exp(s / n)


def rouge_l_recall(scorer, examples: List[QAExample], max_new_tokens=64) -> float:
    if not examples:
        return 0.0
    vals = []
    for ex in examples:
        gen = scorer.generate(ex.question, max_new_tokens=max_new_tokens)
        vals.append(_ROUGE.score(ex.answer, gen)["rougeL"].recall)
    return float(np.mean(vals))


def mean_answer_prob(scorer, examples: List[QAExample]) -> float:
    if not examples:
        return 0.0
    return float(np.mean([_norm_prob(scorer, ex.question, ex.answer) for ex in examples]))


def truth_ratios(scorer, examples: List[QAExample]) -> List[float]:
    """Per-example TOFU truth ratio. Skips examples lacking perturbed answers."""
    out = []
    for ex in examples:
        if not ex.perturbed_answers:
            continue
        para = ex.paraphrased_answer or ex.answer
        p_para = _norm_prob(scorer, ex.question, para)
        p_perts = [_norm_prob(scorer, ex.question, p) for p in ex.perturbed_answers]
        mean_pert = float(np.mean(p_perts)) if p_perts else 0.0
        out.append(mean_pert / (p_para + 1e-12))
    return out


def forget_quality(scorer, forget: List[QAExample], retain: List[QAExample]) -> float:
    """KS-test p-value between forget and retain truth-ratio distributions."""
    tf = truth_ratios(scorer, forget)
    tr = truth_ratios(scorer, retain)
    if len(tf) < 2 or len(tr) < 2:
        return float("nan")
    return float(ks_2samp(tf, tr).pvalue)


def model_utility(scorer, utility_examples: List[QAExample]) -> float:
    """Simple utility aggregate: mean of answer-prob and ROUGE-L on utility splits."""
    if not utility_examples:
        return float("nan")
    p = mean_answer_prob(scorer, utility_examples)
    r = rouge_l_recall(scorer, utility_examples)
    return float((p + r) / 2)


def evaluate(scorer, forget, retain, utility=None, forget_metric="rouge", do_rouge=True) -> dict:
    """Full metric bundle for one (model, compression) condition.

    do_rouge=False skips greedy generation (used for 7B, where generation is slow/OOM-prone);
    forget_score then comes from probability. Truth-Ratio / Forget Quality are unaffected.
    """
    res = {
        "forget_prob": mean_answer_prob(scorer, forget),
        "retain_prob": mean_answer_prob(scorer, retain),
        "forget_quality": forget_quality(scorer, forget, retain),
    }
    if do_rouge:
        res["forget_rouge"] = rouge_l_recall(scorer, forget)
        res["retain_rouge"] = rouge_l_recall(scorer, retain)
    if utility is not None:
        res["model_utility"] = mean_answer_prob(scorer, utility) if not do_rouge else model_utility(scorer, utility)
    # Headline scalar used by the recovery ratio.
    if forget_metric == "rouge" and do_rouge:
        res["forget_score"] = res["forget_rouge"]
    else:
        res["forget_score"] = res["forget_prob"]
    return res
