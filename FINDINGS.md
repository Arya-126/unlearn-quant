# Findings (work in progress)

**Question:** does int4/int8 quantization of an *unlearned* LLM resurface the forgotten
knowledge, as reported by [Catastrophic Failure of LLM Unlearning via Quantization](https://arxiv.org/abs/2410.16454)
(ICLR 2025; ~21%→~83% recovery after 4-bit)?

**Short answer so far: we do NOT reproduce the recovery effect** with bitsandbytes RTN
(NF4/int8) on TOFU, at either 1.3B or 7B, for NPO or GradDiff. Recovery ≈ 0 by both
probability and ROUGE. This is a *non-reproduction under our conditions*, not yet a refutation
— see threats to validity.

## Setup

- Benchmark: TOFU (`locuslab/TOFU`). Metrics: greedy-generation ROUGE-L recall and
  length-normalized answer probability; recovery ratio `R=(q-u)/(b-u)` (b=base/knows,
  u=unlearned fp16, q=quantized).
- Compute: Kaggle T4, via the harness in this repo. Quantization: bitsandbytes NF4 + int8 (RTN).

## Results

### 1.3B — Phi-1.5, TOFU forget05, our own unlearning (lr 5e-5, 10 ep, n_eval=60)

| method | cond | forget ROUGE | forget prob | retain ROUGE | utility | recovery |
|---|---|---|---|---|---|---|
| — | base (knows) | 0.796 | 0.89 | — | — | — |
| graddiff | fp16 | 0.071 | 0.000 | 0.366 | 0.03 | — |
| graddiff | bnb_nf4 | 0.060 | 0.000 | 0.334 | 0.02 | −0.01 |
| npo | fp16 | 0.289 | 0.028 | 0.758 | 0.54 | — |
| npo | bnb_int8 | 0.292 | 0.028 | 0.762 | 0.55 | 0.006 |
| npo | bnb_nf4 | 0.302 | 0.031 | 0.709 | 0.56 | 0.026 |

- **NPO** is the clean utility-constrained operating point (forgets: ROUGE 0.80→0.29, prob
  0.89→0.03; keeps utility 0.54). Quantization leaves it flat → **no recovery** (≤0.03), and
  **probability and ROUGE agree**.
- **GradDiff** collapsed at this lr (unbounded ascent destroyed utility, 0.03). Not a valid
  operating point; its near-zero recovery is uninformative.

### 7B — Llama-2-7b-chat, TOFU forget10, external full-FT checkpoints (n_eval=50, prob)

Checkpoints: `the-jb/tofu_Llama-2-7b-chat-hf_forget10_{NPO,GradDiff}` (full fine-tuning, not
LoRA); base `open-unlearning/tofu_Llama-2-7b-chat-hf_full`.

| method | cond | forget prob | retain prob | recovery |
|---|---|---|---|---|
| — | base (knows) | 0.992 | — | — |
| npo | fp16 | 0.153 | 0.60 | — |
| npo | bnb_int8 | 0.155 | 0.60 | 0.003 |
| npo | bnb_nf4 | 0.153 | 0.58 | 0.001 |
| graddiff | fp16 | ~1e-32 | 0.71 | — |
| graddiff | bnb_nf4 | ~1e-32 | 0.66 | 0.000 |

- NPO forgot to ~15% of base (close to the paper's "21% retained") with utility preserved, and
  the recovery denominator is healthy (0.99−0.15=0.84) — so **recovery ≈ 0 is meaningful, not a
  small-denominator artifact.** Quantizing did not bring the knowledge back.
- **Scale is therefore ruled out** as the explanation for the 1.3B null result.

## Interpretation

Across **two scales (1.3B, 7B)**, **two methods (NPO, GradDiff)**, and **two metrics (prob,
ROUGE)**, bitsandbytes NF4/int8 quantization of the unlearned model does **not** recover forgotten
TOFU knowledge. If this holds up, it is a genuine (if less flashy) result with a clear
edge-deployment reading: **NF4-quantizing a well-unlearned small model does not, by itself,
resurrect the forgotten facts** — contrary to what the large-model RTN result might lead you to
expect.

## Threats to validity (resolve before claiming a refutation)

1. **Not the paper's exact artifacts.** We used public the-jb/open-unlearning checkpoints + our own
   1.3B unlearning, not [zzwjames/FailureLLMUnlearning](https://github.com/zzwjames/FailureLLMUnlearning)'s
   exact unlearned models / hyperparameters. Their effect is strongest for *their* utility-constrained
   models. **Highest-priority next step: run their repo end-to-end and quantize that.**
2. **Quantizer.** We used bitsandbytes RTN (NF4/int8). The paper's "4-bit" recipe should be matched;
   our planned edge formats (GPTQ/AWQ/GGUF, Phase B) are not yet run.
3. **Shallow forget quality.** KS-test forget-quality is low (≈5e-3 at 7B), i.e. not the deep-forget
   regime; the recovery effect may require deeper forgetting.
4. **Small eval.** n_eval=50–60, single seed.

## Next steps (cheap → expensive)

- [ ] `FINDINGS.md` ← (this file)
- [ ] Phase B edge formats (GPTQ/AWQ/GGUF) on the npo_7b checkpoint — download once, quantize/eval.
- [ ] Reproduce with zzwjames's exact unlearning recipe + their quantization, to close threat #1.
- [ ] More seeds + full eval set; sweep forget-quality (depth of unlearning) vs recovery.
