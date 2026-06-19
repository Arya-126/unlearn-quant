# unlearn-quant — Does LLM unlearning survive edge compression?

Experimental harness for the question: **when you make an LLM forget a fact and then compress
it for edge deployment (int4 quantization or distillation), does the forgotten knowledge come
back?**

## Where this sits in the literature

The *basic* effect is already known — quantization can resurrect "forgotten" knowledge:

- [Catastrophic Failure of LLM Unlearning via Quantization](https://arxiv.org/abs/2410.16454)
  (ICLR 2025): ~21% forget-retention at fp16 → **~83% after 4-bit RTN**. Code:
  [zzwjames/FailureLLMUnlearning](https://github.com/zzwjames/FailureLLMUnlearning).
- [Distillation Robustifies Unlearning](https://arxiv.org/abs/2506.06278) (Jun 2025).

**Our delta (the contribution):** prior work quantizes almost entirely with bitsandbytes
round-to-nearest (RTN). Real edge deployment uses *calibration-based* int4 — **GPTQ, AWQ, and
llama.cpp GGUF k-quants**. This harness:

1. **Replicates** the RTN effect as a sanity anchor (Phase A), then
2. **Extends** it to realistic edge formats (GPTQ / AWQ / GGUF) and the
   **distillation × quantization** interaction (distill an unlearned teacher, *then* int4 the
   student) (Phase B), reporting a single **recovery ratio** across the whole sweep.

A null result ("recovery is RTN-specific") and a positive one ("recovery is universal across edge
formats; distillation doesn't save you under int4") are both publishable.

## Pipeline

```
load tofu_ft_phi-1.5  ->  unlearn {GA, GradDiff, NPO}  ->  compress {fp16, bnb, gptq, awq, gguf, distill+quant}  ->  eval (TOFU metrics + recovery ratio)
```

## Quickstart (Kaggle T4)

Open `notebooks/kaggle_run.ipynb` on Kaggle (GPU T4, internet ON). It clones this repo, installs
deps, and runs a phase. Or locally:

```bash
pip install -r requirements.txt
# Phase A: replicate the RTN recovery jump (bnb only)
python -m scripts.run_replication --method graddiff
# Phase B: edge int4 formats + distillation
python -m scripts.run_edge_extension --method graddiff
```

Outputs land in `results/` as per-run JSON, a combined `summary.csv`, and heatmaps.

## Recovery ratio

For a forget score `s` (ROUGE-L recall / mean answer prob / accuracy):

```
R = (s_compressed - s_unlearned) / (s_base - s_unlearned)
```

`R≈0` → unlearning holds under compression. `R≈1` → knowledge fully resurfaced.

## Layout

See `configs/` for all knobs. Core code in `src/{data,unlearn,compress,eval}` and `src/pipeline.py`.

## Status

- [x] Harness runs end-to-end on Kaggle T4 (`scripts/run_replication.py`, `scripts/run_reproduce.py`)
- [x] Phase A (bnb) at 1.3B Phi-1.5 — **no NF4 recovery** for NPO; GradDiff collapsed at high lr
- [x] 7B scale validation (Llama-2-7b, external full-FT checkpoints) — **also no recovery** (rules out scale)
- [x] Results written up in [FINDINGS.md](FINDINGS.md) — current result is a *non-reproduction*
- [ ] Phase B edge formats (GPTQ/AWQ/GGUF) + distillation
- [ ] Close threats to validity: run zzwjames's exact recipe/quantizer; more seeds; deeper forget

See [FINDINGS.md](FINDINGS.md) for the data and threats to validity.
