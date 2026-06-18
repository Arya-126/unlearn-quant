"""7B faithful reproduction of the ICLR 2025 quantization-recovery effect.

Uses pre-computed, FULL-fine-tuned unlearned Llama-2-7B-chat checkpoints (no training
needed -> fits a T4 for inference/quant). Validates whether our harness sees the recovery
where it is known to exist, so we can interpret the 1.3B null result (scale dependence).

  python -m scripts.run_reproduce --method npo --n-eval 50

forget split is forget10 (these checkpoints are forget10). We measure probability-based
forget score (generation skipped for speed/memory); recovery = (q - u)/(b - u).
"""
import argparse
import gc

import torch

from src.compress import load_compressed
from src.data.tofu import load_split
from src.eval import tofu_metrics
from src.eval.recovery import plot_heatmap, save_results

BASE_ID = "open-unlearning/tofu_Llama-2-7b-chat-hf_full"
UNLEARNED = {
    "npo": "the-jb/tofu_Llama-2-7b-chat-hf_forget10_NPO",
    "graddiff": "the-jb/tofu_Llama-2-7b-chat-hf_forget10_GradDiff",
    "ga": "the-jb/tofu_Llama-2-7b-chat-hf_forget10_GradAscent",
}
CONDITIONS = {
    "fp16": {"backend": "none"},
    "bnb_int8": {"backend": "bnb", "bits": 8},
    "bnb_nf4": {"backend": "bnb", "bits": 4, "nf4": True, "double_quant": True},
}


def llama2_prompt(question: str) -> str:
    # TOFU Llama-2-chat template (question_start="[INST] ", question_end=" [/INST]").
    return f"[INST] {question} [/INST]"


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="npo", choices=list(UNLEARNED))
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--rouge", action="store_true",
                    help="also measure greedy-generation ROUGE (recovery may show here, not in prob)")
    args = ap.parse_args()
    forget_metric = "rouge" if args.rouge else "prob"

    forget = load_split("forget10", perturbed=True)[: args.n_eval]
    retain = load_split("retain_perturbed", perturbed=True)[: args.n_eval]

    def evaluate(model_id, spec):
        scorer = load_compressed(
            model_id, spec, device="cuda", prompt_fn=llama2_prompt,
            fp16_device_map="auto" if spec.get("backend", "none") in ("none", None) else None,
        )
        m = tofu_metrics.evaluate(scorer, forget, retain, forget_metric=forget_metric, do_rouge=args.rouge)
        del scorer
        _free()
        return m

    # base (knows it) -> reference forget score
    base_forget = evaluate(BASE_ID, {"backend": "none"})["forget_score"]
    print(f"[reproduce] base forget_prob={base_forget:.4f}")

    unlearned_id = UNLEARNED[args.method]
    records = []
    fp16 = evaluate(unlearned_id, CONDITIONS["fp16"])
    unlearned_forget = fp16["forget_score"]
    print(f"[reproduce] {args.method} fp16 forget_prob={unlearned_forget:.4f}")

    for cond, spec in CONDITIONS.items():
        m = fp16 if cond == "fp16" else evaluate(unlearned_id, spec)
        rec = {"method": f"{args.method}_7b", "condition": cond,
               "base_forget_score": base_forget, "unlearned_forget_score": unlearned_forget}
        rec.update(m)
        records.append(rec)
        print(f"[reproduce] {cond}: forget_prob={m['forget_score']:.4f}")

    csv = save_results(records, args.out_dir, tag=f"reproduce_{args.method}_7b")
    try:
        plot_heatmap(csv, f"{args.out_dir}/recovery_heatmap.png")
    except Exception as e:  # noqa: BLE001
        print(f"[reproduce] heatmap skipped: {e}")
    print("[reproduce] done.")


if __name__ == "__main__":
    main()
