"""Phase B — edge int4 extension (the contribution).

Full compression sweep across realistic edge formats (GPTQ/AWQ/GGUF k-quants) plus the
distillation arm (distill the unlearned teacher, then quantize the student). Reports the
recovery ratio for every condition.

    python -m scripts.run_edge_extension --method graddiff --n-eval 100 --distill
"""
import argparse

from src.pipeline import run

EDGE_CONDITIONS = [
    "fp16",
    "bnb_int8", "bnb_nf4",
    "gptq_4bit", "gptq_3bit",
    "awq_4bit",
    "gguf_q4_k_m", "gguf_q4_0", "gguf_q3_k_m",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="graddiff", choices=["ga", "graddiff", "npo"])
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--distill", action="store_true", help="also run the distillation arm")
    ap.add_argument("--forget-metric", default="rouge", choices=["rouge", "prob"])
    ap.add_argument(
        "--conditions", nargs="*", default=None,
        help="subset of conditions to run (default: full edge sweep)",
    )
    args = ap.parse_args()

    run(
        method=args.method,
        conditions=args.conditions or EDGE_CONDITIONS,
        n_eval=args.n_eval,
        with_distill=args.distill,
        device=args.device,
        forget_metric=args.forget_metric,
    )


if __name__ == "__main__":
    main()
