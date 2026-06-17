"""Phase A — replication anchor.

Reproduce the ICLR 2025 effect with bitsandbytes RTN only: forget knowledge should be low
at fp16 and recover after 4-bit. Run the utility-constrained methods (graddiff, npo) where
the effect is strong; ga is the weak-recovery control.

    python -m scripts.run_replication --method graddiff --n-eval 100
"""
import argparse

from src.pipeline import run

REPLICATION_CONDITIONS = ["fp16", "bnb_int8", "bnb_nf4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="graddiff", choices=["ga", "graddiff", "npo"])
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--forget-metric", default="rouge", choices=["rouge", "prob"])
    args = ap.parse_args()

    run(
        method=args.method,
        conditions=REPLICATION_CONDITIONS,
        n_eval=args.n_eval,
        with_distill=False,
        device=args.device,
        forget_metric=args.forget_metric,
    )


if __name__ == "__main__":
    main()
