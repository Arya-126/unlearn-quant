"""Recovery ratio + results aggregation.

Recovery ratio for a compressed condition, given the forget score of the original
(base, pre-unlearning) model and the unlearned fp16 model:

    R = (s_compressed - s_unlearned) / (s_base - s_unlearned)

R≈0  -> unlearning still holds after compression.
R≈1  -> the forgotten knowledge has fully resurfaced.
R can exceed 1 (compression made it *more* knowledgeable than the original) or go
slightly negative (compression degraded it further); both are clamped only for display.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import pandas as pd


def recovery_ratio(s_compressed: float, s_unlearned: float, s_base: float) -> float:
    denom = s_base - s_unlearned
    if abs(denom) < 1e-9:
        return float("nan")
    return (s_compressed - s_unlearned) / denom


def build_summary(records: List[Dict]) -> pd.DataFrame:
    """records: list of dicts with keys method, condition, forget_score, ... plus
    'base_forget_score' and 'unlearned_forget_score' for ratio computation."""
    rows = []
    for r in records:
        row = dict(r)
        row["recovery_ratio"] = recovery_ratio(
            r["forget_score"], r["unlearned_forget_score"], r["base_forget_score"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def save_results(records: List[Dict], out_dir: str, tag: str = "run") -> str:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(records, f, indent=2)
    df = build_summary(records)
    csv_path = os.path.join(out_dir, "summary.csv")
    # Append to a combined CSV across runs.
    if os.path.exists(csv_path):
        prev = pd.read_csv(csv_path)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return csv_path


def plot_heatmap(csv_path: str, out_png: str):
    """method x condition -> recovery_ratio heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    piv = df.pivot_table(index="method", columns="condition", values="recovery_ratio", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.4 * len(piv.columns) + 2, 1.0 * len(piv.index) + 2))
    im = ax.imshow(piv.values, vmin=0, vmax=1, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not pd.isna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Knowledge recovery ratio (1 = fully resurfaced)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[recovery] wrote {out_png}")
