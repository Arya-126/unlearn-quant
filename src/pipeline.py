"""End-to-end pipeline: load -> unlearn -> compress sweep -> eval -> recovery ratio.

Each (method, compression-condition) pair becomes one row in results/summary.csv with its
recovery ratio. Compression conditions that fail (e.g. a missing GGUF toolchain) are
recorded with an error string instead of crashing the sweep.
"""
from __future__ import annotations

import gc
import os
import traceback
from typing import List, Optional

import torch
import yaml

from .compress import load_compressed
from .compress.distill import distill
from .data.tofu import load_split
from .eval import tofu_metrics
from .eval.recovery import plot_heatmap, save_results
from .eval.scorers import HFScorer
from .unlearn.trainer import UnlearnConfig, unlearn

# retain split has its own perturbed config name in TOFU
_RETAIN_PERTURBED = "retain_perturbed"


def load_cfg(cfg_dir="configs"):
    out = {}
    for name in ("model", "unlearn", "compress"):
        with open(os.path.join(cfg_dir, f"{name}.yaml")) as f:
            out[name] = yaml.safe_load(f)
    return out


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _eval_dir_or_scorer(scorer, forget, retain, utility, forget_metric):
    res = tofu_metrics.evaluate(scorer, forget, retain, utility, forget_metric)
    _free()
    return res


def run(
    method: str,
    conditions: List[str],
    cfg_dir: str = "configs",
    n_eval: int = 100,
    n_calib: int = 512,
    with_distill: bool = False,
    device: str = "cuda",
    work_root: str = "./.work",
    forget_metric: str = "rouge",
):
    cfg = load_cfg(cfg_dir)
    mcfg, ucfg, ccfg = cfg["model"], cfg["unlearn"], cfg["compress"]
    base_name = mcfg["model"]["name"]
    cache_dir = mcfg["paths"]["cache_dir"]
    out_dir = mcfg["paths"]["out_dir"]
    hf_ds = mcfg["data"]["hf_dataset"]
    forget_split = mcfg["data"]["forget_split"]

    # --- data (perturbed configs carry the answers needed for Truth Ratio) ---
    forget_all = load_split(forget_split, hf_ds, cache_dir, perturbed=True)
    retain_all = load_split(_RETAIN_PERTURBED, hf_ds, cache_dir, perturbed=True)
    utility_all = []
    for us in mcfg["data"]["utility_splits"]:
        utility_all += load_split(us, hf_ds, cache_dir, perturbed=True)

    forget_eval, retain_eval = forget_all[:n_eval], retain_all[:n_eval]
    utility_eval = utility_all[:n_eval]
    calib = retain_all[:n_calib]

    method_spec = ucfg["methods"][method]
    common = ucfg["common"]
    ucfg_obj = UnlearnConfig(
        method_type=method_spec["type"],
        lr=float(common["lr"]),
        epochs=int(common["epochs"]),
        batch_size=int(common["per_device_batch_size"]),
        grad_accum=int(common["grad_accum"]),
        warmup_ratio=float(common["warmup_ratio"]),
        weight_decay=float(common["weight_decay"]),
        beta=float(method_spec.get("beta", 0.1)),
        retain_weight=float(method_spec.get("retain_weight", 1.0)),
        use_8bit_adam=bool(common["use_8bit_adam"]),
        gradient_checkpointing=bool(common["gradient_checkpointing"]),
        seed=int(common["seed"]),
        max_length=int(mcfg["model"]["max_length"]),
    )

    work = os.path.join(work_root, method)
    unlearned_dir = os.path.join(work, "unlearned")

    # --- base (pre-unlearning) forget score: the "knows it" reference ---
    base_model = __import__("transformers").AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.float16, cache_dir=cache_dir    ).to(device)
    base_tok = __import__("transformers").AutoTokenizer.from_pretrained(
        base_name, cache_dir=cache_dir    )
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token
    base_metrics = _eval_dir_or_scorer(
        HFScorer(base_model, base_tok, device), forget_eval, retain_eval, utility_eval, forget_metric
    )
    base_forget = base_metrics["forget_score"]
    del base_model
    _free()

    # --- unlearn (cached if already present) ---
    if not os.path.exists(os.path.join(unlearned_dir, "config.json")):
        unlearn(
            base_name, forget_all, retain_all, ucfg_obj, unlearned_dir,
            cache_dir=cache_dir, device=device,
        )
        _free()

    records = []

    def record(condition, metrics, unlearned_forget, error=None):
        rec = {
            "method": method,
            "condition": condition,
            "base_forget_score": base_forget,
            "unlearned_forget_score": unlearned_forget,
        }
        if error:
            rec["error"] = error
        else:
            rec.update(metrics)
        records.append(rec)

    # fp16 unlearned condition first -> defines unlearned_forget_score baseline
    fp16_metrics = _eval_dir_or_scorer(
        load_compressed(unlearned_dir, {"backend": "none"}, device),
        forget_eval, retain_eval, utility_eval, forget_metric,
    )
    unlearned_forget = fp16_metrics["forget_score"]
    record("fp16", fp16_metrics, unlearned_forget)

    quant_specs = ccfg["quantization"]
    for cond in conditions:
        if cond == "fp16":
            continue
        spec = quant_specs[cond]
        try:
            scorer = load_compressed(unlearned_dir, spec, device, calib=calib, work_dir=os.path.join(work, cond))
            m = _eval_dir_or_scorer(scorer, forget_eval, retain_eval, utility_eval, forget_metric)
            record(cond, m, unlearned_forget)
        except Exception as e:  # noqa: BLE001 - isolate per-condition failures
            print(f"[pipeline] condition {cond} failed: {e}")
            traceback.print_exc()
            record(cond, None, unlearned_forget, error=str(e))
        _free()

    # --- distillation arm: distill unlearned teacher, then rerun quant sweep on the student ---
    if with_distill and ccfg["distillation"]["enabled"]:
        dcfg = ccfg["distillation"]
        student_dir = os.path.join(work, "student")
        if not os.path.exists(os.path.join(student_dir, "config.json")):
            distill(
                unlearned_dir, mcfg["model"]["base_arch"], retain_all, student_dir,
                temperature=float(dcfg["temperature"]), alpha_kd=float(dcfg["alpha_kd"]),
                alpha_ce=float(dcfg["alpha_ce"]), epochs=int(dcfg["epochs"]),
                lr=float(dcfg["lr"]), cache_dir=cache_dir, device=device,
            )
            _free()
        for cond in ["fp16"] + [c for c in conditions if c != "fp16"]:
            spec = {"backend": "none"} if cond == "fp16" else quant_specs[cond]
            try:
                scorer = load_compressed(student_dir, spec, device, calib=calib, work_dir=os.path.join(work, "student_" + cond))
                m = _eval_dir_or_scorer(scorer, forget_eval, retain_eval, utility_eval, forget_metric)
                record("distill+" + cond, m, unlearned_forget)
            except Exception as e:  # noqa: BLE001
                print(f"[pipeline] distill+{cond} failed: {e}")
                record("distill+" + cond, None, unlearned_forget, error=str(e))
            _free()

    csv_path = save_results(records, out_dir, tag=f"{method}")
    try:
        plot_heatmap(csv_path, os.path.join(out_dir, "recovery_heatmap.png"))
    except Exception as e:  # noqa: BLE001
        print(f"[pipeline] heatmap skipped: {e}")
    print(f"[pipeline] done. base_forget={base_forget:.3f} unlearned_forget={unlearned_forget:.3f}")
    return records
