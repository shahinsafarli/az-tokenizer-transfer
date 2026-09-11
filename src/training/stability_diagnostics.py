"""Run the pre-full xlm15 stability diagnosis and stop after step 3.

The diagnostic matrix is intentionally fixed:
  1. repeat one healthy and one collapsed mean run with epoch loss logging;
  2. on mean/seed42, change fp16 only, then learning rate only;
  3. run original-tokenizer Condition 1 at n=2000 for all three seeds.
"""
from __future__ import annotations

import copy
import json
import math
from collections import Counter
from pathlib import Path

from src.training.finetune import run_single, write_run_result
from src.utils import (base_argparser, ensure_dir, load_config, resolve_bases,
                       setup_logging, write_json)


def _loss_curve(history: list[dict]) -> list[dict]:
    by_epoch: dict[float, dict] = {}
    for row in history:
        if "epoch" not in row:
            continue
        epoch = float(row["epoch"])
        item = by_epoch.setdefault(epoch, {"epoch": epoch})
        if "loss" in row:
            item["train_loss"] = float(row["loss"])
        if "eval_loss" in row:
            item["val_loss"] = float(row["eval_loss"])
    return [by_epoch[e] for e in sorted(by_epoch)]


def _summary(result: dict, overrides: dict) -> dict:
    counts = Counter(int(x) for x in result["test_predictions"])
    curve = _loss_curve(result["az_training_history"])
    loss_values = [v for row in curve for k, v in row.items() if k.endswith("_loss")]
    return {
        "name": result["diagnostic_name"],
        "condition": result["condition"],
        "seed": result["seed"],
        "overrides": overrides,
        "test_macro_f1": result["test"]["test_macro_f1"],
        "prediction_counts": {str(k): v for k, v in sorted(counts.items())},
        "all_predictions_same": len(counts) == 1,
        "loss_curve": curve,
        "has_nan_or_inf": any(not math.isfinite(v) for v in loss_values),
    }


def main() -> None:
    raise SystemExit(
        "Cancelled: the earlier diagnostic/preflight is absorbed by the frozen "
        "src.training.orchestrate queue.")
    # Historical implementation retained below for provenance; unreachable.
    ap = base_argparser(__doc__)
    args = ap.parse_args()
    setup_logging(args.log_level)
    root_cfg = load_config(args.config)
    if int(root_cfg.training.batch_size) != 32:
        raise SystemExit("Stability diagnostics are locked to training.batch_size=32")

    base = resolve_bases(root_cfg, ["primary"])[0]
    conds = {c["name"]: c for c in root_cfg.conditions}
    mean_artifact = (Path(root_cfg.experiment.artifacts_dir) /
                     f"transplanted__{base.short}__mean_k{root_cfg.transplant.k}_rescaled")
    if not mean_artifact.exists():
        raise SystemExit(f"Missing required mean artifact: {mean_artifact}")

    # logging_strategy changes only instrumentation, not optimization.
    specs = [
        ("step1_mean_collapsed_seed42", "tokenizator", 42, {}, mean_artifact),
        ("step1_mean_healthy_seed13", "tokenizator", 13, {}, mean_artifact),
        ("step2_fp16_off", "tokenizator", 42, {"fp16": False}, mean_artifact),
        ("step2_lr_5e-6", "tokenizator", 42, {"lr": 5e-6}, mean_artifact),
        ("step3_condition1_seed13", "baza", 13, {}, None),
        ("step3_condition1_seed42", "baza", 42, {}, None),
        ("step3_condition1_seed1337", "baza", 1337, {}, None),
    ]
    runs_dir = ensure_dir(Path(root_cfg.experiment.results_dir) /
                          "stability_diagnostics" / "runs")
    summaries = []
    for name, condition_name, seed, overrides, artifact in specs:
        out = runs_dir / f"{name}.json"
        if bool(root_cfg.run.skip_existing) and out.exists():
            with open(out, "r", encoding="utf-8") as f:
                result = json.load(f)
        else:
            cfg = copy.deepcopy(root_cfg)
            cfg.training["logging_strategy"] = "epoch"
            for key, value in overrides.items():
                cfg.training[key] = value
            result = run_single(
                cfg, base, conds[condition_name], 2000, seed,
                transplanted_path=artifact,
                transplant_variant=("mean_k64_rescaled" if artifact else None),
            )
            result["diagnostic_name"] = name
            result["diagnostic_overrides"] = overrides
            write_run_result(result, out)
        summaries.append(_summary(result, overrides))

    c1 = [s["test_macro_f1"] for s in summaries if s["condition"] == "baza"]
    c1_mean = sum(c1) / len(c1)
    if c1_mean > 0.90:
        gate = "STOP_SATURATED"
    elif c1_mean < 0.45:
        gate = "STOP_SUSPECT_NOISE_OR_GENERAL_TRAINING_FAILURE"
    elif 0.60 <= c1_mean <= 0.85:
        gate = "PROCEED"
    else:
        gate = "BORDERLINE"

    write_json({
        "status": "steps_1_to_3_complete_full_run_not_started",
        "reference_size": 2000,
        "batch_size": 32,
        "diagnostics": summaries,
        "condition1": {"values": c1, "mean": c1_mean, "gate": gate},
    }, Path(root_cfg.experiment.results_dir) / "stability_diagnostics.json")


if __name__ == "__main__":
    main()
