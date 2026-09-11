#!/usr/bin/env python3
"""Verify the 2026-09-07 amendment WITHOUT training anything.

Run this before renting a GPU and again on the rented box:

    python scripts/verify_amendment_20260907.py

Checks, in order:

  1. Grid size is 164 and the Turkish cache still needs only 30 checkpoints
     (the n=500 Turkish runs must REUSE the cache, not enlarge it — the cache
     key deliberately excludes the Azerbaijani train size).
  2. Schema v4 accepts a well-formed run record and REJECTS every way of
     smuggling in a val/test comparison that compares different weights.
  3. `cross_base_quality` withholds delta BPC when the controls file it found
     measured a different transplant method than the one it is reporting on.

Nothing here touches the GPU, the network, or `results/`; every check is
pure bookkeeping, so a failure means a real defect, never a flaky run.
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _stub_heavy_imports() -> None:
    """Allow the schema check to run on a machine with no torch installed.

    Only used when torch is genuinely absent; on the rented GPU box the real
    library is imported and these stubs never appear.
    """
    try:
        import torch  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (object,), {})
    torch.long = "long"
    torch.tensor = lambda *a, **k: None
    torch.cuda = types.SimpleNamespace(is_available=lambda: False,
                                       empty_cache=lambda: None)
    torch.utils = types.SimpleNamespace(
        data=types.SimpleNamespace(Dataset=type("Dataset", (object,), {})))
    sys.modules["torch"] = torch
    transformers = types.ModuleType("transformers")
    for attr in ["AutoModelForSequenceClassification", "AutoTokenizer",
                 "DataCollatorWithPadding", "Trainer", "TrainerCallback",
                 "TrainingArguments"]:
        setattr(transformers, attr, type(attr, (object,), {}))
    sys.modules["transformers"] = transformers


FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


# ------------------------------------------------------------------ 1. grid
def check_grid(cfg_path: str) -> None:
    from src.training.orchestrate import build_queue, tr_stage_cache_summary
    from src.utils import load_config, resolve_bases, verify_config_hashes

    verify_config_hashes()
    cfg = load_config(cfg_path)
    queue = build_queue(cfg, resolve_bases(cfg))
    requests, distinct = tr_stage_cache_summary(queue)

    print("\n[1] grid shape")
    check("total run count is 164", len(queue) == 164, f"got {len(queue)}")
    check("distinct Turkish checkpoints still 30", distinct == 30, f"got {distinct}")
    check("n=500 Turkish runs reuse the cache", requests == 60,
          f"{requests} requests over {distinct} checkpoints")

    turkish = {c["name"] for c in cfg.conditions if c.get("turkish") != "none"}
    at_500 = {cond["name"] for _b, cond, size, _s in queue if size == 500}
    check("all three Turkish conditions present at n=500",
          turkish <= at_500, f"missing: {sorted(turkish - at_500)}")
    bases_500 = {b.short for b, _c, size, _s in queue
                 if size == 500 and _c["name"] in turkish}
    check("Turkish n=500 covers both bases", bases_500 == {"xlm15", "xlmr"},
          f"got {sorted(bases_500)}")
    check("best-checkpoint selection is on",
          bool(cfg.training.get("load_best_model_at_end")) is True
          and str(cfg.training.get("metric_for_best_model")) == "eval_macro_f1",
          f"load_best={cfg.training.get('load_best_model_at_end')} "
          f"metric={cfg.training.get('metric_for_best_model')}")
    check("early stopping still disabled",
          cfg.training.get("early_stopping_patience") is None,
          str(cfg.training.get("early_stopping_patience")))
    check("eval interval divides the step budget (terminal step measurable)",
          int(cfg.training.max_steps) % int(cfg.training.eval_every_steps) == 0,
          f"{cfg.training.max_steps} % {cfg.training.eval_every_steps}")


# ---------------------------------------------------------------- 2. schema
def _record(eval_split="val", test_f1=None, history=None):
    history = history or [
        {"step": 200, "epoch": 1.0, "eval_loss": 1.0, "eval_macro_f1": 0.3334,
         "train_loss": 0.70, "validation_prediction_counts": {"0": 2791, "1": 0}},
        {"step": 400, "epoch": 2.0, "eval_loss": 0.9, "eval_macro_f1": 0.681,
         "train_loss": 0.60, "validation_prediction_counts": {"0": 1400, "1": 1391}},
        {"step": 600, "epoch": 3.0, "eval_loss": 0.95, "eval_macro_f1": 0.643,
         "train_loss": 0.60, "validation_prediction_counts": {"0": 1380, "1": 1411}},
    ]
    best = max(r["eval_macro_f1"] for r in history)
    best_step = min(r["step"] for r in history if r["eval_macro_f1"] == best)
    terminal = history[-1]
    escaped = any(r["eval_macro_f1"] > 0.40 for r in history)
    fallback = bool(escaped and terminal["eval_macro_f1"] <= 0.40)
    return {
        "run_result_schema_version": 4, "n_labels": 2, "step_history": history,
        "escaped": escaped,
        "escape_step": min((r["step"] for r in history
                            if r["eval_macro_f1"] > 0.40), default=None),
        "selected_step": best_step, "selected_validation_macro_f1": best,
        "terminal_step": terminal["step"],
        "terminal_validation_macro_f1": terminal["eval_macro_f1"],
        "escaped_but_terminal_collapsed": fallback,
        "max_steps": 2000, "completed_steps": 2000, "steps_per_epoch": 63,
        "effective_epochs": 2000 / 63,
        "per_run_verdict": "ESCAPED" if escaped else "NOT_ESCAPED",
        "eval_split": eval_split,
        "training_settings": {"max_steps": 2000, "eval_every_steps": 200,
                              "metric_for_best_model": None,
                              "early_stopping_patience": None,
                              "load_best_model_at_end": False},
        "test_comparison": {
            "eval_split": eval_split,
            "selection": "best_validation_checkpoint",
            "matched_step": best_step,
            "validation_macro_f1_matched": best,
            "validation_macro_f1_selected_max": best,
            "validation_macro_f1_terminal": terminal["eval_macro_f1"],
            "restored_validation_macro_f1": best,
            "selected_step": best_step,
            "terminal_step": terminal["step"],
            "test_macro_f1": test_f1,
            "delta_test_minus_validation_matched": (
                None if test_f1 is None else round(test_f1 - best, 6)),
            "validation_drift_selected_minus_terminal": round(
                best - terminal["eval_macro_f1"], 6),
            "escaped_on_validation": escaped,
            "escaped_but_terminal_collapsed": fallback,
            "estimand_note": "see configs/FROZEN.md 2026-09-07",
        },
    }


def check_schema() -> None:
    from src.training.finetune import (RUN_RESULT_SCHEMA_VERSION,
                                       assert_run_result_schema)

    print("\n[2] run-result schema")
    check("schema version is 4", RUN_RESULT_SCHEMA_VERSION == 4,
          f"got {RUN_RESULT_SCHEMA_VERSION}")

    def accepts(record, label):
        try:
            assert_run_result_schema(record)
            check(label, True)
        except AssertionError as exc:
            check(label, False, str(exc))

    def rejects(record, label):
        try:
            assert_run_result_schema(record)
        except AssertionError:
            check(label, True)
            return
        check(label, False, "accepted a record it should have refused")

    accepts(_record(), "accepts a validation-only run")
    good = _record("test", 0.635)
    accepts(good, "accepts a test run")
    comparison = good["test_comparison"]
    print(f"        selection        = {comparison['selection']}")
    print(f"        matched step     = {comparison['matched_step']} "
          f"(val {comparison['validation_macro_f1_matched']:.3f})")
    print(f"        test - val       = "
          f"{comparison['delta_test_minus_validation_matched']:+.3f}  "
          f"<- true generalisation gap")
    print(f"        val drift        = "
          f"{comparison['validation_drift_selected_minus_terminal']:+.3f}  "
          f"(selected vs terminal, context only)")

    fallback_history = [
        {"step": 200, "epoch": 1.0, "eval_loss": 1.0, "eval_macro_f1": 0.3334,
         "train_loss": 0.7, "validation_prediction_counts": {"0": 2791, "1": 0}},
        {"step": 400, "epoch": 2.0, "eval_loss": 0.9, "eval_macro_f1": 0.62,
         "train_loss": 0.6, "validation_prediction_counts": {"0": 1400, "1": 1391}},
        {"step": 600, "epoch": 3.0, "eval_loss": 1.2, "eval_macro_f1": 0.3334,
         "train_loss": 0.7, "validation_prediction_counts": {"0": 2791, "1": 0}},
    ]
    accepts(_record("test", 0.3334, fallback_history),
            "accepts escaped-then-collapsed and flags it")

    bad = _record("test", 0.635)
    bad["test_comparison"]["delta_test_minus_validation_matched"] = -0.008
    rejects(bad, "refuses a matched delta measured against the terminal step")
    bad = _record("test", 0.635)
    bad["test_comparison"]["restored_validation_macro_f1"] = None
    rejects(bad, "refuses an unverified best-checkpoint restore")
    bad = _record("test", 0.635)
    bad["test_comparison"]["restored_validation_macro_f1"] = 0.643
    rejects(bad, "refuses a restore that landed on different weights")
    bad = _record("test", 0.635)
    bad["test_comparison"]["matched_step"] = 600
    rejects(bad, "refuses a matched step that is not the selected step")
    bad = _record("test", 0.635)
    bad["test_comparison"]["selection"] = "whatever"
    rejects(bad, "refuses an unknown selection rule")
    bad = _record("test", 0.635)
    bad["test_comparison"]["test_macro_f1"] = None
    rejects(bad, "refuses eval_split=test with no test figure")
    bad = _record("val")
    bad["test_comparison"]["test_macro_f1"] = 0.61
    rejects(bad, "refuses a test figure on a validation-only run")
    bad = _record()
    bad["terminal_validation_macro_f1"] = 0.681
    rejects(bad, "refuses a terminal figure silently set to the maximum")
    bad = _record()
    bad["terminal_step"] = 400
    rejects(bad, "refuses a terminal step that is not the last row")
    bad = _record()
    bad["test_comparison"]["validation_macro_f1_matched"] = 0.643
    rejects(bad, "refuses a matched validation figure from another step")
    bad = _record("test", 0.3334, fallback_history)
    bad["escaped_but_terminal_collapsed"] = False
    rejects(bad, "refuses a suppressed collapse-fallback flag")
    bad = _record()
    bad["run_result_schema_version"] = 3
    rejects(bad, "refuses a schema-v3 record")


# -------------------------------------------------------------- 3. controls
def check_controls(cfg_path: str) -> None:
    from src.transplant.cross_base_quality import gather_one_base
    from src.utils import load_config, resolve_bases

    print("\n[3] C1c provenance")
    cfg = load_config(cfg_path)
    base = resolve_bases(cfg, ["primary"])[0]
    tmp = Path(tempfile.mkdtemp())
    (tmp / "runs").mkdir()
    cfg["experiment"]["results_dir"] = str(tmp)

    def controls_payload(tag, delta, top1):
        return {
            "base": "xlm15", "base_role": "primary",
            "transplanted_dir": f"artifacts/transplanted__xlm15__{tag}",
            "transplanted_tag": tag,
            "canonical_transplant_tag": "omp_k64_rescaled",
            "is_canonical_artifact": tag == "omp_k64_rescaled",
            "C1a_identity": {"passed": True},
            "C1c_bpc_base": {"bits_per_character": 9.1,
                             "top1_n_correct": 2, "top1_n_masked": 2323},
            "C1c_bpc_transplanted": {"bits_per_character": 9.1 + delta,
                                     "top1_n_correct": top1,
                                     "top1_n_masked": 1502},
            "C1c_delta_bpc": delta,
        }

    def write(name, payload):
        (tmp / name).write_text(json.dumps(payload), encoding="utf-8")

    # The old clobbering bug's end state: a placebo occupying the base-only name.
    write("controls__xlm15.json",
          controls_payload("mean_k64_rescaled", -1.3820, 2))
    out = gather_one_base(cfg, base)
    check("withholds delta BPC when the file measured another method",
          "C1c_delta_bpc" not in out and "error_controls_tag_mismatch" in out,
          f"reported {out.get('C1c_delta_bpc')}")

    write("controls__xlm15__omp_k64_rescaled.json",
          controls_payload("omp_k64_rescaled", -1.1935, 2))
    out = gather_one_base(cfg, base)
    check("prefers the method-tagged file",
          out.get("controls_measured_tag") == "omp_k64_rescaled",
          str(out.get("controls_source_path")))
    check("reports the correct OMP delta BPC",
          out.get("C1c_delta_bpc") == -1.1935, str(out.get("C1c_delta_bpc")))
    check("carries top-1 alongside delta BPC",
          out.get("C1c_top1_transplanted", {}).get("masked") == 1502,
          str(out.get("C1c_top1_transplanted")))

    for path in tmp.glob("controls__*.json"):
        path.unlink()
    out = gather_one_base(cfg, base)
    check("errors clearly when no controls file exists",
          "error_controls" in out and "C1c_delta_bpc" not in out)


def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/experiment.yaml"
    _stub_heavy_imports()
    check_grid(cfg_path)
    check_schema()
    check_controls(cfg_path)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for name in FAILURES:
            print(f"  - {name}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED — safe to launch the grid.")


if __name__ == "__main__":
    main()
