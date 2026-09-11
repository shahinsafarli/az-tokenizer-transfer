"""Synthetic end-to-end exercise of the complete 164-run frozen grid."""
from __future__ import annotations

import copy

from src.analysis.aggregate import aggregate_cells, load_runs
from src.analysis.report import generate_report
from src.training.finetune import assert_run_result_schema
from src.training.orchestrate import build_queue, execute_queue, queue_label
from src.utils import load_config, resolve_bases


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


def test_complete_frozen_grid_budget_resume_schema_and_report(tmp_path):
    cfg = copy.deepcopy(load_config("configs/experiment.yaml"))
    cfg["experiment"]["results_dir"] = str(tmp_path / "results")
    cfg["experiment"]["figures_dir"] = str(tmp_path / "figures")
    cfg["analysis"]["bootstrap_iters"] = 100
    cfg["run"]["budget_seconds_per_step"] = 0.0
    cfg["run"]["budget_eval_overhead_sec"] = 1.0
    # `fake_run` writes no checkpoint and no model weights, so the real
    # per-checkpoint projection (30 x 1.2 GB) is meaningless here and would
    # make this test pass or fail on the *host's free disk* rather than on
    # code correctness. Zeroed deliberately; the guard itself is covered by
    # test_disk_preflight_refuses_when_projection_exceeds_free_space below.
    cfg["run"]["disk_budget_bytes_per_tr_checkpoint"] = 0
    cfg["run"]["disk_budget_bytes_per_run_overhead"] = 0
    queue = build_queue(cfg, resolve_bases(cfg))
    assert len(queue) == 164
    assert [queue_label(x) for x in queue[:3]] == [
        "base=xlm15 condition=Condition 1 — no transplant n=2000 seed=42",
        "base=xlm15 condition=Condition 1 — no transplant n=2000 seed=1337",
        "base=xlm15 condition=Condition 1 — no transplant n=2000 seed=13",
    ]
    # The 2026-09-07 amendment opened `run.contrast_sizes` to [500, 2000], so
    # the contrast base gained an n=500 block. The size gate is per-base and is
    # applied after the primary base's remaining sizes, which puts that block —
    # 35 runs, including the whole n=500 Turkish comparison on the contrast
    # base — at the very END of the queue. Asserted explicitly because a run
    # cut short loses exactly the half that makes the amendment symmetric.
    assert queue_label(queue[-1]) == (
        "base=xlmr condition=turk_qarisiq n=500 seed=2024")
    assert sum(1 for b, _c, size, _s in queue[-35:]
               if b.short == "xlmr" and size == 500) == 35

    clock = FakeClock()

    def fake_run(config, base, condition, size, seed):
        clock.value += 1.0
        steps = list(range(65, 2000, 65)) + [2000]
        escaped = condition["name"] in {"baza", "tokenizator"}
        f1 = 0.55 if escaped else 0.333
        history = [{
            "step": step,
            "epoch": step / max(1, (size + 31) // 32),
            "eval_loss": 1.0,
            "eval_macro_f1": f1,
            "train_loss": 1.0,
            "validation_prediction_counts": {"0": 5, "1": 5},
        } for step in steps]
        result = {
            "run_result_schema_version": 4,
            "base": base.short,
            "base_role": base.role,
            "condition": condition["name"],
            "condition_id": condition["id"],
            "tokenizer": condition["tokenizer"],
            "turkish": condition["turkish"],
            "train_size": size,
            "seed": seed,
            "n_labels": 2,
            "escaped": escaped,
            "escape_step": 65 if escaped else None,
            "selected_step": 65,
            "selected_validation_macro_f1": f1,
            "terminal_step": steps[-1],
            "terminal_validation_macro_f1": f1,
            "escaped_but_terminal_collapsed": False,
            "test_comparison": {
                "eval_split": "val",
                "selection": "best_validation_checkpoint",
                "matched_step": 65,
                "validation_macro_f1_matched": f1,
                "validation_macro_f1_selected_max": f1,
                "validation_macro_f1_terminal": f1,
                "restored_validation_macro_f1": f1,
                "selected_step": 65,
                "terminal_step": steps[-1],
                "test_macro_f1": None,
                "delta_test_minus_validation_matched": None,
                "validation_drift_selected_minus_terminal": 0.0,
                "escaped_on_validation": escaped,
                "escaped_but_terminal_collapsed": False,
                "estimand_note": "synthetic fixture",
            },
            "max_steps": 2000,
            "completed_steps": 2000,
            "steps_per_epoch": (size + 31) // 32,
            "effective_epochs": 2000 / ((size + 31) // 32),
            "step_history": history,
            "per_run_verdict": "ESCAPED" if escaped else "NOT_ESCAPED",
            "eval_split": "val",
            "training_settings": {
                "max_steps": 2000,
                "eval_every_steps": 65,
                "metric_for_best_model": None,
                "early_stopping_patience": None,
                "load_best_model_at_end": False,
            },
            "runtime_sec": 1.0,
        }
        assert_run_result_schema(result)
        return result

    # Budget admits one synthetic run, then stops at the clean run boundary.
    first = execute_queue(cfg, queue, fake_run, budget_hours=1.5 / 3600,
                          now_fn=clock)
    assert len(first["completed"]) == 1

    # Resume skips that durable result and completes every remaining grid slot.
    second = execute_queue(cfg, queue, fake_run, budget_hours=None, now_fn=clock)
    assert len(second["skipped_existing"]) == 1
    assert len(second["completed"]) == 163
    assert len(load_runs(cfg.experiment.results_dir)) == 164

    sample = load_runs(cfg.experiment.results_dir).iloc[0]
    assert sample["max_steps"] == 2000
    cells = aggregate_cells(load_runs(cfg.experiment.results_dir), cfg)
    placebo = next(r for r in cells if r["condition"] == "transplant_mean")
    assert placebo["escape_rate"] == 0.0
    assert placebo["conditional_macro_f1"] is None
    real = next(r for r in cells if r["condition"] == "tokenizator")
    assert real["conditional_macro_f1"] == 0.55

    report = generate_report(cfg)
    # 36 cells after the 2026-09-07 amendment: 14 at n=2000 (7 conditions x
    # 2 bases), 7 at n=500 on each base, and 4 each at n=10000 and n=20914
    # on the primary base only.
    assert len(report["cells"]) == 36
    assert (tmp_path / "results" / "paper_tables.md").exists()
    assert (tmp_path / "figures" / "escape_rate.png").exists()


def test_disk_preflight_refuses_when_projection_exceeds_free_space(tmp_path):
    """The guard the previous test deliberately disables, exercised on its own.

    An absurd per-checkpoint budget makes the projection exceed any real
    filesystem, so this asserts the refusal without depending on how much
    disk the host happens to have free.
    """
    import pytest

    cfg = copy.deepcopy(load_config("configs/experiment.yaml"))
    cfg["experiment"]["results_dir"] = str(tmp_path / "results")
    cfg["run"]["tr_stage_cache_dir"] = str(tmp_path / "cache")
    cfg["run"]["disk_budget_bytes_per_tr_checkpoint"] = 10**15  # 1 PB per checkpoint
    queue = build_queue(cfg, resolve_bases(cfg))

    def never_called(*_args, **_kwargs):  # pragma: no cover - must not run
        raise AssertionError("execute_queue started an item despite failing pre-flight")

    with pytest.raises(SystemExit, match="DISK PRE-FLIGHT FAILED"):
        execute_queue(cfg, queue, never_called, budget_hours=None)
