"""A100 micro-benchmark — NOT a training run. Two 30-step benchmarks
(Azerbaijani stage, Turkish stage), validation disabled, no test evaluation,
no checkpoint saved, nothing written under results/runs/. Run only after
scripts/a100_bootstrap.sh has passed (test suite + --dry-run green).

Mirrors the RTX 4060 diagnostic's Step 4 benchmark exactly in method (see
HANDOFF.md §12) so the two are comparable, plus a second benchmark for the
Turkish stage (not measured on the 4060) and explicit wall-clock
projections for the actual current queue shapes (not the stale "12-run"
figure — see the projection section below for why).

Usage:
    PYTHONUNBUFFERED=1 python scripts/a100_micro_benchmark.py
"""
from __future__ import annotations

import subprocess
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, ".")

import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainerCallback,
                          TrainingArguments)

from src.data.splits import read_jsonl
from src.training.finetune import ListDataset, compute_metrics, subsample
from src.training.orchestrate import build_queue, tr_stage_cache_summary
from src.utils import load_config, resolve_bases, set_all_seeds, verify_config_hashes

MODEL_PATH = "artifacts/transplanted__xlm15__omp_k64_rescaled"
DATADIR = "artifacts/data"
BENCH_STEPS = 30


class BenchCallback(TrainerCallback):
    def __init__(self):
        self.step_times: list[float] = []

    def on_train_begin(self, args, state, control, **kw):
        self._t0 = time.time()
        self._prev = self._t0

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs or "loss" not in logs:
            return
        now = time.time()
        dt = now - self._prev
        self._prev = now
        self.step_times.append(dt)
        print(f"STEP step={state.global_step} loss={logs['loss']:.4f} "
              f"dt_sec={dt:.3f} cum_wall_sec={now - self._t0:.1f}", flush=True)


def nvidia_smi_snapshot() -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,"
                        "power.max_limit,memory.used,utilization.gpu,temperature.gpu",
         "--format=csv"], capture_output=True, text=True)
    return out.stdout


def stability_verdict(step_times: list[float]) -> str:
    """Mean of the first half (after 2 warmup steps) vs. the second half."""
    warm = step_times[2:]
    mid = len(warm) // 2
    first_half, second_half = warm[:mid], warm[mid:]
    if not first_half or not second_half:
        return "INCONCLUSIVE (too few steps)"
    m1 = sum(first_half) / len(first_half)
    m2 = sum(second_half) / len(second_half)
    ratio = m2 / m1 if m1 else float("inf")
    if ratio > 1.3:
        return f"DEGRADING (2nd-half mean {m2:.3f}s is {ratio:.2f}x the 1st-half mean {m1:.3f}s)"
    return f"STABLE (1st-half mean {m1:.3f}s, 2nd-half mean {m2:.3f}s, ratio {ratio:.2f}x)"


def run_benchmark(label: str, model, tokenizer, train_recs, val_recs, cfg, max_length: int) -> dict:
    args = TrainingArguments(
        output_dir=f"_bench_out_{label}",
        seed=42, data_seed=42,
        learning_rate=float(cfg.training.lr),
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=int(cfg.training.get("eval_batch_size", cfg.training.batch_size * 2)),
        gradient_accumulation_steps=cfg.training.grad_accum,
        max_steps=BENCH_STEPS,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        fp16=bool(cfg.training.fp16) and torch.cuda.is_available(),
        eval_strategy="no",           # validation disabled, per spec
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=1,
        report_to=[],
        disable_tqdm=True,
    )
    cb = BenchCallback()
    trainer = Trainer(
        model=model, args=args,
        train_dataset=ListDataset(train_recs, tokenizer, max_length),
        eval_dataset=ListDataset(val_recs, tokenizer, max_length),  # loaded, unused (eval disabled)
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[cb],
    )

    torch.cuda.reset_peak_memory_stats()
    caught = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        t0 = time.time()
        trainer.train()
        torch.cuda.synchronize()
        total_wall = time.time() - t0
        caught = [str(x.message) for x in w]
    expandable_warning = [m for m in caught if "expandable_segments" in m]

    peak_reserved = torch.cuda.max_memory_reserved()
    total_capacity = torch.cuda.get_device_properties(0).total_memory
    warm = cb.step_times[2:]
    mean_step = sum(warm) / len(warm) if warm else float("nan")
    smi = nvidia_smi_snapshot()

    print(f"\n=== {label} BENCHMARK RESULT ===")
    print(f"steps={len(cb.step_times)}  total_wall_sec={total_wall:.2f}")
    print(f"per-step seconds (all {len(cb.step_times)}): "
          f"min={min(cb.step_times):.3f} max={max(cb.step_times):.3f} "
          f"mean={sum(cb.step_times)/len(cb.step_times):.3f}")
    print(f"per-step seconds (excl. first 2 warmup): mean={mean_step:.3f}")
    print(f"stability: {stability_verdict(cb.step_times)}")
    print(f"peak memory_reserved={peak_reserved/1e9:.2f} GB of "
          f"{total_capacity/1e9:.2f} GB device capacity "
          f"({100*peak_reserved/total_capacity:.1f}%)")
    print(f"expandable_segments UserWarning this run: "
          f"{'YES -> ' + expandable_warning[0] if expandable_warning else 'NO (not emitted)'}")
    print(f"nvidia-smi snapshot:\n{smi}")

    return {
        "label": label, "step_times": cb.step_times, "mean_step_sec": mean_step,
        "total_wall_sec": total_wall, "peak_reserved_bytes": peak_reserved,
        "total_capacity_bytes": total_capacity,
        "expandable_segments_warning": bool(expandable_warning),
    }


def main() -> None:
    cfg = load_config("configs/experiment.yaml")
    verify_config_hashes()
    set_all_seeds(42, cfg.experiment.deterministic)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # ---------------------------------------------------------- AZ-stage benchmark
    az_train_all = read_jsonl(f"{DATADIR}/az_train.jsonl")
    az_val = read_jsonl(f"{DATADIR}/az_val.jsonl")
    n_az_labels = len({r["label"] for r in az_train_all} | {r["label"] for r in az_val})
    az_train = subsample(az_train_all, 2000, seed=42)
    az_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=n_az_labels)
    az_result = run_benchmark("AZERBAIJANI", az_model, tokenizer, az_train, az_val,
                              cfg, cfg.models.max_length)
    del az_model
    torch.cuda.empty_cache()

    # ---------------------------------------------------------- TR-stage benchmark
    tr_recs = read_jsonl(f"{DATADIR}/tr_train.jsonl")
    n_tr_labels = len({r["label"] for r in tr_recs})
    cut = int(len(tr_recs) * 0.9)
    tr_train, tr_val = tr_recs[:cut], tr_recs[cut:]
    tr_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=n_tr_labels)
    tr_result = run_benchmark("TURKISH", tr_model, tokenizer, tr_train, tr_val,
                              cfg, cfg.models.max_length)
    del tr_model
    torch.cuda.empty_cache()

    # ---------------------------------------------------------- wall-clock projections
    bases = resolve_bases(cfg)
    full_queue = build_queue(cfg, bases)
    tr_requests, tr_distinct = tr_stage_cache_summary(full_queue)
    az_only_runs = len(full_queue) - tr_requests

    print("\n=== WALL-CLOCK PROJECTIONS (measured numbers only) ===")
    print(f"Full grid: {len(full_queue)} runs total "
          f"({az_only_runs} AZ-only + {tr_requests} Turkish-bearing, "
          f"{tr_distinct} distinct Turkish checkpoints).")

    az_2000_steps_sec = az_result["mean_step_sec"] * int(cfg.training.max_steps)
    tr_1126_steps_sec = tr_result["mean_step_sec"] * 1126  # measured TR step count, see HANDOFF §13
    full_grid_sec = (az_only_runs * az_2000_steps_sec +
                     tr_requests * (az_2000_steps_sec) +
                     tr_distinct * tr_1126_steps_sec)
    print(f"Projected full {len(full_queue)}-run grid (AZ-stage-rate-only per run, "
          f"plus {tr_distinct} distinct TR-stage computations, EXCLUDES eval overhead): "
          f"{full_grid_sec:.0f}s = {full_grid_sec/3600:.2f}h")

    # "12-run preflight": the figure in run_all.sh's retired transplant_controls.py
    # comment predates the 5-seed amendment (that script used the pre-amendment
    # 3-seed list: 2 methods x 2 bases x 3 seeds = 12) and is now dead code
    # (transplant_controls.main() raises SystemExit; mean/random_coef are the
    # transplant_mean/transplant_random_coef conditions in the unified queue).
    # Reporting the actual current equivalent instead of the stale count.
    controls_queue = build_queue(
        cfg, bases, cond_filter={"transplant_mean", "transplant_random_coef"},
        size_filter={int(cfg.analysis.reference_size)})
    controls_sec = len(controls_queue) * az_2000_steps_sec
    print(f"\nNOTE: 'the 12-run preflight' is stale — it referred to the retired "
          f"src.training.transplant_controls script under the pre-amendment "
          f"3-seed list. Under the current frozen 5-seed queue, the equivalent "
          f"(transplant_mean + transplant_random_coef, both bases, reference "
          f"size {int(cfg.analysis.reference_size)}) is "
          f"{len(controls_queue)} runs, not 12.")
    print(f"Projected {len(controls_queue)}-run controls subset (AZ-stage-rate-only, "
          f"EXCLUDES eval overhead): {controls_sec:.0f}s = {controls_sec/3600:.2f}h")

    # ------------------------------------------------------- VRAM -> streams
    # The binding limit is the brief's ALLOCATION (~10-12 GB per team), not the
    # card's capacity. Measuring peak against `total_memory` answers "does it
    # fit the GPU"; the question that carries an automatic deduction is "does
    # it fit our budget". Answer that one explicitly.
    per_process_gb = max(az_result["peak_reserved_bytes"],
                         tr_result["peak_reserved_bytes"]) / 1e9
    ceiling_gb = float(cfg.run.get("vram_ceiling_gb", 10.0))
    configured_streams = int(cfg.run.get("parallel_streams", 1))
    affordable = max(1, int(ceiling_gb // per_process_gb)) if per_process_gb else 1

    print("\n=== VRAM AGAINST THE TEAM ALLOCATION (not device capacity) ===")
    print(f"Measured per-process peak (max of AZ/TR stages): {per_process_gb:.2f} GB")
    print(f"Team ceiling from run.vram_ceiling_gb:           {ceiling_gb:.2f} GB")
    print(f"Streams that fit the ceiling:                    {affordable}")
    print(f"run.parallel_streams currently configured:       {configured_streams}")
    if per_process_gb > ceiling_gb:
        print("  *** A SINGLE RUN ALREADY EXCEEDS THE ALLOCATION. Do not launch "
              "the grid; re-check gradient accumulation (configs/FROZEN.md).")
    elif affordable < configured_streams:
        print(f"  *** LOWER run.parallel_streams TO {affordable} (or raise the "
              "ceiling only if the brief's allocation actually permits it).")
    print("\nSet these in configs/experiment.yaml from THIS measurement:")
    print(f"  run.vram_per_stream_gb: {per_process_gb:.2f}")
    print(f"  run.parallel_streams:   {min(configured_streams, affordable)}")

    # ------------------------------------------------- projection at N streams
    streams = min(configured_streams, affordable)
    print(f"\n=== GRID PROJECTION AT {streams} STREAM(S), MEASURED s/step ===")
    print(f"AZ stage measured: {az_result['mean_step_sec']:.3f} s/optimizer-step "
          f"(per_device_batch={cfg.training.batch_size} x "
          f"grad_accum={cfg.training.grad_accum})")
    print(f"TR stage measured: {tr_result['mean_step_sec']:.3f} s/optimizer-step")
    for n in sorted({1, 2, 3, streams}):
        if n > affordable:
            continue
        # Concurrency fills kernel-launch idle time, so throughput scales
        # sub-linearly; 0.85 is a deliberately conservative efficiency factor
        # and Tranche A should replace it with the observed number.
        hours = full_grid_sec / 3600.0 / (n * 0.85 if n > 1 else 1.0)
        fits = "FITS ~2-day window" if hours <= 44 else "DOES NOT FIT"
        print(f"  {n} stream(s): {hours:6.1f} h  ({n * per_process_gb:5.2f} GB peak)  {fits}")

    print("\nThese projections exclude validation-pass overhead "
          f"(eval_batch_size={cfg.training.get('eval_batch_size')}, ~10 passes/run) "
          "and TR-cache-hit reuse across resumed grid passes — both make later "
          "passes/runs faster than a cold-cache first pass.")

    from src.utils import write_json
    write_json({
        "az_stage": {k: v for k, v in az_result.items() if k != "step_times"},
        "tr_stage": {k: v for k, v in tr_result.items() if k != "step_times"},
        "per_process_peak_gb": round(per_process_gb, 3),
        "vram_ceiling_gb": ceiling_gb,
        "streams_that_fit_ceiling": affordable,
        "recommended_parallel_streams": streams,
        "recommended_vram_per_stream_gb": round(per_process_gb, 2),
        "full_grid_serial_sec": full_grid_sec,
        "training": {"batch_size": int(cfg.training.batch_size),
                     "grad_accum": int(cfg.training.grad_accum),
                     "effective_batch": int(cfg.training.batch_size) * int(cfg.training.grad_accum)},
    }, Path(cfg.experiment.results_dir) / "hardware_benchmark.json")
    print(f"\nWritten: {Path(cfg.experiment.results_dir) / 'hardware_benchmark.json'}")


if __name__ == "__main__":
    main()
