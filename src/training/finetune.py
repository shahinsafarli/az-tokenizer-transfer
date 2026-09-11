"""
BİR FINE-TUNING RUN-U:  (şərt, data_həcmi, seed) → metriklər

Şərtlərin pipeline-ı:
    turkish: none       →  AZ fine-tune
    turkish: real       →  TR fine-tune → [BAŞ ATILIR]  → AZ fine-tune
    turkish: scrambled  →  qarışıq TR   → [BAŞ ATILIR]  → AZ fine-tune

[C3] Təsnifat başı (classification head) TR mərhələsindən sonra TAM ATILIR:
`ignore_mismatched_sizes=True` YALNIZ ölçüsü uyğunsuz olan parametrləri
sıfırlayır (xlmr-də bu, `classifier.out_proj`-dur), `classifier.dense` isə
EYNİ ÖLÇÜDƏ olduğu üçün türk mərhələsindən SƏSSİZCƏ keçirdi. Ona görə
`reset_task_head()` encoder-dən KƏNAR bütün üst-səviyyə modulları modelin öz
initializer-i ilə yenidən qurur (bax həmin funksiyanın izahı). Beləliklə
köçürülən HƏQİQƏTƏN yalnız ENCODER GÖVDƏSİDİR.

Baza model rolu (--base primary | contrast) hansı modeldən başlanacağını
təyin edir. Nəticə faylının adı bazanı da daşıyır, ona görə iki baza modelin
run-ları bir-birini üstələmir.

Bu modul birbaşa da çağırıla bilər (debug üçün):
    python -m src.training.finetune --config configs/experiment.yaml \
        --base primary --condition tokenizator --train-size 2000 --seed 42
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Must be set before `import torch` — the CUDA caching allocator reads this
# at first CUDA init. expandable_segments reduces fragmentation-driven
# over-reservation; not a pre-registered scientific parameter (see
# configs/FROZEN.md's Step-3 memory-pressure note).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainerCallback,
                          TrainingArguments)

from src.data.splits import read_jsonl
from src.utils import (BaseModel, RunKey, base_argparser, canonical_transplant_tag,
                       ensure_dir, get_logger, load_config, resolve_bases,
                       set_all_seeds, setup_logging, tee_run_log, transplant_dir,
                       verify_config_hashes, write_json)

log = get_logger(__name__)

# Diagnostic root cause (2026-09-04): a run stalled for 41 minutes without a
# single 65-step validation pass, while nvidia-smi showed
# utilization.gpu=100% and falling temperature/power draw. utilization.gpu
# only reports whether *any* kernel was resident in the sample window — it is
# not a throughput signal and must never again stand in for step-level
# progress evidence. STEP_LOG_EVERY makes that evidence always available.
STEP_LOG_EVERY = 10


class StepTimingCallback(TrainerCallback):
    """Unbuffered, step-indexed progress: global step, loss, dt, cumulative
    wall-clock — logged every ``STEP_LOG_EVERY`` optimizer steps via
    ``logging_steps``, independent of the (possibly much sparser)
    evaluation cadence."""

    def on_train_begin(self, args, state, control, **kwargs):
        self._t_start = time.time()
        self._t_prev = self._t_start

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return  # eval-only / no-op log lines are reported elsewhere
        now = time.time()
        prev = getattr(self, "_t_prev", now)
        start = getattr(self, "_t_start", now)
        self._t_prev = now
        log.info("STEP step=%d loss=%.4f dt_sec=%.2f cum_wall_sec=%.1f",
                 state.global_step, logs["loss"], now - prev, now - start)


# ---------------------------------------------------------------- dataset
class ListDataset(torch.utils.data.Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.enc = tokenizer([r["text"] for r in records],
                             truncation=True, max_length=max_length)
        self.labels = [int(r["label"]) for r in records]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


# Source-stage competence (added 2026-09-08). A Turkish stage that ends
# predicting one class for every Turkish validation example is a FAILED
# intervention, not a clean test of "Turkish warm-up did not help": the
# manipulation never occurred. Several primary-base seeds did exactly this in
# the previous grid (macro-F1 0.1595 on three classes = one constant class).
# The metric was already recorded; nothing made it COUNTABLE downstream, so it
# never reached an aggregate or a table. This flag does, and it deliberately
# does NOT gate the run: dropping failed source stages would be outcome-based
# exclusion. It makes them reportable.
TR_STAGE_DEGENERATE_MAX_F1 = 0.40


def summarise_tr_stage(tr_val_metrics: dict | None, n_tr_labels: int | None = None) -> dict:
    """Flatten source-stage competence into fields an aggregate can group on."""
    if not tr_val_metrics:
        return {"tr_stage_macro_f1": None, "tr_stage_degenerate": None}
    f1 = tr_val_metrics.get("eval_macro_f1")
    f1 = None if f1 is None else float(f1)
    counts = {k: v for k, v in tr_val_metrics.items()
              if k.startswith("eval_prediction_count_class_")}
    constant = None
    if counts:
        nonzero = [v for v in counts.values() if v]
        constant = bool(len(nonzero) <= 1)
    return {
        "tr_stage_macro_f1": f1,
        "tr_stage_prediction_counts": counts or None,
        "tr_stage_constant_prediction": constant,
        "tr_stage_degenerate": (
            None if f1 is None
            else bool(f1 <= TR_STAGE_DEGENERATE_MAX_F1 or constant)),
        "tr_stage_degenerate_threshold": TR_STAGE_DEGENERATE_MAX_F1,
    }


def reset_task_head(model) -> dict:
    """Re-initialise the ENTIRE task head, leaving the encoder untouched.

    Added 2026-09-08. `ignore_mismatched_sizes=True` only re-initialises
    parameters whose SHAPE differs from the checkpoint. That is not the same as
    discarding the head, and for the contrast base it demonstrably is not:
    `XLMRobertaClassificationHead` is `dense` (hidden -> hidden, shape
    INDEPENDENT of num_labels) followed by `out_proj` (hidden -> num_labels).
    Going from the 3-label Turkish stage to the 2-label Azerbaijani stage
    re-initialises `out_proj` and silently CARRIES `dense` OVER — verified by
    marking `dense`, saving a 3-label checkpoint and reloading it with
    num_labels=2: the marker survives, and transformers' own warning mentions
    only `out_proj`. So a Turkish-trained task projection was leaking into the
    Azerbaijani stage, which contradicts this module's own docstring ("only the
    encoder body transfers") and confounds encoder transfer with classifier
    transfer in exactly the arms the M1/M2 comparison depends on.

    Everything that is not the encoder is the head: `base_model_prefix` names
    the encoder child ('roberta' for xlmr, 'transformer' for xlm15), so every
    OTHER top-level child is re-initialised with the model's own initialiser.
    Applied to EVERY condition, not only the Turkish ones, so the head is
    constructed identically everywhere and no condition gets a differently-aged
    head. Deterministic: `set_all_seeds` has already run.
    """
    prefix = getattr(model, "base_model_prefix", None)
    modules, n_params = [], 0
    for name, child in model.named_children():
        if name == prefix:
            continue
        child.apply(model._init_weights)
        modules.append(name)
        n_params += sum(p.numel() for p in child.parameters())
    if not modules:
        raise AssertionError(
            f"No task head found outside base_model_prefix={prefix!r}; refusing "
            "to train a model whose head could not be verified as reset")
    return {"modules": modules, "n_params": int(n_params),
            "encoder_prefix": prefix}


def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
    }
    # Trainer prefixes these with ``eval_``. Persisting them in log_history lets
    # us identify majority-class collapse on every validation epoch.
    for label in range(int(logits.shape[-1])):
        metrics[f"prediction_count_class_{label}"] = int(np.sum(preds == label))
    return metrics


def subsample(records, n, seed):
    """Sinif paylanmasını qoruyaraq n nümunə seçir."""
    if n >= len(records):
        return records
    rng = np.random.default_rng(seed)
    by_label = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)

    out, labels = [], sorted(by_label)
    per = max(1, n // len(labels))
    for lab in labels:
        items = by_label[lab]
        idx = rng.choice(len(items), size=min(per, len(items)), replace=False)
        out.extend(items[i] for i in idx)

    # qalanı təsadüfi doldur
    if len(out) < n:
        chosen = {id(o) for o in out}
        rest = [r for r in records if id(r) not in chosen]
        if rest:
            extra = rng.choice(len(rest), size=min(n - len(out), len(rest)), replace=False)
            out.extend(rest[i] for i in extra)
    rng.shuffle(out)
    return out[:n]


# ---------------------------------------------------------------- training
def _train_stage(model, tokenizer, train_recs, val_recs, cfg, seed,
                 workdir: Path, tag: str, *, az_stage: bool):
    """Train one stage; only AZ uses the frozen, size-invariant step budget."""
    if az_stage:
        eval_strategy = "steps"
        eval_steps = int(cfg.training.eval_every_steps)
        max_steps = int(cfg.training.max_steps)
        epochs = 1.0  # ignored by transformers when max_steps > 0
    else:
        eval_strategy = "epoch"
        eval_steps = None
        max_steps = -1
        epochs = float(cfg.training.epochs_tr)
    eval_batch_size = int(cfg.training.get("eval_batch_size", cfg.training.batch_size * 2))
    # ---- best-validation checkpoint selection, AZ STAGE ONLY (2026-09-07) ----
    # Standard protocol: select the checkpoint on validation, then report THAT
    # checkpoint's held-out test score. Until today `save_strategy: "no"` kept
    # no checkpoint, so the only weights that survived training were the
    # final-step ones and a test evaluation could not see the selected model.
    #
    # This changes WHICH WEIGHTS ARE KEPT, not how they are produced: the
    # optimisation trajectory, RNG consumption, gradients and every logged
    # validation figure are bit-identical to before. `save_only_model=True`
    # skips optimizer/scheduler/RNG state (we never resume from a checkpoint),
    # so a saved checkpoint is just the model and the writes stay small.
    #
    # The TURKISH stage keeps `save_strategy: "no"` deliberately. It is an
    # intermediate stage whose product is the END of Turkish training, not its
    # best Turkish checkpoint; selecting there would silently redefine what the
    # Turkish arm means, and it is what `_tr_cache_spec` records.
    if az_stage:
        select_metric = str(cfg.training.get("metric_for_best_model")
                            or "eval_macro_f1")
        select_best = bool(cfg.training.get("load_best_model_at_end", True))
        save_strategy = "steps" if select_best else "no"
        save_steps = eval_steps if select_best else None
    else:
        select_metric = None
        select_best = False
        save_strategy = "no"
        save_steps = None
    args = TrainingArguments(
        output_dir=str(workdir / tag),
        seed=seed,
        data_seed=seed,
        learning_rate=float(cfg.training.lr),
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        dataloader_num_workers=int(cfg.training.get("dataloader_num_workers", 0)),
        gradient_accumulation_steps=cfg.training.grad_accum,
        num_train_epochs=epochs,
        max_steps=max_steps,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        fp16=bool(cfg.training.fp16) and torch.cuda.is_available(),
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=(1 if select_best else None),
        save_only_model=select_best,
        load_best_model_at_end=select_best,
        metric_for_best_model=select_metric,
        greater_is_better=(True if select_best else None),
        # Step-level logging cadence is intentionally decoupled from eval
        # cadence: STEP_LOG_EVERY=10 gives progress evidence far more often
        # than the (frozen, possibly sparse) validation interval, in both
        # the Turkish and Azerbaijani stages.
        logging_strategy="steps",
        logging_steps=STEP_LOG_EVERY,
        report_to=[],
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=ListDataset(train_recs, tokenizer, cfg.models.max_length),
        eval_dataset=ListDataset(val_recs, tokenizer, cfg.models.max_length),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[StepTimingCallback()],
    )
    trainer.train()
    # Add the terminal point when the step budget is not divisible by the
    # evaluation interval, so every completed AZ run ends on a measured step.
    #
    # MUST NOT FIRE while best-checkpoint selection is on: `trainer.train()`
    # has already restored the BEST weights by this point, so an evaluation
    # here would append the selected model's score to `step_history` under the
    # terminal step number — a row describing weights that step never held.
    # With max_steps=2000 and eval_every_steps=200 the terminal step is always
    # already measured, so this is a guard against a future interval change,
    # not a live branch.
    if az_stage:
        eval_steps_seen = {
            int(row.get("step", -1)) for row in trainer.state.log_history
            if "eval_macro_f1" in row
        }
        if int(trainer.state.global_step) not in eval_steps_seen:
            if select_best:
                raise SystemExit(
                    f"max_steps ({max_steps}) is not a multiple of "
                    f"eval_every_steps ({eval_steps}), so the terminal step is "
                    "unmeasured — but best-checkpoint selection has already "
                    "restored the selected weights, so it cannot be measured "
                    "now without mislabelling step_history. Set "
                    "eval_every_steps to a divisor of max_steps.")
            trainer.evaluate()
    return trainer


# v4 (2026-09-07): adds the terminal-step validation figure and the
# `test_comparison` block. Rationale: `selected_validation_macro_f1` is a
# MAXIMUM over evaluation points, whereas a test evaluation can only ever see
# the FINAL-step model (`save_strategy: "no"` keeps no checkpoint). Comparing
# those two directly is comparing two different sets of weights under two
# different rules, and the max is upward-biased by construction, so the
# comparison manufactures a val->test "drop" that is pure definition. v4 makes
# the matched-estimand pair a REQUIRED field so the interpretable comparison is
# always on hand and the misleading one is explicitly labelled as such.
RUN_RESULT_SCHEMA_VERSION = 4
RUN_RESULT_REQUIRED_KEYS = {
    "run_result_schema_version",
    "escaped",
    "escape_step",
    "selected_step",
    "selected_validation_macro_f1",
    "terminal_step",
    "terminal_validation_macro_f1",
    "escaped_but_terminal_collapsed",
    "test_comparison",
    "max_steps",
    "steps_per_epoch",
    "effective_epochs",
    "step_history",
    "per_run_verdict",
    "eval_split",
    "training_settings",
}
ESCAPE_THRESHOLD = 0.40  # pre-registered collapse diagnostic (configs/FROZEN.md)
STEP_HISTORY_REQUIRED_KEYS = {
    "step",
    "eval_loss",
    "eval_macro_f1",
    "validation_prediction_counts",
}
TRAINING_SETTINGS_REQUIRED_KEYS = {
    "max_steps",
    "eval_every_steps",
    "metric_for_best_model",
    "early_stopping_patience",
    "load_best_model_at_end",
}


def _step_training_evidence(trainer, n_labels: int, train_size: int, cfg) -> dict:
    """Extract step-indexed evidence and select by max validation macro-F1."""
    raw_history = list(trainer.state.log_history)
    train_loss_by_step: dict[int, float] = {}
    eval_rows: list[dict] = []

    for row in raw_history:
        if "loss" in row:
            train_loss_by_step[int(row.get("step", -1))] = float(row["loss"])
        if "eval_loss" in row:
            eval_rows.append(row)

    step_history = []
    for row in eval_rows:
        step = int(row["step"])
        counts = {}
        for label in range(n_labels):
            key = f"eval_prediction_count_class_{label}"
            if key not in row:
                raise AssertionError(f"Trainer history is missing {key} at step {step}")
            counts[str(label)] = int(row[key])
        step_history.append({
            "step": step,
            "epoch": float(row.get("epoch", 0.0)),
            "eval_loss": float(row["eval_loss"]),
            "eval_macro_f1": float(row["eval_macro_f1"]),
            "train_loss": train_loss_by_step.get(step),
            "validation_prediction_counts": counts,
        })

    if not step_history:
        raise AssertionError("AZ training produced no validation history")
    best_f1 = max(row["eval_macro_f1"] for row in step_history)
    selected_step = min(
        row["step"] for row in step_history if row["eval_macro_f1"] == best_f1)
    escaped_rows = [row for row in step_history
                    if row["eval_macro_f1"] > ESCAPE_THRESHOLD]
    # The TERMINAL row is the only validation figure that describes the same
    # weights a test evaluation can see, so it is recorded as a first-class
    # field rather than left for a downstream reader to dig out of the history.
    terminal_row = step_history[-1]
    completed_steps = int(trainer.state.global_step)
    steps_per_epoch = int(math.ceil(
        train_size / (int(cfg.training.batch_size) * int(cfg.training.grad_accum))))
    escaped = bool(escaped_rows)
    complete = completed_steps >= int(cfg.training.max_steps)
    return {
        "escaped": escaped,
        "escape_step": min(row["step"] for row in escaped_rows) if escaped else None,
        "selected_step": selected_step,
        "selected_validation_macro_f1": best_f1,
        "terminal_step": int(terminal_row["step"]),
        "terminal_validation_macro_f1": float(terminal_row["eval_macro_f1"]),
        # A seed can cross the escape threshold mid-training and fall back into
        # the collapse basin by the terminal step. Such a run is ESCAPED (the
        # criterion is "at any evaluation point") while its terminal — and
        # therefore its test — figure is a collapsed one. That is not an
        # inconsistency to explain away in prose; it is flagged here so the
        # aggregation can count these seeds explicitly.
        "escaped_but_terminal_collapsed": bool(
            escaped and terminal_row["eval_macro_f1"] <= ESCAPE_THRESHOLD),
        "escape_threshold": ESCAPE_THRESHOLD,
        "max_steps": int(cfg.training.max_steps),
        "completed_steps": completed_steps,
        "steps_per_epoch": steps_per_epoch,
        "effective_epochs": completed_steps / steps_per_epoch,
        "step_history": step_history,
        "per_run_verdict": ("ESCAPED" if escaped else
                            "NOT_ESCAPED" if complete else "UNRESOLVED"),
        # Recorded from the CONFIG, not hard-coded: these three used to be
        # literals here, which meant the run file asserted
        # `load_best_model_at_end: false` no matter what the trainer actually
        # did. A provenance field that cannot disagree with reality is not
        # provenance.
        "training_settings": {
            "max_steps": int(cfg.training.max_steps),
            "eval_every_steps": int(cfg.training.eval_every_steps),
            "metric_for_best_model": cfg.training.get("metric_for_best_model"),
            "early_stopping_patience": cfg.training.get(
                "early_stopping_patience"),
            "load_best_model_at_end": bool(
                cfg.training.get("load_best_model_at_end", True)),
        },
        "az_training_history": raw_history,
    }


def assert_run_result_schema(result: dict) -> None:
    """Fail closed before a run record with incomplete evidence is persisted."""
    missing = RUN_RESULT_REQUIRED_KEYS - set(result)
    if missing:
        raise AssertionError(f"Run result missing required keys: {sorted(missing)}")
    if result["run_result_schema_version"] != RUN_RESULT_SCHEMA_VERSION:
        raise AssertionError("Unexpected run_result_schema_version")
    settings_missing = TRAINING_SETTINGS_REQUIRED_KEYS - set(result["training_settings"])
    if settings_missing:
        raise AssertionError(
            f"Run training_settings missing required keys: {sorted(settings_missing)}")
    if not isinstance(result["escaped"], bool):
        raise AssertionError("escaped must be bool")
    if not result["step_history"]:
        raise AssertionError("step_history must not be empty")
    if result["per_run_verdict"] not in {"ESCAPED", "NOT_ESCAPED", "UNRESOLVED"}:
        raise AssertionError("invalid per_run_verdict")

    expected_classes = {str(i) for i in range(int(result["n_labels"]))}
    observed_steps = set()
    previous_step = -1
    for i, row in enumerate(result["step_history"]):
        row_missing = STEP_HISTORY_REQUIRED_KEYS - set(row)
        if row_missing:
            raise AssertionError(
                f"step_history[{i}] missing required keys: {sorted(row_missing)}")
        step = int(row["step"])
        if step <= previous_step:
            raise AssertionError("step_history must be strictly step ordered")
        previous_step = step
        observed_steps.add(step)
        counts = row["validation_prediction_counts"]
        if set(counts) != expected_classes:
            raise AssertionError(
                f"step_history[{i}] validation classes are incomplete")
        if any(not isinstance(value, int) or value < 0 for value in counts.values()):
            raise AssertionError("validation prediction counts must be non-negative ints")
    if int(result["selected_step"]) not in observed_steps:
        raise AssertionError("selected_step must identify a point in step_history")
    escaped_steps = [r["step"] for r in result["step_history"]
                     if r["eval_macro_f1"] > 0.40]
    if result["escaped"] != bool(escaped_steps):
        raise AssertionError("escaped disagrees with step_history")
    expected_escape = min(escaped_steps) if escaped_steps else None
    if result["escape_step"] != expected_escape:
        raise AssertionError("escape_step disagrees with step_history")

    # ---- v4: the matched-estimand pair must describe the same weights ----
    terminal = result["step_history"][-1]
    if int(result["terminal_step"]) != int(terminal["step"]):
        raise AssertionError("terminal_step is not the last step in step_history")
    if float(result["terminal_validation_macro_f1"]) != float(terminal["eval_macro_f1"]):
        raise AssertionError(
            "terminal_validation_macro_f1 disagrees with the last step_history row")
    expected_fallback = bool(
        result["escaped"] and terminal["eval_macro_f1"] <= ESCAPE_THRESHOLD)
    if bool(result["escaped_but_terminal_collapsed"]) != expected_fallback:
        raise AssertionError(
            "escaped_but_terminal_collapsed disagrees with step_history")

    comparison = result["test_comparison"]
    if not isinstance(comparison, dict):
        raise AssertionError("test_comparison must be a dict")
    if comparison.get("eval_split") != result["eval_split"]:
        raise AssertionError("test_comparison.eval_split disagrees with eval_split")

    selection = comparison.get("selection")
    if selection not in {"best_validation_checkpoint", "final_step"}:
        raise AssertionError(
            "test_comparison.selection must be 'best_validation_checkpoint' "
            "or 'final_step'")
    # The matched pair must name the step the test evaluation actually saw.
    if selection == "best_validation_checkpoint":
        expected_step = int(result["selected_step"])
        expected_val = float(result["selected_validation_macro_f1"])
        restored = comparison.get("restored_validation_macro_f1")
        if restored is None:
            raise AssertionError(
                "best-checkpoint selection claimed but the restore was never "
                "verified (restored_validation_macro_f1 is null)")
        if abs(float(restored) - expected_val) > 1e-4:
            raise AssertionError(
                "restored_validation_macro_f1 does not match the selected "
                "step's figure — the checkpoint in memory is not the selected "
                "one, so any test number describes the wrong weights")
    else:
        expected_step = int(result["terminal_step"])
        expected_val = float(result["terminal_validation_macro_f1"])
    if comparison.get("matched_step") != expected_step:
        raise AssertionError(
            f"test_comparison.matched_step must be {expected_step} for "
            f"selection='{selection}'")
    if float(comparison.get("validation_macro_f1_matched")) != expected_val:
        raise AssertionError(
            "test_comparison.validation_macro_f1_matched is not the validation "
            "figure of the matched step")

    if result["eval_split"] == "test":
        if comparison.get("test_macro_f1") is None:
            raise AssertionError(
                "eval_split='test' but test_comparison carries no test_macro_f1")
        expected_matched = round(
            float(comparison["test_macro_f1"]) - expected_val, 6)
        if comparison.get("delta_test_minus_validation_matched") != expected_matched:
            raise AssertionError(
                "test_comparison matched delta is not test minus the matched "
                "validation figure")
    elif comparison.get("test_macro_f1") is not None:
        raise AssertionError(
            "eval_split='val' must not carry a test_macro_f1")


def write_run_result(result: dict, path: str | Path) -> None:
    """The only supported writer for per-run result JSON files."""
    assert_run_result_schema(result)
    write_json(result, path)


TR_CACHE_SCHEMA = 1


def _tr_cache_spec(cfg, base: BaseModel, condition: dict, model_path: str,
                   seed: int, tr_data_path: Path) -> dict:
    """Everything that can change the reusable Turkish encoder checkpoint."""
    stat = tr_data_path.stat()
    return {
        "schema": TR_CACHE_SCHEMA,
        "base": base.short,
        "base_model": base.hf_id,
        "model_path": str(model_path),
        "tokenizer": condition["tokenizer"],
        "turkish": condition["turkish"],
        "seed": int(seed),
        "tr_data": {
            "path": str(tr_data_path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "deterministic": bool(cfg.experiment.deterministic),
        "max_length": int(cfg.models.max_length),
        "training": {
            "lr": float(cfg.training.lr),
            "epochs_tr": float(cfg.training.epochs_tr),
            "batch_size": int(cfg.training.batch_size),
            "grad_accum": int(cfg.training.grad_accum),
            "warmup_ratio": float(cfg.training.warmup_ratio),
            "weight_decay": float(cfg.training.weight_decay),
            "fp16": bool(cfg.training.fp16),
            "early_stopping_patience": None,
            "eval_strategy": "epoch",
            "metric": str(cfg.training.metric),
            "metric_for_best_model": None,
            "load_best_model_at_end": False,
        },
    }


def _tr_cache_path(cfg, spec: dict) -> Path:
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    readable = (f"base={spec['base']}__tok={spec['tokenizer']}__"
                f"tr={spec['turkish']}__seed={spec['seed']}__{digest}")
    root = Path(cfg.run.get("tr_stage_cache_dir", "artifacts/tr_stage_cache"))
    return root / readable


def _load_cached_tr_stage(cache_dir: Path, spec: dict) -> dict | None:
    manifest = cache_dir / "cache_manifest.json"
    if not manifest.exists() or not (cache_dir / "config.json").exists():
        return None
    try:
        with open(manifest, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return saved if saved.get("spec") == spec else None


def _publish_tr_stage(trainer, tokenizer, cache_dir: Path, spec: dict,
                      tr_val_metrics: dict) -> None:
    """Publish last: an interrupted save never counts as a cache hit."""
    ensure_dir(cache_dir.parent)
    staging = Path(tempfile.mkdtemp(prefix=cache_dir.name + ".tmp-",
                                    dir=str(cache_dir.parent)))
    try:
        trainer.save_model(str(staging))
        tokenizer.save_pretrained(str(staging))
        write_json({"spec": spec, "tr_val_metrics": tr_val_metrics},
                   staging / "cache_manifest.json")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        os.replace(staging, cache_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def resolve_model_path(cfg, base: BaseModel, condition: dict,
                       transplanted_path: str | Path | None = None,
                       transplant_variant: str | None = None
                       ) -> tuple[str, str | None]:
    """Hansi cekilerden baslanir — orijinal baza, yoxsa transplant artefakti.

    `run_single` ve `build_tr_stage_only` HER IKISI bunu cagirir: Faza 1-in
    qurdugu checkpoint Faza 2-nin gozledigi ile eyni model yolundan
    baslamalidir, eks halda `_tr_cache_spec` (o, `model_path`-i acara daxil
    edir) kes promahi verer ve Faza 2 yalniz-oxunan kesde dayanar.
    """
    if transplanted_path is not None:
        if condition["tokenizer"] == "original":
            raise ValueError(
                "transplanted_path is only valid for a transplanted-tokenizer condition")
        model_path = str(transplanted_path)
        if not Path(model_path).exists():
            raise SystemExit(f"{model_path} tapilmadi")
        return model_path, transplant_variant
    if condition.get("transplant", "none") == "none" and condition["tokenizer"] == "original":
        return base.hf_id, transplant_variant
    # `canonical_transplant_tag()` PRE-REGISTERED `rescale_reconstructed`
    # qerarini nezere alir (bax HANDOFF 3.16) — `default_tag()` DEYIL,
    # cunki o, hemise no-rescale ablasiya-baseline tag-ini qaytarir.
    tag = condition.get("transplant_tag") or canonical_transplant_tag(cfg)
    model_path = str(transplant_dir(cfg, base.short, tag))
    if not Path(model_path).exists():
        raise SystemExit(
            f"{model_path} tapilmadi — evvelce "
            f"`python -m src.transplant.build --bases {base.role}` "
            f"(ve lazimdirsa `src.transplant.build_rescale_variant`) isledin.")
    return model_path, (transplant_variant or tag)


def _run_tr_stage(cfg, base: BaseModel, condition: dict, model_path: str,
                  seed: int, tokenizer, workdir: Path, *,
                  readonly_cache: bool = False) -> tuple[str, dict]:
    """Turk ara-merhelesi: kesden oxu, yoxdursa qur (icaze varsa).

    `readonly_cache=True` (Faza 2) kes PROMAHINDA qurmaga cehd ETMIR — LOUD
    dayanir. Paralel axinlarda "yoxdursa qur" davranisi mehz iki prosesin
    eyni yola yazmasina aparan yoldur; Faza 1 onsuz da her seyi qurmus
    olmalidir, ona gore promah burada bir SEHVDIR, is deyil.
    """
    datadir = Path(cfg.experiment.artifacts_dir) / "data"
    fname = ("tr_train_scrambled.jsonl" if condition["turkish"] == "scrambled"
             else "tr_train.jsonl")
    tr_data_path = datadir / fname
    tr_recs = read_jsonl(tr_data_path)
    n_tr_labels = len({r["label"] for r in tr_recs})
    cut = int(len(tr_recs) * 0.9)
    tr_train, tr_val = tr_recs[:cut], tr_recs[cut:]

    use_cache = bool(cfg.run.get("cache_turkish_stages", True))
    cache_spec = _tr_cache_spec(cfg, base, condition, model_path, seed, tr_data_path)
    cache_dir = _tr_cache_path(cfg, cache_spec)
    cached = _load_cached_tr_stage(cache_dir, cache_spec) if use_cache else None

    stage_info: dict = {}
    if cached is not None:
        log.info("STAGE=turkish_intermediate cache_hit=true ts=%s path=%s",
                 datetime.now(timezone.utc).isoformat(), cache_dir)
        stage_info["tr_val_metrics"] = cached["tr_val_metrics"]
        stage_info.update(summarise_tr_stage(cached["tr_val_metrics"], n_tr_labels))
        stage_info["tr_cache_hit"] = True
        stage_info["tr_cache_path"] = str(cache_dir)
        if stage_info.get("tr_stage_degenerate"):
            log.warning(
                "SOURCE STAGE DEGENERATE (cache hit): tr macro-F1=%.4f "
                "constant_prediction=%s — this run's Turkish manipulation did "
                "NOT take effect. Recorded, not excluded.",
                stage_info["tr_stage_macro_f1"],
                stage_info.get("tr_stage_constant_prediction"))
        return str(cache_dir), stage_info

    if readonly_cache:
        raise SystemExit(
            "PHASE-2 CACHE MISS: turk checkpointi kesde yoxdur:\n"
            f"  {cache_dir}\n"
            "Faza 2 kesi YALNIZ-OXUNAN acir. Promah o demekdir ki, Faza 1 bu "
            "spesifikasiyani qurmayib (ve ya konfiq Faza 1-den sonra deyisib). "
            "Burada qurmaq paralel axinlarin eyni yola yazmasi riskini geri "
            "getirerdi — dayanilir.")

    log.info("STAGE=turkish_intermediate start ts=%s n=%d n_classes=%d",
             datetime.now(timezone.utc).isoformat(), len(tr_train), n_tr_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=n_tr_labels)
    tr_t0 = time.time()
    tr_trainer = _train_stage(model, tokenizer, tr_train, tr_val, cfg, seed,
                              workdir, "tr", az_stage=False)
    tr_val_metrics = tr_trainer.evaluate()
    stage_info["tr_stage_runtime_sec"] = round(time.time() - tr_t0, 1)
    stage_info["tr_val_metrics"] = tr_val_metrics
    stage_info.update(summarise_tr_stage(tr_val_metrics, n_tr_labels))
    stage_info["tr_cache_hit"] = False
    if stage_info.get("tr_stage_degenerate"):
        log.warning(
            "SOURCE STAGE DEGENERATE: tr macro-F1=%.4f constant_prediction=%s "
            "— the Turkish manipulation did NOT take effect for this seed. "
            "Recorded, not excluded.",
            stage_info["tr_stage_macro_f1"],
            stage_info.get("tr_stage_constant_prediction"))
    log.info("STAGE=turkish_intermediate end ts=%s runtime_sec=%.1f",
             datetime.now(timezone.utc).isoformat(), stage_info["tr_stage_runtime_sec"])

    if use_cache:
        _publish_tr_stage(tr_trainer, tokenizer, cache_dir, cache_spec, tr_val_metrics)
        stage_info["tr_cache_path"] = str(cache_dir)
        az_init_from = str(cache_dir)
    else:
        tr_ckpt = workdir / "tr_final"
        tr_trainer.save_model(str(tr_ckpt))
        tokenizer.save_pretrained(str(tr_ckpt))
        az_init_from = str(tr_ckpt)

    # Step-2 diagnostic finding (2026-09-04): `tr_trainer` and its
    # model/optimizer were previously left as live locals for the rest of
    # run_single(), so the Turkish-stage model + AdamW state stayed resident
    # on the GPU throughout the whole AZ stage — roughly doubling peak VRAM
    # for any Turkish-bearing condition. Release explicitly.
    del tr_trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return az_init_from, stage_info


def build_tr_stage_only(cfg, spec: dict) -> dict:
    """FAZA 1 giris noqtesi: YALNIZ turk checkpointini qurur, AZ merhelesi YOX.

    Serial cagirilir (bax src/training/phases.py) — bu funksiyanin paralel
    isledilmesi mehz aradan qaldirmaq istediyimiz yarisi geri getirer.
    """
    base = next(b for b in resolve_bases(cfg) if b.short == spec["base"])
    condition = next(c for c in cfg.conditions if c["name"] == spec["condition"])
    seed = int(spec["seed"])
    set_all_seeds(seed, cfg.experiment.deterministic)
    model_path, _variant = resolve_model_path(cfg, base, condition)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    workdir = Path(tempfile.mkdtemp(prefix="phase1_"))
    try:
        cache_path, stage_info = _run_tr_stage(
            cfg, base, condition, model_path, seed, tokenizer, workdir,
            readonly_cache=False)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return {
        "path": cache_path,
        "model_path": model_path,
        "cache_hit": bool(stage_info.get("tr_cache_hit")),
        "tr_stage_runtime_sec": stage_info.get("tr_stage_runtime_sec"),
        "status": "CACHED" if stage_info.get("tr_cache_hit") else "BUILT",
    }


def run_single(cfg, base: BaseModel, condition: dict,
               train_size: int, seed: int,
               transplanted_path: str | Path | None = None,
               transplant_variant: str | None = None,
               eval_split: str = "val", allow_test_eval: bool = False,
               test_ledger_path: str | Path | None = None,
               streams_active: int = 1,
               readonly_tr_cache: bool = False) -> dict:
    """Bir run-ı ucdan-uca icra edir və metrikləri qaytarır."""
    t0 = time.time()
    set_all_seeds(seed, cfg.experiment.deterministic)

    datadir = Path(cfg.experiment.artifacts_dir) / "data"
    az_train_all = read_jsonl(datadir / "az_train.jsonl")
    az_val = read_jsonl(datadir / "az_val.jsonl")
    az_test = None
    if eval_split not in {"val", "test"}:
        raise ValueError("eval_split must be 'val' or 'test'")
    if eval_split == "test":
        if not allow_test_eval:
            raise PermissionError(
                "Test evaluation requires eval_split='test' and explicit allow_test_eval=True")
        az_test = read_jsonl(datadir / "az_test.jsonl")
    n_az_labels = len({r["label"] for r in az_train_all} |
                      {r["label"] for r in az_val})

    # ---- model yolu: orijinal, yoxsa transplant olunmuş?
    #      Transplant artefaktı BAZA MODELƏ SPESİFİKDİR (embedding fəzası fərqlidir).
    model_path, transplant_variant = resolve_model_path(
        cfg, base, condition, transplanted_path, transplant_variant)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    workdir = Path(tempfile.mkdtemp(prefix="ft_"))
    stage_info = {}

    try:
        # ==================================================== TURK MERHELESI
        if condition["turkish"] != "none":
            az_init_from, stage_info = _run_tr_stage(
                cfg, base, condition, model_path, seed, tokenizer, workdir,
                readonly_cache=readonly_tr_cache)
        else:
            az_init_from = model_path

        # ==================================================== AZƏRBAYCAN MƏRHƏLƏSİ
        az_train = subsample(az_train_all, train_size, seed)
        log.info("  [AZ mərhələsi] %d nümunə | %d sinif | init=%s",
                 len(az_train), n_az_labels,
                 "TR checkpoint" if condition["turkish"] != "none" else "baza")

        # [C3] BAŞ ATILIR — ignore_mismatched_sizes yalnız uyğunsuz başı sıfırlayır,
        #      encoder gövdəsi olduğu kimi köçürülür.
        model = AutoModelForSequenceClassification.from_pretrained(
            az_init_from, num_labels=n_az_labels, ignore_mismatched_sizes=True)
        head_reset = reset_task_head(model)
        log.info("Task head reset: modules=%s params=%d (encoder=%r untouched)",
                 head_reset["modules"], head_reset["n_params"],
                 head_reset["encoder_prefix"])

        log.info("STAGE=azerbaijani start ts=%s n=%d n_classes=%d",
                 datetime.now(timezone.utc).isoformat(), len(az_train), n_az_labels)
        az_t0 = time.time()
        az_trainer = _train_stage(model, tokenizer, az_train, az_val, cfg, seed,
                                  workdir, "az", az_stage=True)
        az_runtime_sec = round(time.time() - az_t0, 1)
        log.info("STAGE=azerbaijani end ts=%s runtime_sec=%.1f",
                 datetime.now(timezone.utc).isoformat(), az_runtime_sec)
        training_evidence = _step_training_evidence(
            az_trainer, n_az_labels, len(az_train), cfg)

        # ---- prove the selected checkpoint is the one now in memory --------
        # `load_best_model_at_end` restoring silently no-op would make every
        # test number in the grid describe the wrong weights, with nothing in
        # the output to reveal it. So the restored model is re-scored on
        # VALIDATION and checked against the recorded maximum. Evaluation is
        # deterministic at fixed weights (no dropout, sequential sampler), so
        # agreement is exact up to float noise; disagreement means the restore
        # did not happen and the run fails here rather than producing a
        # plausible wrong result. `_step_training_evidence` has already taken
        # its snapshot of the history, so this extra pass cannot alter it.
        select_best = bool(cfg.training.get("load_best_model_at_end", True))
        restored_validation_macro_f1 = None
        if select_best:
            restored = az_trainer.evaluate()
            restored_validation_macro_f1 = float(restored["eval_macro_f1"])
            expected = float(training_evidence["selected_validation_macro_f1"])
            if abs(restored_validation_macro_f1 - expected) > 1e-4:
                raise AssertionError(
                    "Best-checkpoint restore FAILED: re-scoring the model left "
                    f"in memory gives validation macro-F1 "
                    f"{restored_validation_macro_f1:.6f}, but the selected step "
                    f"({training_evidence['selected_step']}) recorded "
                    f"{expected:.6f}. Any test evaluation would describe the "
                    "wrong weights. Refusing to write this run.")
            log.info("Best checkpoint restored and verified: step %d, "
                     "val macro-F1 = %.4f",
                     training_evidence["selected_step"],
                     restored_validation_macro_f1)

        evaluation = {
            "eval_macro_f1": training_evidence["selected_validation_macro_f1"],
            "selected_step": training_evidence["selected_step"],
        }
        test_metrics = None
        pred_labels = None
        if eval_split == "test":
            assert az_test is not None
            test_metrics = az_trainer.evaluate(
                eval_dataset=ListDataset(az_test, tokenizer, cfg.models.max_length),
                metric_key_prefix="test")
            preds = az_trainer.predict(
                ListDataset(az_test, tokenizer, cfg.models.max_length))
            pred_labels = np.argmax(preds.predictions, axis=-1).tolist()
            ledger = Path(test_ledger_path or
                          (Path(cfg.experiment.results_dir) / "test_evaluation_ledger.jsonl"))
            ensure_dir(ledger.parent)
            entry = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "base": base.short,
                "condition": condition["name"],
                "train_size": int(train_size),
                "seed": int(seed),
                "call_site": "src.training.finetune.run_single",
            }
            with open(ledger, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    # ---------------------------------------------------------- v4 comparison
    # Built here, not in the paper, so that every run file already carries the
    # interpretable pair and the non-interpretable one is named as such.
    test_macro_f1 = None
    if test_metrics is not None:
        raw = test_metrics.get("test_macro_f1")
        if raw is None:
            raise AssertionError(
                "test evaluation returned no test_macro_f1 — refusing to write "
                "a run file that claims a test split it cannot report")
        test_macro_f1 = float(raw)
    terminal_val = float(training_evidence["terminal_validation_macro_f1"])
    selected_val = float(training_evidence["selected_validation_macro_f1"])
    # With best-checkpoint selection the SELECTED step is the matched one: the
    # test evaluation ran on exactly the weights that produced
    # selected_validation_macro_f1, verified above. The terminal figure is kept
    # for information (it shows post-peak drift) but is NOT the comparison.
    matched_step = int(training_evidence["selected_step"] if select_best
                       else training_evidence["terminal_step"])
    matched_val = selected_val if select_best else terminal_val
    test_comparison = {
        "eval_split": eval_split,
        "selection": ("best_validation_checkpoint" if select_best
                      else "final_step"),
        "matched_step": matched_step,
        # Validation figure for the SAME weights the test split saw.
        "validation_macro_f1_matched": matched_val,
        "validation_macro_f1_selected_max": selected_val,
        "validation_macro_f1_terminal": terminal_val,
        "restored_validation_macro_f1": restored_validation_macro_f1,
        "selected_step": int(training_evidence["selected_step"]),
        "terminal_step": int(training_evidence["terminal_step"]),
        "test_macro_f1": test_macro_f1,
        # The reportable generalisation gap: same weights on both sides.
        "delta_test_minus_validation_matched": (
            None if test_macro_f1 is None
            else round(test_macro_f1 - matched_val, 6)),
        # Post-peak drift on validation, for context only. Never a test number.
        "validation_drift_selected_minus_terminal": round(
            selected_val - terminal_val, 6),
        "escaped_on_validation": bool(training_evidence["escaped"]),
        "escaped_but_terminal_collapsed": bool(
            training_evidence["escaped_but_terminal_collapsed"]),
        "estimand_note": (
            "Standard protocol: the checkpoint is SELECTED on validation "
            "(max macro-F1 over the evaluation points) and the held-out test "
            "split is then scored on that same checkpoint, which is verified "
            "by re-scoring the restored model on validation "
            "(restored_validation_macro_f1 must equal "
            "validation_macro_f1_selected_max). So "
            "delta_test_minus_validation_matched is a true generalisation gap, "
            "not an artefact of comparing different weights. Escape rate stays "
            "a validation-only quantity: it is defined over the ten evaluation "
            "points, and the test split is measured once."),
    }

    result = {
        "run_result_schema_version": RUN_RESULT_SCHEMA_VERSION,
        "base": base.short,
        "base_role": base.role,
        "base_model": base.hf_id,
        "az_in_pretraining": base.az_in_pretraining,
        "condition": condition["name"],
        "condition_id": condition["id"],
        "tokenizer": condition["tokenizer"],
        "turkish": condition["turkish"],
        "train_size": train_size,
        "actual_train_size": len(az_train),
        "seed": seed,
        "n_labels": n_az_labels,
        "model_path": model_path,
        "transplant_variant": transplant_variant,
        "eval_split": eval_split,
        "task_head_reset": head_reset,
        "test_comparison": test_comparison,
        # Reqabet altinda olculen wall-clock hemin run-un TEK QALDIQDA
        # cekeceyi vaxt DEYIL. Bunu her fayla yaziriq ki, meqalede vaxt
        # reqemi verilerken hansi seraitde olculdüyü ITMESIN.
        "streams_active": int(streams_active),
        "evaluation": evaluation,
        "test": ({k: float(v) for k, v in test_metrics.items()
                  if isinstance(v, (int, float))} if test_metrics else None),
        "test_predictions": pred_labels,
        "test_gold": ([int(r["label"]) for r in az_test] if az_test else None),
        "az_stage_runtime_sec": az_runtime_sec,
        "runtime_sec": round(time.time() - t0, 1),
        **training_evidence,
        **stage_info,
    }
    assert_run_result_schema(result)
    return result


def main() -> None:
    ap = base_argparser(__doc__)
    ap.add_argument("--base", default="primary", help="primary | contrast")
    ap.add_argument("--condition", required=True)
    ap.add_argument("--train-size", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--transplanted", default=None,
                    help="optional explicit transplanted-model path (ablation controls)")
    ap.add_argument("--result-tag", default=None,
                    help="suffix that prevents an ablation result overwriting the main run")
    ap.add_argument("--eval-split", choices=["val", "test"], default="val")
    ap.add_argument("--allow-test-eval", action="store_true",
                    help="required explicit opt-in when --eval-split test")
    # Phase-2 worker flags (set by src/training/phases.py, not by hand).
    ap.add_argument("--stream-id", type=int, default=0,
                    help="identifier of the parallel stream running this item")
    ap.add_argument("--streams-active", type=int, default=1,
                    help="how many streams ran concurrently; recorded in the result "
                         "because a contended wall-clock is not an isolated cost")
    ap.add_argument("--phase2-readonly-cache", action="store_true",
                    help="refuse to build a missing Turkish checkpoint; Phase 1 owns "
                         "cache writes, so a miss here is a bug, not work to do")
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)
    verify_config_hashes()  # fail loudly before this machine trains anything

    conds = {c["name"]: c for c in cfg.conditions}
    if args.condition not in conds:
        raise SystemExit(f"naməlum şərt: {args.condition}. Mövcud: {list(conds)}")

    base = resolve_bases(cfg, [args.base])[0]
    key = RunKey(base=base.short, condition=args.condition,
                 train_size=args.train_size, seed=args.seed)
    with tee_run_log(cfg.experiment.results_dir, key.filename.removesuffix(".json")):
        res = run_single(cfg, base, conds[args.condition], args.train_size, args.seed,
                         transplanted_path=args.transplanted,
                         transplant_variant=args.result_tag,
                         eval_split=args.eval_split,
                         allow_test_eval=args.allow_test_eval,
                         streams_active=args.streams_active,
                         readonly_tr_cache=args.phase2_readonly_cache)
    filename = key.filename
    if args.result_tag:
        safe_tag = "".join(c if c.isalnum() or c in "-_" else "_"
                           for c in args.result_tag)
        filename = filename.removesuffix(".json") + f"__variant={safe_tag}.json"
    out = ensure_dir(Path(cfg.experiment.results_dir) / "runs") / filename
    write_run_result(res, out)
    log.info("selected val %s = %.4f at step %d (%.0fs)", cfg.training.metric,
             res["selected_validation_macro_f1"], res["selected_step"],
             res["runtime_sec"])


if __name__ == "__main__":
    main()
