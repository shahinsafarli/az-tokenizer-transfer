"""Aggregate post-freeze runs into escape rate and conditional macro-F1 only."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.training.orchestrate import build_queue
from src.utils import (base_argparser, ensure_dir, load_config, read_json,
                       resolve_bases, setup_logging, write_json)


SUPPORTED_RUN_SCHEMA = 4


def load_runs(results_dir: str | Path) -> pd.DataFrame:
    """Load only current-schema records; older files are never pooled.

    The version gate is deliberately exact rather than ">=": two grids run
    under different schemas are two different samples, and silently averaging
    them would be the one mistake no downstream statistic could reveal. Files
    at another version are counted and reported by version, so pointing this
    at a mixed directory produces a visible refusal, not a quiet subset.
    """
    rows: list[dict] = []
    skipped_by_version: dict[int, int] = {}
    for path in sorted((Path(results_dir) / "runs").glob("*.json")):
        record = read_json(path)
        version = int(record.get("run_result_schema_version", 0))
        if version != SUPPORTED_RUN_SCHEMA:
            skipped_by_version[version] = skipped_by_version.get(version, 0) + 1
            continue
        comparison = record.get("test_comparison") or {}
        rows.append({
            "base": record["base"],
            "base_role": record.get("base_role", "unknown"),
            "condition": record["condition"],
            "condition_id": int(record["condition_id"]),
            "tokenizer": record["tokenizer"],
            "turkish": record["turkish"],
            "train_size": int(record["train_size"]),
            "seed": int(record["seed"]),
            "escaped": bool(record["escaped"]),
            "escape_step": record["escape_step"],
            "selected_step": int(record["selected_step"]),
            "selected_validation_macro_f1": float(
                record["selected_validation_macro_f1"]),
            # v4 matched-estimand fields. `selected_...` is a MAXIMUM over
            # evaluation points; `terminal_...` is the single measurement of
            # the same weights the test split sees. Only the latter may be
            # compared against `test_macro_f1`.
            "terminal_step": int(record["terminal_step"]),
            "terminal_validation_macro_f1": float(
                record["terminal_validation_macro_f1"]),
            "escaped_but_terminal_collapsed": bool(
                record["escaped_but_terminal_collapsed"]),
            "eval_split": record["eval_split"],
            "test_macro_f1": comparison.get("test_macro_f1"),
            "tr_stage_macro_f1": record.get("tr_stage_macro_f1"),
            "tr_stage_degenerate": record.get("tr_stage_degenerate"),
            "delta_test_minus_validation_matched": comparison.get(
                "delta_test_minus_validation_matched"),
            "max_steps": int(record["max_steps"]),
            "steps_per_epoch": int(record["steps_per_epoch"]),
            "effective_epochs": float(record["effective_epochs"]),
            "runtime_sec": float(record.get("runtime_sec", float("nan"))),
            "source_path": str(path),
        })
    if skipped_by_version:
        detail = ", ".join(f"v{v}: {n} file(s)"
                           for v, n in sorted(skipped_by_version.items()))
        print(f"[load_runs] NOT POOLED (schema != {SUPPORTED_RUN_SCHEMA}) — "
              f"{detail}. Two grids at different schemas are two samples; "
              f"analyse them separately.")
    return pd.DataFrame(rows)


def wilson_interval(successes: int, total: int, alpha: float = 0.05) -> list[float] | None:
    if total == 0:
        return None
    # 1.95996398454 is the two-sided 95% standard-normal quantile.
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-12 else 1.959963984540054
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def bootstrap_mean_ci(values: list[float], iters: int, alpha: float,
                      seed: int = 0) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    means = rng.choice(array, size=(int(iters), len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2))]


def summarize_cell(records: list[dict], *, iters: int = 10000,
                   alpha: float = 0.05) -> dict:
    total = len(records)
    escaped = [r for r in records if bool(r["escaped"])]
    n_escaped = len(escaped)
    values = [float(r["selected_validation_macro_f1"]) for r in escaped]
    all_values = [float(r["selected_validation_macro_f1"]) for r in records]
    n_fallback = sum(1 for r in records
                     if bool(r.get("escaped_but_terminal_collapsed")))
    n_sustained = n_escaped - n_fallback
    n_tr_degenerate = sum(1 for r in records if r.get("tr_stage_degenerate"))
    tr_f1s = [float(r["tr_stage_macro_f1"]) for r in records
              if r.get("tr_stage_macro_f1") is not None]
    return {
        "status": "MEASURED" if total else "NOT MEASURED",
        "escape_rate": (n_escaped / total) if total else None,
        "escape_rate_count": f"{n_escaped}/{total}" if total else "0/0",
        "escape_rate_wilson_ci": wilson_interval(n_escaped, total, alpha),
        "conditional_macro_f1": (float(np.mean(values)) if values else None),
        "conditional_macro_f1_bootstrap_ci": bootstrap_mean_ci(
            values, iters, alpha),
        "n_escaped": n_escaped,
        "n_total": total,
        # ---- UNCONDITIONAL estimand, added 2026-09-08 --------------------
        # `conditional_macro_f1` averages ONLY the runs that escaped, so two
        # conditions can be compared on different survivor sets — selection on
        # outcome. It answers "how good are this condition's successes", which
        # is a legitimate question but is NOT "how good is this condition".
        # The all-seed mean answers the second one: every declared seed
        # counts, failures at their actual score. Reported side by side
        # because the two can RANK CONDITIONS DIFFERENTLY, and a paper that
        # shows only the conditional one can state a ranking its data reverses.
        "all_seed_macro_f1": (float(np.mean(all_values)) if all_values else None),
        "all_seed_macro_f1_bootstrap_ci": bootstrap_mean_ci(
            all_values, iters, alpha),
        "all_seed_minus_conditional": (
            round(float(np.mean(all_values)) - float(np.mean(values)), 4)
            if all_values and values else None),
        # ---- SUSTAINED escape, added 2026-09-08 --------------------------
        # `escaped` is true if the run crossed 0.40 at ANY evaluation point,
        # so a seed that peaked mid-training and fell back into the collapse
        # basin counts as escaped while its final model is collapsed. Both
        # readings are reported; neither replaces the pre-registered one.
        "n_escaped_sustained": n_sustained,
        "escape_rate_sustained": (n_sustained / total) if total else None,
        "escape_rate_sustained_wilson_ci": wilson_interval(
            n_sustained, total, alpha),
        "n_escaped_then_collapsed": n_fallback,
        # ---- source-stage competence, added 2026-09-08 -------------------
        "n_tr_stage_degenerate": n_tr_degenerate,
        "tr_stage_macro_f1_mean": (float(np.mean(tr_f1s)) if tr_f1s else None),
        "source_paths": [str(r["source_path"]) for r in records],
    }


def aggregate_cells(df: pd.DataFrame, cfg) -> list[dict]:
    bases = resolve_bases(cfg)
    queue = build_queue(cfg, bases)
    keys: list[tuple[str, str, int]] = []
    metadata: dict[tuple[str, str, int], dict] = {}
    for base, condition, size, _seed in queue:
        key = (base.short, condition["name"], int(size))
        if key not in metadata:
            keys.append(key)
            metadata[key] = {
                "base": base.short,
                "base_role": base.role,
                "condition": condition["name"],
                "condition_id": int(condition["id"]),
                "train_size": int(size),
            }
    rows = df.to_dict(orient="records") if not df.empty else []
    out = []
    for key in keys:
        records = [row for row in rows
                   if (row["base"], row["condition"], row["train_size"]) == key]
        out.append({**metadata[key], **summarize_cell(
            records, iters=int(cfg.analysis.bootstrap_iters),
            alpha=float(cfg.analysis.alpha))})
    return out


def main() -> None:
    parser = base_argparser(__doc__)
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)
    results_dir = ensure_dir(cfg.experiment.results_dir)
    df = load_runs(results_dir)
    df.to_csv(results_dir / "aggregate.csv", index=False)
    summary = aggregate_cells(df, cfg)
    pd.DataFrame(summary).to_csv(results_dir / "summary.csv", index=False)
    write_json(summary, results_dir / "summary.json")
    for row in summary:
        conditional = ("null" if row["conditional_macro_f1"] is None else
                       f"{row['conditional_macro_f1']:.4f}")
        print(f"{row['base']} {row['condition']} n={row['train_size']}: "
              f"{row['status']}; escape_rate={row['escape_rate_count']} "
              f"CI={row['escape_rate_wilson_ci']}; "
              f"conditional_macro_f1={conditional} "
              f"n_escaped={row['n_escaped']}; sources={row['source_paths']}")


if __name__ == "__main__":
    main()
