"""Dual-metric mechanism summaries without pooling escaped and non-escaped runs."""
from __future__ import annotations

from pathlib import Path

from src.analysis.aggregate import aggregate_cells, load_runs
from src.utils import (base_argparser, ensure_dir, load_config, setup_logging,
                       write_json)


def _cell_map(cells: list[dict]) -> dict[tuple[str, str, int], dict]:
    return {(r["base"], r["condition"], int(r["train_size"])): r for r in cells}


def compare_bases(cells: list[dict], size: int, cfg) -> dict:
    """Compare both escape probability and escaped-only performance by base."""
    by_key = _cell_map(cells)
    primary = cfg.models.primary.short
    contrast = cfg.models.contrast.short

    def arm(base: str, condition: str):
        return by_key.get((base, condition, int(size)))

    needed = {(base, condition): arm(base, condition)
              for base in (primary, contrast)
              for condition in ("baza", "tokenizator")}
    missing = [f"{b}/{c}" for (b, c), row in needed.items()
               if row is None or row["status"] == "NOT MEASURED"]
    if missing:
        return {
            "status": "NOT MEASURED",
            "missing": missing,
            "source_paths": [],
        }

    result = {"status": "MEASURED", "train_size": int(size), "bases": {}}
    all_sources: list[str] = []
    for base in (primary, contrast):
        baseline = needed[(base, "baza")]
        omp = needed[(base, "tokenizator")]
        b_cond = baseline["conditional_macro_f1"]
        o_cond = omp["conditional_macro_f1"]
        conditional_delta = None if b_cond is None or o_cond is None else o_cond - b_cond
        result["bases"][base] = {
            "baseline_escape_rate": baseline["escape_rate"],
            "omp_escape_rate": omp["escape_rate"],
            "escape_rate_difference": omp["escape_rate"] - baseline["escape_rate"],
            "baseline_conditional_macro_f1": b_cond,
            "omp_conditional_macro_f1": o_cond,
            "conditional_macro_f1_difference": conditional_delta,
            "baseline_n_escaped": baseline["n_escaped"],
            "omp_n_escaped": omp["n_escaped"],
            "source_paths": baseline["source_paths"] + omp["source_paths"],
        }
        all_sources.extend(result["bases"][base]["source_paths"])

    p = result["bases"][primary]
    c = result["bases"][contrast]
    result["cross_base"] = {
        "escape_rate_effect_difference": (
            p["escape_rate_difference"] - c["escape_rate_difference"]),
        "conditional_macro_f1_effect_difference": (
            None if p["conditional_macro_f1_difference"] is None or
            c["conditional_macro_f1_difference"] is None else
            p["conditional_macro_f1_difference"] -
            c["conditional_macro_f1_difference"]),
    }
    result["source_paths"] = sorted(set(all_sources))
    return result


def main() -> None:
    parser = base_argparser(__doc__)
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)
    results_dir = ensure_dir(cfg.experiment.results_dir)
    cells = aggregate_cells(load_runs(results_dir), cfg)
    result = {
        "metric_policy": (
            "escape_rate with Wilson CI; conditional_macro_f1 over escaped seeds "
            "only with bootstrap CI; never a mixed macro-F1 mean"),
        "cells": cells,
        "cross_base_comparison": compare_bases(
            cells, int(cfg.analysis.reference_size), cfg),
    }
    write_json(result, Path(results_dir) / "decompose.json")
    cross = result["cross_base_comparison"]
    print("compare_bases escape quantity:",
          cross.get("cross_base", {}).get("escape_rate_effect_difference", "NOT MEASURED"))
    print("compare_bases conditional macro-F1 quantity:",
          cross.get("cross_base", {}).get(
              "conditional_macro_f1_effect_difference", "NOT MEASURED"))
    print("sources:", cross.get("source_paths", []))


if __name__ == "__main__":
    main()
