"""Regenerate every frozen paper table and figure from ``results/``.

Usage: python -m src.analysis.report --config configs/experiment.yaml
"""
from __future__ import annotations

from pathlib import Path

from src.analysis.aggregate import aggregate_cells, load_runs
from src.analysis.decompose import compare_bases
from src.analysis.figures import generate_figures
from src.utils import (base_argparser, ensure_dir, load_config, setup_logging,
                       write_json)


def _fmt(value, digits: int = 4) -> str:
    return "NOT MEASURED" if value is None else f"{float(value):.{digits}f}"


def generate_report(cfg) -> dict:
    results_dir = ensure_dir(cfg.experiment.results_dir)
    runs = load_runs(results_dir)
    cells = aggregate_cells(runs, cfg)
    cross = compare_bases(cells, int(cfg.analysis.reference_size), cfg)
    figures = generate_figures(cfg)
    report = {
        "policy": "No macro-F1 pooling across escaped and non-escaped seeds.",
        "cells": cells,
        "cross_base_comparison": cross,
        "figures": figures,
    }
    write_json(report, results_dir / "paper_report.json")

    lines = [
        "# Frozen paper tables",
        "",
        "Every measured number lists all backing result files. Empty slots are",
        "printed as `NOT MEASURED`.",
        "",
        "| Base | Condition | n | Escape rate (Wilson CI) | Conditional macro-F1 (bootstrap CI; n escaped) | Sources |",
        "|---|---|---:|---|---|---|",
    ]
    for row in cells:
        if row["status"] == "NOT MEASURED":
            escape = conditional = "NOT MEASURED"
        else:
            wi = row["escape_rate_wilson_ci"]
            escape = (f"{row['escape_rate_count']} = {row['escape_rate']:.4f} "
                      f"[{wi[0]:.4f}, {wi[1]:.4f}]")
            if row["conditional_macro_f1"] is None:
                conditional = f"null (n_escaped={row['n_escaped']})"
            else:
                ci = row["conditional_macro_f1_bootstrap_ci"]
                conditional = (f"{row['conditional_macro_f1']:.4f} "
                               f"[{ci[0]:.4f}, {ci[1]:.4f}]; "
                               f"n_escaped={row['n_escaped']}")
        sources = "<br>".join(row["source_paths"]) or "NOT MEASURED"
        lines.append(f"| {row['base']} | {row['condition']} | {row['train_size']} | "
                     f"{escape} | {conditional} | {sources} |")

    lines.extend(["", "## Cross-base comparison", ""])
    if cross.get("status") != "MEASURED":
        lines.append("NOT MEASURED")
    else:
        cr = cross["cross_base"]
        lines.append(
            "Escape-rate effect difference: " +
            _fmt(cr["escape_rate_effect_difference"]) + "; sources: " +
            ", ".join(cross["source_paths"]))
        lines.append("")
        lines.append(
            "Conditional-macro-F1 effect difference: " +
            _fmt(cr["conditional_macro_f1_effect_difference"]) +
            "; sources: " + ", ".join(cross["source_paths"]))
    (results_dir / "paper_tables.md").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")
    return report


def main() -> None:
    parser = base_argparser(__doc__)
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)
    report = generate_report(cfg)
    for row in report["cells"]:
        sources = row["source_paths"] or ["NOT MEASURED"]
        print(f"{row['base']} {row['condition']} n={row['train_size']}: "
              f"escape_rate={_fmt(row['escape_rate'])}; "
              f"conditional_macro_f1={_fmt(row['conditional_macro_f1'])}; "
              f"sources={sources}")
    print("paper tables:", Path(cfg.experiment.results_dir) / "paper_tables.md")
    for figure in report["figures"]:
        print(f"figure: {figure['figure']} {figure['status']} "
              f"sources={figure['source_paths'] or ['NOT MEASURED']}")


if __name__ == "__main__":
    main()
