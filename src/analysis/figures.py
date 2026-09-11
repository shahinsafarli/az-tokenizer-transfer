"""Generate the frozen dual-metric paper figures from post-freeze results."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.aggregate import aggregate_cells, load_runs
from src.utils import (base_argparser, ensure_dir, load_config, setup_logging,
                       write_json)


def _plot(cells: list[dict], field: str, ylabel: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    measured = [r for r in cells if r["status"] == "MEASURED" and r[field] is not None]
    if not measured:
        ax.text(0.5, 0.5, "NOT MEASURED", ha="center", va="center",
                transform=ax.transAxes, fontsize=18)
        ax.set_axis_off()
    else:
        groups: dict[tuple[str, str], list[dict]] = {}
        for row in measured:
            groups.setdefault((row["base"], row["condition"]), []).append(row)
        for (base, condition), rows in groups.items():
            rows.sort(key=lambda r: r["train_size"])
            ax.plot([r["train_size"] for r in rows], [r[field] for r in rows],
                    marker="o", label=f"{base}/{condition}")
        ax.set_xscale("log")
        ax.set_xlabel("Azerbaijani training examples")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def generate_figures(cfg) -> list[dict]:
    results_dir = Path(cfg.experiment.results_dir)
    figures_dir = ensure_dir(cfg.experiment.figures_dir)
    cells = aggregate_cells(load_runs(results_dir), cfg)
    specs = [
        ("escape_rate", "Escape rate", figures_dir / "escape_rate.png"),
        ("conditional_macro_f1", "Validation macro-F1 (escaped seeds only)",
         figures_dir / "conditional_macro_f1.png"),
    ]
    manifest = []
    for field, ylabel, path in specs:
        _plot(cells, field, ylabel, path)
        sources = sorted({source for row in cells for source in row["source_paths"]
                          if row[field] is not None})
        manifest.append({"figure": str(path), "quantity": field,
                         "source_paths": sources,
                         "status": "MEASURED" if sources else "NOT MEASURED"})
    write_json(manifest, figures_dir / "source_manifest.json")
    return manifest


def main() -> None:
    parser = base_argparser(__doc__)
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)
    for row in generate_figures(cfg):
        print(f"{row['figure']}: {row['status']}; sources={row['source_paths']}")


if __name__ == "__main__":
    main()
