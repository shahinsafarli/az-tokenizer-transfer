"""Run the required Condition-3 mean/random-coefficient controls.

This is deliberately separate from the 75-run experiment matrix. It expects the
rescaled control artifacts to exist and writes one result per method/base/seed to
``results/transplant_controls/runs`` plus an aggregate JSON report.
"""
from __future__ import annotations

import statistics
from pathlib import Path

from src.training.finetune import run_single, write_run_result
from src.utils import (base_argparser, ensure_dir, load_config, resolve_bases,
                       setup_logging, write_json)


def control_filename(base: str, method: str, n: int, seed: int) -> str:
    return f"base={base}__method={method}__cond=tokenizator__n={n}__seed={seed}.json"


def main() -> None:
    raise SystemExit(
        "Retired: mean and random_coef are conditions in the single "
        "src.training.orchestrate queue.")
    # Historical implementation retained below for provenance; unreachable.
    ap = base_argparser(__doc__)
    ap.add_argument("--methods", default="mean,random_coef")
    ap.add_argument("--bases", default=None, help="comma-separated: primary,contrast")
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    if int(cfg.training.batch_size) != 32:
        raise SystemExit("Required controls are locked to training.batch_size=32")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if set(methods) != {"mean", "random_coef"}:
        raise SystemExit("Required method set is exactly: mean,random_coef")
    roles = [r.strip() for r in args.bases.split(",")] if args.bases else None
    bases = resolve_bases(cfg, roles)
    ref = int(cfg.analysis.reference_size)
    seeds = [int(s) for s in cfg.experiment.seeds]
    condition = next(c for c in cfg.conditions if c["name"] == "tokenizator")
    runs_dir = ensure_dir(Path(cfg.experiment.results_dir) / "transplant_controls" / "runs")

    records = []
    for method in methods:
        tag = f"{method}_k{cfg.transplant.k}_rescaled"
        for base in bases:
            artifact = Path(cfg.experiment.artifacts_dir) / f"transplanted__{base.short}__{tag}"
            if not artifact.exists():
                raise SystemExit(
                    f"Missing {artifact}; build and rescale the {method} artifacts first")
            for seed in seeds:
                out = runs_dir / control_filename(base.short, method, ref, seed)
                if bool(cfg.run.skip_existing) and out.exists():
                    import json
                    with open(out, "r", encoding="utf-8") as f:
                        records.append(json.load(f))
                    continue
                result = run_single(
                    cfg, base, condition, ref, seed,
                    transplanted_path=artifact, transplant_variant=tag)
                result["control_method"] = method
                write_run_result(result, out)
                records.append(result)

    metric = f"test_{cfg.training.metric}"
    aggregate = {}
    for method in methods:
        aggregate[method] = {}
        for base in bases:
            vals = [r["test"][metric] for r in records
                    if r["control_method"] == method and r["base"] == base.short]
            aggregate[method][base.short] = {
                "values": vals,
                "mean": statistics.fmean(vals),
                "sample_sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            }
    report = {
        "status": "controls_complete_full_75_not_started",
        "condition": "tokenizator",
        "reference_size": ref,
        "batch_size": int(cfg.training.batch_size),
        "seeds": seeds,
        "n_runs": len(records),
        "metric": metric,
        "aggregate": aggregate,
    }
    write_json(report, Path(cfg.experiment.results_dir) /
               "condition3_transplant_controls.json")


if __name__ == "__main__":
    main()
