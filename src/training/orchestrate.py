"""One resumable, time-boxed queue for every frozen experimental condition.

Dry-run queue inspection deliberately imports neither torch nor transformers.
"""
from __future__ import annotations

import shutil
import statistics
import time
import traceback
from pathlib import Path

from src.utils import (BaseModel, RunKey, base_argparser, condition_applies,
                       ensure_dir, get_logger, load_config, read_json, resolve_bases,
                       seeds_for_size, setup_logging, sizes_for_base, tee_run_log,
                       verify_config_hashes, write_json)

log = get_logger(__name__)
Item = tuple[BaseModel, dict, int, int]


def build_queue(cfg, bases: list[BaseModel], only_priority: bool = False,
                cond_filter: set[str] | None = None,
                size_filter: set[int] | None = None,
                seed_filter: set[int] | None = None) -> list[Item]:
    """Build the sole queue in frozen priority, condition, then seed order."""
    conditions = [c for c in cfg.conditions
                  if cond_filter is None or c["name"] in cond_filter]
    reference = int(cfg.analysis.reference_size)
    queue: list[Item] = []

    def add_size(base: BaseModel, size: int) -> None:
        if size_filter is not None and size not in size_filter:
            return
        for condition in conditions:
            if not condition_applies(condition, size):
                continue
            for seed in seeds_for_size(cfg, size):
                if seed_filter is None or seed in seed_filter:
                    queue.append((base, condition, size, seed))

    if bool(cfg.run.priority_first):
        for base in bases:
            if reference in sizes_for_base(cfg, base):
                add_size(base, reference)
        if only_priority:
            return queue

    for base in bases:
        for size in sizes_for_base(cfg, base):
            if bool(cfg.run.priority_first) and size == reference:
                continue
            add_size(base, int(size))
    return queue


def tr_stage_cache_summary(queue: list[Item]) -> tuple[int, int]:
    """Return (TR requests, scientifically distinct reusable TR stages)."""
    requests = [item for item in queue if item[1].get("turkish") != "none"]
    keys = {(base.short, condition.get("tokenizer"), condition.get("turkish"), seed)
            for base, condition, _size, seed in requests}
    return len(requests), len(keys)


def _disk_free_bytes(*paths: str | Path) -> int:
    """Minimum free bytes across `paths`' filesystems — the TR checkpoint
    cache and the results dir aren't guaranteed to share a mount, and the
    binding constraint is whichever is tightest. shutil.disk_usage needs an
    existing path, so each is walked up to its nearest existing ancestor."""
    free_values = []
    for path in paths:
        p = Path(path).resolve()
        while not p.exists():
            parent = p.parent
            if parent == p:
                raise SystemExit(f"DISK CHECK FAILED: no existing ancestor for {path}")
            p = parent
        free_values.append(shutil.disk_usage(p).free)
    return min(free_values)


def estimate_grid_disk_bytes(cfg, queue: list[Item]) -> dict:
    """Projected ADDITIONAL disk bytes still needed to run everything in
    `queue` from here — not counting cache/results already on disk.

    Deliberately dependency-light (2026-09-04 disk audit, HANDOFF.md §13):
    the per-checkpoint byte figure is a static config constant (the measured
    maximum across (base, tokenizer) combos — xlmr with its 250,002-token
    original vocabulary), not a live model-config lookup, so this stays
    import-cheap (no torch/transformers) and runs before every single item
    without adding meaningful overhead. It also means the projection is
    conservative for the smaller combos (e.g. an xlmr+omp checkpoint is
    ~445 MB, well under the 1.2 GB ceiling used here for all of them) —
    intentional: better to over-project free disk than under-project it.

    "Already cached" is approximated by counting cache directories under
    `run.tr_stage_cache_dir` that contain a `cache_manifest.json`, rather
    than replicating `_tr_cache_spec`'s exact content hash (which would
    require importing torch/transformers here). A spec mismatch would make
    the real code recompute a checkpoint this function counted as already
    done — the margin_ratio below is partly there to absorb that.
    """
    per_checkpoint = int(cfg.run.get("disk_budget_bytes_per_tr_checkpoint", 1_200_000_000))
    per_run_overhead = int(cfg.run.get("disk_budget_bytes_per_run_overhead", 250_000))
    margin = float(cfg.run.get("disk_budget_margin_ratio", 0.15))

    # TRANSIENT AZ checkpoints (added 2026-09-07 with best-checkpoint
    # selection). Each in-flight run holds at most `save_total_limit=1` saved
    # checkpoint plus the one being written, in a temp dir deleted when the run
    # ends — so the peak is bounded by the number of concurrent streams, not by
    # the run count. Counted once, as headroom, because it never accumulates.
    streams = max(1, int(cfg.run.get("parallel_streams", 1)))
    select_best = bool(cfg.training.get("load_best_model_at_end", True))
    transient_az = (2 * streams * per_checkpoint) if select_best else 0

    _requests, distinct_tr = tr_stage_cache_summary(queue)
    tr_cache_dir = Path(cfg.run.get("tr_stage_cache_dir", "artifacts/tr_stage_cache"))
    already_cached = 0
    if tr_cache_dir.exists():
        already_cached = sum(
            1 for d in tr_cache_dir.iterdir()
            if d.is_dir() and (d / "cache_manifest.json").exists())
    missing_tr = max(0, distinct_tr - already_cached)

    remaining_runs = len(queue)
    raw = (missing_tr * per_checkpoint
           + remaining_runs * per_run_overhead
           + transient_az)
    return {
        "distinct_tr_checkpoints_needed": distinct_tr,
        "tr_checkpoints_already_cached": already_cached,
        "missing_tr_checkpoints": missing_tr,
        "remaining_runs": remaining_runs,
        "per_checkpoint_bytes": per_checkpoint,
        "per_run_overhead_bytes": per_run_overhead,
        "transient_az_checkpoint_bytes": transient_az,
        "raw_projected_bytes": raw,
        "projected_bytes_with_margin": int(raw * (1 + margin)),
    }


def result_path(results_dir: str | Path, item: Item) -> Path:
    base, condition, size, seed = item
    return (Path(results_dir) / "runs" /
            RunKey(base.short, condition["name"], size, seed).filename)


def queue_label(item: Item) -> str:
    base, condition, size, seed = item
    label = condition.get("label", condition["name"])
    return f"base={base.short} condition={label} n={size} seed={seed}"


def _estimate_next_seconds(cfg, completed: list[float]) -> float:
    if completed:
        return float(statistics.median(completed))
    step_sec = float(cfg.run.get("budget_seconds_per_step", 2.8081))
    overhead = float(cfg.run.get("budget_eval_overhead_sec", 600.0))
    return step_sec * int(cfg.training.max_steps) + overhead


def execute_queue(cfg, queue: list[Item], run_fn, *, budget_hours: float | None,
                  now_fn=time.monotonic) -> dict:
    """Run atomically persisted jobs; never begin one predicted to miss budget."""
    ensure_dir(Path(cfg.experiment.results_dir) / "runs")
    started = now_fn()
    completed_durations: list[float] = []
    completed: list[str] = []
    skipped: list[str] = []
    failures: list[dict] = []
    budget_sec = None if budget_hours is None else max(0.0, budget_hours * 3600.0)

    # Disk pre-flight, mirroring the time-budget check below: refuse to start
    # at all if the whole queue can't fit, and re-check the shrinking
    # remainder before every item so the grid stops cleanly between runs
    # instead of dying mid-write inside one. See HANDOFF.md §13.
    disk_check_paths = (
        Path(cfg.experiment.results_dir),
        Path(cfg.run.get("tr_stage_cache_dir", "artifacts/tr_stage_cache")),
    )
    initial_projection = estimate_grid_disk_bytes(cfg, queue)
    initial_free = _disk_free_bytes(*disk_check_paths)
    if initial_free < initial_projection["projected_bytes_with_margin"]:
        raise SystemExit(
            "DISK PRE-FLIGHT FAILED: refusing to start. "
            f"free={initial_free/1e9:.2f}GB < "
            f"projected(+margin)={initial_projection['projected_bytes_with_margin']/1e9:.2f}GB "
            f"(missing_tr_checkpoints={initial_projection['missing_tr_checkpoints']}, "
            f"remaining_runs={initial_projection['remaining_runs']}). "
            "Free disk or reduce the queue before retrying.")
    log.info("Disk pre-flight OK: free=%.2fGB, projected(+margin)=%.2fGB",
             initial_free / 1e9, initial_projection["projected_bytes_with_margin"] / 1e9)

    for queue_pos, item in enumerate(queue, 1):
        path = result_path(cfg.experiment.results_dir, item)
        if bool(cfg.run.skip_existing) and path.exists():
            try:
                existing = read_json(path)
                if int(existing.get("run_result_schema_version", 0)) == 4:
                    skipped.append(str(path))
                    continue
                log.warning("Ignoring incompatible existing result: %s", path)
            except Exception:  # noqa: BLE001
                log.warning("Ignoring unreadable existing result: %s", path)
        remaining_projection = estimate_grid_disk_bytes(cfg, queue[queue_pos - 1:])
        remaining_free = _disk_free_bytes(*disk_check_paths)
        if remaining_free < remaining_projection["projected_bytes_with_margin"]:
            log.error(
                "DISK PRE-FLIGHT FAILED before %s: free=%.2fGB < "
                "projected(+margin)=%.2fGB for the %d run(s) left. Stopping "
                "cleanly rather than risking a mid-write failure.",
                queue_label(item), remaining_free / 1e9,
                remaining_projection["projected_bytes_with_margin"] / 1e9,
                remaining_projection["remaining_runs"])
            break
        elapsed = now_fn() - started
        estimate = _estimate_next_seconds(cfg, completed_durations)
        if budget_sec is not None and elapsed + estimate > budget_sec:
            log.info("Budget stop before %s: elapsed=%.1fs estimate=%.1fs cap=%.1fs",
                     queue_label(item), elapsed, estimate, budget_sec)
            break
        base, condition, size, seed = item
        run_tag = RunKey(base.short, condition["name"], size, seed).filename.removesuffix(".json")
        t0 = now_fn()
        try:
            log.info("QUEUE start %s (%d/%d)", queue_label(item), queue_pos, len(queue))
            with tee_run_log(cfg.experiment.results_dir, run_tag):
                result = run_fn(cfg, base, condition, size, seed)
            result["launcher_wall_clock_sec"] = round(now_fn() - t0, 3)
            from src.training.finetune import assert_run_result_schema
            assert_run_result_schema(result)
            write_json(result, path)
            completed_durations.append(float(result["launcher_wall_clock_sec"]))
            completed.append(str(path))
        except KeyboardInterrupt:
            log.warning("Interrupted; completed result files are resumable.")
            break
        except Exception as exc:  # noqa: BLE001
            failures.append({"run": queue_label(item), "error": str(exc)})
            log.error("FAILED %s: %s", queue_label(item), exc)
            log.debug(traceback.format_exc())

    state = {
        "queue_count": len(queue),
        "completed": completed,
        "skipped_existing": skipped,
        "failures": failures,
        "elapsed_sec": round(now_fn() - started, 3),
        "budget_hours": budget_hours,
        "disk_preflight_initial": initial_projection,
        "disk_free_bytes_at_start": initial_free,
    }
    write_json(state, Path(cfg.experiment.results_dir) / "launcher_state.json")
    return state


def _print_queue(queue: list[Item]) -> None:
    print(f"TOTAL RUN COUNT: {len(queue)}")
    print("QUEUE ORDER:")
    for index, item in enumerate(queue, 1):
        print(f"{index:3d}. {queue_label(item)}")


def main() -> None:
    parser = base_argparser(__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-priority", action="store_true")
    parser.add_argument("--bases", default=None, help="comma-separated model roles")
    parser.add_argument("--conditions", default=None, help="comma-separated names")
    parser.add_argument("--sizes", default=None, help="comma-separated sizes")
    parser.add_argument("--seeds", default=None, help="comma-separated seeds")
    parser.add_argument("--budget-hours", type=float, default=None)
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)
    verify_config_hashes()  # fail loudly before building/running the queue, even in --dry-run
    roles = args.bases.split(",") if args.bases else None
    bases = resolve_bases(cfg, roles)
    cond_filter = set(args.conditions.split(",")) if args.conditions else None
    size_filter = {int(x) for x in args.sizes.split(",")} if args.sizes else None
    seed_filter = {int(x) for x in args.seeds.split(",")} if args.seeds else None
    queue = build_queue(cfg, bases, args.only_priority, cond_filter,
                        size_filter, seed_filter)
    _print_queue(queue)
    requests, distinct = tr_stage_cache_summary(queue)
    print(f"TURKISH STAGES: {requests} requests, {distinct} distinct checkpoints")
    if args.dry_run:
        return

    from src.training.finetune import run_single

    def run_fn(config, base, condition, size, seed):
        return run_single(config, base, condition, size, seed, eval_split="val")

    state = execute_queue(cfg, queue, run_fn, budget_hours=args.budget_hours)
    print(f"COMPLETED: {len(state['completed'])}; "
          f"SKIPPED: {len(state['skipped_existing'])}; "
          f"FAILED: {len(state['failures'])}")


if __name__ == "__main__":
    main()
