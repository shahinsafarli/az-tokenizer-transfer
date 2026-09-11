"""
QRİDİN GİRİŞ NÖQTƏSİ  —  iki fazalı, isteğe bağlı paralel

    # Faza 1 — türk keşini SERİAL qur və möhürlə (elmi heç nə ölçmür)
    python -m src.training.run_grid --config configs/experiment.yaml --phase 1

    # Faza 2 — qridi işlət (keş YALNIZ-OXUNAN)
    python -m src.training.run_grid --config configs/experiment.yaml --phase 2 --streams 3

    # Hər ikisi ardıcıl
    python -m src.training.run_grid --config configs/experiment.yaml --phase all

    # Tranche A (run plan §7): xlm15 · n=2000 · baza + tokenizator · 5 seed
    python -m src.training.run_grid --config configs/experiment.yaml --tranche A

`--streams 1` (default) tam serial icra deməkdir və köhnə davranışla eynidir;
paralellik ancaq açıq şəkildə istənildikdə işə düşür.

TRANCHE-lər run plan §7-dəki mərhələli plandır. A və B türk mərhələsi
DAŞIMIR, ona görə Faza 1 onlar üçün LAZIM DEYİL — bu, məhz A-nın 2 saatda
işləyə bilməsinin səbəbidir.
"""
from __future__ import annotations

import argparse
import sys

from src.training.orchestrate import (build_queue, execute_queue, queue_label,
                                      tr_stage_cache_summary)
from src.training.phases import (build_phase1_cache, resolve_stream_count,
                                 run_phase2, verify_phase1_manifest)
from src.utils import (get_logger, load_config, resolve_bases, setup_logging,
                       verify_config_hashes)

log = get_logger(__name__)

# Run plan §7. Hər tranche bir (bases, conditions, sizes) filtridir.
TRANCHES: dict[str, dict] = {
    "A": {"bases": ["primary"], "conditions": {"baza", "tokenizator"},
          "sizes": {2000},
          "what_it_decides": ("Real s/step; şərtlərin basin-dən çıxıb-çıxmadığı; "
                              "per-proses VRAM; yazma/escape/seçim/aqreqasiyanın "
                              "real data üzərində ucdan-uca doğrulanması. "
                              "Türk mərhələsi YOX.")},
    "B": {"bases": ["contrast"], "conditions": {"baza", "tokenizator"},
          "sizes": {2000},
          "what_it_decides": ("Tokenizator effekti AZ görmüş modeldə YOXDURmu. "
                              "A+B birlikdə tam çarpaz-baza rəqəmini verir.")},
    "D": {"bases": ["primary", "contrast"],
          "conditions": {"turk", "her_ikisi", "turk_qarisiq"}, "sizes": {2000},
          "what_it_decides": ("M1 vs M2: real türkcə vs qarışdırılmış türkcə. "
                              "Faza 1 TƏLƏB OLUNUR.")},
    "E": {"bases": ["primary"], "conditions": None, "sizes": {500, 10000, 20914},
          "what_it_decides": "Data əyrisi — effektin harada itdiyi."},
}


def _resolve_filters(args, cfg):
    if args.tranche:
        spec = TRANCHES[args.tranche]
        log.info("TRANCHE %s — %s", args.tranche, spec["what_it_decides"])
        roles = spec["bases"]
        conditions = spec["conditions"]
        sizes = spec["sizes"]
    else:
        roles = args.bases.split(",") if args.bases else None
        conditions = set(args.conditions.split(",")) if args.conditions else None
        sizes = {int(x) for x in args.sizes.split(",")} if args.sizes else None
    seeds = {int(x) for x in args.seeds.split(",")} if args.seeds else None
    return roles, conditions, sizes, seeds


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--phase", default="all", choices=["1", "2", "all"])
    ap.add_argument("--phase1-streams", type=int, default=None,
                    help="parallel workers for Phase 1 (default: run.phase1_streams, "
                         "else 1). Safe only because Phase 1's work list is "
                         "deduplicated, so no two workers share a cache path; "
                         "the invariant is asserted, not assumed.")
    # Internal: one Phase-1 worker builds exactly one Turkish checkpoint.
    ap.add_argument("--phase1-build-one", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--base-role", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--condition", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--tranche", default=None, choices=sorted(TRANCHES))
    ap.add_argument("--streams", type=int, default=None,
                    help="override run.parallel_streams (capped by the VRAM ceiling)")
    ap.add_argument("--bases", default=None)
    ap.add_argument("--conditions", default=None)
    ap.add_argument("--sizes", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--budget-hours", type=float, default=None)
    ap.add_argument("--only-priority", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    # Held-out test split (added 2026-09-07). Training-time evaluation stays on
    # VALIDATION in both cases: src.training.finetune._train_stage always passes
    # az_val as the Trainer's eval_dataset, so escape/selection evidence is
    # unchanged. --eval-split test only adds ONE terminal evaluation of the
    # final-step model on az_test, plus a ledger entry per run. Both flags are
    # required together; run_single raises PermissionError otherwise.
    ap.add_argument("--eval-split", choices=["val", "test"], default="val")
    ap.add_argument("--allow-test-eval", action="store_true",
                    help="required explicit opt-in when --eval-split test")
    args = ap.parse_args()
    if args.eval_split == "test" and not args.allow_test_eval:
        raise SystemExit(
            "--eval-split test requires --allow-test-eval. The test split is "
            "spent once: pass both flags deliberately or neither.")

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    verify_config_hashes()  # fail loudly before anything trains, including --dry-run

    # ---------------------------------------------------- Phase-1 worker
    if args.phase1_build_one:
        from src.training.finetune import build_tr_stage_only

        base = resolve_bases(cfg, [args.base_role])[0]
        info = build_tr_stage_only(cfg, {
            "base": base.short, "base_role": base.role,
            "condition": args.condition, "seed": int(args.seed),
        })
        print(f"PHASE1 {info['status']} {info['path']}")
        return

    roles, conditions, sizes, seeds = _resolve_filters(args, cfg)
    bases = resolve_bases(cfg, roles)
    queue = build_queue(cfg, bases, args.only_priority, conditions, sizes, seeds)

    requests, distinct = tr_stage_cache_summary(queue)
    print(f"TOTAL RUN COUNT: {len(queue)}")
    print(f"TURKISH STAGES: {requests} requests, {distinct} distinct checkpoints")
    if args.dry_run:
        for i, item in enumerate(queue, 1):
            print(f"{i:3d}. {queue_label(item)}")
        return
    if not queue:
        raise SystemExit("Növbə boşdur — filtrləri yoxlayın.")

    # ---------------------------------------------------------------- Faza 1
    if args.phase in {"1", "all"}:
        if distinct == 0:
            log.info("Faza 1 ATLANIR: bu növbədə türk mərhələli run yoxdur.")
        else:
            phase1_streams = int(
                args.phase1_streams if args.phase1_streams is not None
                else cfg.run.get("phase1_streams", 1))
            # Phase 1 holds one model per worker, same footprint as Phase 2,
            # so it obeys the same combined VRAM ceiling.
            capped = resolve_stream_count(cfg, phase1_streams)["streams"]
            build_phase1_cache(cfg, queue, args.config, streams=capped)
        if args.phase == "1":
            print("PHASE 1 COMPLETE — cache sealed. Nothing scientific was measured.")
            return

    # ---------------------------------------------------------------- Faza 2
    verify_phase1_manifest(cfg, queue)
    plan = resolve_stream_count(cfg, args.streams)
    streams = plan["streams"]

    if streams == 1:
        # Tək axın: alt-proses yükü olmadan mövcud serial icraçı işlədilir.
        from src.training.finetune import run_single

        def run_fn(config, base, condition, size, seed):
            return run_single(config, base, condition, size, seed,
                              eval_split=args.eval_split,
                              allow_test_eval=args.allow_test_eval,
                              streams_active=1,
                              readonly_tr_cache=True)

        state = execute_queue(cfg, queue, run_fn, budget_hours=args.budget_hours)
    else:
        state = run_phase2(cfg, args.config, queue, streams=streams,
                           budget_hours=args.budget_hours,
                           eval_split=args.eval_split,
                           allow_test_eval=args.allow_test_eval)

    print(f"COMPLETED: {len(state['completed'])}; "
          f"SKIPPED: {len(state['skipped_existing'])}; "
          f"FAILED: {len(state['failures'])}")
    if state["failures"]:
        for failure in state["failures"]:
            print("  FAILED:", failure)
        sys.exit(1)


if __name__ == "__main__":
    main()
