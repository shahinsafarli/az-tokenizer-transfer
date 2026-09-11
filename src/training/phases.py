"""
İKİ FAZALI İCRA + PARALEL AXINLAR  (run plan §6.2 / §6.3)

--------------------------------------------------------------------------
FAZA 1 — türk keşinin SERİAL qurulması, sonra MÖHÜRLƏNMƏSİ
--------------------------------------------------------------------------
30 türk checkpointi 114 run arasında PAYLAŞILIR. İki paralel axın eyni
keşlənməmiş checkpointi istəsə, HƏR İKİSİ onu qurur və HƏR İKİSİ eyni yola
yazır. Nəticə: kəsilmiş və ya bir-birinə qarışmış çəki faylı — və bu fayl
HƏLƏ DƏ yüklənə bilər, sadəcə dəyərləri səhv olar. Heç bir yerdə xəta
mesajı olmaz; nəticələr səssizcə etibarsız olar.

Bunu kilidlə "azaltmaq" olardı. Əvəzinə fazaları AYIRIRIQ: Faza 1 hər şeyi
serial qurur və doğrulama manifesti yazır; Faza 2 keşi YALNIZ-OXUNAN kimi
açır və manifest doğrulanmasa BAŞLAMIR. Yarış EHTİMALSIZ deyil, MÜMKÜNSÜZ
olur.

--------------------------------------------------------------------------
FAZA 2 — 2-3 paralel axın, dörd düzgünlük tələbi ilə
--------------------------------------------------------------------------
Azərbaycan cümlələri orta 29.6 token (p95=79), yəni bir batch təxminən
2,000 token — GPU hesablama ilə deyil, kernel-launch overhead-i ilə
məhdudlaşır və kiçik kernel-lər arasında boş dayanır. Paralellik MƏHZ o boş
vaxtı doldurur; böyük batch-lı iş yükündə bu, demək olar heç nə verməzdi.

Dördü də GÜMAN EDİLMİR, TƏMİN OLUNUR:
  1. Növbə elementləri DEQUEUE zamanı ATOMAR tutulur (başlanğıcda
     "yoxlanmır" — bu, yarışdır). `os.open(..., O_CREAT | O_EXCL)` POSIX-də
     də, Windows-da da atomardır.
  2. AYRI OS prosesləri — heç bir qlobal random state paylaşılmır. (Thread
     işlətsəydik, `set_all_seeds` prosesə-qlobal olduğu üçün axınlar
     bir-birinin seed-ini üstələyərdi.)
  3. BİRLƏŞMİŞ VRAM tavanı — nadir 256-token batch-ları eyni anda zirvəyə
     çıxa bilər, ona görə tavan per-proses deyil, CƏMİ üzərindədir.
  4. `streams_active` HƏR nəticə faylına yazılır — rəqabət altında ölçülən
     per-run wall-clock həmin run-un TƏK QALDIQDA çəkəcəyi vaxt DEYİL və
     məqalədə vaxt rəqəmi kimi VERİLMƏMƏLİDİR.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.training.orchestrate import (Item, queue_label, result_path,
                                      tr_stage_cache_summary)
from src.utils import (ensure_dir, get_logger, read_json, write_json)

log = get_logger(__name__)

PHASE1_MANIFEST = "phase1_manifest.json"
PHASE1_SCHEMA = 1
CLAIM_SUFFIX = ".claim"
# Tutulmuş, amma prosesi ölmüş elementi yenidən növbəyə qaytarmaq üçün yaş həddi.
STALE_CLAIM_SEC = 6 * 3600


# ==================================================================== FAZA 1
def phase1_required_specs(cfg, queue: list[Item]) -> list[dict]:
    """Növbənin tələb etdiyi FƏRQLİ türk checkpointləri (elmi açar üzrə)."""
    seen: dict[tuple, dict] = {}
    for base, condition, _size, seed in queue:
        if condition.get("turkish") == "none":
            continue
        key = (base.short, condition.get("tokenizer"), condition.get("turkish"), seed)
        seen.setdefault(key, {
            "base": base.short, "base_role": base.role,
            "tokenizer": condition["tokenizer"], "turkish": condition["turkish"],
            "seed": int(seed), "condition": condition["name"],
        })
    return [seen[k] for k in sorted(seen)]


def _assert_specs_are_injective(specs: list[dict]) -> None:
    """
    PARALEL FAZA 1-in TƏHLÜKƏSİZLİK ŞƏRTİ — güman edilmir, YOXLANILIR.

    İki fazalı icranın aradan qaldırdığı yarış TƏKRAR-İSTƏK yarışıdır: iki
    axın EYNİ keşlənməmiş checkpointi istəyir və HƏR İKİSİ eyni yola yazır.
    Faza 1-in öz iş siyahısı `phase1_required_specs()` tərəfindən ARTIQ
    dedup edilib, keş yolu isə spesifikasiyanın sha256-sıdır — yəni fərqli
    spesifikasiyalar fərqli yollara düşür və PAYLAŞILAN YAZI HƏDƏFİ YOXDUR.

    Bu şərt pozulsaydı (məsələn `_tr_cache_spec` sonradan bir sahəni açardan
    çıxarsaydı), paralel Faza 1 səssizcə korlanmış çəkilər istehsal edərdi.
    Ona görə şərt hər dəfə yoxlanılır və pozulanda LOUD dayanır.
    """
    keys = {(s["base"], s["tokenizer"], s["turkish"], int(s["seed"])) for s in specs}
    if len(keys) != len(specs):
        raise SystemExit(
            "PARALEL FAZA 1 RƏDD EDİLDİ: iş siyahısı dedup edilməyib "
            f"({len(specs)} spesifikasiya, {len(keys)} fərqli açar). İki işçi "
            "eyni keş yoluna yaza bilər — bu, səssizcə korlanmış çəkilər "
            "deməkdir. --phase1-streams 1 ilə serial işlədin.")


def _phase1_worker_command(cfg_path: str, spec: dict) -> list[str]:
    return [
        sys.executable, "-u", "-m", "src.training.run_grid",
        "--config", cfg_path, "--phase1-build-one",
        "--base-role", spec["base_role"],
        "--condition", spec["condition"],
        "--seed", str(spec["seed"]),
    ]


def build_phase1_cache(cfg, queue: list[Item], cfg_path: str | None = None, *,
                       streams: int = 1, dry_run: bool = False) -> dict:
    """
    Hər fərqli türk checkpointini qurur və möhürlü manifest yazır.

    AZ mərhələsi keşin açarına DAXİL DEYİL (`_tr_cache_spec` onu saymır),
    ona görə burada AZ mərhələsi TAMAMİLƏ ATLANIR — yalnız TR mərhələsi
    icra olunur (`build_tr_stage_only`).

    `streams > 1` yalnız `_assert_specs_are_injective()` keçdikdə mümkündür
    (bax orada: paylaşılan yazı hədəfi yoxdur). Default SERİALDIR — mövcud
    zəmanət dəyişmir, paralellik AÇIQ şəkildə istənilməlidir. Kirayə
    götürülmüş saatlıq GPU-da bu blok başqa cür azaldıla bilməyən ~4 saatlıq
    serial gecikmədir (bax docs/COMPUTE_ESTIMATES.md).
    """
    from src.training.finetune import build_tr_stage_only

    cache_dir = ensure_dir(Path(cfg.run.get("tr_stage_cache_dir",
                                            "artifacts/tr_stage_cache")))
    specs = phase1_required_specs(cfg, queue)
    requests, distinct = tr_stage_cache_summary(queue)
    streams = max(1, int(streams))
    if streams > 1:
        _assert_specs_are_injective(specs)
        if cfg_path is None:
            raise SystemExit("Paralel Faza 1 üçün konfiq yolu lazımdır")
    log.info("FAZA 1: %d TR istəyi → %d fərqli checkpoint (%s)",
             requests, distinct, "serial" if streams == 1 else f"{streams} axın")

    entries: list[dict] = []
    if dry_run:
        entries = [{**s, "status": "DRY_RUN", "path": None} for s in specs]
    elif streams == 1:
        for i, spec in enumerate(specs, 1):
            log.info("FAZA 1 [%d/%d] base=%s tok=%s tr=%s seed=%d",
                     i, len(specs), spec["base"], spec["tokenizer"],
                     spec["turkish"], spec["seed"])
            entries.append({**spec, **build_tr_stage_only(cfg, spec)})
    else:
        entries = _build_phase1_parallel(cfg, cfg_path, specs, streams)

    manifest = {
        "schema": PHASE1_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_distinct_checkpoints": len(specs),
        "phase1_streams": streams,
        "sealed": not dry_run,
        "entries": entries,
    }
    write_json(manifest, cache_dir / PHASE1_MANIFEST)
    built = sum(1 for e in entries if e.get("status") == "BUILT")
    log.info("FAZA 1 tamamlandı: %s (%d checkpoint, %d yeni qurulub)",
             cache_dir / PHASE1_MANIFEST, len(entries), built)
    failed = [e for e in entries if e.get("status") == "FAILED"]
    if failed:
        raise SystemExit(
            f"FAZA 1 TAMAMLANMADI: {len(failed)} checkpoint qurulmadı, məsələn "
            f"{failed[0]}. Manifest MÖHÜRLƏNMƏYİB kimi oxunmalıdır — Faza 2 "
            "onsuz da çatışmayan girişdə dayanacaq.")
    return manifest


def _build_phase1_parallel(cfg, cfg_path: str, specs: list[dict],
                           streams: int) -> list[dict]:
    """Faza 1-i ayrı proseslərdə işlədir; hər spesifikasiya DƏQİQ bir dəfə."""
    from src.training.finetune import _tr_cache_path, _tr_cache_spec  # noqa: F401

    pending = list(specs)
    running: list[tuple[subprocess.Popen, dict, float]] = []
    done: list[dict] = []

    def _reap(block: bool) -> None:
        nonlocal running
        still = []
        for proc, spec, t0 in running:
            code = proc.wait() if block else proc.poll()
            if code is None:
                still.append((proc, spec, t0))
                continue
            elapsed = round(time.monotonic() - t0, 1)
            if code == 0:
                log.info("FAZA 1 done base=%s tok=%s tr=%s seed=%d (%.0fs)",
                         spec["base"], spec["tokenizer"], spec["turkish"],
                         spec["seed"], elapsed)
                done.append({**spec, "status": "BUILT",
                             "tr_stage_runtime_sec": elapsed,
                             "path": spec.get("_expected_path")})
            else:
                log.error("FAZA 1 FAILED base=%s seed=%d returncode=%s",
                          spec["base"], spec["seed"], code)
                done.append({**spec, "status": "FAILED", "path": None})
        running = still

    while pending or running:
        while pending and len(running) < streams:
            spec = pending.pop(0)
            log.info("FAZA 1 start base=%s tok=%s tr=%s seed=%d (%d/%d)",
                     spec["base"], spec["tokenizer"], spec["turkish"],
                     spec["seed"], len(done) + len(running) + 1, len(specs))
            proc = subprocess.Popen(
                _phase1_worker_command(cfg_path, spec),
                env={**os.environ, "PYTHONUNBUFFERED": "1"})
            running.append((proc, spec, time.monotonic()))
        _reap(block=not pending)
        if pending and len(running) >= streams:
            time.sleep(2.0)

    # The worker wrote the cache; recover each real path from the cache dir so
    # the manifest points at what exists rather than at what we predicted.
    cache_root = Path(cfg.run.get("tr_stage_cache_dir", "artifacts/tr_stage_cache"))
    by_prefix = {d.name: d for d in cache_root.iterdir()
                 if d.is_dir() and (d / "cache_manifest.json").exists()}
    for entry in done:
        if entry.get("path"):
            continue
        prefix = (f"base={entry['base']}__tok={entry['tokenizer']}__"
                  f"tr={entry['turkish']}__seed={entry['seed']}__")
        match = next((str(p) for name, p in by_prefix.items()
                      if name.startswith(prefix)), None)
        entry["path"] = match
        if match is None and entry.get("status") == "BUILT":
            entry["status"] = "FAILED"
    return done


def verify_phase1_manifest(cfg, queue: list[Item]) -> dict:
    """
    Faza 2 BAŞLAMAZDAN ƏVVƏL: manifest var, möhürlənib, və növbənin tələb
    etdiyi HƏR checkpointi əhatə edirmi. Uyğunsuzluqda LOUD SystemExit.
    """
    cache_dir = Path(cfg.run.get("tr_stage_cache_dir", "artifacts/tr_stage_cache"))
    path = cache_dir / PHASE1_MANIFEST
    needed = phase1_required_specs(cfg, queue)
    if not needed:
        return {"required": 0, "verified": True,
                "note": "Növbədə türk mərhələli run yoxdur — Faza 1 lazım deyil."}

    if not path.exists():
        raise SystemExit(
            f"FAZA 2 RƏDD EDİLDİ: {path} yoxdur. Türk keşi möhürlənməyib, ona "
            "görə paralel axınlar eyni checkpointi eyni anda qura bilər — bu, "
            "səssiz şəkildə korlanmış çəkilər deməkdir.\n"
            "Əvvəlcə: python -m src.training.run_grid --config <cfg> --phase 1")
    manifest = read_json(path)
    if int(manifest.get("schema", 0)) != PHASE1_SCHEMA:
        raise SystemExit(f"FAZA 2 RƏDD EDİLDİ: {path} sxem versiyası uyğun deyil.")
    if not manifest.get("sealed"):
        raise SystemExit(f"FAZA 2 RƏDD EDİLDİ: {path} möhürlənməyib (dry-run?).")

    have = {(e["base"], e["tokenizer"], e["turkish"], int(e["seed"]))
            for e in manifest.get("entries", []) if e.get("path")}
    missing = [s for s in needed
               if (s["base"], s["tokenizer"], s["turkish"], s["seed"]) not in have]
    if missing:
        raise SystemExit(
            f"FAZA 2 RƏDD EDİLDİ: manifestdə {len(missing)} checkpoint çatışmır, "
            f"məsələn {missing[0]}. Faza 1-i tamamlayın.")

    bad = [e for e in manifest["entries"]
           if e.get("path") and not (Path(e["path"]) / "cache_manifest.json").exists()]
    if bad:
        raise SystemExit(
            f"FAZA 2 RƏDD EDİLDİ: {len(bad)} manifest girişinin faktiki keş "
            f"qovluğu yoxdur, məsələn {bad[0].get('path')}.")

    log.info("FAZA 1 manifesti doğrulandı: %d checkpoint, keş YALNIZ-OXUNAN.",
             len(needed))
    return {"required": len(needed), "verified": True, "manifest_path": str(path)}


# ==================================================================== iddia
def _claim_path(results_dir: str | Path, item: Item) -> Path:
    return result_path(results_dir, item).with_suffix(".json" + CLAIM_SUFFIX)


def claim_item(results_dir: str | Path, item: Item, stream_id: int) -> bool:
    """
    Elementi ATOMAR tutur. Tutula bildisə True.

    `O_CREAT | O_EXCL` — faylın mövcudluğunu YOXLAYIB sonra yaratmaq DEYİL;
    yoxla-sonra-yarat məhz aradan qaldırmaq istədiyimiz yarışdır. Bu bayraq
    cütü həm POSIX-də, həm Windows-da atomardır: iki proses eyni anda
    çağırsa, DƏQİQ biri uğur qazanır.
    """
    path = _claim_path(results_dir, item)
    ensure_dir(path.parent)
    if path.exists():
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            age = 0.0
        if age > STALE_CLAIM_SEC:
            log.warning("Köhnəlmiş iddia silinir (%.0f saat): %s", age / 3600, path.name)
            try:
                path.unlink()
            except OSError:
                return False
        else:
            return False
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"stream_id": stream_id, "pid": os.getpid(),
                   "claimed_utc": datetime.now(timezone.utc).isoformat(),
                   "item": queue_label(item)}, f)
    return True


def release_claim(results_dir: str | Path, item: Item) -> None:
    """Uğursuz run-un iddiasını buraxır ki, təkrar işə salınma onu götürə bilsin."""
    try:
        _claim_path(results_dir, item).unlink()
    except OSError:
        pass


# ==================================================================== VRAM
def resolve_stream_count(cfg, requested: int | None = None) -> dict:
    """
    BİRLƏŞMİŞ VRAM tavanına görə axın sayını təyin edir.

    Tavan cəmə tətbiq olunur, per-prosesə yox: nadir 256-token batch-ları
    axınlarda EYNİ ANDA düşə bilər, ona görə "hər biri 4 GB, kart 20 GB"
    hesabı deyil, "3 x 4 = 12 GB, tavan 10 GB" hesabı aparılır.
    """
    ceiling = float(cfg.run.get("vram_ceiling_gb", 10.0))
    per_stream = float(cfg.run.get("vram_per_stream_gb", 4.0))
    configured = int(requested if requested is not None
                     else cfg.run.get("parallel_streams", 1))
    if per_stream <= 0:
        raise SystemExit("run.vram_per_stream_gb müsbət olmalıdır")

    affordable = max(1, int(ceiling // per_stream))
    streams = max(1, min(configured, affordable))
    out = {
        "configured_streams": configured,
        "vram_ceiling_gb": ceiling,
        "vram_per_stream_gb": per_stream,
        "affordable_streams": affordable,
        "streams": streams,
        "projected_peak_gb": round(streams * per_stream, 2),
    }
    if streams < configured:
        log.warning(
            "Axın sayı %d → %d AZALDILDI: %d x %.1f GB = %.1f GB birləşmiş "
            "tavanı (%.1f GB) keçərdi. Brifdə ayrılmış büdcəni keçmək "
            "AVTOMATİK bal itkisidir.",
            configured, streams, configured, per_stream,
            configured * per_stream, ceiling)
    if per_stream > ceiling:
        raise SystemExit(
            f"TƏK run ({per_stream} GB) tavanı ({ceiling} GB) keçir — "
            "gradient accumulation-ı yoxlayın (configs/FROZEN.md).")
    log.info("Axın: %d (proqnozlaşdırılan zirvə %.1f GB / tavan %.1f GB)",
             streams, out["projected_peak_gb"], ceiling)
    return out


# ==================================================================== FAZA 2
def _worker_command(cfg_path: str, item: Item, stream_id: int, streams: int,
                    *, eval_split: str = "val",
                    allow_test_eval: bool = False) -> list[str]:
    base, condition, size, seed = item
    cmd = [
        sys.executable, "-u", "-m", "src.training.finetune",
        "--config", cfg_path,
        "--base", base.role,
        "--condition", condition["name"],
        "--train-size", str(size),
        "--seed", str(seed),
        "--stream-id", str(stream_id),
        "--streams-active", str(streams),
        "--phase2-readonly-cache",
    ]
    # Added 2026-09-07. Forwarded verbatim so a parallel Phase 2 evaluates the
    # same splits as the serial path; the worker re-checks the opt-in itself.
    if eval_split != "val":
        cmd += ["--eval-split", eval_split]
        if allow_test_eval:
            cmd.append("--allow-test-eval")
    return cmd


def run_phase2(cfg, cfg_path: str, queue: list[Item], *, streams: int,
               budget_hours: float | None = None,
               eval_split: str = "val",
               allow_test_eval: bool = False) -> dict:
    """
    Növbəni `streams` ayrı OS prosesi ilə işlədir.

    Hər element bir dəfə DEQUEUE zamanı atomar tutulur; işçi proses onu icra
    edir və nəticəni özü yazır. Launcher yalnız planlaşdırma edir — modelə
    və ya torch-a HEÇ TOXUNMUR, ona görə heç bir qlobal state paylaşılmır.
    """
    results_dir = Path(cfg.experiment.results_dir)
    ensure_dir(results_dir / "runs")
    started = time.monotonic()
    budget_sec = None if budget_hours is None else max(0.0, budget_hours * 3600.0)

    pending = list(queue)
    running: list[tuple[subprocess.Popen, Item, int, float]] = []
    completed: list[str] = []
    skipped: list[str] = []
    failures: list[dict] = []
    next_stream_id = 0

    def _reap(block: bool) -> None:
        nonlocal running
        still: list[tuple[subprocess.Popen, Item, int, float]] = []
        for proc, item, stream_id, t0 in running:
            code = proc.wait() if block else proc.poll()
            if code is None:
                still.append((proc, item, stream_id, t0))
                continue
            path = result_path(results_dir, item)
            if code == 0 and path.exists():
                completed.append(str(path))
                log.info("QUEUE done  %s (%.0fs, stream=%d)",
                         queue_label(item), time.monotonic() - t0, stream_id)
            else:
                failures.append({"run": queue_label(item), "returncode": code})
                release_claim(results_dir, item)
                log.error("FAILED %s (returncode=%s)", queue_label(item), code)
        running = still

    while pending or running:
        if budget_sec is not None and time.monotonic() - started > budget_sec:
            log.info("Büdcə bitdi — yeni element başladılmır; işləyənlər gözlənilir.")
            pending = []

        while pending and len(running) < streams:
            item = pending.pop(0)
            path = result_path(results_dir, item)
            if bool(cfg.run.skip_existing) and path.exists():
                try:
                    if int(read_json(path).get("run_result_schema_version", 0)) == 4:
                        skipped.append(str(path))
                        continue
                except Exception:  # noqa: BLE001
                    pass
            if not claim_item(results_dir, item, next_stream_id):
                log.info("Başqa axın tutub, atlanır: %s", queue_label(item))
                continue
            stream_id = next_stream_id
            next_stream_id += 1
            log.info("QUEUE start %s (stream=%d, active=%d)",
                     queue_label(item), stream_id, len(running) + 1)
            proc = subprocess.Popen(
                _worker_command(cfg_path, item, stream_id, streams,
                                eval_split=eval_split,
                                allow_test_eval=allow_test_eval),
                env={**os.environ, "PYTHONUNBUFFERED": "1"})
            running.append((proc, item, stream_id, time.monotonic()))

        _reap(block=not pending)
        if pending and len(running) >= streams:
            time.sleep(2.0)

    state = {
        "queue_count": len(queue),
        "streams_active": streams,
        "completed": completed,
        "skipped_existing": skipped,
        "failures": failures,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "budget_hours": budget_hours,
        "timing_caveat": (
            "Per-run wall-clock in these results was measured with "
            f"streams_active={streams}. Under contention that is NOT the "
            "run's isolated cost and must not be reported as a timing figure."),
    }
    write_json(state, results_dir / "launcher_state_phase2.json")
    return state
