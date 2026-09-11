"""
PARALELLİK DÜZGÜNLÜK TESTİ  —  run plan §6.3, "təsdiqlənir, güman edilmir"

Run plan dörd tələb qoyur və hər birinin GÜMAN edilməsini deyil, TEST
edilməsini tələb edir:

  1. növbə elementləri DEQUEUE zamanı ATOMAR tutulur (başlanğıcda yoxlanmır);
  2. AYRI OS prosesləri — qlobal random state paylaşılmır;
  3. BİRLƏŞMİŞ VRAM tavanı — axınlar eyni anda zirvəyə çıxa bilər;
  4. `streams_active` hər nəticə faylına yazılır.

Bu fayl 1, 3 və 4-ü birbaşa yoxlayır. 2-ci (ayrı proses) `run_phase2`-nin
`subprocess.Popen` ilə `src.training.finetune`-u çağırması ilə struktur
olaraq təmin olunur — burada həmin əmrin FORMASI yoxlanılır (torch tələb
etmədən).
"""
from __future__ import annotations

import copy
import multiprocessing as mp
from pathlib import Path

import pytest

from src.training.orchestrate import build_queue, result_path
from src.training.phases import (PHASE1_MANIFEST, _worker_command, claim_item,
                                 phase1_required_specs, release_claim,
                                 resolve_stream_count, verify_phase1_manifest)
from src.utils import load_config, resolve_bases, write_json


def _cfg(tmp_path, **run_overrides):
    cfg = copy.deepcopy(load_config("configs/experiment.yaml"))
    cfg["experiment"]["results_dir"] = str(tmp_path / "results")
    cfg["run"]["tr_stage_cache_dir"] = str(tmp_path / "cache")
    cfg["run"].update(run_overrides)
    return cfg


# ============================================================ 1 · atomar iddia
def test_claim_is_exclusive_within_a_process(tmp_path):
    cfg = _cfg(tmp_path)
    item = build_queue(cfg, resolve_bases(cfg))[0]
    results_dir = cfg["experiment"]["results_dir"]
    assert claim_item(results_dir, item, stream_id=0) is True
    assert claim_item(results_dir, item, stream_id=1) is False
    release_claim(results_dir, item)
    assert claim_item(results_dir, item, stream_id=2) is True


def _claim_worker(args):
    """Ayrı prosesdə işləyir — modul səviyyəsində olmalıdır ki, picklenə bilsin."""
    import copy as _copy

    from src.training.orchestrate import build_queue as _bq
    from src.training.phases import claim_item as _claim
    from src.utils import load_config as _lc, resolve_bases as _rb

    results_dir, index = args
    cfg = _copy.deepcopy(_lc("configs/experiment.yaml"))
    item = _bq(cfg, _rb(cfg))[0]
    return _claim(results_dir, item, stream_id=index)


def test_claim_is_exclusive_across_real_processes(tmp_path):
    """
    Əsl tələb budur: eyni elementi EYNİ ANDA istəyən N proses arasında
    DƏQİQ biri uğur qazanmalıdır. "Mövcuddurmu?" yoxlayıb sonra yaratmaq
    bu testi keçə bilməz — `os.open(O_CREAT|O_EXCL)` keçir.
    """
    results_dir = str(tmp_path / "results")
    Path(results_dir, "runs").mkdir(parents=True)
    n = 8
    with mp.Pool(n) as pool:
        outcomes = pool.map(_claim_worker, [(results_dir, i) for i in range(n)])
    assert sum(outcomes) == 1, f"tam bir qalib gözlənilirdi, alındı {sum(outcomes)}"


def test_every_queue_item_is_claimed_exactly_once(tmp_path):
    """Bütün növbə üzərində: hər element bir dəfə tutulur, təkrar tutulmur."""
    cfg = _cfg(tmp_path)
    queue = build_queue(cfg, resolve_bases(cfg))
    results_dir = cfg["experiment"]["results_dir"]
    first = [claim_item(results_dir, item, 0) for item in queue]
    second = [claim_item(results_dir, item, 1) for item in queue]
    assert all(first) and not any(second)
    claims = list(Path(results_dir, "runs").glob("*.claim"))
    assert len(claims) == len(queue) == 164
    # Fayl adları unikaldır → iki element eyni nəticə faylına yaza bilməz.
    assert len({result_path(results_dir, i).name for i in queue}) == len(queue)


# ============================================================ 3 · VRAM tavanı
def test_stream_count_is_capped_by_the_combined_vram_ceiling(tmp_path):
    """3 x 4 GB = 12 GB > 10 GB tavan → 2 axına ENDİRİLİR."""
    cfg = _cfg(tmp_path, parallel_streams=3, vram_ceiling_gb=10.0,
               vram_per_stream_gb=4.0)
    plan = resolve_stream_count(cfg)
    assert plan["streams"] == 2
    assert plan["projected_peak_gb"] == 8.0


def test_three_streams_allowed_when_they_fit(tmp_path):
    cfg = _cfg(tmp_path, parallel_streams=3, vram_ceiling_gb=12.0,
               vram_per_stream_gb=3.5)
    plan = resolve_stream_count(cfg)
    assert plan["streams"] == 3
    assert plan["projected_peak_gb"] == 10.5


def test_single_run_over_the_ceiling_is_a_hard_stop(tmp_path):
    """Brifdə ayrılmış büdcəni keçmək avtomatik bal itkisidir — sükutla davam etmirik."""
    cfg = _cfg(tmp_path, parallel_streams=1, vram_ceiling_gb=10.0,
               vram_per_stream_gb=12.0)
    with pytest.raises(SystemExit, match="tavan"):
        resolve_stream_count(cfg)


# ============================================================ 2 · ayrı proses
def test_worker_command_launches_a_separate_process_with_stream_metadata(tmp_path):
    cfg = _cfg(tmp_path)
    item = next(i for i in build_queue(cfg, resolve_bases(cfg))
                if i[1]["turkish"] != "none")
    cmd = _worker_command("configs/experiment.yaml", item, stream_id=2, streams=3)
    assert "-m" in cmd and "src.training.finetune" in cmd
    assert "--streams-active" in cmd and cmd[cmd.index("--streams-active") + 1] == "3"
    assert "--stream-id" in cmd and cmd[cmd.index("--stream-id") + 1] == "2"
    # Faza 2 keşə YAZA bilməməlidir.
    assert "--phase2-readonly-cache" in cmd


# ============================================================ Faza 1 möhürü
def test_phase2_refuses_to_start_without_a_sealed_manifest(tmp_path):
    cfg = _cfg(tmp_path)
    queue = build_queue(cfg, resolve_bases(cfg))
    with pytest.raises(SystemExit, match="FAZA 2 RƏDD EDİLDİ"):
        verify_phase1_manifest(cfg, queue)


def test_phase2_refuses_an_incomplete_manifest(tmp_path):
    cfg = _cfg(tmp_path)
    queue = build_queue(cfg, resolve_bases(cfg))
    specs = phase1_required_specs(cfg, queue)
    assert len(specs) == 30
    cache = Path(cfg["run"]["tr_stage_cache_dir"])
    entry_dir = cache / "one"
    (entry_dir).mkdir(parents=True)
    (entry_dir / "cache_manifest.json").write_text("{}", encoding="utf-8")
    write_json({"schema": 1, "sealed": True, "n_distinct_checkpoints": 1,
                "entries": [{**specs[0], "path": str(entry_dir)}]},
               cache / PHASE1_MANIFEST)
    with pytest.raises(SystemExit, match="çatışmır"):
        verify_phase1_manifest(cfg, queue)


def test_phase1_is_not_required_for_a_turkish_free_tranche(tmp_path):
    """Tranche A/B türk mərhələsi daşımır — məhz buna görə 2 saatda işləyir."""
    cfg = _cfg(tmp_path)
    queue = build_queue(cfg, resolve_bases(cfg, ["primary"]),
                        cond_filter={"baza", "tokenizator"}, size_filter={2000})
    assert len(queue) == 10
    assert phase1_required_specs(cfg, queue) == []
    assert verify_phase1_manifest(cfg, queue)["verified"] is True


# ============================================================ paralel Faza 1
def test_phase1_specs_are_injective_so_workers_cannot_share_a_path():
    """
    Paralel Faza 1-in bütün təhlükəsizlik arqumenti budur: 30 spesifikasiya →
    30 fərqli keş açarı → 30 fərqli yol. Ona görə iki işçi eyni fayla yaza
    bilməz və iki fazalı icranın aradan qaldırdığı TƏKRAR-İSTƏK yarışı Faza
    1-in öz siyahısında BAŞ VERƏ BİLMƏZ.
    """
    from src.training.phases import _assert_specs_are_injective

    cfg = load_config("configs/experiment.yaml")
    specs = phase1_required_specs(cfg, build_queue(cfg, resolve_bases(cfg)))
    assert len(specs) == 30
    keys = {(s["base"], s["tokenizer"], s["turkish"], s["seed"]) for s in specs}
    assert len(keys) == 30
    _assert_specs_are_injective(specs)  # must not raise


def test_parallel_phase1_refuses_a_non_deduplicated_work_list():
    """Şərt pozulsa, paralel Faza 1 səssizcə davam etməməli — dayanmalıdır."""
    from src.training.phases import _assert_specs_are_injective

    duplicated = [
        {"base": "xlm15", "tokenizer": "original", "turkish": "real", "seed": 42},
        {"base": "xlm15", "tokenizer": "original", "turkish": "real", "seed": 42},
    ]
    with pytest.raises(SystemExit, match="dedup edilməyib"):
        _assert_specs_are_injective(duplicated)


def test_phase1_worker_command_builds_exactly_one_checkpoint():
    from src.training.phases import _phase1_worker_command

    cfg = load_config("configs/experiment.yaml")
    spec = phase1_required_specs(cfg, build_queue(cfg, resolve_bases(cfg)))[0]
    cmd = _phase1_worker_command("configs/experiment.yaml", spec)
    assert "src.training.run_grid" in cmd
    assert "--phase1-build-one" in cmd
    assert cmd[cmd.index("--seed") + 1] == str(spec["seed"])
    assert cmd[cmd.index("--condition") + 1] == spec["condition"]
    # No --streams / --phase: a worker builds one checkpoint and exits.
    assert "--streams" not in cmd and "--phase" not in cmd


def test_phase1_fails_closed_when_a_worker_does_not_produce_a_checkpoint(tmp_path,
                                                                         monkeypatch):
    """
    Faza 1 QISMƏN uğurla bitə bilməz: möhürlənmiş, amma natamam manifest Faza
    2-nin bütün zəmanətini pozardı (o, manifestə güvənir). Bir işçi belə
    uğursuz olarsa, LOUD dayanılır.
    """
    import src.training.phases as phases

    cfg = _cfg(tmp_path, phase1_streams=2, vram_ceiling_gb=100.0,
               vram_per_stream_gb=1.0)
    queue = build_queue(cfg, resolve_bases(cfg), cond_filter={"turk"},
                        size_filter={2000})
    specs = phase1_required_specs(cfg, queue)
    assert specs

    def fake_parallel(_cfg, _cfg_path, _specs, _streams):
        return [{**s, "status": "FAILED", "path": None} for s in _specs]

    monkeypatch.setattr(phases, "_build_phase1_parallel", fake_parallel)
    with pytest.raises(SystemExit, match="FAZA 1 TAMAMLANMADI"):
        phases.build_phase1_cache(cfg, queue, "configs/experiment.yaml", streams=2)


def test_phase1_manifest_records_the_worker_count(tmp_path, monkeypatch):
    """`phase1_streams` manifestə yazılır — sonradan necə qurulduğu bilinsin."""
    import src.training.phases as phases
    from src.utils import read_json

    cfg = _cfg(tmp_path, vram_ceiling_gb=100.0, vram_per_stream_gb=1.0)
    queue = build_queue(cfg, resolve_bases(cfg), cond_filter={"turk"},
                        size_filter={2000})
    built = {}

    def fake_parallel(_cfg, _cfg_path, _specs, _streams):
        built["streams"] = _streams
        out = []
        for s in _specs:
            d = Path(cfg["run"]["tr_stage_cache_dir"]) / f"ck_{s['seed']}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "cache_manifest.json").write_text("{}", encoding="utf-8")
            out.append({**s, "status": "BUILT", "path": str(d)})
        return out

    monkeypatch.setattr(phases, "_build_phase1_parallel", fake_parallel)
    phases.build_phase1_cache(cfg, queue, "configs/experiment.yaml", streams=3)
    assert built["streams"] == 3
    on_disk = read_json(Path(cfg["run"]["tr_stage_cache_dir"]) / PHASE1_MANIFEST)
    assert on_disk["phase1_streams"] == 3
    assert on_disk["sealed"] is True
