"""End-to-end execution of the frozen pipeline on a tiny local model.

The project state document lists "One complete run under the frozen config" as
**Never executed** — result writing, escape detection and post-hoc selection had
never run together on real data, and the first real run would have exercised all
three for the first time. This test does exactly that, in seconds, by building a
2-layer XLM-R around the real tokenizer:

  1. `run_single` for an AZ-only condition, through Trainer, escape detection,
     post-hoc selection and the schema assertion;
  2. Phase 1 `build_tr_stage_only` publishes a Turkish checkpoint to the cache;
  3. Phase 2 `run_single(readonly_tr_cache=True)` HITS that cache and records
     `streams_active`;
  4. Phase 2 REFUSES a cache miss instead of rebuilding it — the concurrent
     duplicate-write failure mode that two-phase execution exists to remove;
  5. `aggregate` + `report` regenerate the paper tables and figures from the
     real result files.

Needs the Hub for the tokenizer, so it skips when offline.
"""
from __future__ import annotations

import copy
import json
import random
import shutil
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


def _hub_reachable() -> bool:
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("xlm-roberta-base")
        return True
    except Exception:  # noqa: BLE001 — offline, rate-limited, gated, anything
        return False


AZ_POSITIVE = ["Bu film həqiqətən əla idi və çox bəyəndim",
               "Xidmət mükəmməl idi, hamıya tövsiyə edirəm",
               "Məhsul gözlədiyimdən qat-qat yaxşı çıxdı"]
AZ_NEGATIVE = ["Tamamilə bərbad idi, pulumu havaya sovurdum",
               "Heç işləmir, dəfələrlə şikayət etdim",
               "Keyfiyyət çox aşağıdır, məmnun qalmadım"]
TR_SENTENCES = ["Bu ürün gerçekten harikaydı ve çok beğendim",
                "Tamamen berbattı, param boşa gitti",
                "Hizmet fena değildi ama beklediğim gibi olmadı"]


def _write_jsonl(rows, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_tiny_base(work: Path):
    """A real tokenizer on a randomly-initialised 2-layer body — seconds on CPU."""
    from transformers import (AutoTokenizer, XLMRobertaConfig,
                              XLMRobertaForMaskedLM)

    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = XLMRobertaForMaskedLM(XLMRobertaConfig(
        vocab_size=tokenizer.vocab_size + 2, hidden_size=64, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=128,
        max_position_embeddings=512, type_vocab_size=1))
    tiny = work / "tiny_base"
    model.save_pretrained(tiny)
    tokenizer.save_pretrained(tiny)
    return tiny


def _write_synthetic_data(work: Path) -> Path:
    rng = random.Random(0)
    datadir = work / "artifacts" / "data"
    datadir.mkdir(parents=True)

    def az_rows(n):
        rows = []
        for i in range(n):
            positive = i % 2 == 0
            pool = AZ_POSITIVE if positive else AZ_NEGATIVE
            rows.append({"text": f"{rng.choice(pool)} {i}",
                         "label": 1 if positive else 0})
        rng.shuffle(rows)
        return rows

    _write_jsonl(az_rows(80), datadir / "az_train.jsonl")
    _write_jsonl(az_rows(24), datadir / "az_val.jsonl")
    _write_jsonl(az_rows(24), datadir / "az_test.jsonl")
    tr_rows = [{"text": f"{rng.choice(TR_SENTENCES)} {i}", "label": i % 3}
               for i in range(60)]
    _write_jsonl(tr_rows, datadir / "tr_train.jsonl")
    _write_jsonl([{"text": " ".join(rng.sample(r["text"].split(),
                                               len(r["text"].split()))),
                   "label": r["label"]} for r in tr_rows],
                 datadir / "tr_train_scrambled.jsonl")
    return datadir


def _smoke_config(work: Path, tiny: Path):
    from src.utils import load_config

    cfg = copy.deepcopy(load_config("configs/experiment.yaml"))
    cfg["experiment"].update(
        artifacts_dir=str(work / "artifacts"), results_dir=str(work / "results"),
        figures_dir=str(work / "figures"), deterministic=False, seeds=[42],
        seeds_high_resource=[42])
    cfg["models"]["primary"]["id"] = str(tiny)
    cfg["models"]["contrast"]["id"] = str(tiny)
    cfg["models"]["max_length"] = 64
    cfg["training"].update(max_steps=12, eval_every_steps=4, epochs_tr=1,
                           batch_size=4, grad_accum=2, eval_batch_size=8,
                           fp16=False)
    cfg["data"]["train_sizes"] = [40]
    cfg["analysis"]["reference_size"] = 40
    cfg["run"].update(contrast_sizes=[40],
                      tr_stage_cache_dir=str(work / "artifacts" / "tr_stage_cache"),
                      disk_budget_bytes_per_tr_checkpoint=0,
                      disk_budget_bytes_per_run_overhead=0)
    # The frozen config scopes Turkish-bearing conditions to n=2000; this runs
    # at n=40, so re-scope them or `condition_applies` filters them all out.
    for condition in cfg["conditions"]:
        if condition.get("sizes") != "all":
            condition["sizes"] = [40]
    return cfg


@pytest.mark.slow
@pytest.mark.skipif(not _hub_reachable(),
                    reason="needs the Hub for the xlm-roberta-base tokenizer")
def test_frozen_pipeline_completes_a_real_run_end_to_end():
    from src.analysis.aggregate import aggregate_cells, load_runs
    from src.analysis.report import generate_report
    from src.training.finetune import (build_tr_stage_only, run_single,
                                       write_run_result)
    from src.training.orchestrate import build_queue
    from src.training.phases import phase1_required_specs
    from src.utils import RunKey, resolve_bases

    work = Path(tempfile.mkdtemp(prefix="e2e_"))
    try:
        tiny = _build_tiny_base(work)
        _write_synthetic_data(work)
        cfg = _smoke_config(work, tiny)
        base = resolve_bases(cfg, ["primary"])[0]
        conditions = {c["name"]: c for c in cfg.conditions}
        runs_dir = Path(cfg["experiment"]["results_dir"]) / "runs"
        runs_dir.mkdir(parents=True)

        # --- 1. a complete AZ-only run -------------------------------------
        result = run_single(cfg, base, conditions["baza"], 40, 42,
                            eval_split="val", streams_active=1,
                            readonly_tr_cache=True)
        assert result["run_result_schema_version"] == 4
        assert result["streams_active"] == 1
        # ceil(40 / (batch 4 * accum 2)) — accumulation folded into the epoch count
        assert result["steps_per_epoch"] == 5
        assert len(result["step_history"]) >= 3
        assert result["per_run_verdict"] in {"ESCAPED", "NOT_ESCAPED", "UNRESOLVED"}
        write_run_result(result,
                         runs_dir / RunKey(base.short, "baza", 40, 42).filename)

        # --- 2. Phase 1 builds and publishes the Turkish checkpoint --------
        queue = build_queue(cfg, [base], cond_filter={"turk"}, size_filter={40})
        specs = phase1_required_specs(cfg, queue)
        assert len(specs) == 1
        info = build_tr_stage_only(cfg, specs[0])
        assert info["status"] == "BUILT"
        assert (Path(info["path"]) / "cache_manifest.json").exists()

        # --- 3. Phase 2 reuses it and records the stream count -------------
        turkish = run_single(cfg, base, conditions["turk"], 40, 42,
                             eval_split="val", streams_active=3,
                             readonly_tr_cache=True)
        assert turkish["tr_cache_hit"] is True, "Phase 2 rebuilt instead of reusing"
        assert turkish["streams_active"] == 3
        write_run_result(turkish,
                         runs_dir / RunKey(base.short, "turk", 40, 42).filename)

        # --- 4. a read-only cache miss is a loud refusal, not a rebuild ----
        shutil.rmtree(cfg["run"]["tr_stage_cache_dir"])
        with pytest.raises(SystemExit, match="PHASE-2 CACHE MISS"):
            run_single(cfg, base, conditions["turk"], 40, 1337,
                       eval_split="val", readonly_tr_cache=True)

        # --- 5. tables and figures regenerate from results/ ----------------
        frame = load_runs(cfg["experiment"]["results_dir"])
        assert len(frame) == 2
        cells = aggregate_cells(frame, cfg)
        assert sum(1 for c in cells if c["status"] == "MEASURED") == 2
        generate_report(cfg)
        assert (Path(cfg["experiment"]["results_dir"]) / "paper_tables.md").exists()
        assert (Path(cfg["experiment"]["figures_dir"]) / "escape_rate.png").exists()
    finally:
        shutil.rmtree(work, ignore_errors=True)
