"""Ümumi köməkçilər: konfiq, seed, loglama, IO."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import logging
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOG_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    # Defensive against buffered stdout even when PYTHONUNBUFFERED isn't set
    # by the launcher (e.g. an IDE run configuration): force line buffering
    # so log lines appear as they're emitted, not in a batch at exit.
    # Also force UTF-8 with replacement on stdout/stderr. Log messages across
    # this codebase are written in Azerbaijani; a Windows console defaults to
    # cp1252 and raises UnicodeEncodeError mid-log, which prints a spurious
    # traceback *after* the work already succeeded (observed on the stats
    # entry point). `errors="replace"` degrades a character, never the run.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace",
                               line_buffering=True)
        except (AttributeError, ValueError):
            pass  # not a real TextIOWrapper (e.g. captured stdout in tests)
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=LOG_FMT,
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextlib.contextmanager
def tee_run_log(results_dir: str | Path, run_tag: str):
    """Attach a per-run FileHandler to the root logger for one run's duration.

    Every log line already emitted to stdout (STAGE transitions, step-timing
    lines, eval results) is duplicated verbatim into
    ``results_dir/logs/<run_tag>__<utc-timestamp>.log`` so a stalled or
    interrupted run leaves a durable, step-level record — not just whatever
    survived in a terminal scrollback. ``logging.FileHandler`` flushes on
    every emitted record by default, so the file is current even if the
    process is killed mid-run.
    """
    log_dir = ensure_dir(Path(results_dir) / "logs")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{run_tag}__{ts}.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FMT, datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield log_path
    finally:
        handler.close()
        root.removeHandler(handler)


# ---------------------------------------------------------------- konfiq
class Config(dict):
    """Nöqtə ilə müraciət edilə bilən dict:  cfg.models.primary.id"""

    def __getattr__(self, item: str) -> Any:
        try:
            v = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        return Config(v) if isinstance(v, dict) else v

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path, require_complete: bool = True) -> Config:
    """
    require_complete=False → 'FILL' sahələri yoxlanılmır.

    Bu, tokenizator analizi (anchors, fertility) üçün lazımdır: onlar dataset
    konfiqindən asılı deyil və GO/NO-GO qapısı olduğu üçün datasetlər
    seçilməmişdən ƏVVƏL işlədilməlidir.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Config(raw)
    if require_complete:
        _check_fill(cfg)
    return cfg


def verify_config_hashes(root: str | Path = ".",
                         lock_path: str | Path = "configs/CONFIG_HASHES.lock") -> None:
    """Fail loudly if the frozen config files don't match the committed hashes.

    Guards against the one thing that would silently invalidate a
    cross-machine comparison: a config drift between the machine that
    produced one half of the results and the machine producing the other
    half (e.g. the 4060 dev checkout vs. an A100 clone). Every real training
    entry point (src.training.orchestrate.main, src.training.finetune.main)
    calls this immediately after load_config(), including in --dry-run.
    """
    import hashlib

    root = Path(root)
    lock_file = root / lock_path
    if not lock_file.exists():
        raise SystemExit(
            f"CONFIG HASH CHECK FAILED: {lock_file} does not exist. "
            "Refusing to proceed without a hash lock — this is not a "
            "missing-file edge case to skip past.")
    with open(lock_file, "r", encoding="utf-8") as f:
        expected = json.load(f)

    mismatches = []
    for rel_path, expected_hash in expected.items():
        if rel_path.startswith("_"):
            continue  # e.g. "_comment"
        target = root / rel_path
        if not target.exists():
            mismatches.append(f"  {rel_path}: MISSING (expected sha256 {expected_hash})")
            continue
        actual_hash = hashlib.sha256(target.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if actual_hash != expected_hash:
            mismatches.append(
                f"  {rel_path}: expected {expected_hash}, got {actual_hash}")
    if mismatches:
        raise SystemExit(
            "CONFIG HASH CHECK FAILED — this machine's frozen config files do "
            "not match the committed versions in configs/CONFIG_HASHES.lock. "
            "Nothing scientific may differ between machines; refusing to "
            "proceed.\n" + "\n".join(mismatches))


def _check_fill(node: Any, trail: str = "") -> None:
    """'FILL' qalıbsa erkən və aydın xəta ver."""
    if isinstance(node, dict):
        for k, v in node.items():
            _check_fill(v, f"{trail}.{k}" if trail else k)
    elif node == "FILL":
        raise ValueError(
            f"configs/experiment.yaml sahəsi doldurulmayıb: '{trail}'. "
            "README-nin ADDIM 0 bölməsinə baxın."
        )


def base_argparser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", required=True, help="configs/experiment.yaml")
    p.add_argument("--log-level", default="INFO")
    return p


# ---------------------------------------------------------------- seed
def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """Python / NumPy / PyTorch / CUDA seed-ləri."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:  # köhnə torch
                pass
    except ImportError:
        pass


# ---------------------------------------------------------------- IO
def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sanitise_nonfinite(obj: Any) -> Any:
    """Replace non-finite floats with null so the output is valid JSON.

    Added 2026-09-08. `json.dump` defaults to `allow_nan=True`, which emits the
    bare tokens `Infinity` / `-Infinity` / `NaN`. Those are NOT valid JSON
    (RFC 8259 has no non-finite numbers): Python re-reads them, but strict
    parsers — `jq`, most JS/Go/Rust readers, many notebook loaders — reject the
    whole file. The previous grid wrote 60 `Infinity` gradient-norm records
    across 48 run files, so a third party auditing those results could not
    parse them with standard tooling. Non-finite gradient norms are real and
    informative (AMP overflow), so they are preserved as an explicit COUNT and
    as `null` in place, never silently dropped.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, dict):
        return {k: _sanitise_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise_nonfinite(v) for v in obj]
    return obj


def count_nonfinite(obj: Any) -> int:
    """How many non-finite floats `_sanitise_nonfinite` would replace."""
    if isinstance(obj, (float, np.floating)):
        return 0 if math.isfinite(float(obj)) else 1
    if isinstance(obj, dict):
        return sum(count_nonfinite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(count_nonfinite(v) for v in obj)
    return 0


def write_json(obj: Any, path: str | Path) -> None:
    """Atomically publish VALID JSON so a killed process cannot leave a partial
    result and a strict parser can always read the finished one."""
    path = Path(path)
    ensure_dir(path.parent)
    n_nonfinite = count_nonfinite(obj)
    payload = _sanitise_nonfinite(obj) if n_nonfinite else obj
    if n_nonfinite and isinstance(payload, dict):
        payload = {**payload, "n_nonfinite_values_nulled": n_nonfinite}
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(temp_path, "w", encoding="utf-8") as f:
        # allow_nan=False: if anything non-finite survived the sanitiser we
        # want a loud TypeError here, not an unparseable file on disk.
        json.dump(payload, f, ensure_ascii=False, indent=2,
                  default=_default, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, path)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"JSON-a çevrilə bilmir: {type(o)}")


@dataclass(frozen=True)
class BaseModel:
    """Bir baza modelin tərifi (konfiqdəki models.primary / models.contrast)."""

    role: str          # "primary" | "contrast"
    short: str         # fayl adlarında işlənən qısa ad
    hf_id: str
    az_in_pretraining: bool
    note: str = ""


def resolve_bases(cfg, roles: list[str] | None = None) -> list[BaseModel]:
    """
    Konfiqdən baza modelləri oxuyur.

    İki baza modeli doza-cavab dizaynının əsasıdır:
      primary  — AZ pretraining-də yox  → tokenizasiya defisiti real
      contrast — AZ pretraining-də var  → sıfır-effekt nəzarəti
    """
    if roles is None:
        roles = list(cfg.get("run", {}).get("bases", ["primary", "contrast"]))
    out: list[BaseModel] = []
    for role in roles:
        node = cfg["models"].get(role)
        if node is None:
            raise ValueError(
                f"configs-da models.{role} yoxdur. Mövcud rollar: "
                f"{[k for k in cfg['models'] if isinstance(cfg['models'][k], dict)]}"
            )
        out.append(BaseModel(
            role=role,
            short=node.get("short", role),
            hf_id=node["id"],
            az_in_pretraining=bool(node.get("az_in_pretraining", False)),
            note=node.get("note", ""),
        ))
    return out


def transplant_tag(base_short: str, tag: str) -> str:
    """Transplant artefaktının tam adı — baza modelin adı ilə prefikslənir."""
    return f"{base_short}__{tag}"


def transplant_dir(cfg, base_short: str, tag: str) -> Path:
    """artifacts/transplanted__<base>__<tag>"""
    return (Path(cfg.experiment.artifacts_dir) /
            f"transplanted__{transplant_tag(base_short, tag)}")


def default_tag(cfg) -> str:
    """Konfiqdən default transplant tag-ı: '<method>_k<k>'."""
    return f"{cfg.transplant.method}_k{cfg.transplant.k}"


def canonical_transplant_tag(cfg) -> str:
    """
    HANGI transplant artefaktı FAKTİKİ İSTİFADƏ olunmalıdır (fine-tuning və s.
    üçün) — `default_tag()`-dan fərqli olaraq, `transplant.rescale_reconstructed`
    pre-registered qərarını NƏZƏRƏ ALIR. Bax HANDOFF.md §3.16: rescale-to-
    base-anchor-median variantı hər iki bazada həm BPC-ni, həm norm nisbətini
    yaxşılaşdırdığı üçün seçildi — kanonik artefakt indi
    `transplanted__<base>__<method>_k<k>_rescaled`-dir, no-rescale build
    YALNIZ ablasiya baseline kimi saxlanılır. `default_tag()`-ı özü DƏYİŞMİRİK
    (build_rescale_variant.py onu "mənbə" tag-ı kimi istifadə edir).
    """
    tag = default_tag(cfg)
    if cfg.transplant.get("rescale_reconstructed", False):
        tag += "_rescaled"
    return tag


def sizes_for_base(cfg, base: "BaseModel") -> list[int]:
    """
    Hansı data həcmləri bu baza model üçün işlədilir.

    primary  → bütün train_sizes (tam doza-cavab əyrisi)
    contrast → yalnız run.contrast_sizes (compute qənaəti; sıfır-effekt nəzarəti
               üçün bir referans həcm kifayətdir)
    """
    all_sizes = list(cfg.data.train_sizes)
    if base.role == "primary":
        return all_sizes
    limited = list(cfg.get("run", {}).get("contrast_sizes", all_sizes))
    return [n for n in all_sizes if n in limited] or limited


def seeds_for_size(cfg, train_size: int) -> list[int]:
    """Frozen low-resource replication: five seeds through n=2000, else three."""
    if int(train_size) <= 2000:
        return [int(s) for s in cfg.experiment.seeds]
    return [int(s) for s in cfg.experiment.seeds_high_resource]


def condition_applies(condition: dict, train_size: int) -> bool:
    """Honor an optional condition-specific size scope (``all`` or a list)."""
    sizes = condition.get("sizes", "all")
    return sizes == "all" or int(train_size) in [int(n) for n in sizes]


@dataclass(frozen=True)
class RunKey:
    """Bir run-ı birmənalı təyin edən açar → fayl adı."""

    base: str          # baza modelin qısa adı (xlm15 / xlmr)
    condition: str
    train_size: int
    seed: int

    @property
    def filename(self) -> str:
        return (f"base={self.base}__cond={self.condition}"
                f"__n={self.train_size}__seed={self.seed}.json")
