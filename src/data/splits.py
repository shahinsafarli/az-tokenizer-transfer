"""
SABİT TRAIN / VAL / TEST BÖLGÜSÜ  —  bir dəfə yaradılır, sonra DƏYİŞMİR

Sıra (STEP D, HANDOFF §3.13) — bu ardıcıllıq MƏCBURİDİR:

    _load_records → apply_exclude_labels → _dedup → _encode_labels → split

`apply_exclude_labels` DEDUP-dan ƏVVƏL çağırılır ki (a) etiket indeksləri
ardıcıl qalsın (0/1, "0/_/2" yox) və (b) dedup statistikası YEKUN korpusu
təsvir etsin, ondan əvvəlkini yox.

⚠️ DEDUP BÖLGÜDƏN ƏVVƏL OLMALIDIR. Əks halda eyni mətn həm train, həm test
dəstinə düşür — bu, gizli test-set sızmasıdır və brifdə AVTOMATİK bal
itkisidir. `_dedup()` boşluq/böyük-kiçik hərfi normallaşdırır, təkrarları
atır və etiket-ziddiyyətli mətnləri TAMAMİLƏ atır.

Bölgü seed-i (`data.split.seed`) MODEL seed-lərindən (`experiment.seeds`)
AYRIDIR — fərqli model seed-ləri EYNİ bölgü üzərində işləsin deyə.

İşlətmə:
    python -m src.data.splits --config configs/experiment.yaml

Çıxış:
    artifacts/data/az_train.jsonl · az_val.jsonl · az_test.jsonl
    artifacts/data/tr_train.jsonl
    results/splits.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.utils import (base_argparser, ensure_dir, get_logger, load_config,
                       setup_logging, write_json)

log = get_logger(__name__)

_WS = re.compile(r"\s+")

# `_dedup` xəbərdarlıq həddləri (STEP D) — dataseti rədd etmir, yalnız bildirir.
DUP_WARN_PCT = 25.0
DUP_RECONSIDER_PCT = 40.0


# ---------------------------------------------------------------- IO
def read_jsonl(path: str | Path) -> list[dict]:
    """JSONL faylını oxuyur. `finetune.py` və analiz modulları bunu işlədir."""
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    """Dedup/sızma yoxlaması üçün kanonik forma: boşluq + hərf reqistri."""
    return _WS.sub(" ", str(text).strip()).casefold()


# ---------------------------------------------------------------- yükləmə
def _load_records(node, *, text_column: str, label_column: str) -> list[dict]:
    """`source: hf` və ya `source: csv` konfiqindən xam qeydlər."""
    source = node.get("source", "hf")
    if source == "csv":
        import csv

        path = node.get("csv_path")
        if not path:
            raise SystemExit("data.*.source=csv üçün csv_path lazımdır")
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    elif source == "hf":
        from datasets import load_dataset

        ds = load_dataset(node["hf_name"], node.get("hf_config"),
                          split=node.get("hf_split", "train"))
        missing = [c for c in (text_column, label_column) if c not in ds.column_names]
        if missing:
            raise SystemExit(
                f"{node['hf_name']}: sütun tapılmadı {missing}. "
                f"Mövcud sütunlar: {ds.column_names}")
        # `int2str` yalnız ClassLabel sütunlarında var; olmayanda xam dəyər qalır.
        feature = ds.features[label_column]
        to_name = getattr(feature, "int2str", None)
        rows = [{text_column: t, label_column: (to_name(l) if to_name else l)}
                for t, l in zip(ds[text_column], ds[label_column])]
    else:
        raise SystemExit(f"naməlum data source: {source!r} (hf | csv)")

    records = []
    for r in rows:
        text = r.get(text_column)
        label = r.get(label_column)
        if text is None or label is None:
            continue
        text = str(text).strip()
        if text:
            records.append({"text": text, "label": label})
    return records


# ---------------------------------------------------------------- exclude
def apply_exclude_labels(records: list[dict], node) -> tuple[list[dict], dict]:
    """
    `data.az.exclude_labels`-dakı sinifləri atır (T6 auditinin qərarı:
    3-sinifli tapşırıqda razılaşma 66.7% idi, neutral-sərhəd anlaşılmazlığı
    səbəbindən; qütb alt-çoxluğunda 88.1%). Bax HANDOFF §3.12–3.13.

    `_dedup`/`_encode_labels`-dan ƏVVƏL çağırılır — modul dosstring-inə bax.
    """
    excluded = [str(x) for x in (node.get("exclude_labels") or [])]
    n_before = len(records)
    if not excluded:
        return records, {"excluded_labels": [], "n_before": n_before,
                         "n_after": n_before, "n_excluded": 0}
    drop = set(excluded)
    out = [r for r in records if str(r["label"]) not in drop]
    return out, {"excluded_labels": excluded, "n_before": n_before,
                 "n_after": len(out), "n_excluded": n_before - len(out)}


# ---------------------------------------------------------------- dedup
def _dedup(records: list[dict]) -> tuple[list[dict], dict]:
    """
    Normallaşdırılmış mətnə görə təkrarları atır VƏ etiket-ziddiyyətli
    mətnləri TAMAMİLƏ atır (hansı etiketin düz olduğunu bilmirik — saxlamaq
    etiket küyü əlavə etmək olardı).

    `duplicates_removed` ATILAN SƏTİRLƏRİN ÜMUMİ sayıdır (təkrarlar +
    ziddiyyətli mətnlərin bütün sətirləri), `conflicting_texts` isə neçə
    FƏRQLİ mətnin ziddiyyətli olduğudur.
    """
    before = len(records)
    first: dict[str, dict] = {}
    labels_seen: dict[str, set] = {}
    for r in records:
        key = normalize_text(r["text"])
        labels_seen.setdefault(key, set()).add(str(r["label"]))
        first.setdefault(key, r)

    conflicting = {k for k, labs in labels_seen.items() if len(labs) > 1}
    out = [r for k, r in first.items() if k not in conflicting]
    after = len(out)
    return out, {
        "before": before,
        "after": after,
        "duplicates_removed": before - after,
        "duplicate_pct": round(100.0 * (before - after) / max(before, 1), 2),
        "conflicting_texts": len(conflicting),
        "label_aware": True,
    }


# ---------------------------------------------------------------- etiketlər
def _encode_labels(records: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Etiket adlarını sabit (əlifba sırası ilə) tam ədədlərə çevirir."""
    names = sorted({str(r["label"]) for r in records})
    mapping = {name: i for i, name in enumerate(names)}
    out = [{"text": r["text"], "label": mapping[str(r["label"])]} for r in records]
    return out, mapping


# ---------------------------------------------------------------- bölgü
def stratified_split(records: list[dict], seed: int, val_ratio: float,
                     test_ratio: float) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Sinif paylanmasını hər üç dəstdə qoruyan bölgü.

    Stratifikasiya seçimdir, təsadüf deyil: `analysis` macro-F1 işlədir, ona
    görə val/test-də sinif nisbətinin sürüşməsi metrikanı birbaşa pozardı.
    """
    rng = np.random.default_rng(seed)
    by_label: dict[Any, list[dict]] = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for label in sorted(by_label):
        items = by_label[label]
        order = rng.permutation(len(items))
        shuffled = [items[i] for i in order]
        n = len(shuffled)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        val.extend(shuffled[:n_val])
        test.extend(shuffled[n_val:n_val + n_test])
        train.extend(shuffled[n_val + n_test:])

    for part in (train, val, test):
        rng.shuffle(part)
    return train, val, test


def stratified_subsample(records: list[dict], n: int, seed: int) -> list[dict]:
    """
    Sinif nisbətini qoruyan `n`-ölçülü alt-çoxluq — TR ara-mərhələsi üçün
    (`data.tr.max_examples`). Bölgü seed-i işlədilir, MODEL seed-i yox:
    türk mərhələsi bütün model seed-lərində EYNİ mətnləri görməlidir, əks
    halda TR checkpoint keşi elmi cəhətdən fərqli şeyləri eyni sayardı.
    """
    if n >= len(records):
        return records
    rng = np.random.default_rng(seed)
    by_label: dict[Any, list[dict]] = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)

    out: list[dict] = []
    for label in sorted(by_label):
        items = by_label[label]
        quota = int(round(n * len(items) / len(records)))
        order = rng.permutation(len(items))[:quota]
        out.extend(items[i] for i in order)
    rng.shuffle(out)
    return out[:n]


def _label_dist(records: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        out[str(r["label"])] = out.get(str(r["label"]), 0) + 1
    return out


def split_content_manifest(train: list[dict], val: list[dict],
                           test: list[dict]) -> dict:
    """Content hash of each split, so dataset drift is DETECTED, not inferred.

    Added 2026-09-08. `data.az.hf_name` is not revision-pinned (unlike the
    donor model, which is), so a fresh `load_dataset` can return different or
    reordered rows. That is not hypothetical: two artifacts of this project
    record 20,936/2,791/4,187 and 20,937/2,791/4,186 for the same nominal
    split, a one-example drift with no code change to explain it. Counts alone
    cannot tell "same holdout" from "different holdout", which is exactly the
    question a held-out claim depends on.

    The hash is over the SORTED normalised texts, so it is invariant to row
    order but sensitive to membership: two runs with the same hash saw the same
    examples. Labels are hashed alongside the text so a relabelling is caught
    too. Cheap enough to compute unconditionally.
    """
    import hashlib

    def digest(rows: list[dict]) -> dict:
        items = sorted(f"{normalize_text(r['text'])}\t{r['label']}" for r in rows)
        h = hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()
        return {"n": len(rows), "sha256": h}

    return {
        "algorithm": "sha256 of sorted(normalize_text(text) + TAB + label)",
        "order_invariant": True,
        "train": digest(train), "val": digest(val), "test": digest(test),
        "note": ("Compare `test.sha256` across runs to prove the holdout is "
                 "the SAME holdout. A changed hash with unchanged code means "
                 "the upstream dataset moved; pin data.az.hf_revision."),
    }


def leakage_check(train: list[dict], val: list[dict], test: list[dict]) -> dict:
    """Normallaşdırılmış mətn üzərində üç cüt kəsişmə — hamısı 0 olmalıdır."""
    t = {normalize_text(r["text"]) for r in train}
    v = {normalize_text(r["text"]) for r in val}
    s = {normalize_text(r["text"]) for r in test}
    return {"train_test": len(t & s), "train_val": len(t & v), "val_test": len(v & s)}


# ---------------------------------------------------------------- main
def main() -> None:
    ap = base_argparser(__doc__)
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    datadir = ensure_dir(Path(cfg.experiment.artifacts_dir) / "data")
    split_cfg = cfg.data.split
    seed = int(split_cfg.seed)
    report: dict = {"split_seed": seed}

    # ======================================================== AZ
    az = cfg.data.az
    log.info("AZ dataseti yüklənir: %s", az.get("hf_name") or az.get("csv_path"))
    az_records = _load_records(az, text_column=az.text_column,
                               label_column=az.label_column)
    log.info("  xam qeyd: %d", len(az_records))

    az_records, exclude_stats = apply_exclude_labels(az_records, az)
    if exclude_stats["n_excluded"]:
        log.info("  exclude_labels=%s → %d qeyd atıldı (%d qaldı)",
                 exclude_stats["excluded_labels"], exclude_stats["n_excluded"],
                 exclude_stats["n_after"])

    az_records, az_dedup = _dedup(az_records)
    log.info("  dedup: %d → %d (%.2f%% təkrar, %d ziddiyyətli mətn)",
             az_dedup["before"], az_dedup["after"], az_dedup["duplicate_pct"],
             az_dedup["conflicting_texts"])
    if az_dedup["duplicate_pct"] >= DUP_RECONSIDER_PCT:
        log.error("Təkrar nisbəti %.1f%% ≥ %.0f%% — bu dataseti YENİDƏN NƏZƏRDƏN KEÇİRİN.",
                  az_dedup["duplicate_pct"], DUP_RECONSIDER_PCT)
    elif az_dedup["duplicate_pct"] >= DUP_WARN_PCT:
        log.warning("Təkrar nisbəti %.1f%% ≥ %.0f%% — məqalədə qeyd edin.",
                    az_dedup["duplicate_pct"], DUP_WARN_PCT)

    az_records, az_label_map = _encode_labels(az_records)
    az_train, az_val, az_test = stratified_split(
        az_records, seed, float(split_cfg.val_ratio), float(split_cfg.test_ratio))

    write_jsonl(az_train, datadir / "az_train.jsonl")
    write_jsonl(az_val, datadir / "az_val.jsonl")
    write_jsonl(az_test, datadir / "az_test.jsonl")

    report["az"] = {
        "source": az.get("hf_name") or az.get("csv_path"),
        "total": len(az_records),
        "n_labels": len(az_label_map),
        "label_map": az_label_map,
        "exclude_labels": exclude_stats,
        "dedup": az_dedup,
        "train": len(az_train),
        "val": len(az_val),
        "test": len(az_test),
        "train_label_dist": _label_dist(az_train),
        "test_label_dist": _label_dist(az_test),
    }
    log.info("AZ bölgüsü: train=%d val=%d test=%d", len(az_train), len(az_val), len(az_test))

    largest = max(int(n) for n in cfg.data.train_sizes)
    if len(az_train) < largest:
        log.error("train=%d < ən böyük train_sizes girişi (%d) — "
                  "`data.train_sizes` siyahısını kiçildin.", len(az_train), largest)

    # ======================================================== TR
    tr = cfg.data.tr
    log.info("TR dataseti yüklənir: %s", tr.get("hf_name") or tr.get("csv_path"))
    tr_records = _load_records(tr, text_column=tr.text_column,
                               label_column=tr.label_column)
    tr_records, tr_dedup = _dedup(tr_records)
    log.info("  dedup: %d → %d (%.2f%%, %d ziddiyyətli)", tr_dedup["before"],
             tr_dedup["after"], tr_dedup["duplicate_pct"], tr_dedup["conflicting_texts"])
    tr_records, tr_label_map = _encode_labels(tr_records)

    max_tr = tr.get("max_examples")
    if max_tr and len(tr_records) > int(max_tr):
        tr_records = stratified_subsample(tr_records, int(max_tr), seed)
    write_jsonl(tr_records, datadir / "tr_train.jsonl")

    report["tr"] = {
        "source": tr.get("hf_name") or tr.get("csv_path"),
        "total": len(tr_records),
        "n_labels": len(tr_label_map),
        "label_map": tr_label_map,
        "dedup": tr_dedup,
    }
    log.info("TR ara-mərhələ: %d cümlə", len(tr_records))

    # ======================================================== sızma
    leak = leakage_check(az_train, az_val, az_test)
    report["leakage_check"] = leak
    report["az_content_manifest"] = split_content_manifest(az_train, az_val, az_test)
    report["az_hf_revision_requested"] = az.get("hf_revision")
    if not az.get("hf_revision"):
        log.warning(
            "data.az.hf_revision is NOT pinned — a future download may return "
            "different rows. The content manifest in results/splits.json is "
            "what makes such drift detectable; record test.sha256 in the paper.")
    log.info("AZ content manifest: test sha256=%s (n=%d)",
             report["az_content_manifest"]["test"]["sha256"][:16],
             report["az_content_manifest"]["test"]["n"])
    write_json(report, Path(cfg.experiment.results_dir) / "splits.json")

    if any(leak.values()):
        raise SystemExit(
            f"SIZMA AŞKARLANDI: {leak}. Bölgü etibarsızdır — brifdə bu, "
            "AVTOMATİK bal itkisidir. Davam ETMƏYİN.")
    log.info("Sızma yoxlaması TƏMİZ: %s", leak)
    log.info("Yazıldı: %s", Path(cfg.experiment.results_dir) / "splits.json")


if __name__ == "__main__":
    main()
