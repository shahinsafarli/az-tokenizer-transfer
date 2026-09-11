"""
T6 · ƏL İLƏ ETİKET AUDİTİ  —  export / verify / score

Niyə brifin orijinal planından FƏRQLİDİR (qəsdən): brif tək CSV-də `label`
sütunu + boş `verdict` sütunu nəzərdə tuturdu. Amma auditora gold etiketi
GÖSTƏRMƏK onun qərarını həmin etiketi TƏSDİQLƏMƏYƏ yönəldir — ölçülən
razılaşma nisbəti süni şəkildə şişər və MÜSTƏQİL auditin bütün mənası itər.
Ona görə:

  * İKİ ayrı, TAM KOR vərəq (`idx, text, your_label`) — gold HEÇ YERDƏ yoxdur.
  * Gold ayrıca `label_audit_key.json`-da saxlanılır (auditora verilmir).
  * `verify` vərəqləri STRUKTURA görə yoxlayır: qadağan olunmuş sütun varmı.
    DİQQƏT — auditorun DÜZ tapması sızma DEYİL; köhnə (səhv) dizayn
    `your_label == gold` sətirlərini sızma sayırdı və hər düzgün cavabı
    yalançı-müsbətə çevirirdi.

KODLAŞDIRMA (real vərəqlərdə baş verən hadisə): BOM-suz UTF-8 CSV-ni Excel
sistem lokalı (cp1254) kimi açıb saxlayır və `ə` kimi hərflər İTİR
("bilmirəm" → "bilmir?m"). İki nəticə: (1) `_write_csv` `utf-8-sig` yazır —
BOM Excel-ə faylın UTF-8 olduğunu bildirir; (2) `_read_csv_robust` bir neçə
kodlaşdırmanı sıra ilə sınayır və `_is_unclear` ön-şəkilçi ilə uyğunlaşdırır
ki, korlanmış "bilmir?m" də abstinensiya kimi sayılsın — əks halda o sətir
"cavab" sayılıb SƏHV kimi qiymətləndirilərdi.

QƏRAR QAYDASI (HD HH-ə NİSBƏTƏN, mütləq həddə görə YOX):
    HH < 60%                → STOP_TASK_TOO_SUBJECTIVE  (HD-dən ASILI OLMAYARAQ)
    HD − HH ≥ −5pp          → PROCEED
    −12pp ≤ HD − HH < −5pp  → PROCEED_WITH_CAVEAT
    HD − HH < −12pp         → STOP_LABELS_UNRELIABLE
Yekun verdikt İKİ auditorun tier-lərindən PİS olanıdır ("zəif həlqə bağlayıcıdır").

İşlətmə:
    python -m src.data.audit export --config configs/experiment.yaml --n 100
    python -m src.data.audit verify --config configs/experiment.yaml
    python -m src.data.audit score  --config configs/experiment.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.utils import (ensure_dir, get_logger, load_config, setup_logging,
                       write_json)

log = get_logger(__name__)

RESULTS_ANN1 = "label_audit_annotator1.csv"
RESULTS_ANN2 = "label_audit_annotator2.csv"
RESULTS_KEY = "label_audit_key.json"
RESULTS_SCORE = "label_audit.json"

BLIND_COLUMNS = ["idx", "text", "your_label"]

# Vərəqdə görünsə SIZMA sayılan sütun adları (gold daşıya bilənlər).
FORBIDDEN_COLUMNS = {"label", "labels", "gold", "gold_label", "gold_label_name",
                     "label_name", "verdict", "answer", "correct"}

# Real vərəqlər ad yox, RƏQƏM işlədib. Xəritə `build_label_mapping()` ilə
# real probe sətirlərinə qarşı YOXLANILIR — güman edilmir.
DIGIT_LABEL_MAP = {"1": "positive", "2": "neutral", "3": "negative"}

# Rəqəm↔ad xəritəsini təsdiqləmək üçün İSTİFADƏ OLUNAN sətirlər: mənası
# mübahisəsiz olan üç element, hər sinifdən biri.
LABEL_MAPPING_PROBES = [
    {"idx": 38, "expected_gold": "positive", "expected_digit": "1"},
    {"idx": 48, "expected_gold": "negative", "expected_digit": "3"},
    {"idx": 57, "expected_gold": "neutral", "expected_digit": "2"},
]

# "Bilmirəm" / abstinensiya dəyərləri. Ön-şəkilçi uyğunlaşması Excel-in
# korladığı variantları da tutur ("bilmir?m", "bilmirem", ...).
UNCLEAR_PREFIXES = ("bilmir", "unclear", "n/a", "na", "?", "-")

# Bu nisbətdən çox "unclear" → mətn keyfiyyəti şübhəlidir (auditorun
# bacarıqsızlığı deyil, korpusun oxunaqlılığı problemi).
UNCLEAR_CONCERN_PCT = 15.0

HH_TOO_SUBJECTIVE_CEILING = 0.60
HD_PROCEED_MARGIN_PP = 5.0
HD_CAVEAT_MARGIN_PP = 12.0

BOOTSTRAP_ITERS = 10000


# ================================================================ IO
def _read_csv_robust(path: str | Path) -> tuple[list[dict], str]:
    """
    CSV-ni bir neçə kodlaşdırma ilə sıra ilə sınayır; İLK uğurlu oxunuşu və
    HANSI kodlaşdırmanın işlədiyini qaytarır.

    `utf-8-sig` BİRİNCİDİR — biz özümüz BOM ilə yazırıq, və `utf-8-sig`
    BOM-suz UTF-8-i də düzgün oxuyur. `cp1254` (türk) real vərəqlərdə
    Excel-in saxladığı kodlaşdırmadır.
    """
    encodings = ("utf-8-sig", "utf-8", "cp1254", "cp1252", "latin-1")
    last_error: Exception | None = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                rows = list(csv.DictReader(f))
        except (UnicodeDecodeError, LookupError) as e:
            last_error = e
            continue
        return rows, enc
    raise SystemExit(f"{path} heç bir sınanan kodlaşdırma ilə oxunmadı: {last_error}")


def _write_csv(path: str | Path, rows: list[dict], columns: list[str]) -> None:
    """`utf-8-sig` (BOM ilə) yazır — Excel-in lossy yenidən-kodlaşdırmasına qarşı."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


# ================================================================ etiketlər
def _is_unclear(value: str | None) -> bool:
    """
    Abstinensiya yoxlaması — DƏQİQ uyğunluq DEYİL, ön-şəkilçi.

    Səbəb (real bug): Excel "bilmirəm"-i "bilmir?m"-ə çevirir ('ə'
    xəritələnmir). Dəqiq uyğunluq bunu buraxardı və sətir "cavab" kimi
    sayılıb gold ilə müqayisə edilərdi — yəni abstinensiya SƏHV kimi.
    """
    if value is None:
        return False
    v = str(value).strip().casefold()
    if not v:
        return False
    return any(v.startswith(p) for p in UNCLEAR_PREFIXES)


def _canonical_label(value: str | None) -> str | None:
    """
    Xam vərəq dəyərini kanonik etiket adına çevirir.

    None      → sətir boşdur (cavab verilməyib)
    'unclear' → auditor bilərəkdən abstinensiya edib
    əks halda → sinif adı (rəqəm işlədilibsə DIGIT_LABEL_MAP ilə)
    """
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None
    if _is_unclear(v):
        return "unclear"
    if v in DIGIT_LABEL_MAP:
        return DIGIT_LABEL_MAP[v]
    return v.casefold()


def _normalize_text(text: str) -> str:
    return " ".join(str(text).split()).casefold()


# ================================================================ export
def _assert_blind_rows_unfilled(rows: list[dict]) -> None:
    """
    İXRAC BUGUNU TUTUR: kor vərəqin `your_label` sütunu HAMISI boş olmalıdır.
    Bir dənə də əvvəlcədən doldurulmuş sətir auditoru "düzəliş" rejiminə
    salar və audit müstəqil olmaqdan çıxar.
    """
    filled = [r for r in rows if str(r.get("your_label", "")).strip()]
    if filled:
        raise AssertionError(
            f"Kor vərəqdə {len(filled)} sətir ƏVVƏLCƏDƏN doldurulub "
            f"(ilk idx={filled[0].get('idx')}). İxrac buqu — auditora "
            "verməyin.")


def assert_sample_disjoint_from(sample_texts, other_texts, name: str) -> None:
    """
    Audit nümunəsi val/test dəstləri ilə ÜST-ÜSTƏ DÜŞMƏMƏLİDİR.

    Auditə çıxarılan mətni oxuyub etiketini müzakirə etmək — həmin mətn test
    dəstindədirsə — test dəstinə baxmaqdır. Normallaşdırılmış (boşluq/reqistr)
    müqayisə işlədilir, çünki eyni mətnin fərqli yazılışı da eyni mətndir.
    """
    a = {_normalize_text(t) for t in sample_texts}
    b = {_normalize_text(t) for t in other_texts}
    overlap = a & b
    if overlap:
        raise AssertionError(
            f"Audit nümunəsi '{name}' dəsti ilə {len(overlap)} mətndə üst-üstə "
            f"düşür. Nümunə YALNIZ train dəstindən götürülməlidir.")


def export_audit(cfg, n: int = 100) -> dict:
    """Stratifikasiya olunmuş `n` elementlik nümunə → iki KOR vərəq + gold açarı."""
    from src.data.splits import read_jsonl

    results_dir = ensure_dir(cfg.experiment.results_dir)
    datadir = Path(cfg.experiment.artifacts_dir) / "data"
    train = read_jsonl(datadir / "az_train.jsonl")
    val = read_jsonl(datadir / "az_val.jsonl")
    test = read_jsonl(datadir / "az_test.jsonl")

    splits = json.loads((results_dir / "splits.json").read_text(encoding="utf-8"))
    id_to_name = {v: k for k, v in splits["az"]["label_map"].items()}

    rng = np.random.default_rng(int(cfg.data.split.seed))
    by_label: dict[int, list[dict]] = {}
    for r in train:
        by_label.setdefault(int(r["label"]), []).append(r)

    per = max(1, n // max(len(by_label), 1))
    sample: list[dict] = []
    for label in sorted(by_label):
        items = by_label[label]
        idx = rng.permutation(len(items))[:min(per, len(items))]
        sample.extend(items[i] for i in idx)
    rng.shuffle(sample)
    sample = sample[:n]

    # Test dəstinə toxunmamaq — nümunə YALNIZ train-dəndir, amma bunu
    # güman etmirik, YOXLAYIRIQ.
    texts = [r["text"] for r in sample]
    assert_sample_disjoint_from(texts, [r["text"] for r in test], "test")
    assert_sample_disjoint_from(texts, [r["text"] for r in val], "val")

    blind_rows = [{"idx": i, "text": r["text"], "your_label": ""}
                  for i, r in enumerate(sample)]
    _assert_blind_rows_unfilled(blind_rows)
    for name in (RESULTS_ANN1, RESULTS_ANN2):
        _write_csv(results_dir / name, blind_rows, BLIND_COLUMNS)

    key = {str(i): {"text": r["text"],
                    "gold_label_name": id_to_name[int(r["label"])]}
           for i, r in enumerate(sample)}
    write_json(key, results_dir / RESULTS_KEY)

    log.info("İxrac edildi: %d element → %s, %s (KOR) + %s (gold, auditora VERİLMİR)",
             len(sample), RESULTS_ANN1, RESULTS_ANN2, RESULTS_KEY)
    return {"n": len(sample), "files": [RESULTS_ANN1, RESULTS_ANN2, RESULTS_KEY]}


# ================================================================ verify
def verify_no_leak(cfg) -> dict:
    """
    STRUKTUR yoxlaması: vərəqdə gold daşıya bilən sütun varmı.

    Məzmuna (auditorun cavabının gold ilə üst-üstə düşməsinə) BAXMIR — bu,
    hər DÜZGÜN cavabı sızma kimi bildirən köhnə səhv dizayn idi.
    """
    results_dir = Path(cfg.experiment.results_dir)
    files: dict[str, dict] = {}
    all_clean = True
    for name in (RESULTS_ANN1, RESULTS_ANN2):
        path = results_dir / name
        if not path.exists():
            files[name] = {"present": False, "forbidden_columns_found": [],
                           "n_rows": 0, "encoding": None}
            continue
        rows, enc = _read_csv_robust(path)
        columns = list(rows[0].keys()) if rows else []
        forbidden = sorted({c for c in columns
                            if c is not None and c.strip().casefold() in FORBIDDEN_COLUMNS})
        if forbidden:
            all_clean = False
        files[name] = {
            "present": True,
            "encoding": enc,
            "n_rows": len(rows),
            "columns": columns,
            "forbidden_columns_found": forbidden,
        }
    return {"all_clean": all_clean, "files": files,
            "expected_columns": list(BLIND_COLUMNS),
            "note": ("Struktur-yalnız yoxlama: auditorun gold ilə üst-üstə "
                     "düşən cavabı sızma DEYİL.")}


# ================================================================ score
def _cohens_kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    if len(labels) < 2 or not a:
        return float("nan")
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if abs(1.0 - expected) < 1e-12:
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def _bootstrap_ci(values: list[float], stat, iters: int = BOOTSTRAP_ITERS,
                  seed: int = 0) -> list[float] | None:
    if not values:
        return None
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        draws.append(stat(idx))
    return [round(float(np.percentile(draws, 2.5)), 4),
            round(float(np.percentile(draws, 97.5)), 4)]


def _pairwise(reference: list[str], other: list[str], ref_name: str) -> dict:
    """`other`-in `reference`-ə uyğunluğu — 'unclear' MƏXRƏCDƏN atılır."""
    pairs = [(r, o) for r, o in zip(reference, other) if o != "unclear"]
    n_unclear = len(other) - len(pairs)
    n = len(pairs)
    n_correct = sum(1 for r, o in pairs if r == o)

    per_class: dict[str, dict] = {}
    for label in dict.fromkeys(r for r, _ in pairs):
        subset = [(r, o) for r, o in pairs if r == label]
        correct = sum(1 for r, o in subset if r == o)
        per_class[label] = {
            "n": len(subset), "correct": correct,
            "accuracy": round(correct / max(len(subset), 1), 4),
        }

    ref = [r for r, _ in pairs]
    oth = [o for _, o in pairs]
    agree_ci = _bootstrap_ci(
        list(range(n)),
        lambda idx: 100.0 * np.mean([ref[i] == oth[i] for i in idx])) if n else None
    kappa_ci = _bootstrap_ci(
        list(range(n)),
        lambda idx: _cohens_kappa([ref[i] for i in idx],
                                  [oth[i] for i in idx])) if n else None

    total = len(other)
    unclear_pct = 100.0 * n_unclear / max(total, 1)
    return {
        "n": n,
        "n_correct": n_correct,
        "accuracy": round(n_correct / max(n, 1), 4),
        "per_class": per_class,
        "kappa": round(_cohens_kappa(ref, oth), 4) if n else float("nan"),
        "bootstrap_ci_95_agreement_pct": agree_ci,
        "bootstrap_ci_95_kappa": kappa_ci,
        "n_unclear_excluded": n_unclear,
        "unclear_rate_pct": round(unclear_pct, 2),
        "text_quality_concern": bool(unclear_pct > UNCLEAR_CONCERN_PCT),
        "reference": ref_name,
    }


def _decompose_disagreements(a: list[str], b: list[str]) -> dict:
    """
    Uyğunsuzluqları İKİ növə ayırır — bu, bookkeeping deyil, ölçüdür:

      neutral_boundary — biri "neutral" deyib, digəri qütb (harada həddin
                         çəkilməsi məsələsi; qütbü ayırd etmək bacarığına aid DEYİL)
      polarity_flip    — biri positive, digəri negative (ƏSL ziddiyyət)

    Binary tapşırığa keçid qərarı MƏHZ buna əsaslanır: uyğunsuzluqların
    böyük hissəsi neutral-sərhəddirsə, qütb alt-tapşırığı sağlamdır.
    """
    n_agree = n_disagree = neutral_boundary = polarity_flip = 0
    for x, y in zip(a, b):
        if x == y:
            n_agree += 1
            continue
        n_disagree += 1
        if x == "neutral" or y == "neutral":
            neutral_boundary += 1
        else:
            polarity_flip += 1
    return {
        "n_agreements": n_agree,
        "n_disagreements": n_disagree,
        "neutral_boundary": neutral_boundary,
        "neutral_boundary_pct_of_disagreements": round(
            100.0 * neutral_boundary / max(n_disagree, 1), 2),
        "polarity_flip": polarity_flip,
        "polarity_flip_pct_of_disagreements": round(
            100.0 * polarity_flip / max(n_disagree, 1), 2),
    }


def build_label_mapping(cfg) -> dict:
    """
    Rəqəm↔ad xəritəsini REAL probe sətirlərinə qarşı yoxlayır.

    Xəritəni güman etmək təhlükəlidir: 1/2/3 sırası tərs olsaydı, hər skor
    səssizcə yanlış çıxardı. Ona görə hər probe üçün (a) gold-un gözlənilən
    adla, (b) auditorun rəqəminin gözlənilən rəqəmlə uyğunluğu yazılır.
    """
    results_dir = Path(cfg.experiment.results_dir)
    key_path = results_dir / RESULTS_KEY
    key = json.loads(key_path.read_text(encoding="utf-8")) if key_path.exists() else {}

    sheets: dict[str, dict] = {}
    encodings: dict[str, str] = {}
    missing: list[str] = []
    for name in (RESULTS_ANN1, RESULTS_ANN2):
        path = results_dir / name
        if not path.exists():
            missing.append(name)
            sheets[name] = {}
            continue
        rows, enc = _read_csv_robust(path)
        encodings[name] = enc
        sheets[name] = {str(r.get("idx")): r.get("your_label", "") for r in rows}

    probes = []
    fully_verified = True
    for probe in LABEL_MAPPING_PROBES:
        idx = str(probe["idx"])
        entry = key.get(idx, {})
        gold_name = entry.get("gold_label_name")
        record = {
            "idx": probe["idx"],
            "text": entry.get("text"),
            "expected_gold": probe["expected_gold"],
            "expected_digit": probe["expected_digit"],
            "gold_label_name": gold_name,
            "gold_matches_expected": gold_name == probe["expected_gold"],
            "annotators": {},
        }
        if not record["gold_matches_expected"]:
            fully_verified = False
        for name in (RESULTS_ANN1, RESULTS_ANN2):
            raw = sheets.get(name, {}).get(idx, "")
            available = bool(str(raw).strip())
            mapped = _canonical_label(raw)
            record["annotators"][name] = {
                "available": available,
                "raw_value": raw if available else None,
                "mapped_label": mapped,
                "matches_expected_digit": available and str(raw).strip() == probe["expected_digit"],
                "matches_expected_gold": mapped == probe["expected_gold"],
            }
            if not available or mapped != probe["expected_gold"]:
                fully_verified = False
        probes.append(record)

    return {
        "digit_to_label": dict(DIGIT_LABEL_MAP),
        "probes": probes,
        "sheet_encodings_detected": encodings,
        "missing_sheets": missing,
        "fully_verified_both_annotators": fully_verified,
        "note": ("Xəritə HƏR ÜÇ probe üçün gold etiketlə VƏ mövcud "
                 "auditor(lar)ın rəqəmi ilə təsdiqlənib."
                 if fully_verified else
                 "Xəritə QİSMƏN təsdiqlənib — aşağıdakı probe-lara baxın."),
    }


def _decision_gate(hh_pct: float, hd1_pct: float, hd2_pct: float) -> dict:
    """HD HH-ə NİSBƏTƏN qiymətləndirilir, mütləq həddə görə yox."""
    order = ["PROCEED", "PROCEED_WITH_CAVEAT", "STOP_LABELS_UNRELIABLE",
             "STOP_TASK_TOO_SUBJECTIVE"]

    def tier(hd: float) -> str:
        diff = hd - hh_pct
        if diff >= -HD_PROCEED_MARGIN_PP:
            return "PROCEED"
        if diff >= -HD_CAVEAT_MARGIN_PP:
            return "PROCEED_WITH_CAVEAT"
        return "STOP_LABELS_UNRELIABLE"

    t1, t2 = tier(hd1_pct), tier(hd2_pct)
    if hh_pct < HH_TOO_SUBJECTIVE_CEILING * 100:
        verdict = "STOP_TASK_TOO_SUBJECTIVE"
        note = (f"HH={hh_pct:.1f}% < {HH_TOO_SUBJECTIVE_CEILING*100:.0f}% — auditorlar "
                "bir-biri ilə DƏ razılaşmır; tapşırıq bu formada çox subyektivdir. "
                "HD-dən ASILI OLMAYARAQ bağlayıcıdır.")
    else:
        verdict = max((t1, t2), key=order.index)
        note = (f"HH={hh_pct:.2f}% (tapşırıq tavanı). Annotator1 HD={hd1_pct:.1f}% "
                f"({hd1_pct - hh_pct:+.1f}pp → {t1}); Annotator2 HD={hd2_pct:.1f}% "
                f"({hd2_pct - hh_pct:+.1f}pp → {t2}). Yekun (bağlayıcı, "
                f"ikisindən PİS olan): {verdict}.")
    return {
        "verdict": verdict,
        "note": note,
        "rule": ("HD judged RELATIVE TO HH (task ceiling), not against an "
                 "absolute floor"),
        "tier_annotator1": t1,
        "tier_annotator2": t2,
        "hh_pct": round(hh_pct, 2),
        "thresholds": {
            "hh_too_subjective_ceiling": HH_TOO_SUBJECTIVE_CEILING,
            "hd_proceed_margin_pp": HD_PROCEED_MARGIN_PP,
            "hd_caveat_margin_pp": HD_CAVEAT_MARGIN_PP,
        },
    }


def score_audit(cfg) -> dict:
    """Doldurulmuş vərəqləri oxuyur, HD1/HD2/HH hesablayır və qərar qapısını tətbiq edir."""
    results_dir = ensure_dir(cfg.experiment.results_dir)
    key = json.loads((results_dir / RESULTS_KEY).read_text(encoding="utf-8"))

    sheets = {}
    for name in (RESULTS_ANN1, RESULTS_ANN2):
        rows, _enc = _read_csv_robust(results_dir / name)
        sheets[name] = {str(r.get("idx")): r.get("your_label", "") for r in rows}

    indices = sorted(key, key=lambda k: int(k))
    gold = [key[i]["gold_label_name"] for i in indices]
    ann1 = [_canonical_label(sheets[RESULTS_ANN1].get(i)) or "unclear" for i in indices]
    ann2 = [_canonical_label(sheets[RESULTS_ANN2].get(i)) or "unclear" for i in indices]

    hd1 = _pairwise(gold, ann1, "dataset_gold")
    hd2 = _pairwise(gold, ann2, "dataset_gold")

    # Auditor↔auditor: HƏR İKİSİNİN cavab verdiyi sətirlər.
    both = [(a, b) for a, b in zip(ann1, ann2)
            if a != "unclear" and b != "unclear"]
    n_excluded_either = len(ann1) - len(both)
    hh = _pairwise([a for a, _ in both], [b for _, b in both], "annotator1")
    hh["n_excluded_either_unclear"] = n_excluded_either
    hh.pop("n_unclear_excluded", None)
    hh.pop("unclear_rate_pct", None)
    hh.pop("text_quality_concern", None)

    hh_pct = 100.0 * hh["accuracy"]
    hd1_pct = 100.0 * hd1["accuracy"]
    hd2_pct = 100.0 * hd2["accuracy"]

    # Qütb alt-çoxluğu: hər üç mənbənin qütb (neutral OLMAYAN) dediyi sətirlər.
    polarity = [(g, a, b) for g, a, b in zip(gold, ann1, ann2)
                if "neutral" not in (g, a, b) and "unclear" not in (a, b)]
    polarity_hh = (100.0 * sum(1 for _g, a, b in polarity if a == b) / len(polarity)
                   if polarity else None)

    result = {
        "label_mapping": build_label_mapping(cfg),
        "n_sampled": len(indices),
        "annotator1_vs_gold_pct": round(hd1_pct, 2),
        "annotator2_vs_gold_pct": round(hd2_pct, 2),
        "hd_spread_pct": round(abs(hd1_pct - hd2_pct), 2),
        "hd_spread_note": ("HD1 və HD2 AYRI bildirilir, ORTALANMIR — böyük fərq "
                           "auditorların EYNİ həddi tətbiq etmədiyini göstərir."),
        "annotator1_vs_gold": hd1,
        "annotator2_vs_gold": hd2,
        "annotator1_vs_annotator2": hh,
        "human_human_agreement_pct": round(hh_pct, 2),
        "abstention_counts": {
            "annotator1": sum(1 for a in ann1 if a == "unclear"),
            "annotator2": sum(1 for b in ann2 if b == "unclear"),
        },
        "disagreement_decomposition": {
            "annotator1_vs_annotator2": _decompose_disagreements(
                [a for a, _ in both], [b for _, b in both]),
            "dataset_vs_annotator1": _decompose_disagreements(
                [g for g, a in zip(gold, ann1) if a != "unclear"],
                [a for a in ann1 if a != "unclear"]),
            "dataset_vs_annotator2": _decompose_disagreements(
                [g for g, b in zip(gold, ann2) if b != "unclear"],
                [b for b in ann2 if b != "unclear"]),
        },
        "polarity_only_subset": {
            "n": len(polarity),
            "hh_agreement_pct": None if polarity_hh is None else round(polarity_hh, 2),
            "note": ("Şərtlənib (hər üçü qütb deyib) — OPTİMİST nöqtə qiyməti, "
                     "tapşırıq tavanı kimi çılpaq sitat gətirilməməlidir."),
        },
        "decision_gate": _decision_gate(hh_pct, hd1_pct, hd2_pct),
    }
    write_json(result, results_dir / RESULTS_SCORE)
    return result


# ================================================================ CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["export", "verify", "score"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--n", type=int, default=100, help="export: nümunə ölçüsü")
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    if args.command == "export":
        export_audit(cfg, args.n)
    elif args.command == "verify":
        check = verify_no_leak(cfg)
        for name, info in check["files"].items():
            log.info("%s: mövcud=%s sətir=%s kodlaşdırma=%s qadağan_sütun=%s",
                     name, info.get("present"), info.get("n_rows"),
                     info.get("encoding"), info.get("forbidden_columns_found"))
        if not check["all_clean"]:
            raise SystemExit("SIZMA: vərəqdə gold daşıya bilən sütun var — "
                             "auditora VERMƏYİN, yenidən ixrac edin.")
        log.info("Vərəqlər TƏMİZ (struktur yoxlaması).")
    else:
        out = score_audit(cfg)
        gate = out["decision_gate"]
        log.info("HH=%.2f%%  HD1=%.2f%%  HD2=%.2f%%",
                 out["human_human_agreement_pct"],
                 out["annotator1_vs_gold_pct"], out["annotator2_vs_gold_pct"])
        log.info("QƏRAR: %s — %s", gate["verdict"], gate["note"])


if __name__ == "__main__":
    main()
