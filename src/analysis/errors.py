"""
SƏHV TAKSONOMİYASI  —  komandanızın dil biliyindən gələn ÜSTÜNLÜK

Test dəstindəki səhvləri üç kateqoriyaya bölür:

  1. yalanci_dost  — cümlədə TR/AZ arasında məna fərqi olan söz var
                     (sabah = TR "səhər" / AZ "gələn gün";  ilan = TR "elan" / AZ "ilan")
  2. texniki_lugat — dil islahatından gələn lüğət fərqi
                     (bilgisayar/kompüter, uçak/təyyarə)
  3. morfoloji     — uzun şəkilçi zənciri (fertility yüksək olan sözlər)
  4. diger         — yuxarıdakılara düşməyən

Bu, avtomatik HEURİSTİKADIR — məqalədə "əl ilə yoxlanılmış alt-nümunə"
ilə birlikdə təqdim edin (skript `--manual-sample` ilə nümunə çıxarır).

İşlətmə:
    python -m src.analysis.errors --config configs/experiment.yaml
    python -m src.analysis.errors --config ... --manual-sample 50
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from src.data.splits import read_jsonl
from src.utils import (base_argparser, ensure_dir, get_logger, load_config, read_json, resolve_bases, setup_logging, write_json)

log = get_logger(__name__)

FALLBACK_FALSE_FRIENDS = [
    ("sabah", "səhər (TR)", "gələn gün (AZ)"),
    ("ilan", "elan (TR)", "ilan/heyvan (AZ)"),
    ("düşmək", "yıxılmaq (TR)", "düşmək/enmək (AZ)"),
    ("subay", "zabit/subay (TR)", "subay (AZ)"),
    ("qonaq", "qonaq (TR)", "qonaq (AZ)"),
]
FALLBACK_TECHNICAL = [
    ("kompüter", "bilgisayar"), ("təyyarə", "uçak"), ("sual", "soru"),
    ("məsələ", "sorun"), ("nəticə", "sonuç"), ("imkan", "olanak"),
    ("hadisə", "olay"), ("cavab", "yanıt"),
]
MORPH_FERTILITY_THRESHOLD = 4   # bir sözə bundan çox token düşürsə "morfoloji"


def _load_false_friends(path: str | None):
    if path and Path(path).exists():
        out = []
        with open(path, encoding="utf-8") as f:
            for row in csv.reader(f):
                if row and not row[0].startswith("#"):
                    out.append(tuple(row))
        return out
    return FALLBACK_FALSE_FRIENDS


def classify(text: str, tokenizer, ff_words: set[str], tech_words: set[str]) -> str:
    """tokenizer None ola bilər — o halda morfoloji kateqoriya yoxlanılmır."""
    low = text.lower()
    words = low.split()
    if any(w.strip(".,!?;:()\"'") in ff_words for w in words):
        return "yalanci_dost"
    if any(w.strip(".,!?;:()\"'") in tech_words for w in words):
        return "texniki_lugat"
    if tokenizer is not None:
        n_tok = len(tokenizer.tokenize(text))
        if n_tok / max(len(words), 1) > MORPH_FERTILITY_THRESHOLD:
            return "morfoloji"
    return "diger"


def main() -> None:
    ap = base_argparser(__doc__)
    ap.add_argument("--manual-sample", type=int, default=30,
                    help="əl ilə yoxlama üçün neçə səhv nümunəsi çıxarılsın")
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    rdir = ensure_dir(cfg.experiment.results_dir)
    runsdir = rdir / "runs"
    test_path = Path(cfg.experiment.artifacts_dir) / "data" / "az_test.jsonl"
    if not test_path.exists():
        raise SystemExit(
            f"{test_path} tapılmadı — səhv taksonomiyası test mətnini tələb edir.\n"
            "Əvvəlcə `python -m src.data.splits --config <cfg>` işlədin.")
    test = read_jsonl(test_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(resolve_bases(cfg)[0].hf_id)
    except Exception as e:  # noqa: BLE001  (offline / şəbəkəsiz mühit)
        log.warning("Tokenizator yüklənmədi (%s) — 'morfoloji' kateqoriyası "
                    "atlanır, qalan təsnifat işləyir.", type(e).__name__)
        tokenizer = None

    ff = _load_false_friends(cfg.analysis.false_friends_file)
    ff_words = {r[0].lower() for r in ff}
    tech_words = {a.lower() for a, _ in FALLBACK_TECHNICAL} | \
                 {b.lower() for _, b in FALLBACK_TECHNICAL}

    ref = cfg.analysis.reference_size
    per_condition: dict[str, Counter] = defaultdict(Counter)
    totals: dict[str, int] = defaultdict(int)
    samples: dict[str, list] = defaultdict(list)

    n_skipped_schema = n_skipped_no_test = 0
    for f in sorted(runsdir.glob(f"*__n={ref}__*.json")):
        r = read_json(f)
        if int(r.get("run_result_schema_version", 0)) != 4:
            n_skipped_schema += 1
            continue
        cond = r["condition"]
        preds, gold = r.get("test_predictions"), r.get("test_gold")
        if not preds or not gold or len(preds) != len(test):
            n_skipped_no_test += 1
            continue
        for i, (p, g) in enumerate(zip(preds, gold)):
            totals[cond] += 1
            if p != g:
                cat = classify(test[i]["text"], tokenizer, ff_words, tech_words)
                per_condition[cond][cat] += 1
                if len(samples[cond]) < args.manual_sample:
                    samples[cond].append({
                        "text": test[i]["text"][:300],
                        "gold": g, "pred": p, "auto_category": cat,
                        "manual_category": "",   # <<< ƏL İLƏ DOLDURUN
                    })

    if not per_condition:
        # Dondurulmuş protokolda qrid `eval_split="val"` ilə işləyir, ona görə
        # `test_predictions` YOXDUR. Bu, xəta deyil — səhv taksonomiyası
        # YALNIZ yekun, açıq opt-in test qiymətləndirməsindən SONRA mənalıdır.
        log.warning(
            "Test proqnozu olan schema-v4 run tapılmadı (schema-görə atılan: %d, "
            "test-proqnozsuz: %d). Qrid EVAL_SPLIT=test ilə işlədilibsə, bu "
            "GÖZLƏNİLMƏZDİR və araşdırılmalıdır; EVAL_SPLIT=val ilə "
            "işlədilibsə, normaldır — bu analiz yalnız yekun test "
            "qiymətləndirməsindən sonra dolur.",
            n_skipped_schema, n_skipped_no_test)

    out = {"reference_size": ref,
           "status": "MEASURED" if per_condition else "NOT MEASURED",
           "n_skipped_wrong_schema": n_skipped_schema,
           "n_skipped_no_test_predictions": n_skipped_no_test,
           "conditions": {}}
    for cond, counter in per_condition.items():
        n_err = sum(counter.values())
        out["conditions"][cond] = {
            "n_evaluated": totals[cond],
            "n_errors": n_err,
            "error_rate_pct": round(100 * n_err / max(totals[cond], 1), 2),
            "counts": dict(counter),
            "shares_pct": {k: round(100 * v / max(n_err, 1), 1)
                           for k, v in counter.items()},
        }
        log.info("%-16s səhv=%d (%.1f%%)  %s", cond, n_err,
                 out["conditions"][cond]["error_rate_pct"],
                 out["conditions"][cond]["shares_pct"])

    write_json(out, rdir / "errors.json")

    # əl ilə yoxlama üçün CSV
    man = rdir / "errors_manual_sample.csv"
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "text", "gold", "pred", "auto_category", "manual_category"])
        for cond, rows in samples.items():
            for r in rows:
                w.writerow([cond, r["text"], r["gold"], r["pred"],
                            r["auto_category"], r["manual_category"]])
    log.info("Əl ilə yoxlama faylı: %s  ← 'manual_category' sütununu DOLDURUN", man)


if __name__ == "__main__":
    main()
