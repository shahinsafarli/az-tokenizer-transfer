"""
T5 · KÜY × ŞƏRT QARŞILIQLI TƏSİRİ TESTİ  —  bookkeeping DEYİL, ixtiyari deyil

Layihənin "küy tolerantlığı" arqumenti belədir: etiket səhvləri BÜTÜN 5
şərtdə EYNİDİR (ÜMUMİ-REJİM), ona görə hər fərqi EYNİ amillə SIXIŞDIRIR,
YENİ FƏRQ YARADA BİLMƏZ. Bu arqument YALNIZ etiket səhvləri öyrənilən
DİLÇİLİK XÜSUSİYYƏTİNDƏN ASILI OLMADIQDA doğrudur. Türk-mənşəli sözlərlə
zəngin (yaxud sarkastik) elementlər SİSTEMLİ şəkildə səhv etiketlənibsə,
küy ŞƏRTLƏ QARŞILIQLI TƏSİRDƏDİR və sıxışdırma arqumenti ÇÖKÜR.

Test: audit-un 100 elementi üzərində (yalnız iki auditorun KONSENSUS
verdiyi, "unclear" olmayan elementlər), `overlap_control.py`-ın istehsal
etdiyi Türk-spesifik paylaşılan alt-söz tiplərini (AZ∩TR, AZ∩(EN∪FI)
nəzarət-dillərindən ÇIXILARAQ — HANDOFF §3.2-nin "+12.6 xal lift"-inin
ARXASINDAKI TİPLƏRİN ÖZÜ, sadəcə sayı yox) işlədərək: dataset ilə insan
konsensusunun RAZILAŞMADIĞI elementlər bu tiplərlə ZƏNGİNLƏŞDİRİLİBMİ
(Fisher dəqiq test, 2×2 contingency)?

İşlətmə:
    python -m src.tokenization.overlap_control --config configs/experiment.yaml   # istinad üçün
    python -m src.analysis.noise_interaction --config configs/experiment.yaml

Çıxış:
    results/noise_interaction.json

QƏRAR QAYDASI:
    p ≥ 0.05  → assosiasiya YOXDUR → Limitations-a bir cümlə, davam et.
    p < 0.05  → assosiasiya VAR    → DAYAN, insana bildir, davam ETMƏ.
"""
from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import fisher_exact
from transformers import AutoTokenizer

from src.data.audit import RESULTS_ANN1, RESULTS_ANN2, RESULTS_KEY, _canonical_label, _read_csv_robust
from src.tokenization.corpus_fertility import load_wikipedia
from src.tokenization.overlap_control import _clean, type_sets
from src.utils import base_argparser, ensure_dir, get_logger, load_config, setup_logging, write_json

log = get_logger(__name__)

TOKENIZER_NAME = "xlm-roberta-base"    # overlap_control.py-ın öz istinad ölçmələri bu tokenizatorla
BASELINE_CONTROL_LANGS = ("en", "fi")  # qohum OLMAYAN Latin-əlifba dilləri — script/baseline effekti üçün çıxılır
N_SENTENCES = 5000
ALPHA = 0.05


def compute_turkish_specific_types(n_sentences: int = N_SENTENCES) -> set[str]:
    """
    AZ∩TR (hərfli tip) MINUS AZ∩(EN∪FI) — "türk-spesifik" leksik körpünün
    ÖZÜ (HANDOFF §3.2-nin ≈1,264 tip rəqəminin ARXASINDAKI faktiki tiplər).
    `overlap_control.py`-ın öz funksiyalarını (type_sets, load_wikipedia)
    təkrarlamadan işlədir.
    """
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    log.info("AZ korpusu yüklənir...")
    _, az_alpha = type_sets(tok, load_wikipedia("az", n_sentences))
    log.info("TR korpusu yüklənir...")
    _, tr_alpha = type_sets(tok, load_wikipedia("tr", n_sentences))

    baseline_shared: set[str] = set()
    for lang in BASELINE_CONTROL_LANGS:
        log.info("%s korpusu yüklənir (nəzarət)...", lang)
        _, alpha = type_sets(tok, load_wikipedia(lang, n_sentences))
        baseline_shared |= (az_alpha & alpha)

    az_tr_shared = az_alpha & tr_alpha
    turkish_specific = az_tr_shared - baseline_shared
    log.info("AZ∩TR=%d  AZ∩(EN∪FI)=%d  türk-spesifik=%d",
             len(az_tr_shared), len(baseline_shared), len(turkish_specific))
    return turkish_specific


def _item_has_turkish_specific_token(text: str, tok, turkish_specific: set[str]) -> bool:
    return any(_clean(t) in turkish_specific for t in tok.tokenize(text))


def build_consensus_items(cfg) -> tuple[list[dict], int]:
    """
    Hər audit elementi üçün (mövcuddursa) insan KONSENSUSUNU (hər iki auditor
    EYNİ, unclear olmayan cavab verdikdə) dataset gold-u ilə müqayisə edir.

    Konsensussuz (auditorlar fərqli deyib VƏ YA biri/hər ikisi "unclear")
    elementlər ATILIR — onlar üçün "insan həqiqəti" TƏYİN OLUNA BİLMİR.
    Qaytarır: (konsensuslu elementlər, atılan say).
    """
    results_dir = Path(cfg.experiment.results_dir)
    key = json.loads((results_dir / RESULTS_KEY).read_text(encoding="utf-8"))
    rows1, _ = _read_csv_robust(results_dir / RESULTS_ANN1)
    rows2, _ = _read_csv_robust(results_dir / RESULTS_ANN2)
    ann1_by_idx = {r["idx"]: _canonical_label(r.get("your_label", "")) for r in rows1}
    ann2_by_idx = {r["idx"]: _canonical_label(r.get("your_label", "")) for r in rows2}

    items, n_excluded = [], 0
    for idx, entry in key.items():
        a1, a2 = ann1_by_idx.get(idx), ann2_by_idx.get(idx)
        if a1 is None or a2 is None or a1 == "unclear" or a2 == "unclear" or a1 != a2:
            n_excluded += 1
            continue
        items.append({
            "idx": idx, "text": entry["text"], "gold": entry["gold_label_name"],
            "consensus": a1, "agrees_with_dataset": a1 == entry["gold_label_name"],
        })
    return items, n_excluded


def run_fisher_test(items: list[dict]) -> dict:
    """
    SAF funksiya (şəbəkə/fayl YOXDUR) — hər elementdə artıq
    `has_turkish_specific_token` VƏ `agrees_with_dataset` olmalıdır. Testlə
    birbaşa (sintetik `items` ilə) yoxlana bilər.
    """
    a = sum(1 for it in items if it["has_turkish_specific_token"] and not it["agrees_with_dataset"])
    b = sum(1 for it in items if not it["has_turkish_specific_token"] and not it["agrees_with_dataset"])
    c = sum(1 for it in items if it["has_turkish_specific_token"] and it["agrees_with_dataset"])
    d = sum(1 for it in items if not it["has_turkish_specific_token"] and it["agrees_with_dataset"])

    odds_ratio, p_value = fisher_exact([[a, b], [c, d]])
    n_disagree, n_agree = a + b, c + d
    rate_disagree = a / max(n_disagree, 1)
    rate_agree = c / max(n_agree, 1)
    association = bool(p_value < ALPHA)

    return {
        "n_items": len(items),
        "contingency_table": {
            "disagreement_with_turkish_specific_token": a,
            "disagreement_without": b,
            "agreement_with_turkish_specific_token": c,
            "agreement_without": d,
        },
        "rate_turkish_specific_given_disagreement_pct": round(100 * rate_disagree, 2),
        "rate_turkish_specific_given_agreement_pct": round(100 * rate_agree, 2),
        "fisher_odds_ratio": round(float(odds_ratio), 4),
        "fisher_p_value": round(float(p_value), 4),
        "alpha": ALPHA,
        "association_found": association,
        "conclusion": (
            "ASSOSİASİYA TAPILDI (p<0.05) — küy dilçilik xüsusiyyəti ilə QARŞILIQLI "
            "TƏSİRDƏDİR, ümumi-rejim (common-mode) arqumenti ÇÖKÜR. Bu, ÖZÜ bir tapıntıdır "
            "— DAYAN, insana bildir, davam ETMƏ."
            if association else
            "Assosiasiya tapılmadı (p≥0.05) — küy öyrənilən dilçilik xüsusiyyətindən ASILI "
            "DEYİL, ümumi-rejim arqumenti dayanır. Limitations bölməsinə bir cümlə əlavə "
            "edib davam edin — etiraz həmişəlik bağlanır."
        ),
    }


def main() -> None:
    ap = base_argparser(__doc__)
    ap.add_argument("--n-sentences", type=int, default=N_SENTENCES)
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config, require_complete=False)

    items, n_excluded = build_consensus_items(cfg)
    log.info("Konsensuslu element sayı: %d (konsensussuz/unclear atıldı: %d)",
             len(items), n_excluded)
    if len(items) < 10:
        raise SystemExit(
            f"Konsensuslu element sayı çox azdır ({len(items)}) — Fisher testi mənasız "
            "olardı. Real auditor məlumatı (hər iki vərəq doldurulmuş) olmadan bu analiz "
            "işlədilə bilməz."
        )

    turkish_specific = compute_turkish_specific_types(args.n_sentences)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    for it in items:
        it["has_turkish_specific_token"] = _item_has_turkish_specific_token(
            it["text"], tok, turkish_specific)

    result = run_fisher_test(items)
    result["n_items_excluded_no_consensus_or_unclear"] = n_excluded
    result["turkish_specific_type_count"] = len(turkish_specific)

    path = ensure_dir(cfg.experiment.results_dir) / "noise_interaction.json"
    write_json(result, path)
    log.info("=" * 60)
    log.info("Fisher exact: OR=%.3f  p=%.4f  (%s)",
             result["fisher_odds_ratio"], result["fisher_p_value"],
             "ASSOSİASİYA TAPILDI" if result["association_found"] else "assosiasiya yoxdur")
    log.info("Yazıldı: %s", path)


if __name__ == "__main__":
    main()
