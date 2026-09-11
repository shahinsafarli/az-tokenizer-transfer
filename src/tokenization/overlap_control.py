"""
TOKEN ÖRTÜŞMƏSİ + NƏZARƏT DİLLƏRİ  —  layihənin yeni ƏSAS ölçməsi

Real korpusda XLM-R-in tokenizatorunda Azərbaycan mətninin token növlərinin
~41%-i türk mətnində də görünür. Amma bu rəqəm təkbaşına mənasızdır:

    Bu, DİL QOHUMLUĞUNDAN gəlir, yoxsa istənilən iki dil arasında belədir?

Bu skript AZ↔TR örtüşməsini NƏZARƏT dilləri ilə müqayisə edir:
    tr — qohum (oğuz qrupu)          → yüksək gözlənilir
    en — qohum deyil, latın əlifba   → nəzarət
    ru — qohum deyil, kiril əlifba   → nəzarət (əlifba effekti)
    fi — aqlütinativ, qohum deyil    → nəzarət (tipologiya effekti)

Oxunuş:
    AZ↔TR >> AZ↔nəzarət   →  örtüşmə əsl qohumluqdandır  (mexanizm təsdiqlənir)
    AZ↔TR ≈ AZ↔nəzarət    →  örtüşmə ümumi artefaktdır   (mexanizm zəifdir)

Həmçinin iki filtr tətbiq edilir:
    raw   — bütün tokenlər (durğu, rəqəm, simvol daxil)
    alpha — yalnız hərfdən ibarət tokenlər (əsas rəqəm BUDUR)

İşlətmə:
    python -m src.tokenization.overlap_control --config configs/experiment.yaml \
        --n-sentences 5000 --langs tr,en,ru,fi
"""
from __future__ import annotations


import numpy as np
from transformers import AutoTokenizer

from src.tokenization.canon import byte_decode
from src.tokenization.corpus_fertility import (fertility_stats,
                                               load_wikipedia)
from src.utils import (base_argparser, ensure_dir, get_logger, load_config,
                       resolve_bases, setup_logging, write_json)

log = get_logger(__name__)

LANG_LABEL = {
    "az": "azərbaycan (hədəf)",
    "tr": "türk (qohum — oğuz)",
    "en": "ingilis (nəzarət)",
    "ru": "rus (nəzarət, kiril)",
    "fi": "fin (nəzarət, aqlütinativ)",
    "hu": "macar (nəzarət, aqlütinativ)",
    "kk": "qazax (uzaq qohum — qıpçaq)",
}

# FLORES-200 dil kodları (paralel korpus üçün)
FLORES_CODE = {
    "az": "azj_Latn", "tr": "tur_Latn", "en": "eng_Latn",
    "ru": "rus_Cyrl", "fi": "fin_Latn", "hu": "hun_Latn", "kk": "kaz_Cyrl",
}


def load_flores(lang: str, n_sentences: int) -> list[str]:
    """
    FLORES-200 — EYNİ cümlələrin 200 dilə tərcüməsi.

    Bu, mövzu effektini tamamilə aradan qaldırır: Wikipedia-da AZ və TR
    məqalələri fərqli mövzulardadır, ona görə örtüşmənin bir hissəsi dilə
    yox, məzmuna aid ola bilər. FLORES-də məzmun eynidir.
    """
    from datasets import load_dataset
    code = FLORES_CODE.get(lang)
    if code is None:
        raise SystemExit(f"FLORES üçün '{lang}' kodu təyin edilməyib.")

    attempts = [
        ("facebook/flores", code, "dev"),
        ("Muennighoff/flores200", code, "dev"),
        ("openlanguagedata/flores_plus", code, "dev"),
    ]
    for ds_name, cfg_name, split in attempts:
        try:
            log.info("FLORES yüklənir: %s / %s", ds_name, cfg_name)
            ds = load_dataset(ds_name, cfg_name, split=split,
                              trust_remote_code=True)
            col = next((c for c in ("sentence", "text") if c in ds.column_names),
                       ds.column_names[0])
            return [str(r[col]) for r in ds][:n_sentences]
        except Exception as e:  # noqa: BLE001
            log.warning("  alınmadı (%s): %s", ds_name, type(e).__name__)
    raise SystemExit("FLORES yüklənmədi — --source wikipedia işlədin.")


def _clean(tok_str: str) -> str:
    """Byte-level tokenləri açır və sərhəd işarəsini atır."""
    d = byte_decode(tok_str) or tok_str
    return d.lstrip("▁").lstrip("##").strip()


def type_sets(tok, sentences: list[str]) -> tuple[set[str], set[str]]:
    """(bütün tokenlər, yalnız hərfli tokenlər) — kanonik formada."""
    raw: set[str] = set()
    alpha: set[str] = set()
    for s in sentences:
        for t in tok.tokenize(s):
            c = _clean(t)
            if not c:
                continue
            raw.add(c)
            if c.isalpha():
                alpha.add(c)
    return raw, alpha


def overlap(a: set[str], b: set[str]) -> dict:
    inter = a & b
    return {
        "a_types": len(a),
        "b_types": len(b),
        "shared": len(inter),
        "a_covered_pct": round(100 * len(inter) / max(len(a), 1), 2),
        "jaccard": round(len(inter) / max(len(a | b), 1), 4),
    }


def main() -> None:
    ap = base_argparser(__doc__)
    ap.add_argument("--n-sentences", type=int, default=5000)
    ap.add_argument("--langs", default="tr,en,ru,fi",
                    help="müqayisə dilləri (vergüllə)")
    ap.add_argument("--source", default="wikipedia",
                    choices=["wikipedia", "flores"],
                    help="flores = EYNİ cümlələr bütün dillərdə (mövzu effekti yoxdur)")
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config, require_complete=False)

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    loader = load_flores if args.source == "flores" else load_wikipedia

    # ---- korpuslar (hər dildən eyni sayda cümlə)
    log.info("Korpuslar yüklənir (mənbə: %s) ...", args.source)
    corpora = {"az": loader("az", args.n_sentences)}
    for l in langs:
        corpora[l] = loader(l, args.n_sentences)
    n_min = min(len(v) for v in corpora.values())
    corpora = {k: v[:n_min] for k, v in corpora.items()}
    log.info("Hər dildən %d cümlə götürüldü.", n_min)
    if args.source == "flores":
        log.info("PARALEL korpus — məzmun bütün dillərdə eynidir, "
                 "yəni fərq YALNIZ dildəndir.")

    out = {"n_sentences_per_lang": n_min, "langs": langs,
           "source": args.source, "models": []}

    _bases = resolve_bases(cfg)
    for name in [b.hf_id for b in _bases] + [cfg.models.donor]:
        log.info("=" * 66)
        log.info("Model: %s", name)
        tok = AutoTokenizer.from_pretrained(name)

        sets = {k: type_sets(tok, v) for k, v in corpora.items()}
        az_raw, az_alpha = sets["az"]

        rec = {
            "model": name,
            "az_fertility": fertility_stats(tok, corpora["az"])["fertility_micro"],
            "az_types_raw": len(az_raw),
            "az_types_alpha": len(az_alpha),
            "overlaps": {},
        }

        log.info("  AZ token növü: raw=%d  alpha=%d", len(az_raw), len(az_alpha))
        log.info("  %-30s %10s %10s", "müqayisə", "raw %", "alpha %")
        for l in langs:
            o_raw = overlap(az_raw, sets[l][0])
            o_alpha = overlap(az_alpha, sets[l][1])
            rec["overlaps"][l] = {"raw": o_raw, "alpha": o_alpha}
            log.info("  AZ ↔ %-24s %9.1f%% %9.1f%%",
                     LANG_LABEL.get(l, l), o_raw["a_covered_pct"],
                     o_alpha["a_covered_pct"])

        # ---- örtüşmənin PARÇALANMASI: əlifba bazası vs qohumluq artımı
        #
        # Kritik müşahidə: latın əlifbalı, amma qohum OLMAYAN dillər (en, fi)
        # praktiki olaraq eyni örtüşmə verir → bu, "əlifba/beynəlxalq söz bazası"dır.
        # Türk dilinin həmin bazadan YUXARISI əsl qohumluq siqnalıdır.
        if "tr" in langs and len(langs) > 1:
            tr_a = rec["overlaps"]["tr"]["alpha"]["a_covered_pct"]
            latin_ctrls = {l: rec["overlaps"][l]["alpha"]["a_covered_pct"]
                           for l in langs if l != "tr" and l != "ru"}
            nonlatin = {l: rec["overlaps"][l]["alpha"]["a_covered_pct"]
                        for l in langs if l == "ru"}

            baseline = (float(np.mean(list(latin_ctrls.values())))
                        if latin_ctrls else 0.0)
            spread = (max(latin_ctrls.values()) - min(latin_ctrls.values())
                      if len(latin_ctrls) > 1 else 0.0)
            lift = tr_a - baseline
            ratio = tr_a / max(baseline, 1e-9)
            headroom = 100.0 - baseline
            lift_share = 100.0 * lift / max(headroom, 1e-9)

            rec["relatedness_signal"] = {
                "tr_alpha_pct": tr_a,
                "latin_baseline_pct": round(baseline, 2),
                "latin_controls": {k: round(v, 2) for k, v in latin_ctrls.items()},
                "latin_baseline_spread_points": round(spread, 2),
                "nonlatin_controls": {k: round(v, 2) for k, v in nonlatin.items()},
                "relatedness_lift_points": round(lift, 2),
                "ratio_vs_baseline": round(ratio, 3),
                "lift_share_of_headroom_pct": round(lift_share, 2),
                "n_turkic_specific_types": int(round(
                    rec["az_types_alpha"] * lift / 100.0)),
            }
            log.info("  ── örtüşmənin parçalanması ──")
            log.info("     latın bazası (qohum olmayan) ... %.1f%%  (yayılma %.1f punkt)",
                     baseline, spread)
            if nonlatin:
                for k, v in nonlatin.items():
                    log.info("     latın olmayan (%s) ............ %.1f%%   "
                             "← əlifba effektini göstərir", k, v)
            log.info("     türk ........................... %.1f%%", tr_a)
            log.info("     >>> QOHUMLUQ ARTIMI ............ +%.1f punkt  (%.2f×)",
                     lift, ratio)
            log.info("     ≈ %d türkə xas paylaşılan token növü",
                     rec["relatedness_signal"]["n_turkic_specific_types"])
        out["models"].append(rec)

    # ---- yekun qərar (baza model üzrə)
    base = out["models"][0]
    sig = base.get("relatedness_signal")
    if sig:
        lift = sig["relatedness_lift_points"]
        spread = sig["latin_baseline_spread_points"]
        # Artım nəzarət dillərinin öz aralarındakı yayılmadan xeyli böyük olmalıdır,
        # əks halda "siqnal" sadəcə küydür.
        if lift >= 10 and lift >= 3 * max(spread, 0.5):
            v = ("RELATEDNESS_CONFIRMED",
                 f"Türk örtüşməsi latın bazasından +{lift:.1f} punkt yüksəkdir və bu, "
                 f"nəzarət dillərinin öz yayılmasından ({spread:.1f} punkt) qat-qat "
                 "böyükdür — leksik paylaşma mexanizmi təsdiqlənir.")
        elif lift >= 5:
            v = ("RELATEDNESS_MODERATE",
                 f"Türkə xas artım +{lift:.1f} punkt — real, amma orta səviyyədə. "
                 "Downstream effekti kiçik gözlənilməlidir.")
        else:
            v = ("RELATEDNESS_WEAK",
                 "Türk örtüşməsi nəzarət dillərindən əhəmiyyətli dərəcədə fərqlənmir — "
                 "yüksək AZ↔TR rəqəmi əsasən əlifba/beynəlxalq söz artefaktıdır.")
        out["decision"] = {"verdict": v[0], "note": v[1], **sig}
        log.info("=" * 66)
        log.info("QƏRAR: %s", v[0])
        log.info("  %s", v[1])
        log.info("  Şəkil 1 üçün əsas rəqəmlər: latın bazası %.1f%% · türk %.1f%% · "
                 "artım +%.1f punkt", sig["latin_baseline_pct"],
                 sig["tr_alpha_pct"], lift)

    path = ensure_dir(cfg.experiment.results_dir) / "overlap_control.json"
    write_json(out, path)
    log.info("Yazıldı: %s", path)


if __name__ == "__main__":
    main()
