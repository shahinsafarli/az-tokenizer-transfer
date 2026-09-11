"""
FERTILITY və TOKEN ÖRTÜŞMƏSİ analizi  —  məqalənin BİRİNCİ şəkli

Ölçülənlər:
  1. Fertility        — hər sözə düşən token sayı (aşağı = yaxşı)
  2. Chars-per-token  — hər tokenə düşən hərf sayı (yüksək = yaxşı)
  3. ə/q/x statistikası — Azərbaycana xas hərflərin lüğətdə təmsili
  4. AZ↔TR token örtüşməsi (Jaccard) — fərziyyənin birinci empirik dəlili:
     Azərbaycan mətninin tokenlərinin nə qədəri türk mətnində də görünür?
     Yüksək örtüşmə → türk fine-tune-u elə AZ-ın işlətdiyi embedding-ləri isindirir.

İşlətmə:
    python -m src.tokenization.fertility --config configs/experiment.yaml
"""
from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer

from src.tokenization.canon import byte_decode, decode_vocab
from src.utils import (base_argparser, ensure_dir, get_logger, load_config, resolve_bases, setup_logging, write_json)

log = get_logger(__name__)

AZ_SPECIFIC = ["ə", "q", "x", "ğ", "ş", "ç", "ö", "ü", "ı"]

# Nümunə mətnlər — öz korpusunuzla əvəz edin (resources/*.txt)
FALLBACK_AZ_COLLOQUIAL = """
Uşaqlar məktəbdən gəlmişdilər və evdə gözləyirdilər.
Bu gün hava çox gözəldir, ona görə də gəzməyə çıxmaq istəyirəm.
Bacım universitetdə oxuyur, qardaşım isə işləyir.
Mən onu görməmişdim, amma haqqında çox eşitmişdim.
"""
FALLBACK_AZ_TECHNICAL = """
Qanunvericilik sahəsindəki dəyişikliklər haqqında məlumat əldə etmək üçün rəsmi mənbələrə müraciət edin.
Kompüter elmləri fakültəsində süni intellekt üzərində tədqiqatlar aparılır.
Təyyarənin uçuşu meteoroloji şəraitə görə təxirə salınmışdır.
Elektrik enerjisinin istehsalı və paylanması sahəsində islahatlar davam etdirilir.
"""
FALLBACK_TR = """
Çocuklar okuldan gelmişlerdi ve evde bekliyorlardı.
Bugün hava çok güzel, o yüzden yürüyüşe çıkmak istiyorum.
Bilgisayar bilimleri fakültesinde yapay zeka üzerine araştırmalar yapılıyor.
Uçağın kalkışı meteorolojik koşullar nedeniyle ertelenmiştir.
"""

PROBE_WORDS = [
    "gəlmişdilər", "qanunvericilik", "təyyarə", "kompüter",
    "xəstəxanalarımızda", "əməkdaşlıq", "işləyirlər",
]


def _load_or_fallback(path: str | None, fallback: str) -> str:
    if path and Path(path).exists():
        return Path(path).read_text(encoding="utf-8")
    return fallback


def measure(tok, text: str) -> dict:
    words = text.split()
    toks = tok.tokenize(text)
    n_chars = sum(len(w) for w in words)
    unk = tok.unk_token
    return {
        "n_words": len(words),
        "n_tokens": len(toks),
        "fertility": round(len(toks) / max(len(words), 1), 3),
        "chars_per_token": round(n_chars / max(len(toks), 1), 3),
        "n_unk": sum(1 for t in toks if t == unk),
    }


def vocab_letter_stats(tok) -> dict:
    """
    Azərbaycana xas hərflərin lüğətdə neçə tokendə göründüyü.

    DİQQƏT: byte-level lüğətlərdə 'ə' hərfi 'ÉĻ' kimi saxlanılır — açmadan
    saysaq, nəticə səhvən 0 çıxır. Ona görə əvvəlcə decode edirik.
    """
    vocab, was_byte_level = decode_vocab(tok.get_vocab())
    out = {"_byte_level_decoded": was_byte_level}
    for ch in AZ_SPECIFIC:
        out[ch] = sum(1 for k in vocab if ch in k)
    return out


def token_overlap(tok, az_text: str, tr_text: str) -> dict:
    """AZ və TR mətnlərinin token çoxluqlarının örtüşməsi."""
    az = set(tok.tokenize(az_text))
    tr = set(tok.tokenize(tr_text))
    inter = az & tr
    union = az | tr
    return {
        "az_unique_tokens": len(az),
        "tr_unique_tokens": len(tr),
        "shared": len(inter),
        "az_covered_by_tr_pct": round(100 * len(inter) / max(len(az), 1), 2),
        "jaccard": round(len(inter) / max(len(union), 1), 4),
    }


def analyse_one(name: str, az_coll: str, az_tech: str, tr: str,
                revision: str | None = None) -> dict:
    # Donor üçün revizyon pinlənir — `build.py` ilə EYNİ lüğət ölçülsün.
    tok = AutoTokenizer.from_pretrained(
        name, **({"revision": revision} if revision else {}))
    res = {
        "model": name,
        "revision": revision,
        "vocab_size": len(tok.get_vocab()),
        "az_colloquial": measure(tok, az_coll),
        "az_technical": measure(tok, az_tech),
        "tr": measure(tok, tr),
        "letters": vocab_letter_stats(tok),
        "overlap_az_tr": token_overlap(tok, az_coll + az_tech, tr),
        "probe_words": {
            w: {
                "n_pieces": len(tok.tokenize(w)),
                "pieces_raw": tok.tokenize(w),
                # byte-level tokenizatorlarda xam forma oxunmur: 'âĸģgÉĻl' -> '▁gəl'
                "pieces": [byte_decode(t) or t for t in tok.tokenize(w)],
            }
            for w in PROBE_WORDS
        },
    }
    az_f = (res["az_colloquial"]["fertility"] + res["az_technical"]["fertility"]) / 2
    res["az_over_tr_fertility_ratio"] = round(az_f / max(res["tr"]["fertility"], 1e-9), 3)
    return res


def main() -> None:
    ap = base_argparser(__doc__)
    ap.add_argument("--az-colloquial-file", default="resources/az_colloquial.txt")
    ap.add_argument("--az-technical-file", default="resources/az_technical.txt")
    ap.add_argument("--tr-file", default="resources/tr_sample.txt")
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config, require_complete=False)

    az_coll = _load_or_fallback(args.az_colloquial_file, FALLBACK_AZ_COLLOQUIAL)
    az_tech = _load_or_fallback(args.az_technical_file, FALLBACK_AZ_TECHNICAL)
    tr = _load_or_fallback(args.tr_file, FALLBACK_TR)

    donor_revision = cfg.models.get("donor_revision")
    models = ([(b.hf_id, None) for b in resolve_bases(cfg)] +
              [(cfg.models.donor, donor_revision)])
    out = {"models": []}
    for m, rev in models:
        log.info("Analiz: %s (rev=%s)", m, rev or "HEAD")
        r = analyse_one(m, az_coll, az_tech, tr, rev)
        out["models"].append(r)
        log.info("  fertility AZ(danışıq)=%.2f  AZ(texniki)=%.2f  TR=%.2f  → AZ/TR=%.2f",
                 r["az_colloquial"]["fertility"], r["az_technical"]["fertility"],
                 r["tr"]["fertility"], r["az_over_tr_fertility_ratio"])
        log.info("  AZ tokenlərinin %.1f%%-i TR mətnində də var (Jaccard %.3f)",
                 r["overlap_az_tr"]["az_covered_by_tr_pct"],
                 r["overlap_az_tr"]["jaccard"])
        for w, d in r["probe_words"].items():
            log.info("    %-20s %d hissə: %s", w, d["n_pieces"], d["pieces"])

    path = ensure_dir(cfg.experiment.results_dir) / "fertility.json"
    write_json(out, path)
    log.info("Yazıldı: %s", path)


if __name__ == "__main__":
    main()
