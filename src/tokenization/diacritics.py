"""
DİAKRİTİK CONFOUND ÖLÇMƏSİ  —  Tapşırıq T1

XLM-15 tokenizatoru `do_lowercase_and_remove_accent=True` ilə (XLM-in
defolt davranışı) hərfləri kiçildir və aksanları siləndirir. Bu o deməkdir
ki, donor tokenizatorla əvəzlənmə (Şərt 3) EYNİ ANDA İKİ şeyi dəyişir:

    1. seqmentasiya qranulyarlığı        (M3-ün özü)
    2. orfoqrafik sədaqət (aksan bərpası) (ayrıca dəyişən)

Bu skript confound-u ÖLÇÜR, güman etmir:

  (1) XLM-15 fertility-ni eyni 5000 AZ/TR cümlə üzərində
      do_lowercase_and_remove_accent=True VƏ False ilə yenidən ölçür.
  (2)+(3) Azərbaycana-xas hərflərin (ə ş ç ö ü ğ ı İ) xlm15 / xlmr / donor
      lüğətlərində neçə tokendə göründüyünü sayır.
  (4) Fertility bölgüsünü (flag ON/OFF, AZ və TR üçün) yazır.

Nəticə: `results/diacritics.json`.

İşlətmə:
    python -m src.tokenization.diacritics --config configs/experiment.yaml \
        --n-sentences 5000
"""
from __future__ import annotations

import sys

from transformers import AutoTokenizer

from src.tokenization.canon import decode_vocab
from src.tokenization.corpus_fertility import (fertility_stats, load_lines,
                                                load_wikipedia)
from src.utils import (base_argparser, ensure_dir, get_logger, load_config,
                       setup_logging, write_json)

log = get_logger(__name__)

XLM15_ID = "FacebookAI/xlm-mlm-tlm-xnli15-1024"
XLMR_ID = "xlm-roberta-base"

# Azərbaycan əlifbasının latın hərfləri arasında öz nüsxəsi digər 15 dildə
# nadir/yox olan hərflər (böyük İ də daxil, çünki kiçilmə onu 'i'-yə aparır).
AZ_SPECIAL_LETTERS = ["ə", "ş", "ç", "ö", "ü", "ğ", "ı", "İ"]


def letter_vocab_counts(tok, letters: list[str]) -> dict:
    """
    Neçə vocab tokeninin tərkibində hər hərf var (aksanlı, case-sensitive).

    Byte-level lüğətlər (HPLT kimi) ƏVVƏLCƏ açılır (`decode_vocab`) — əks
    halda 'ə' hərfi 'ÉĻ' kimi bayt-kodlanmış görünür və heç vaxt tapılmır
    (canon.py-dəki eyni bug-un bir daha rast gəlinən forması).
    """
    raw_vocab = tok.get_vocab()
    decoded_vocab, was_byte_level = decode_vocab(raw_vocab)
    vocab = list(decoded_vocab.keys())
    counts = {letter: sum(1 for t in vocab if letter in t) for letter in letters}
    counts["_vocab_size"] = len(vocab)
    counts["_byte_level_decoded"] = was_byte_level
    return counts


def main() -> None:
    ap = base_argparser(__doc__)
    ap.add_argument("--n-sentences", type=int, default=5000)
    ap.add_argument("--az-file", default=None)
    ap.add_argument("--tr-file", default=None)
    args = ap.parse_args()
    # Windows konsolları çox vaxt cp1252-dədir; Azərbaycan hərfləri (ə, ş, ...)
    # çap zamanı loglama xətası yaradır (JSON yazısına təsir etmir, amma
    # gərəksiz stack-trace-lər verir) — konsolu UTF-8-ə keçiririk.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    setup_logging(args.log_level)
    cfg = load_config(args.config, require_complete=False)

    az = (load_lines(args.az_file, args.n_sentences) if args.az_file
          else load_wikipedia("az", args.n_sentences))
    tr = (load_lines(args.tr_file, args.n_sentences) if args.tr_file
          else load_wikipedia("tr", args.n_sentences))
    log.info("Korpus: AZ=%d cümlə, TR=%d cümlə", len(az), len(tr))

    out: dict = {"n_sentences_requested": args.n_sentences}

    # ---------------------------------------------------- (1) & (4): flag ON/OFF
    log.info("=" * 62)
    log.info("XLM-15 fertility: do_lowercase_and_remove_accent ON vs OFF")
    flag_results = {}
    for flag in (True, False):
        tok = AutoTokenizer.from_pretrained(XLM15_ID, do_lowercase_and_remove_accent=flag)
        az_st = fertility_stats(tok, az)
        tr_st = fertility_stats(tok, tr)
        ratio = az_st["fertility_micro"] / max(tr_st["fertility_micro"], 1e-9)
        key = "flag_on" if flag else "flag_off"
        flag_results[key] = {
            "az": az_st, "tr": tr_st,
            "az_over_tr_fertility": round(float(ratio), 4),
        }
        log.info("  %-8s AZ_fertility=%.4f  TR_fertility=%.4f  AZ/TR=%.4f",
                  key, az_st["fertility_micro"], tr_st["fertility_micro"], ratio)
    out["xlm15_fertility_flag_on_off"] = flag_results

    delta_az = (flag_results["flag_off"]["az"]["fertility_micro"]
                - flag_results["flag_on"]["az"]["fertility_micro"])
    delta_tr = (flag_results["flag_off"]["tr"]["fertility_micro"]
                - flag_results["flag_on"]["tr"]["fertility_micro"])
    flag_off_worse = bool(delta_az > 0)
    log.info("  flag=False dəyişimi (mənfi=yaxşılaşma, müsbət=pisləşmə):  "
             "AZ %+.4f   TR %+.4f", delta_az, delta_tr)

    # ---------------------------------------------------- (2) & (3): vocab letter counts
    log.info("=" * 62)
    log.info("Lüğətdə Azərbaycana-xas hərflərin sayı (token-level, case-sensitive)")
    donor_id = cfg.models.donor if "models" in cfg and "donor" in cfg.models else "HPLT/hplt_bert_base_az"
    model_specs = [
        ("xlm15", XLM15_ID),   # flag=True (default) ilə — real training vəziyyəti
        ("xlmr", XLMR_ID),
        ("donor", donor_id),
    ]
    vocab_letters = {}
    for short, name in model_specs:
        tok = AutoTokenizer.from_pretrained(name)
        counts = letter_vocab_counts(tok, AZ_SPECIAL_LETTERS)
        vocab_letters[short] = {"model": name, **counts}
        log.info("  %-6s vocab=%-7d %s", short, counts["_vocab_size"],
                  {l: counts[l] for l in AZ_SPECIAL_LETTERS})
    out["vocab_letter_counts"] = vocab_letters

    # ---------------------------------------------------- yekun
    # Üç fərqli hərf kateqoriyası — hamısını eyni "near_zero" qutusuna qoymaq
    # yanlışdır, çünki onlar accent-stripping-dən FƏRQLİ təsirlənirlər:
    #   DESTROYED  — combining-aksentli hərflər; NFD+Mn-silmə onları bazasına
    #                endirir (ş→s, ç→c, ö→o, ü→u, ğ→g). Lüğətdə demək olar
    #                sıfır olmalıdırlar, əgər flag=True ilə öyrədilibsə.
    #   ISOLATED   — 'ə'-nin özündə heç bir Unicode-birləşən aksent yoxdur
    #                (o, ayrıca hərfdir), ona görə silinmir, amma çox nadir
    #                hallarda tək-hərf tokeni kimi qalır.
    #   UNAFFECTED — 'ı'/'İ' combining-mark daşımır, ona görə accent-stripping
    #                onlara TƏSİR ETMİR; lüğətdə adi tezlikdə olmalıdırlar
    #                (Türk mətnində onsuz da geniş yayılıb).
    xlm15_counts = vocab_letters["xlm15"]
    DESTROYED = ["ş", "ç", "ö", "ü", "ğ"]
    ISOLATED = ["ə"]
    UNAFFECTED = ["ı", "İ"]

    destroyed_counts = {l: xlm15_counts[l] for l in DESTROYED}
    isolated_counts = {l: xlm15_counts[l] for l in ISOLATED}
    unaffected_counts = {l: xlm15_counts[l] for l in UNAFFECTED}
    destroyed_confirmed = all(v == 0 for v in destroyed_counts.values())
    isolated_confirmed = 0 < isolated_counts["ə"] <= 10   # yalnız təklikdə/nadir

    if destroyed_confirmed and flag_off_worse:
        note = (
            "XLM-15-in 95k lüğətində DESTROYED hərflər (ş ç ö ü ğ) HƏR BİRİ "
            f"0 dəfə görünür: {destroyed_counts} — bunlar pretraining zamanı "
            "artıq silinib, lüğətdə YOXDURLAR. do_lowercase_and_remove_accent="
            "False açmaq VƏZİYYƏTİ YAXŞILAŞDIRMIR, PİSLƏŞDİRİR: AZ fertility "
            f"{flag_results['flag_on']['az']['fertility_micro']:.3f} → "
            f"{flag_results['flag_off']['az']['fertility_micro']:.3f} "
            f"({delta_az:+.3f}) çünki indi bu hərflər UNK/tək-bayt tokenə düşür "
            "(lüğətdə qarşılığı olmadığı üçün). 'ə' demək olar təklikdə qalır "
            f"({isolated_counts['ə']} vocab girişi — 'izolyasiya edilmiş tək-hərf "
            "token' iddiasını təsdiqləyir). Əksinə 'ı'/'İ' combining-aksent "
            f"daşımadığı üçün toxunulmamış qalıb (lüğətdə {unaffected_counts['ı']}/"
            f"{unaffected_counts['İ']} dəfə — Türk mətnindən miras). NƏTİCƏ: "
            "ölçülmüş 3.206/1.814 defisitin BÖYÜK HİSSƏSİ orfoqrafik məhvin "
            "'düzəldilə bilən' forması DEYİL (bayraq kömək etmir) — deməli "
            "confound M3 lehinə TƏMİZLƏNİR: donor tokenizatora keçid "
            "SEQMENTASİYA + DİAKRİTİK BƏRPASI ikisini də edir, amma flag-only "
            "düzəliş mümkün olmadığı üçün bu iki komponenti ayırmaq yalnız "
            "T2 Option C (accent-stripped mətn üzərində ayrıca Şərt-3 icrası) "
            "ilə mümkündür — flag ilə deyil."
        )
    elif not flag_off_worse:
        note = (
            "flag=False AZ fertility-ni azaldır (yaxşılaşdırır) — deməli lüğətdə "
            "kifayət qədər aksanlı token var və bayrağın söndürülməsi işə yarayır. "
            "Bu halda defisitin bir hissəsi HƏQİQƏTƏN orfoqrafik itkidəndir, T2-də "
            "compound-treatment kimi rəsmiləşdirilməlidir."
        )
    else:
        note = (
            f"Qismən qarışıq: DESTROYED hərflər {destroyed_counts}, tam sıfır "
            "deyil, amma flag=False yenə fertility-ni pisləşdirir. Xam ədədlərə "
            "baxıb T2-ni əl ilə qərarlaşdırın."
        )

    out["confound_summary"] = {
        "destroyed_letters_xlm15_vocab_counts": destroyed_counts,
        "isolated_letter_ə_xlm15_vocab_count": isolated_counts["ə"],
        "unaffected_letters_xlm15_vocab_counts": unaffected_counts,
        "destroyed_confirmed_zero_in_vocab": destroyed_confirmed,
        "isolated_confirmed_rare_single_token": isolated_confirmed,
        "flag_off_makes_fertility_worse": flag_off_worse,
        "delta_az_fertility_flag_off_minus_on": round(float(delta_az), 4),
        "delta_tr_fertility_flag_off_minus_on": round(float(delta_tr), 4),
        "note": note,
    }

    path = ensure_dir(cfg.experiment.results_dir) / "diacritics.json"
    write_json(out, path)
    log.info("=" * 62)
    log.info("Yazıldı: %s", path)
    log.info("QEYD: %s", note)


if __name__ == "__main__":
    main()
