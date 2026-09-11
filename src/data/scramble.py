"""
QARIŞDIRILMIŞ TÜRKCƏ  [Nəzarət C4 · Şərt 5]  —  leksik vs sintaktik ayrımı

Şərt 2 (`turk`) türk mərhələsinin faydasını göstərir, amma o fayda İKİ
şeydən gələ bilər:

  M1 · sintaktik — türk qrammatik strukturu (söz sırası, uzlaşma)
  M2 · leksik    — paylaşılan subword embedding-lərinin "isindirilməsi"

Bu nəzarət söz sırasını DAĞIDIR, leksikonu isə TOXUNMADAN saxlayır. Cümlədəki
söz ÇOXLUĞU eynidir (`word_multiset_preserved` bunu yoxlayır), yalnız SIRA
təsadüfidir. Ona görə:

    Şərt 5 ≈ Şərt 2   →  fayda LEKSİKDİR (sıra lazım deyildi)
    Şərt 5 ≪ Şərt 2   →  fayda SİNTAKTİKDİR (sıra vacib idi)

Söz-səviyyəsində qarışdırırıq, subword səviyyəsində YOX: subword-ləri
qarışdırmaq sözlərin özünü məhv edərdi və leksikonu qorumazdı — nəzarətin
bütün mənası itərdi.

İşlətmə (`src.data.splits`-dən SONRA):
    python -m src.data.scramble --config configs/experiment.yaml

Çıxış:
    artifacts/data/tr_train_scrambled.jsonl
    results/scramble.json
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.splits import read_jsonl, write_jsonl
from src.utils import (base_argparser, get_logger, load_config, setup_logging,
                       write_json)

log = get_logger(__name__)

# Bundan qısa cümlələr olduğu kimi qalır — 1 sözlük "qarışdırma" mənasızdır,
# 2 sözlük isə determinist şəkildə tərsinə çevrilməkdən başqa seçim vermir.
MIN_WORDS_TO_SCRAMBLE = 3
# Eyni permutasiyanı təkrar-təkrar alıb "dəyişmədi" nəticəsinə düşməmək üçün
# neçə dəfə yenidən cəhd edilsin.
MAX_SHUFFLE_ATTEMPTS = 10


def scramble_sentence(sentence: str, rng: np.random.Generator) -> str:
    """
    Söz sırasını təsadüfi dəyişir, söz ÇOXLUĞUNU olduğu kimi saxlayır.

    >>> scramble_sentence("merhaba", np.random.default_rng(0))
    'merhaba'

    Qısa cümlələr (< MIN_WORDS_TO_SCRAMBLE söz) dəyişmədən qaytarılır.
    """
    words = sentence.split()
    if len(words) < MIN_WORDS_TO_SCRAMBLE:
        return sentence
    for _ in range(MAX_SHUFFLE_ATTEMPTS):
        order = rng.permutation(len(words))
        out = [words[i] for i in order]
        if out != words:
            return " ".join(out)
    return " ".join(out)


def main() -> None:
    ap = base_argparser(__doc__)
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    datadir = Path(cfg.experiment.artifacts_dir) / "data"
    src = datadir / "tr_train.jsonl"
    if not src.exists():
        raise SystemExit(f"{src} tapılmadı — əvvəlcə `python -m src.data.splits` işlədin.")

    # Bölgü seed-i işlədilir (MODEL seed-i YOX): qarışdırılmış korpus bütün
    # model seed-lərində EYNİ olmalıdır, əks halda Şərt 5 seed-lər arasında
    # fərqli bir datasetə çevrilər və nəzarət nəzarət olmaqdan çıxar.
    rng = np.random.default_rng(int(cfg.data.split.seed))
    records = read_jsonl(src)

    out_records = []
    preserved = 0
    changed = 0
    for r in records:
        scrambled = scramble_sentence(r["text"], rng)
        if sorted(scrambled.split()) == sorted(r["text"].split()):
            preserved += 1
        if scrambled != r["text"]:
            changed += 1
        out_records.append({"text": scrambled, "label": r["label"]})

    dst = datadir / "tr_train_scrambled.jsonl"
    write_jsonl(out_records, dst)

    n = len(records)
    report = {
        "n": n,
        "word_multiset_preserved": preserved,
        "word_multiset_preserved_pct": round(100.0 * preserved / max(n, 1), 2),
        "order_actually_changed": changed,
        "output": str(dst),
        "note": ("word_multiset_preserved_pct 100-ə yaxın olmalıdır — token "
                 "çoxluğu dəyişməyib."),
    }
    write_json(report, Path(cfg.experiment.results_dir) / "scramble.json")

    log.info("Qarışdırıldı: %d cümlə | söz-çoxluğu qorundu: %d (%.2f%%) | sıra dəyişdi: %d",
             n, preserved, report["word_multiset_preserved_pct"], changed)
    if report["word_multiset_preserved_pct"] < 99.9:
        raise SystemExit(
            "C4 POZULDU: söz çoxluğu qorunmayıb — qarışdırma leksikonu "
            "dəyişdirib, nəzarət etibarsızdır.")


if __name__ == "__main__":
    main()
