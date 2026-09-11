"""
REAL KORPUS ÜZƏRİNDƏ FERTILITY  —  qərar verici ölçmə

Kiçik nümunə mətnlərdə alınan fertility etibarsızdır. Bu skript minlərlə
cümlə üzərində ölçür və fərziyyənin əsasını təsdiqləyir və ya rədd edir.

Əsas sual: XLM-R Azərbaycan dilini türk dilinə NİSBƏTƏN cəzalandırırmı?
    AZ/TR > 1.15  →  bəli, tokenizasiya defisiti var  (fərziyyə sağdır)
    AZ/TR ≈ 1.0   →  xeyr, defisit yoxdur             (fərziyyə yenidən baxılmalı)

İşlətmə (HF Wikipedia ilə — heç nə hazırlamaq lazım deyil):
    python -m src.tokenization.corpus_fertility --config configs/experiment.yaml \
        --n-sentences 5000

Öz mətn fayllarınızla:
    python -m src.tokenization.corpus_fertility --config configs/experiment.yaml \
        --az-file resources/az_corpus.txt --tr-file resources/tr_corpus.txt
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from src.utils import (base_argparser, ensure_dir, get_logger, load_config,
                       resolve_bases, setup_logging, write_json)

log = get_logger(__name__)

WORD_RE = re.compile(r"\w+", re.UNICODE)


def load_wikipedia(lang: str, n_sentences: int) -> list[str]:
    """HF Wikipedia-dan cümlələr çəkir."""
    from datasets import load_dataset
    for cfg_name in (f"20231101.{lang}", f"20220301.{lang}"):
        try:
            log.info("Wikipedia yüklənir: %s", cfg_name)
            ds = load_dataset("wikimedia/wikipedia", cfg_name,
                              split="train", streaming=True)
            out: list[str] = []
            for row in ds:
                for line in row["text"].split("\n"):
                    line = line.strip()
                    if 40 <= len(line) <= 400 and len(WORD_RE.findall(line)) >= 5:
                        out.append(line)
                        if len(out) >= n_sentences:
                            return out
            return out
        except Exception as e:  # noqa: BLE001
            log.warning("  %s alınmadı: %s", cfg_name, type(e).__name__)
    raise SystemExit(f"Wikipedia '{lang}' yüklənmədi — --az-file/--tr-file işlədin.")


def load_lines(path: str, n: int) -> list[str]:
    lines = [l.strip() for l in Path(path).read_text(encoding="utf-8").split("\n")]
    return [l for l in lines if len(WORD_RE.findall(l)) >= 5][:n]


def fertility_stats(tok, sentences: list[str]) -> dict:
    """Cümlə-cümlə fertility → orta + persentillər (tək ədəddən etibarlıdır)."""
    per_sentence, n_tok_total, n_word_total, n_char_total = [], 0, 0, 0
    for s in sentences:
        words = WORD_RE.findall(s)
        if not words:
            continue
        toks = tok.tokenize(s)
        per_sentence.append(len(toks) / len(words))
        n_tok_total += len(toks)
        n_word_total += len(words)
        n_char_total += sum(len(w) for w in words)
    arr = np.array(per_sentence, dtype=float)
    return {
        "n_sentences": len(per_sentence),
        "n_words": n_word_total,
        "n_tokens": n_tok_total,
        "fertility_micro": round(n_tok_total / max(n_word_total, 1), 4),
        "fertility_macro_mean": round(float(arr.mean()), 4),
        "fertility_std": round(float(arr.std(ddof=1)), 4),
        "fertility_p50": round(float(np.percentile(arr, 50)), 4),
        "fertility_p90": round(float(np.percentile(arr, 90)), 4),
        "chars_per_token": round(n_char_total / max(n_tok_total, 1), 4),
    }


def token_overlap(tok, az: list[str], tr: list[str], cap: int = 2000) -> dict:
    """
    AZ və TR mətnlərinin token çoxluqlarının örtüşməsi.

    QEYD: məzmun fərqli olduğu üçün bu, tokenizatorun yox, mövzunun da təsirini
    daşıyır. Ona görə eyni sayda cümlə götürülür və nəticə yalnız İŞARƏ kimi
    oxunmalıdır — qəti sübut kimi yox.
    """
    n = min(len(az), len(tr), cap)
    a = set()
    b = set()
    for s in az[:n]:
        a.update(tok.tokenize(s))
    for s in tr[:n]:
        b.update(tok.tokenize(s))
    inter = a & b
    return {
        "n_sentences_each": n,
        "az_types": len(a),
        "tr_types": len(b),
        "shared_types": len(inter),
        "az_covered_by_tr_pct": round(100 * len(inter) / max(len(a), 1), 2),
        "jaccard": round(len(inter) / max(len(a | b), 1), 4),
    }


def main() -> None:
    ap = base_argparser(__doc__)
    ap.add_argument("--n-sentences", type=int, default=5000)
    ap.add_argument("--az-file", default=None)
    ap.add_argument("--tr-file", default=None)
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config, require_complete=False)

    az = (load_lines(args.az_file, args.n_sentences) if args.az_file
          else load_wikipedia("az", args.n_sentences))
    tr = (load_lines(args.tr_file, args.n_sentences) if args.tr_file
          else load_wikipedia("tr", args.n_sentences))
    log.info("Korpus: AZ=%d cümlə, TR=%d cümlə", len(az), len(tr))

    bases = resolve_bases(cfg)
    names = [(b.short, b.hf_id) for b in bases] + [("donor", cfg.models.donor)]
    out = {"n_sentences_requested": args.n_sentences, "models": []}
    for short, name in names:
        log.info("=" * 62)
        log.info("Model: %s", name)
        tok = AutoTokenizer.from_pretrained(name)
        az_st = fertility_stats(tok, az)
        tr_st = fertility_stats(tok, tr)
        ratio = az_st["fertility_micro"] / max(tr_st["fertility_micro"], 1e-9)
        ov = token_overlap(tok, az, tr)

        log.info("  AZ fertility  micro=%.3f  median=%.3f  p90=%.3f  (n=%d cümlə, %d söz)",
                 az_st["fertility_micro"], az_st["fertility_p50"],
                 az_st["fertility_p90"], az_st["n_sentences"], az_st["n_words"])
        log.info("  TR fertility  micro=%.3f  median=%.3f",
                 tr_st["fertility_micro"], tr_st["fertility_p50"])
        log.info("  >>> AZ/TR = %.3f", ratio)
        log.info("  AZ tokenlərinin %.1f%%-i TR-də də var (Jaccard %.3f)",
                 ov["az_covered_by_tr_pct"], ov["jaccard"])

        out["models"].append({
            "short": short, "model": name, "az": az_st, "tr": tr_st,
            "az_over_tr_fertility": round(float(ratio), 4),
            "overlap_az_tr": ov,
        })

    # ---- qərar
    base_ratio = out["models"][0]["az_over_tr_fertility"]
    b_az = out["models"][0]["az"]["fertility_micro"]
    d_az = out["models"][-1]["az"]["fertility_micro"]
    gain = 100 * (b_az - d_az) / max(b_az, 1e-9)

    if base_ratio > 1.15:
        verdict = ("PREMISE_HOLDS",
                   "Baza model AZ-ı TR-ə nisbətən cəzalandırır — fərziyyə sağdır.")
    elif base_ratio > 1.05:
        verdict = ("PREMISE_WEAK",
                   "Cüzi cəza var — effekt kiçik gözlənilməlidir.")
    else:
        verdict = ("PREMISE_FAILS",
                   "Baza model AZ-ı cəzalandırmır. Tokenizasiya defisiti fərziyyəsi "
                   "bu setup-da təsdiqlənmir — layihəni yenidən çərçivələyin.")

    out["decision"] = {
        "base_az_over_tr": base_ratio,
        "donor_fertility_gain_pct": round(float(gain), 2),
        "verdict": verdict[0], "note": verdict[1],
    }
    log.info("=" * 62)
    log.info("QƏRAR: %s — %s", verdict[0], verdict[1])
    log.info("Donorun fertility qazancı: %.1f%%", gain)

    path = ensure_dir(cfg.experiment.results_dir) / "corpus_fertility.json"
    write_json(out, path)
    log.info("Yazıldı: %s", path)


if __name__ == "__main__":
    main()
