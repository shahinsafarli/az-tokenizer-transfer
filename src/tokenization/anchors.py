"""
ANCHOR (ortaq token) analizi  —  [Nəzarət C2 / GO–NO-GO qapısı]

Bu, layihənin BİRİNCİ icra ediləcək skriptidir. OMP yalnız iki tokenizatorun
ortaq token-ləri ("anchor") üzərində işləyir: hər tanış olmayan token onların
seyrək xətti kombinasiyası kimi ifadə edilir. Anchor azdırsa, OMP-nin tikinti
materialı yoxdur və layihə bu donorla mümkün deyil.

Anchor tapılmasının məntiqi `src/tokenization/anchor_map.py`-dədir — bu
skript və `src/transplant/build.py` EYNİ `compute_anchors()` funksiyasını
çağırır ki, GO/NO-GO qapısı ilə həqiqi transplant heç vaxt uyğunsuz düşməsin
(T3-ün tələbi).

Hər BAZA MODEL üçün ÜÇ rejim də hesablanır və yazılır (`strict`, `surface`,
`functional`) — GO/NO-GO qərarı isə `functional`-a əsaslanır, çünki o
konvensiyadan asılı deyil (bax anchor_map.py modul dosstring-i).

İşlətmə:
    python -m src.tokenization.anchors --config configs/experiment.yaml
    python -m src.tokenization.anchors --config ... --donor <basqa/model>

Çıxış:
    results/anchors.json   — hər üç rejimdə ortaq token sayı, qərar
"""
from __future__ import annotations

import sys

from transformers import AutoTokenizer

from src.tokenization.anchor_map import DEFAULT_MODE, MODES, compute_anchors
from src.tokenization.canon import detect_scheme, is_byte_level
from src.utils import (base_argparser, ensure_dir, get_logger, load_config,
                       resolve_bases, setup_logging, write_json)

log = get_logger(__name__)

# GO / NO-GO həddləri (icra bələdçisi §C2) — `functional` rejiminə tətbiq olunur
THRESH_GOOD = 20_000
THRESH_MIN = 5_000

DECISION_MODE = DEFAULT_MODE  # GO/NO-GO qərarını verən rejim (T3 qərarı) — anchor_map.py-də təyin olunub


def analyse(base_name: str, donor_name: str,
            donor_revision: str | None = None) -> dict:
    log.info("Tokenizatorlar yüklənir: base=%s  donor=%s (rev=%s)",
             base_name, donor_name, donor_revision or "HEAD")
    base_tok = AutoTokenizer.from_pretrained(base_name)
    # Donor revizyonu `build.py`-dəki ilə EYNİ olmalıdır. Əks halda
    # GO/NO-GO qapısı bir lüğəti, faktiki transplant BAŞQASINI görər —
    # `anchor_map.py`-ın aradan qaldırmaq üçün yazıldığı uyğunsuzluğun
    # məhz özü, sadəcə kod səviyyəsində yox, ARTEFAKT səviyyəsində.
    donor_tok = AutoTokenizer.from_pretrained(
        donor_name, **({"revision": donor_revision} if donor_revision else {}))

    base_vocab = base_tok.get_vocab()
    donor_vocab = donor_tok.get_vocab()

    base_scheme = detect_scheme(base_vocab.keys())
    donor_scheme = detect_scheme(donor_vocab.keys())
    base_bl = is_byte_level(base_vocab.keys())
    donor_bl = is_byte_level(donor_vocab.keys())
    log.info("Konvensiyalar: base=%s%s  donor=%s%s",
             base_scheme, " (byte-level)" if base_bl else "",
             donor_scheme, " (byte-level)" if donor_bl else "")

    # --- SƏHV YOL (müqayisə üçün göstərilir): xam sətir kəsişməsi
    naive_overlap = len(set(base_vocab) & set(donor_vocab))

    # --- ÜÇ RЕJİM DƏ — bax anchor_map.py
    modes: dict[str, dict] = {}
    raw_results: dict[str, object] = {}
    for mode in MODES:
        log.info("  rejim: %s ...", mode)
        r = compute_anchors(base_tok, donor_tok, mode=mode)
        raw_results[mode] = r
        modes[mode] = {
            "n_anchors": r.n_anchors,
            "anchor_share_of_donor_pct": round(100 * r.n_anchors / max(len(donor_vocab), 1), 2),
            "base_scheme": r.base_scheme,
            "donor_scheme": r.donor_scheme,
            "diagnostics": r.diagnostics,
        }
        log.info("    %-10s anchor=%-7d (donorun %.1f%%-i)",
                 mode, r.n_anchors, modes[mode]["anchor_share_of_donor_pct"])

    # --- DİAQNOSTİKA-YALNIZ: union(strict, functional) — İSTİFADƏ OLUNMUR.
    #
    # xlmr üçün strict "eyni konvensiya ailəsi" daxilində etibarlıdır (bax
    # yuxarı qeyd). xlm15 üçün isə strict-in 1637-si fastBPE-nin "plain"
    # kataloqundan (canon.py-də qəsdən düzəldilməyib) gəlir — donor söz-başı
    # tokeni baza tərəfinin sərhəd statusu bilinməyən tokeni ilə cütləşir,
    # yəni semantik cəhətdən GEVŞƏKDİR. Bir neçə yüz əlavə "anchor" qazanmaq
    # üçün bu qeyri-müəyyənliyi transplantın ÖZÜNƏ qarışdırmaq buna dəyməz.
    # Buna görə union YALNIZ log/nəticədə görünür, `compute_anchors`-ın heç
    # bir çağırışında istifadə OLUNMUR (build.py DEFAULT_MODE-u dəyişmədən
    # işlədir).
    strict_ids = set(raw_results["strict"].donor_ids.tolist())
    functional_ids = set(raw_results["functional"].donor_ids.tolist())
    union_ids = strict_ids | functional_ids
    only_in_strict = len(strict_ids - functional_ids)
    union_diag = {
        "n_union": len(union_ids),
        "n_strict_only": only_in_strict,
        "n_functional_only": len(functional_ids - strict_ids),
        "n_intersection": len(strict_ids & functional_ids),
        "used_for_transplant": False,
        "note": ("YALNIZ diaqnostika — transplant üçün İSTİFADƏ OLUNMUR. strict-in "
                 "əlavə etdiyi anchor-lar sərhəd statusu bilinməyən (fastBPE→plain "
                 "fallback) donor-baza cütləridir; bir neçə yüz atom qazanmaq üçün "
                 "bu qeyri-müəyyənliyi qəbul etmək dəyməz."),
    }
    log.info("  [diaqnostika] union(strict,functional)=%d  (yalnız strict=%d, yalnız functional=%d) — İŞLƏDİLMİR",
             union_diag["n_union"], union_diag["n_strict_only"], union_diag["n_functional_only"])

    n_decision = modes[DECISION_MODE]["n_anchors"]
    unfamiliar = len(donor_vocab) - n_decision

    if n_decision >= THRESH_GOOD:
        verdict, note = "GO", "Anchor sayı kifayətdir — davam edin."
    elif n_decision >= THRESH_MIN:
        verdict, note = ("GO_WITH_CARE",
                         "Anchor məhduddur — transplant.k dəyərini kiçildin (8–32).")
    else:
        verdict, note = ("NO_GO",
                         "Anchor çox azdır. Donor modeli dəyişin "
                         "(tercihən baza ilə eyni konvensiyalı tokenizator) "
                         "və ya T3b-yə keçin (subword-mean init).")

    res = {
        "base_model": base_name,
        "donor_model": donor_name,
        "donor_revision": donor_revision,
        "base_scheme": base_scheme,
        "donor_scheme": donor_scheme,
        "base_byte_level": base_bl,
        "donor_byte_level": donor_bl,
        "base_vocab_size": len(base_vocab),
        "donor_vocab_size": len(donor_vocab),
        "naive_string_overlap": naive_overlap,
        "modes": modes,
        "union_strict_functional_diagnostic_only": union_diag,
        "decision_mode": DECISION_MODE,
        "canonical_shared_anchors": n_decision,   # geriyə uyğunluq üçün — decision_mode-un sayı
        "unfamiliar_tokens": unfamiliar,
        "anchor_share_of_donor_pct": modes[DECISION_MODE]["anchor_share_of_donor_pct"],
        "verdict": verdict,
        "note": note,
        "thresholds": {"good": THRESH_GOOD, "min": THRESH_MIN},
    }

    log.info("Xam sətir kəsişməsi (naive) ... %d", naive_overlap)
    log.info("Qərar rejimi: %s → anchor=%d, tanış olmayan=%d", DECISION_MODE, n_decision, unfamiliar)
    log.info("QƏRAR: %s — %s", verdict, note)
    strict_n = modes["strict"]["n_anchors"]
    if strict_n and n_decision / max(strict_n, 1) > 2:
        log.info("Qeyd: `functional` rejimi `strict`-dən %.1f× çox anchor tapdı.",
                 n_decision / max(strict_n, 1))
    return res


def main() -> None:
    ap = base_argparser(__doc__)
    ap.add_argument("--donor", default=None, help="konfiqdəki donoru əvəz edir")
    ap.add_argument("--bases", default=None,
                    help="vergüllə: primary,contrast (default konfiqdən)")
    args = ap.parse_args()
    # Windows konsolları çox vaxt cp1252-dədir; Azərbaycan hərfləri loglama
    # zamanı xəta yaradır (JSON yazısına təsir etmir, gərəksiz stack-trace-lər).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    setup_logging(args.log_level)

    cfg = load_config(args.config, require_complete=False)
    donor = args.donor or cfg.models.donor
    # Açıq `--donor` verilibsə revizyon pinini tətbiq etmirik: o, başqa
    # bir modeldir və konfiqdəki SHA ona aid deyil.
    donor_revision = None if args.donor else cfg.models.get("donor_revision")
    roles = [r.strip() for r in args.bases.split(",")] if args.bases else None
    bases = resolve_bases(cfg, roles)

    out = {"donor": donor, "decision_mode": DECISION_MODE, "bases": {}}
    for b in bases:
        log.info("=" * 66)
        log.info("BAZA MODEL: %s (%s) · AZ pretraining-də: %s",
                 b.short, b.role, "VAR" if b.az_in_pretraining else "YOX")
        res = analyse(b.hf_id, donor, donor_revision)
        res["role"] = b.role
        res["az_in_pretraining"] = b.az_in_pretraining
        out["bases"][b.short] = res

    path = ensure_dir(cfg.experiment.results_dir) / "anchors.json"
    write_json(out, path)
    log.info("Yazıldı: %s", path)

    blocked = [k for k, v in out["bases"].items() if v["verdict"] == "NO_GO"]
    if blocked:
        raise SystemExit(
            f"\nNO-GO: {', '.join(blocked)} üçün anchor sayı çox azdır ({DECISION_MODE} rejimi).\n"
            "Həmin baza model üçün OMP transplant mümkün deyil:\n"
            "  --donor <model>   ilə başqa donor sınayın, YA DA\n"
            "  T3b-yə keçin: subword-mean init (bax CLAUDE_CODE_BRIEF.md §T3b)\n"
        )


if __name__ == "__main__":
    main()
