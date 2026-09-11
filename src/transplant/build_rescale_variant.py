"""
RESCALE VARIANT  —  reconstructed vektorları bazanın öz anchor-median normuna gətirir

Kontekst (bax HANDOFF.md, norm-ratio bölməsi): `omp.py`-dəki köhnə səhv artıq
silinib (donor-hədəf normuna miqyaslama), amma bu, "heç bir miqyaslama"
demək deyil — sual açıq qalırdı: OMP-nin reconstruction-u ÖZÜ (heç bir
miqyaslama olmadan) hər iki bazada sağlam normda çıxır, yoxsa yox?

Ölçmə göstərdi: xlm15 sağlam (nisbət 1.13), xlmr DEYİL (nisbət 1.91 —
şişmə). İstifadəçinin qeyd etdiyi kimi, pre-registered prinsip simmetrikdir:
reconstruksiya olunmuş sətirlər BAZA modelin ÖZ miqyasında yaşamalıdır —
bu, 1.91× üçün də 0.7× üçün olduğu kimi POZULUR, nümunə sadəcə bir-tərəfli
idi. Ona görə bu variant HƏR İKİ baza üçün qurulur, YALNIZ xlmr üçün yox.

Bu skript `omp.py`-ni YENİDƏN İŞLƏTMİR (əmsallar artıq düzgündür, ~35 dəq
CPU vaxtına ehtiyac yoxdur) — sadəcə ARTIQ saxlanılmış transplant modelinin
"tanış olmayan" (reconstructed) sətirlərini GÖTÜRÜR, İSTİQAMƏTLƏRİNİ
SAXLAYIR, VƏ hər sətri bazanın ÖZ anchor sətirlərinin median L2 normuna
miqyaslayır — köhnə kodun fikri, bu dəfə DÜZGÜN hədəflə (donor normu yox,
bazanın öz anchor normu).

`k` və `n_candidates` TOXUNULMUR (pre-registered, bu pas-da dəyişdirilmir).

İşlətmə (hər iki `build` ARTIQ tamamlandıqdan sonra):
    python -m src.transplant.build_rescale_variant --config configs/experiment.yaml

Çıxış:
    artifacts/transplanted__<base>__<method>_k<k>_rescaled/
    results/transplant__<base>__<method>_k<k>_rescaled.json
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.tokenization.anchor_map import DEFAULT_MODE, compute_anchors
from src.tokenization.canon import align_special_tokens
from src.transplant.donor_embeddings import load_donor_embeddings
from src.utils import (BaseModel, base_argparser, default_tag, ensure_dir, get_logger,
                       load_config, resolve_bases, setup_logging, transplant_dir,
                       transplant_tag, write_json)

log = get_logger(__name__)

RESCALED_SUFFIX = "_rescaled"


def rescaled_tag(base_tag: str) -> str:
    return base_tag + RESCALED_SUFFIX


def build_variant(cfg, base: BaseModel, src_tag: str | None = None) -> dict:
    src_tag = src_tag or default_tag(cfg)
    src_dir = transplant_dir(cfg, base.short, src_tag)
    dst_tag = rescaled_tag(src_tag)
    dst_dir = transplant_dir(cfg, base.short, dst_tag)
    donor_name = cfg.models.donor
    donor_revision = cfg.models.donor_revision

    log.info("[%s] baza tokenizator: %s", base.short, base.hf_id)
    base_tok = AutoTokenizer.from_pretrained(base.hf_id)

    log.info("[%s] donor tokenizator: %s (rev=%s)", base.short, donor_name, donor_revision)
    donor_tok = AutoTokenizer.from_pretrained(donor_name, revision=donor_revision)
    E_donor, _, _ = load_donor_embeddings(donor_name, donor_revision)

    # build.py-nin İSTİFADƏ ETDİYİ EYNİ bölgü (OMP-ni YENİDƏN İŞLƏTMƏDƏN)
    anchor_result = compute_anchors(base_tok, donor_tok, mode=DEFAULT_MODE)
    special_map = align_special_tokens(base_tok, donor_tok)
    donor_vocab_size = len(donor_tok.get_vocab())
    is_anchor = np.zeros(donor_vocab_size, dtype=bool)
    is_anchor[anchor_result.donor_ids] = True
    for did in special_map:
        if 0 <= did < donor_vocab_size:
            is_anchor[did] = True
    unfamiliar_ids = np.where(~is_anchor)[0]
    anchor_donor_ids = np.where(is_anchor)[0]

    log.info("[%s] transplant (no-rescale) yüklənir: %s", base.short, src_dir)
    model = AutoModelForMaskedLM.from_pretrained(str(src_dir))
    E_new = model.get_input_embeddings().weight.detach().cpu().numpy().copy()

    # ------------------------------------------------------------ miqyaslama
    # Hədəf: BAZANIN ÖZ anchor sətirlərinin median L2 normu (donorun normu
    # DEYİL — köhnə səhvin YENİDƏN təkrarlanmaması üçün bu, açıq şəkildə
    # `E_new[anchor_donor_ids]`-dən, yəni artıq BAZA fəzasında olan
    # sətirlərdən hesablanır).
    anchor_rows = E_new[anchor_donor_ids]
    target_median = float(np.median(np.linalg.norm(anchor_rows, axis=1)))
    log.info("[%s] hədəf median norm (bazanın öz anchor sətirləri) = %.4f",
             base.short, target_median)

    recon_rows = E_new[unfamiliar_ids]
    row_norms = np.linalg.norm(recon_rows, axis=1, keepdims=True)
    safe_norms = np.maximum(row_norms, 1e-8)
    rescaled_rows = recon_rows / safe_norms * target_median
    E_new[unfamiliar_ids] = rescaled_rows

    with torch.no_grad():
        model.get_input_embeddings().weight.copy_(torch.from_numpy(E_new))
        out_emb = model.get_output_embeddings()
        if out_emb is not None and out_emb.weight.shape[0] == donor_vocab_size:
            if out_emb.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
                out_emb.weight.copy_(torch.from_numpy(E_new))
    model.tie_weights()

    ensure_dir(dst_dir)
    model.save_pretrained(dst_dir)
    donor_tok.save_pretrained(dst_dir)
    log.info("[%s] rescale variantı saxlanıldı: %s", base.short, dst_dir)

    # NOTE: `compute_norm_report()` (embedding_norms.py) also wants `E_base`/
    # `E_donor` all-rows stats for context; those are unchanged from the
    # no-rescale build's own report (`results/embedding_norm_report__<base>.
    # json`) and are not recomputed here — only the two numbers that this
    # variant actually changes (reconstructed-row norms, and the ratio built
    # from them) are reported.
    report = {
        "base": base.short, "base_role": base.role,
        "variant": "rescale_to_base_anchor_median",
        "target_median_norm": round(target_median, 4),
        "new_anchor_rows_median": round(float(np.median(np.linalg.norm(E_new[anchor_donor_ids], axis=1))), 4),
        "new_reconstructed_rows_median": round(float(np.median(np.linalg.norm(E_new[unfamiliar_ids], axis=1))), 4),
    }
    report["reconstructed_to_anchor_median_ratio"] = round(
        report["new_reconstructed_rows_median"] / max(report["new_anchor_rows_median"], 1e-8), 4)

    info = {
        "tag": dst_tag, "full_tag": transplant_tag(base.short, dst_tag),
        "base": base.short, "base_role": base.role,
        "variant": "rescale_to_base_anchor_median",
        "source_tag": src_tag, "output_dir": str(dst_dir),
        "n_anchor": int(len(anchor_donor_ids)), "n_unfamiliar": int(len(unfamiliar_ids)),
        "norm_report": report,
    }
    write_json(info, Path(cfg.experiment.results_dir) / f"transplant__{base.short}__{dst_tag}.json")
    return info


def main() -> None:
    ap = base_argparser(__doc__)
    ap.add_argument("--method", default=None, help="omp | mean | random_coef")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--tag", default=None, help="source (unrescaled) transplant tag")
    ap.add_argument("--bases", default=None, help="comma-separated: primary,contrast")
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config, require_complete=False)
    method = args.method or cfg.transplant.method
    k = args.k if args.k is not None else cfg.transplant.k
    src_tag = args.tag or f"{method}_k{k}"
    roles = [r.strip() for r in args.bases.split(",")] if args.bases else None

    for base in resolve_bases(cfg, roles):
        log.info("=" * 70)
        build_variant(cfg, base, src_tag=src_tag)


if __name__ == "__main__":
    main()
