"""
POST-HOC EMBEDDİNG NORM YOXLANIŞI  —  ARTIQ tikilmiş transplantlar üçün

`build.py` artıq işlədilib, model saxlanılıb — bu skript OMP-ni YENİDƏN
İŞLƏTMİR, sadəcə saxlanılmış modelin embedding matrisini oxuyub eyni
anchor/tanış-olmayan bölgüsünü (compute_anchors + align_special_tokens,
build.py-nin İSTİFADƏ ETDİYİ EYNİ funksiyalar, ona görə EYNİ nəticə) təzədən
hesablayır və `src.transplant.embedding_norms.compute_norm_report`-u çağırır.

İşlətmə (hər iki baza üçün `build` tamamlandıqdan SONRA):
    python -m src.transplant.check_embedding_norms --config configs/experiment.yaml

Çıxış:
    results/embedding_norm_report__<base>.json
"""
from __future__ import annotations

import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.tokenization.anchor_map import DEFAULT_MODE, compute_anchors
from src.tokenization.canon import align_special_tokens
from src.transplant.donor_embeddings import load_donor_embeddings
from src.transplant.embedding_norms import compute_norm_report
from src.utils import (BaseModel, base_argparser, default_tag, ensure_dir, get_logger,
                       load_config, resolve_bases, setup_logging, transplant_dir, write_json)

log = get_logger(__name__)


def check_one(cfg, base: BaseModel) -> dict:
    tag = default_tag(cfg)
    outdir = transplant_dir(cfg, base.short, tag)
    donor_name = cfg.models.donor
    donor_revision = cfg.models.donor_revision

    log.info("[%s] baza model yüklənir: %s", base.short, base.hf_id)
    base_tok = AutoTokenizer.from_pretrained(base.hf_id)
    base_model = AutoModelForMaskedLM.from_pretrained(base.hf_id)
    E_base = base_model.get_input_embeddings().weight.detach().cpu().numpy()

    log.info("[%s] donor yüklənir: %s (rev=%s)", base.short, donor_name, donor_revision)
    donor_tok = AutoTokenizer.from_pretrained(donor_name, revision=donor_revision)
    E_donor, _, _ = load_donor_embeddings(donor_name, donor_revision)

    # build.py-nin İSTİFADƏ ETDİYİ EYNİ bölgü — OMP-ni YENİDƏN İŞLƏTMƏDƏN
    anchor_result = compute_anchors(base_tok, donor_tok, mode=DEFAULT_MODE)
    special_map = align_special_tokens(base_tok, donor_tok)
    donor_vocab_size = len(donor_tok.get_vocab())
    is_anchor = np.zeros(donor_vocab_size, dtype=bool)
    is_anchor[anchor_result.donor_ids] = True
    for did in special_map:
        if 0 <= did < donor_vocab_size:
            is_anchor[did] = True
    unfamiliar_ids = np.where(~is_anchor)[0]

    log.info("[%s] transplant olunmuş model yüklənir: %s", base.short, outdir)
    new_model = AutoModelForMaskedLM.from_pretrained(str(outdir))
    E_new = new_model.get_input_embeddings().weight.detach().cpu().numpy()

    report = compute_norm_report(E_base, E_donor, E_new,
                                 anchor_result.donor_ids, unfamiliar_ids)
    report["base"] = base.short
    report["base_role"] = base.role
    log.info("[%s] anchor median=%.4f  reconstructed median=%.4f  nisbət=%.4f  (sağlam=%s)",
             base.short, report["new_anchor_rows"]["median"],
             report["new_reconstructed_rows"]["median"],
             report["reconstructed_to_anchor_median_ratio"], report["in_healthy_range"])
    return report


def main() -> None:
    ap = base_argparser(__doc__)
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config, require_complete=False)

    for base in resolve_bases(cfg):
        report = check_one(cfg, base)
        path = ensure_dir(cfg.experiment.results_dir) / f"embedding_norm_report__{base.short}.json"
        write_json(report, path)
        log.info("Yazıldı: %s", path)


if __name__ == "__main__":
    main()
