"""
TRANSPLANT OLUNMUŞ MODELİN QURULMASI

Baza modelin (XLM-R) çəkilərini saxlayır, tokenizatorunu donorun tokenizatoru
ilə əvəz edir. Yeni embedding matrisi:

    anchor tokenlər        → bazadakı embedding BİRBAŞA köçürülür
    xüsusi tokenlər        → ROLA görə uyğunlaşdırılır (<s>≡[CLS], <pad>≡[PAD], ...)
    tanış olmayan tokenlər → OMP ilə yenidən qurulur

Hər BAZA MODEL üçün ayrı transplant qurulur (xlm15 və xlmr) — çünki
embedding matrisi baza modelin öz fəzasındadır.

İşlətmə:
    python -m src.transplant.build --config configs/experiment.yaml         # bütün bazalar
    python -m src.transplant.build --config ... --bases primary
    python -m src.transplant.build --config ... --method mean --tag mean_init
    python -m src.transplant.build --config ... --k 8   --tag k8

Çıxış:
    artifacts/transplanted__<base>__<tag>/   — tam model + tokenizator (HF formatı)
    results/transplant__<base>__<tag>.json   — diaqnostika
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.tokenization.anchor_map import DEFAULT_MODE, compute_anchors
from src.tokenization.canon import align_special_tokens
from src.transplant.donor_embeddings import load_donor_embeddings
from src.transplant.embedding_norms import compute_norm_report
from src.transplant.omp import reconstruction_error, transplant_embeddings
from src.utils import (BaseModel, base_argparser, ensure_dir, get_logger,
                       load_config, resolve_bases, setup_logging,
                       transplant_dir, transplant_tag, write_json)

log = get_logger(__name__)


def build(cfg, base: BaseModel, method: str, k: int, tag: str) -> dict:
    base_name = base.hf_id
    donor_name = cfg.models.donor
    donor_revision = cfg.models.donor_revision
    full_tag = transplant_tag(base.short, tag)
    # Transplant RNG seed — separate from the model seeds and from the split
    # seed, so it is named separately (2026-09-08). Default 0 preserves the
    # previously built artifacts exactly; it is read from config so it can be
    # varied deliberately rather than by editing a function default.
    transplant_seed = int(cfg.transplant.get("seed", 0))

    log.info("Baza model yüklənir: %s", base_name)
    base_tok = AutoTokenizer.from_pretrained(base_name)
    base_model = AutoModelForMaskedLM.from_pretrained(base_name)

    # Donor MODELİ `AutoModelForMaskedLM` ilə YÜKLƏNMİR — xüsusi arxitektura
    # kodu (`trust_remote_code=True`) tələb edir. Bizə YALNIZ giriş embedding
    # matrisi lazımdır; onu safetensors-dan BİRBAŞA, arxitektura kodunu İCRA
    # ETMƏDƏN oxuyuruq (bax `donor_embeddings.py` modul qeydi).
    log.info("Donor tokenizator yüklənir: %s", donor_name)
    donor_tok = AutoTokenizer.from_pretrained(donor_name, revision=donor_revision)
    E_donor, donor_embedding_key, donor_embedding_diag = load_donor_embeddings(
        donor_name, donor_revision)

    E_base = base_model.get_input_embeddings().weight.detach().cpu().numpy()
    log.info("Embedding ölçüləri: base=%s  donor=%s", E_base.shape, E_donor.shape)

    # ---------------------------------------------------------- anchor tapılması
    # `anchors.py` (GO/NO-GO qapısı) İLƏ EYNİ funksiya və EYNİ default rejim
    # (T3) — ikisi heç vaxt uyğunsuz düşməsin deyə. Bax anchor_map.py.
    donor_vocab = donor_tok.get_vocab()
    anchor_result = compute_anchors(base_tok, donor_tok, mode=DEFAULT_MODE)
    base_scheme, donor_scheme = anchor_result.base_scheme, anchor_result.donor_scheme
    anchor_donor_ids = anchor_result.donor_ids
    anchor_base_ids = anchor_result.base_ids
    log.info("Anchor rejimi: %s (%d anchor)", anchor_result.mode, anchor_result.n_anchors)

    donor_vocab_size = len(donor_vocab)
    is_anchor = np.zeros(donor_vocab_size, dtype=bool)
    is_anchor[anchor_donor_ids] = True

    # xüsusi tokenlər ayrıca — rola görə
    special_map = align_special_tokens(base_tok, donor_tok)
    for did in special_map:
        if 0 <= did < donor_vocab_size:
            is_anchor[did] = True   # OMP-yə getməsin, birbaşa köçürülsün

    unfamiliar_ids = np.where(~is_anchor)[0]
    log.info("anchor=%d  xüsusi=%d  tanış_olmayan=%d",
             len(anchor_donor_ids), len(special_map), len(unfamiliar_ids))

    if len(anchor_donor_ids) < 1000:
        raise SystemExit(
            f"Anchor sayı çox azdır ({len(anchor_donor_ids)}). "
            "Əvvəlcə `python -m src.tokenization.anchors` işlədin və donoru dəyişin."
        )

    # ---------------------------------------------------------- yeni matris
    d_b = E_base.shape[1]
    E_new = np.zeros((donor_vocab_size, d_b), dtype=np.float32)

    # 1) anchorlar — birbaşa köçürmə
    E_new[anchor_donor_ids] = E_base[anchor_base_ids]
    # 2) xüsusi tokenlər — rol xəritəsi ilə (anchor köçürməsini üstələyir)
    for did, bid in special_map.items():
        if 0 <= did < donor_vocab_size:
            E_new[did] = E_base[bid]
    # 3) tanış olmayanlar — OMP
    diag = {}
    if len(unfamiliar_ids):
        if method == "omp":
            diag["reconstruction"] = reconstruction_error(
                E_donor, anchor_donor_ids, unfamiliar_ids,
                k=k, n_candidates=cfg.transplant.n_candidates,
            )
            log.info("Yenidənqurma (donor fəzasında): cos=%.3f  L2=%.3f",
                     diag["reconstruction"]["mean_cosine"],
                     diag["reconstruction"]["mean_l2_error"])
        E_new[unfamiliar_ids] = transplant_embeddings(
            E_donor=E_donor, E_base=E_base,
            anchor_donor_ids=anchor_donor_ids, anchor_base_ids=anchor_base_ids,
            unfamiliar_donor_ids=unfamiliar_ids,
            k=k, n_candidates=cfg.transplant.n_candidates,
            method=method, normalize=cfg.transplant.normalize,
            batch_size=cfg.transplant.batch_size,
            # 2026-09-08: passed and RECORDED rather than left to the
            # function default. `random_coef` is therefore one fixed
            # realisation of the random control, shared by all downstream
            # model seeds — five fine-tuning seeds are NOT five independent
            # random transplants, and the paper must say so. Making the seed
            # explicit is what lets that sentence be written truthfully; a
            # multi-realisation control would need one artifact per draw.
            seed=transplant_seed,
        )

    # C1a-nın YANINDA TƏLƏB OLUNAN yoxlama (T7-də tapılan miqyas-bug-undan
    # sonra, bax HANDOFF §3.16): C1a donor==baza olanda miqyas fərqini HEÇ
    # VAXT tuta bilməz, reconstruction cosine də miqyasdan asılı deyil.
    # Normların BİRBAŞA müqayisəsi bunu tutan YEGANƏ erkən yoxlamadır.
    if len(unfamiliar_ids) and len(anchor_donor_ids):
        diag["embedding_norm_report"] = compute_norm_report(
            E_base, E_donor, E_new, anchor_donor_ids, unfamiliar_ids)
        nr = diag["embedding_norm_report"]
        log.info("Norm hesabatı: anchor median=%.4f  reconstructed median=%.4f  "
                 "nisbət=%.4f  (sağlam=%s)",
                 nr["new_anchor_rows"]["median"], nr["new_reconstructed_rows"]["median"],
                 nr["reconstructed_to_anchor_median_ratio"], nr["in_healthy_range"])

    # ---------------------------------------------------------- modelə yazılması
    base_model.resize_token_embeddings(donor_vocab_size)
    with torch.no_grad():
        base_model.get_input_embeddings().weight.copy_(torch.from_numpy(E_new))
        out_emb = base_model.get_output_embeddings()
        if out_emb is not None and out_emb.weight.shape[0] == donor_vocab_size:
            # bağlı deyilsə, çıxış matrisini də eyni cür doldur
            if out_emb.weight.data_ptr() != base_model.get_input_embeddings().weight.data_ptr():
                out_emb.weight.copy_(torch.from_numpy(E_new))
    base_model.config.vocab_size = donor_vocab_size
    base_model.tie_weights()

    outdir = ensure_dir(transplant_dir(cfg, base.short, tag))
    base_model.save_pretrained(outdir)
    donor_tok.save_pretrained(outdir)
    log.info("Model saxlanıldı: %s", outdir)

    info = {
        "tag": tag, "full_tag": full_tag, "method": method, "k": k,
        "transplant_seed": int(transplant_seed),
        "transplant_seed_note": (
            "One fixed realisation per method, shared across all downstream "
            "model seeds. Variance across model seeds is fine-tuning noise, "
            "NOT transplant-draw noise."),
        "base": base.short, "base_role": base.role,
        "az_in_pretraining": base.az_in_pretraining,
        "base_model": base_name, "donor_model": donor_name,
        "donor_revision": donor_revision, "donor_embedding_diag": donor_embedding_diag,
        "base_scheme": base_scheme, "donor_scheme": donor_scheme,
        "anchor_mode": anchor_result.mode,
        "new_vocab_size": donor_vocab_size,
        "n_anchor": int(len(anchor_donor_ids)),
        "n_special_mapped": len(special_map),
        "n_unfamiliar": int(len(unfamiliar_ids)),
        "output_dir": str(outdir),
        **diag,
    }
    write_json(info, Path(cfg.experiment.results_dir) / f"transplant__{full_tag}.json")
    return info


def main() -> None:
    ap = base_argparser(__doc__)
    ap.add_argument("--method", default=None, help="omp | mean | random_coef")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--tag", default=None, help="çıxış qovluğu üçün ad")
    ap.add_argument("--bases", default=None,
                    help="vergüllə: primary,contrast (default konfiqdən)")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg = load_config(args.config, require_complete=False)
    method = args.method or cfg.transplant.method
    k = args.k if args.k is not None else cfg.transplant.k
    tag = args.tag or f"{method}_k{k}"

    roles = [r.strip() for r in args.bases.split(",")] if args.bases else None
    bases = resolve_bases(cfg, roles)

    ensure_dir(cfg.experiment.results_dir)
    for b in bases:
        log.info("=" * 70)
        log.info("BAZA MODEL: %s (%s) · AZ pretraining-də: %s",
                 b.short, b.role, "VAR" if b.az_in_pretraining else "YOX")
        build(cfg, base=b, method=method, k=k, tag=tag)


if __name__ == "__main__":
    main()
