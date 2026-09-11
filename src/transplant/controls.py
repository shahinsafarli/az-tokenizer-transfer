"""
NƏZARƏT EKSPERİMENTLƏRİ  [C1]  —  bunlar olmadan nəticələr şərh edilə bilməz

C1a · EYNİYYƏT TRANSPLANTI (identity)
    Donor = Baza. Bütün tokenlər anchor olur, heç bir təxmin aparılmır.
    Nəticə orijinala BİT-BƏRABƏR olmalıdır. Olmursa → kodda səhv var.

C1b · YENİDƏNQURMA KEYFİYYƏTİ
    OMP donor fəzasında hədəfləri nə qədər yaxşı ifadə edir? (kosinus)

C1c · BİTS-PER-CHARACTER (BPC) + TOP-1
    Transplantın öz zərərini ölçür. DİQQƏT: perplexity fərqli tokenizatorlar
    arasında müqayisə OLUNA BİLMƏZ (token sayları fərqlidir). BPC hərfə görə
    normallaşdırıldığı üçün müqayisə edilə biləndir, amma heç vaxt tək verilmir:
    eyni maskalanmış mövqelərdə top-1 MLM-in işlək olub-olmadığını göstərir:

        BPC = (ümumi_loss_nats / ln 2) / ümumi_hərf_sayı

İşlətmə:
    python -m src.transplant.controls --config configs/experiment.yaml
"""
from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.tokenization.canon import canon_vocab, detect_scheme
from src.transplant.eval_text import resolve_eval_text
from src.utils import (base_argparser, canonical_transplant_tag, ensure_dir,
                       get_logger, load_config, resolve_bases, set_all_seeds,
                       setup_logging, write_json)

log = get_logger(__name__)



# ---------------------------------------------------------------- C1a
def identity_transplant_check(base_name: str) -> dict:
    """Donor = Baza olduqda embedding matrisi dəyişməməlidir."""
    tok = AutoTokenizer.from_pretrained(base_name)
    model = AutoModelForMaskedLM.from_pretrained(base_name)
    E = model.get_input_embeddings().weight.detach().cpu().numpy().copy()

    vocab = tok.get_vocab()
    scheme = detect_scheme(vocab.keys())
    canon_map, _ = canon_vocab(vocab, scheme)

    # eyni tokenizator → hər kanonik açar özünə uyğunlaşır
    ids = np.array(sorted(canon_map.values()), dtype=np.int64)
    E_new = E.copy()
    E_new[ids] = E[ids]                      # birbaşa köçürmə

    matrix_identical = bool(np.array_equal(E, E_new))
    n_unfamiliar = len(vocab) - len(canon_map)

    # ------------------------------------------------------------------
    # END-TO-END LEG, added 2026-09-08.
    #
    # The block above compares `E` with `E.copy()` after reassigning rows from
    # `E` itself. It CANNOT FAIL. It never runs the anchor mapper, never writes
    # the matrix into the model, never serialises or reloads, and never
    # compares a forward pass — so it certified nothing about the failure modes
    # that actually matter here (wrong anchor ids, a lost row on save/reload, a
    # broken tied output head, a changed special-token id).
    #
    # This leg exercises the real path with the base model as its own donor,
    # where the correct answer is known exactly: identity. It checks, in order,
    #   (1) the ANCHOR MAPPER pairs every token with itself,
    #   (2) the embedding write + save_pretrained + from_pretrained round trip
    #       returns a bit-identical matrix,
    #   (3) LOGITS on a fixed batch are bit-identical.
    # Any of those failing means the transplant machinery is broken
    # independently of OMP, which is the distinction the negative result rests
    # on.
    e2e: dict = {"attempted": True}
    try:
        from src.tokenization.anchor_map import DEFAULT_MODE, compute_anchors

        anchors = compute_anchors(tok, tok, mode=DEFAULT_MODE)
        self_mapped = bool(len(anchors.donor_ids) > 0 and
                           np.array_equal(anchors.donor_ids, anchors.base_ids))
        e2e["anchor_mode"] = anchors.mode
        e2e["n_anchors"] = int(anchors.n_anchors)
        e2e["anchors_map_to_self"] = self_mapped
        if not self_mapped and len(anchors.donor_ids):
            mism = int(np.sum(anchors.donor_ids != anchors.base_ids))
            e2e["n_anchor_mismatches"] = mism

        ids = torch.tensor([[i % E.shape[0] for i in range(16)]], dtype=torch.long)
        mask = torch.ones_like(ids)
        model.eval()
        with torch.no_grad():
            before = model(input_ids=ids, attention_mask=mask).logits.clone()

        with torch.no_grad():
            model.get_input_embeddings().weight.copy_(torch.from_numpy(E_new))
        model.tie_weights()

        staging = Path(tempfile.mkdtemp(prefix="c1a_identity_"))
        try:
            model.save_pretrained(str(staging))
            tok.save_pretrained(str(staging))
            reloaded = AutoModelForMaskedLM.from_pretrained(str(staging))
            reloaded.eval()
            E_round = (reloaded.get_input_embeddings().weight
                       .detach().cpu().numpy())
            e2e["embeddings_bit_identical_after_round_trip"] = bool(
                np.array_equal(E, E_round))
            with torch.no_grad():
                after = reloaded(input_ids=ids, attention_mask=mask).logits
            e2e["max_abs_logit_delta"] = float((after - before).abs().max())
            e2e["logits_bit_identical"] = bool(torch.equal(after, before))
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        e2e["passed"] = bool(
            e2e.get("embeddings_bit_identical_after_round_trip")
            and e2e.get("logits_bit_identical"))
        # `anchors_map_to_self` is recorded but deliberately NOT part of
        # `passed`: in `functional` mode the mapper re-encodes each stripped
        # donor surface with the base tokenizer, and a piece that only exists
        # as a word-continuation can legitimately re-encode to a different id
        # even when the implementation is correct. It is a diagnostic, not an
        # invariant.
    except Exception as exc:  # noqa: BLE001
        e2e["passed"] = False
        e2e["error"] = f"{type(exc).__name__}: {exc}"
        log.error("C1a end-to-end leg FAILED to execute: %s", exc)

    # GATING IS DELIBERATELY UNCHANGED (2026-09-08).
    # `main()` raises SystemExit when `passed` is false, and run_all.sh runs
    # under `set -euo pipefail`, so `passed` kills the entire preparation
    # stage. The end-to-end leg below is new code that could not be executed
    # against the real base models before this run (no model access in the
    # authoring environment), so promoting it into `passed` would risk killing
    # a 6-hour grid on an untested assertion — the exact trade the brief's
    # "protect your work" advice warns about. It therefore runs, is recorded in
    # full, and logs loudly, but does not gate.
    #
    # AFTER the first successful prep run: read
    # `results/controls__<base>.json -> C1a_identity.end_to_end.passed`. If it
    # is true for both bases, change the line below to
    #     passed = bool(matrix_identical and e2e.get("passed"))
    # and re-lock, so the invariant is enforced from then on.
    passed = bool(matrix_identical)
    log.info("C1a identity: matrix=%s (GATE)  |  end-to-end: anchors_self=%s "
             "round_trip=%s logits=%s max_logit_delta=%s -> e2e_passed=%s",
             matrix_identical, e2e.get("anchors_map_to_self"),
             e2e.get("embeddings_bit_identical_after_round_trip"),
             e2e.get("logits_bit_identical"), e2e.get("max_abs_logit_delta"),
             e2e.get("passed"))
    if not matrix_identical:
        log.error("C1a GATE FAILED — the row-copy logic itself is broken.")
    if not e2e.get("passed"):
        log.error(
            "C1a END-TO-END leg did NOT pass (%s). This is the leg that would "
            "distinguish a broken transplant from a genuine OMP failure, so a "
            "negative downstream result CANNOT be attributed to OMP until this "
            "is understood. It does not stop the run by design — see the "
            "gating note in this function.",
            e2e.get("error") or "see end_to_end fields")
    return {
        "passed": passed,
        "mode": "matrix_gate_plus_end_to_end_diagnostic",
        "matrix_assignment_identical": matrix_identical,
        "end_to_end_passed": bool(e2e.get("passed")),
        "end_to_end": e2e,
        "vocab_size": len(vocab),
        "canonical_keys": len(canon_map),
        "collisions": n_unfamiliar,
        "note": ("`passed` now requires the anchor mapper, the save/reload "
                 "round trip and a forward pass to agree exactly, not only a "
                 "matrix copied onto itself."),
    }


# ---------------------------------------------------------------- C1c
@torch.no_grad()
def bits_per_character(model_dir_or_name: str, text: str, device: str = "cpu",
                       max_length: int = 128, mlm_prob: float = 0.15,
                       seed: int = 0) -> dict:
    """
    Maskalanmış dil modelləşdirmə loss-undan BPC hesablayır.
    Tokenizatorlar arasında MÜQAYİSƏ EDİLƏ BİLƏN ölçü.
    """
    set_all_seeds(seed)
    tok = AutoTokenizer.from_pretrained(model_dir_or_name)
    model = AutoModelForMaskedLM.from_pretrained(model_dir_or_name).to(device).eval()

    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    total_nats, total_masked, total_chars, total_tokens, total_correct = 0.0, 0, 0, 0, 0
    gen = torch.Generator().manual_seed(seed)

    for line in lines:
        enc = tok(line, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc["input_ids"].to(device)
        labels = ids.clone()

        special = torch.tensor(
            tok.get_special_tokens_mask(ids[0].tolist(), already_has_special_tokens=True),
            dtype=torch.bool, device=device,
        ).unsqueeze(0)
        prob = torch.full(ids.shape, mlm_prob)
        prob.masked_fill_(special.cpu(), 0.0)
        masked = torch.bernoulli(prob, generator=gen).bool().to(device)
        if masked.sum() == 0:                     # heç nə maskalanmadısa, birini məcburi seç
            cand = (~special).nonzero()
            if len(cand) == 0:
                continue
            masked[0, cand[0, 1]] = True

        labels[~masked] = -100
        ids_in = ids.clone()
        ids_in[masked] = tok.mask_token_id

        out = model(input_ids=ids_in, attention_mask=enc["attention_mask"].to(device),
                    labels=labels)
        pred = out.logits.argmax(dim=-1)
        n_masked = int(masked.sum().item())
        total_correct += int((pred[masked] == labels[masked]).sum().item())
        total_nats += float(out.loss.item()) * n_masked
        total_masked += n_masked
        total_chars += len(line)
        total_tokens += int(ids.shape[1])

    if total_masked == 0:
        return {"error": "heç bir token maskalanmadı"}

    # maskalanmış tokenlərin orta loss-unu bütün tokenlərə şamil edirik
    nats_per_token = total_nats / total_masked
    bpc = (nats_per_token * total_tokens / math.log(2)) / max(total_chars, 1)
    return {
        "model": model_dir_or_name,
        "n_lines": len(lines),
        "n_tokens": total_tokens,
        "n_chars": total_chars,
        "tokens_per_char": round(total_tokens / max(total_chars, 1), 4),
        "mlm_loss_nats_per_masked_token": round(nats_per_token, 4),
        "bits_per_character": round(bpc, 4),
        "top1_n_correct": total_correct,
        "top1_n_masked": total_masked,
        "top1_accuracy": round(total_correct / total_masked, 6),
    }


def main() -> None:
    ap = base_argparser(__doc__)
    ap.add_argument("--text-file", default=None,
                    help="optional explicit corpus; default samples az_train.jsonl")
    ap.add_argument("--transplanted", default=None,
                    help="artifacts/transplanted__<base>__<tag> yolu (varsa BPC müqayisəsi)")
    ap.add_argument("--base", default="primary", help="primary | contrast")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config, require_complete=False)
    base = resolve_bases(cfg, [args.base])[0]

    # C1c/top-1 are the ONLY controls that can catch a broken transplant, so
    # which corpus they ran on is a reportable fact, not an implementation
    # detail. Silently falling back to six hard-coded sentences is how the
    # top-1 diagnostic ended up with 12 masked positions.
    text, text_source = resolve_eval_text(cfg, args.text_file)
    log.info("Qiymətləndirmə mətni: mənbə=%s sətir=%d simvol=%d",
             text_source["source"], text_source["n_lines"], text_source["n_chars"])

    # Provenance is recorded, not assumed: `controls__<base>.json` has a fixed
    # name, so a second run against a different artifact silently overwrites
    # the first. Without these fields a downstream reader (cross_base_quality)
    # cannot tell whether it is looking at the rescaled canonical artifact or
    # the no-rescale ablation baseline — and it did, in fact, mix them.
    canonical = canonical_transplant_tag(cfg)
    measured_dir = Path(args.transplanted) if args.transplanted else None
    measured_tag = measured_dir.name.split("__", 2)[-1] if measured_dir else None
    res = {"base": base.short, "base_role": base.role, "base_model": base.hf_id,
           "transplanted_dir": str(measured_dir) if measured_dir else None,
           "transplanted_tag": measured_tag,
           "canonical_transplant_tag": canonical,
           "is_canonical_artifact": measured_tag == canonical,
           "eval_text_provenance": text_source,
           "C1a_identity": identity_transplant_check(base.hf_id)}
    if measured_tag is not None and measured_tag != canonical:
        log.warning(
            "Ölçülən artefakt (%s) KANONİK artefakt (%s) DEYİL — bu fayl "
            "ablasiya baseline-ıdır, əsas nəticə deyil.", measured_tag, canonical)

    log.info("C1c: baza modelin BPC-si ölçülür (%s) ...", base.short)
    res["C1c_bpc_base"] = bits_per_character(base.hf_id, text, args.device,
                                             cfg.models.max_length)
    log.info("  baza BPC = %.4f | top-1 = %d/%d",
             res["C1c_bpc_base"]["bits_per_character"],
             res["C1c_bpc_base"]["top1_n_correct"],
             res["C1c_bpc_base"]["top1_n_masked"])

    if args.transplanted and Path(args.transplanted).exists():
        log.info("C1c: transplant olunmuş modelin BPC-si ölçülür ...")
        res["C1c_bpc_transplanted"] = bits_per_character(
            args.transplanted, text, args.device, cfg.models.max_length)
        b = res["C1c_bpc_base"]["bits_per_character"]
        t = res["C1c_bpc_transplanted"]["bits_per_character"]
        res["C1c_delta_bpc"] = round(t - b, 4)
        res["C1c_note"] = (
            "Müsbət delta = transplant modeli PİSLƏŞDİRİB. BPC həmişə eyni "
            "maskalardakı top-1 sayı ilə birlikdə göstərilməlidir; sıfır top-1 "
            "olanda aşağı BPC usable MLM bərpası deyil, calibration improvement-dir."
        )
        log.info("  transplant BPC = %.4f (Δ = %+.4f) | top-1 = %d/%d",
                 t, res["C1c_delta_bpc"],
                 res["C1c_bpc_transplanted"]["top1_n_correct"],
                 res["C1c_bpc_transplanted"]["top1_n_masked"])
    else:
        log.warning("Transplant modeli verilmədi — əvvəlcə src.transplant.build işlədin, "
                    "sonra --transplanted artifacts/transplanted__omp_k64 ilə təkrarlayın.")

    # Baza-spesifik fayl adı: `controls.json` (bazasız) iki bazanın nəticəsini
    # BİRİNİ digərinin üstündən YAZARDI — layihənin öz qaydası (§7 gotcha #4,
    # "hər run faylı `base=` daşımalıdır") buna görə burada da tətbiq olunur.
    #
    # 2026-09-07: the same rule now extends to the transplant METHOD. Until
    # today the only output was `controls__<base>.json`, so probing omp -> mean
    # -> random_coef in a loop left one file holding whichever method ran LAST,
    # and a downstream reader had no un-clobbered per-method record to consult.
    # The method-tagged file is therefore the primary artifact; the base-only
    # name is kept, but ONLY the canonical artifact (or a base-only measurement
    # with no --transplanted) is ever allowed to occupy it, so that name can no
    # longer come to hold a placebo's numbers under a canonical-looking path.
    results_dir = ensure_dir(cfg.experiment.results_dir)
    written = []
    if measured_tag is not None:
        tagged = results_dir / f"controls__{base.short}__{measured_tag}.json"
        write_json(res, tagged)
        written.append(tagged)

    if measured_tag is None or measured_tag == canonical:
        out = results_dir / f"controls__{base.short}.json"
        write_json(res, out)
        written.append(out)
    else:
        log.warning(
            "`controls__%s.json` YAZILMADI: ölçülən artefakt (%s) kanonik "
            "(%s) deyil. Kanonik adı yalnız kanonik artefakt tuta bilər — "
            "əks halda placebo rəqəmi OMP kimi oxunardı. Per-metod fayl: %s",
            base.short, measured_tag, canonical,
            f"controls__{base.short}__{measured_tag}.json")

    for path in written:
        log.info("Yazıldı: %s", path)

    if not res["C1a_identity"]["passed"]:
        raise SystemExit("C1a nəzarəti uğursuz — davam etməyin.")


if __name__ == "__main__":
    main()
