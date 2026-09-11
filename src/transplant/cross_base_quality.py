"""
CROSS-BASE TRANSPLANT-KEYFİYYƏTİ MÜQAYİSƏSİ  —  T8-in tam run-larından ƏVVƏL

Anchor PAYI iki baza arasında fərqlidir (xlm15: 24.65% donor vocab-ının,
xlmr: 29.83% — bax `results/anchors.json`), yəni OMP xlm15-in tokenlərini
DAHA SEYREK lüğətdən yenidən qurur. Bu, Şərt 3-ün ölçülmüş effektinə
TRANSPLANT-KEYFİYYƏTİ səbəbindən təsir edə bilər — SEQMENTASİYA səbəbindən
yox. Bu skript bunu ÖLÇÜR (fərz etmir): hər iki baza üçün
`reconstruction.mean_cosine` (`results/transplant__<base>__<tag>.json`),
`C1a_identity.passed` / `C1c_delta_bpc` (`results/controls__<base>.json`), the
matching top-1 counts (`results/top1_accuracy.json`), and
embedding-norm nisbəti (`results/embedding_norm_report__<base>.json`) oxuyur
və AÇIQ müqayisə yazır.

**VACIB — VERDİKT COSINE-A GÖRƏ QƏRAR VERİLMİR (bax HANDOFF §3.16, "geri
alınmış" bölmə).** İlk versiyada `worse_reconstruction` reconstruction cosine
əsasında hesablanırdı və bu, TƏRSİNƏ nəticə verdi: cosine xlm15-i "daha pis"
elan etdi (0.662 < 0.702), amma cosine dəqiq bu skriptin özünün sənədləşdirdiyi
miqyas-bug-una KOR olan metrikadır. `C1c_delta_bpc` və norm nisbəti (heç biri
miqyasa kor DEYİL) TƏRS nəticə verir: xlmr-in transplantı POZULUB (Δ=+16.67,
nisbət=1.91), xlm15-inki YAXŞILAŞIB (Δ=-3.73, nisbət=1.13). xlmr sıfır-effekt
NƏZARƏTİDİR — pozulmuş nəzarət `compare_bases()`-i YANLIŞ MÜSBƏT
(`MECHANISM_SUPPORTED`) nəticəyə apara bilər. Ona görə **`worse_transplant`
İNDİ `C1c_delta_bpc` VƏ norm nisbətinə görə hesablanır — cosine YALNIZ
əlavə diaqnostika kimi göstərilir, verdiktə İŞTİRAK ETMİR.**

QEYD: nə tərəfə çıxarsa, anchor-payı fərqi bu NƏTİCƏNİ e'tibarsız ETMİR — o,
seqmentasiya-dan asılı olmayan, ayrıca ölçülən bir konfoznddur.

İşlətmə (hər iki `build`, `controls` və `check_embedding_norms` ARTIQ
işlədildikdən sonra):
    python -m src.transplant.cross_base_quality --config configs/experiment.yaml

Çıxış:
    results/cross_base_transplant_quality.json
"""
from __future__ import annotations

import json
from pathlib import Path

from src.utils import (base_argparser, canonical_transplant_tag, ensure_dir, get_logger,
                       load_config, resolve_bases, setup_logging, write_json)

log = get_logger(__name__)


def _read_json_or_none(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def gather_one_base(cfg, base) -> dict:
    results_dir = Path(cfg.experiment.results_dir)
    # `canonical_transplant_tag`, NOT `default_tag`: the artifact that every
    # fine-tuning run actually loads is the *rescaled* one. Reading
    # `default_tag` here made this comparison report the no-rescale build's
    # BPC and norm ratio alongside the rescaled build's top-1 counts — three
    # numbers in one verdict, drawn from two different models.
    tag = canonical_transplant_tag(cfg)
    full_tag = f"{base.short}__{tag}"

    transplant_info = _read_json_or_none(results_dir / f"transplant__{full_tag}.json")
    # 2026-09-07: prefer the METHOD-TAGGED controls file and then VERIFY what
    # was actually measured. Previously this read the base-only name and copied
    # `C1c_delta_bpc` out under `full_tag` without ever consulting the
    # provenance fields `controls.py` writes for exactly this purpose — so if
    # a placebo probe had been the last to write that file, a placebo's BPC was
    # reported as the OMP transplant's. The fields are checked now, and a
    # mismatch produces an error field instead of a plausible wrong number.
    controls_candidates = [
        results_dir / f"controls__{base.short}__{tag}.json",
        results_dir / f"controls__{base.short}.json",
    ]
    controls_path = next((p for p in controls_candidates if p.exists()), None)
    controls_info = _read_json_or_none(controls_path) if controls_path else None
    top1_all = _read_json_or_none(results_dir / "top1_accuracy.json")

    out = {"base": base.short, "base_role": base.role, "tag": full_tag}
    if transplant_info is None:
        out["error_transplant"] = f"results/transplant__{full_tag}.json tapılmadı"
    else:
        recon = transplant_info.get("reconstruction", {})
        out["n_anchor"] = transplant_info.get("n_anchor")
        out["n_unfamiliar"] = transplant_info.get("n_unfamiliar")
        out["anchor_share_pct"] = round(
            100 * transplant_info.get("n_anchor", 0) /
            max(transplant_info.get("n_anchor", 0) + transplant_info.get("n_unfamiliar", 0), 1), 2)
        out["reconstruction_mean_cosine"] = recon.get("mean_cosine")
        out["reconstruction_mean_l2_error"] = recon.get("mean_l2_error")

    if controls_info is None:
        out["error_controls"] = (
            "neither results/controls__{b}__{t}.json nor results/controls__{b}.json "
            "was found".format(b=base.short, t=tag))
    else:
        measured_tag = controls_info.get("transplanted_tag")
        out["controls_source_path"] = str(controls_path)
        out["controls_measured_tag"] = measured_tag
        out["C1a_identity_passed"] = controls_info.get("C1a_identity", {}).get("passed")
        if measured_tag is not None and measured_tag != tag:
            # Fail loud, not plausible: emit no BPC rather than the wrong one.
            out["error_controls_tag_mismatch"] = (
                f"{controls_path.name} measured '{measured_tag}', but this "
                f"report is about '{tag}'. BPC fields withheld. Re-run "
                f"src.transplant.controls with --transplanted "
                f"artifacts/transplanted__{base.short}__{tag}.")
        elif measured_tag is None and controls_info.get("C1c_delta_bpc") is not None:
            out["error_controls_tag_mismatch"] = (
                f"{controls_path.name} reports a delta BPC but records no "
                "transplanted_tag, so the measured artifact cannot be "
                "identified. BPC fields withheld.")
        else:
            out["C1c_delta_bpc"] = controls_info.get("C1c_delta_bpc")
            out["C1c_bpc_base"] = controls_info.get("C1c_bpc_base", {}).get("bits_per_character")
            out["C1c_bpc_transplanted"] = controls_info.get("C1c_bpc_transplanted", {}).get("bits_per_character")
            # BPC alone is not evidence of usable MLM recovery: a smoother
            # output distribution lowers cross-entropy without adding any
            # lexical knowledge. Carry the top-1 counts from the SAME masked
            # positions alongside it so the two are never separated in a table.
            transplanted_bpc = controls_info.get("C1c_bpc_transplanted") or {}
            out["C1c_top1_transplanted"] = {
                "correct": transplanted_bpc.get("top1_n_correct"),
                "masked": transplanted_bpc.get("top1_n_masked"),
            }
            base_bpc = controls_info.get("C1c_bpc_base") or {}
            out["C1c_top1_base"] = {
                "correct": base_bpc.get("top1_n_correct"),
                "masked": base_bpc.get("top1_n_masked"),
            }
            out["C1c_interpretation_note"] = (
                "A lower (more negative) delta BPC with a near-zero top-1 "
                "count is a CALIBRATION effect, not recovered lexical "
                "knowledge. Report delta BPC and top-1 together, always.")

    if top1_all is None or base.short not in top1_all:
        out["error_top1"] = "results/top1_accuracy.json missing or base absent"
    else:
        top1 = top1_all[base.short]
        base_top1 = top1.get("base") or {}
        transplanted_top1 = top1.get("transplanted") or {}
        out["top1_base"] = {
            "correct": base_top1.get("n_correct"),
            "masked": base_top1.get("n_masked_tokens"),
            "accuracy": base_top1.get("top1_accuracy"),
        }
        out["top1_transplanted_canonical"] = {
            "correct": transplanted_top1.get("n_correct"),
            "masked": transplanted_top1.get("n_masked_tokens"),
            "accuracy": transplanted_top1.get("top1_accuracy"),
        }

    norm_info = _read_json_or_none(results_dir / f"embedding_norm_report__{base.short}.json")
    if norm_info is None:
        out["error_norm_report"] = f"results/embedding_norm_report__{base.short}.json tapılmadı"
    else:
        out["norm_ratio"] = norm_info.get("reconstructed_to_anchor_median_ratio")
        out["norm_ratio_healthy"] = norm_info.get("in_healthy_range")

    return out


def main() -> None:
    ap = base_argparser(__doc__)
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config, require_complete=False)

    bases = resolve_bases(cfg)
    per_base = {b.short: gather_one_base(cfg, b) for b in bases}

    out = {"per_base": per_base}

    shorts = list(per_base.keys())
    if len(shorts) == 2:
        a, c = per_base[shorts[0]], per_base[shorts[1]]
        cos_a, cos_c = a.get("reconstruction_mean_cosine"), c.get("reconstruction_mean_cosine")
        bpc_a, bpc_c = a.get("C1c_delta_bpc"), c.get("C1c_delta_bpc")
        ratio_a, ratio_c = a.get("norm_ratio"), c.get("norm_ratio")
        comparison = {
            "anchor_share_pct": {shorts[0]: a.get("anchor_share_pct"), shorts[1]: c.get("anchor_share_pct")},
            # Reported for completeness ONLY — NOT used to decide `worse_transplant` (see
            # module docstring: cosine is blind to the cross-space scale bug class, C1c/norm
            # ratio are not; deciding on cosine inverted this exact comparison once already).
            "reconstruction_mean_cosine": {shorts[0]: cos_a, shorts[1]: cos_c},
            "C1c_delta_bpc": {shorts[0]: bpc_a, shorts[1]: bpc_c},
            "top1_base_to_canonical_transplanted": {
                shorts[0]: {"base": a.get("top1_base"),
                            "transplanted": a.get("top1_transplanted_canonical")},
                shorts[1]: {"base": c.get("top1_base"),
                            "transplanted": c.get("top1_transplanted_canonical")},
            },
            "norm_ratio": {shorts[0]: ratio_a, shorts[1]: ratio_c},
        }
        if cos_a is not None and cos_c is not None:
            comparison["cosine_gap"] = round(cos_a - cos_c, 4)

        # ---- AUTHORITATIVE verdict: BPC delta (primary) + norm-ratio health (corroborating).
        # Higher (more positive) C1c_delta_bpc = more degraded transplant. Never decided on cosine.
        if bpc_a is not None and bpc_c is not None:
            worse = shorts[0] if bpc_a > bpc_c else (shorts[1] if bpc_c > bpc_a else None)
            comparison["worse_transplant"] = worse
            healthy_a = a.get("norm_ratio_healthy")
            healthy_c = c.get("norm_ratio_healthy")
            corroborated = (worse == shorts[0] and healthy_a is False and healthy_c is not False) or \
                           (worse == shorts[1] and healthy_c is False and healthy_a is not False)
            comparison["note"] = (
                f"{worse} has the worse (more positive) C1c_delta_bpc "
                f"({shorts[0]}={bpc_a}, {shorts[1]}={bpc_c})"
                + (f" and the unhealthy norm ratio ({shorts[0]}={ratio_a}, {shorts[1]}={ratio_c}, "
                   "corroborating the same conclusion via an independent, non-scale-blind metric)"
                   if corroborated else
                   f" (norm ratios: {shorts[0]}={ratio_a}, {shorts[1]}={ratio_c})")
                + ". Decided on BPC/norm ratio, NOT on reconstruction cosine — cosine is blind to "
                "cross-space scale problems by construction and inverted this exact comparison "
                "once already. BPC must be reported with the top-1 counts above; zero top-1 means "
                "a lower BPC is calibration improvement, not restored usable MLM (HANDOFF §3.16)."
                if worse else
                "Transplant quality (BPC delta, norm ratio) is comparable across bases."
            )
        out["comparison"] = comparison
        log.info("=" * 60)
        log.info("Anchor share: %s=%.2f%%  %s=%.2f%%", shorts[0], a.get("anchor_share_pct") or -1,
                 shorts[1], c.get("anchor_share_pct") or -1)
        log.info("C1c delta BPC: %s=%s  %s=%s", shorts[0], bpc_a, shorts[1], bpc_c)
        log.info("Top-1 base -> canonical transplant: %s=%s -> %s  %s=%s -> %s",
                 shorts[0], a.get("top1_base"), a.get("top1_transplanted_canonical"),
                 shorts[1], c.get("top1_base"), c.get("top1_transplanted_canonical"))
        log.info("Norm ratio: %s=%s  %s=%s", shorts[0], ratio_a, shorts[1], ratio_c)
        log.info("Reconstruction cosine (diagnostic only, NOT the verdict): %s=%s  %s=%s",
                 shorts[0], cos_a, shorts[1], cos_c)
        if "note" in comparison:
            log.info("QEYD: %s", comparison["note"])

    path = ensure_dir(cfg.experiment.results_dir) / "cross_base_transplant_quality.json"
    write_json(out, path)
    log.info("Yazıldı: %s", path)


if __name__ == "__main__":
    main()
