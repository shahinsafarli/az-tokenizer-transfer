"""
TOP-1 MASKED-TOKEN DƏQİQLİYİ  —  BPC-yə əlavə, ucuz diaqnostika

Niyə lazımdır: BPC mütləq şkalada oxunması çətin ədəddir. Random təxmin
(donor lüğəti 32,770, ~3.4 simvol/token) ~4.4 BPC-yə uyğun gəlir — transplant
olunmuş modellərin BPC-si buna YAXIN olanda sual budur: model PROQNOZ EDİR,
sadəcə pis, yoxsa TAMAMILƏ DAĞILIB (təsadüfi təxmindən fərqlənmir)? Top-1
dəqiqlik bunu birbaşa göstərir: 20–40% aralığı "zədələnib amma işləyir"
(fine-tuning düzəldə bilər), ~1/32770 (≈0.003%) isə "dağılıb" deməkdir.

MÜQAYİSƏNİN HƏQİQİ HÜDUDU (əvvəlki qeyd bunu SƏHV ifadə edirdi). Eyni seed,
eyni `mlm_prob` və eyni MƏTN işlədilir, ona görə bu skript ilə
`controls.bits_per_character` EYNİ modeldə EYNİ maskalanmış mövqeləri ölçür —
bu cütlük etibarlıdır. AMMA fərqli TOKENİZATORLU iki model (baza vs
transplant) eyni mətni fərqli sayda tokenə bölür, ona görə onların
maskalanmış mövqeləri EYNİ OLA BİLMƏZ və sayları da fərqlənir (ölçülmüş:
xlm15 baza 37, transplant 12). `n_masked_tokens` MƏHZ buna görə hər nəticədə
saxlanılır: modellər arası top-1 fərqi HƏMİŞƏ öz məxrəci ilə oxunmalıdır,
kiçik məxrəcdə sıfır nəticə "dağılıb" SÜBUTU deyil.

İşlətmə:
    python -m src.transplant.top1_accuracy --config configs/experiment.yaml
"""
from __future__ import annotations


import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.transplant.eval_text import resolve_eval_text
from src.utils import (base_argparser, canonical_transplant_tag, ensure_dir, get_logger,
                       load_config, resolve_bases, set_all_seeds, setup_logging,
                       transplant_dir, write_json)

log = get_logger(__name__)


@torch.no_grad()
def top1_masked_accuracy(model_dir_or_name: str, text: str, device: str = "cpu",
                         max_length: int = 128, mlm_prob: float = 0.15,
                         seed: int = 0) -> dict:
    """
    `controls.bits_per_character`-in EYNİ maskalama prosedurunun bir-başa
    kopyası (məqsədli — eyni maskalanmış tokenlər üzərində müqayisə üçün),
    amma loss əvəzinə top-1 dəqiqliyi (və təsadüfi-təxmin bazası) qaytarır.
    """
    set_all_seeds(seed)
    tok = AutoTokenizer.from_pretrained(model_dir_or_name)
    model = AutoModelForMaskedLM.from_pretrained(model_dir_or_name).to(device).eval()
    vocab_size = model.get_output_embeddings().weight.shape[0]

    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    n_correct, n_masked_total = 0, 0
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
        if masked.sum() == 0:
            cand = (~special).nonzero()
            if len(cand) == 0:
                continue
            masked[0, cand[0, 1]] = True

        labels[~masked] = -100
        ids_in = ids.clone()
        ids_in[masked] = tok.mask_token_id

        out = model(input_ids=ids_in, attention_mask=enc["attention_mask"].to(device))
        pred = out.logits.argmax(dim=-1)
        n_correct += int((pred[masked] == labels[masked]).sum().item())
        n_masked_total += int(masked.sum().item())

    if n_masked_total == 0:
        return {"error": "heç bir token maskalanmadı"}

    acc = n_correct / n_masked_total
    return {
        "model": model_dir_or_name,
        "n_masked_tokens": n_masked_total,
        "n_correct": n_correct,
        "top1_accuracy": round(acc, 6),
        "vocab_size": int(vocab_size),
        "random_guess_accuracy": round(1.0 / vocab_size, 8),
        "accuracy_over_random": round(acc / (1.0 / vocab_size), 2),
    }


def main() -> None:
    ap = base_argparser(__doc__)
    ap.add_argument("--text-file", default=None,
                    help="optional explicit corpus; default samples az_train.jsonl")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config, require_complete=False)

    # EYNİ resolver, EYNİ default → BPC və top-1 eyni korpusdan gəlir.
    text, text_source = resolve_eval_text(cfg, args.text_file)
    log.info("Qiymətləndirmə mətni: mənbə=%s sətir=%d simvol=%d",
             text_source["source"], text_source["n_lines"], text_source["n_chars"])
    variant_tag = canonical_transplant_tag(cfg)

    results = {"eval_text_provenance": text_source}
    for base in resolve_bases(cfg):
        log.info("[%s] baza model top-1 dəqiqliyi ölçülür ...", base.short)
        base_res = top1_masked_accuracy(base.hf_id, text, args.device, cfg.models.max_length)
        log.info("  baza: acc=%.4f  (%.1fx random)", base_res["top1_accuracy"],
                 base_res["accuracy_over_random"])

        outdir = transplant_dir(cfg, base.short, variant_tag)
        transplant_res = None
        if outdir.exists():
            log.info("[%s] transplant (%s) top-1 dəqiqliyi ölçülür ...", base.short, variant_tag)
            transplant_res = top1_masked_accuracy(str(outdir), text, args.device, cfg.models.max_length)
            log.info("  transplant: acc=%.4f  (%.1fx random)", transplant_res["top1_accuracy"],
                     transplant_res["accuracy_over_random"])
        else:
            log.warning("[%s] %s tapılmadı — transplant ölçülmədi", base.short, outdir)

        results[base.short] = {"base_role": base.role, "base": base_res, "transplanted": transplant_res}

    path = ensure_dir(cfg.experiment.results_dir) / "top1_accuracy.json"
    write_json(results, path)
    log.info("Yazıldı: %s", path)


if __name__ == "__main__":
    main()
