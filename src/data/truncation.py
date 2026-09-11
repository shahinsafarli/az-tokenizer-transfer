"""
TRUNCATION KONFOUNDU (T5/T4)  —  `max_length` seçiminin ÖLÇÜLMÜŞ əsası

Problem: Şərt 3/4 donor tokenizatorunu işlədir, o isə Azərbaycan mətnini
DAHA AZ tokenə bölür. Sabit `max_length` altında bu o deməkdir ki, orijinal
tokenizatorlu şərtlər (1/2/5) DAHA ÇOX kəsilir — yəni onlar sadəcə mətnin
DAHA AZINI görür. Ölçülən fərq seqmentasiya keyfiyyətindən yox, GÖRÜLƏN
MƏTN MİQDARINDAN gələ bilər. Bu, konfounddur və ölçülməlidir.

VACİB: `finetune.py` `DataCollatorWithPadding` işlədir (`padding="max_length"`
HEÇ YERDƏ yoxdur) — batch-lar həmin batch-dakı ƏN UZUN nümunəyə görə
dinamik doldurulur. Ona görə `max_length` YALNIZ kəsmə tavanıdır, per-batch
hesablama xərcini TƏYİN ETMİR. 128→256 qaldırmaq real xərci yalnız uzun
nümunə olan batch-larda artırır.

Bu skript `artifacts/data/az_{train,val,test}.jsonl` fayllarını — yəni
`splits.py`-ın YAZDIĞI, təlimin FAKTİKİ gördüyü mətni — BİRBAŞA oxuyur.
Dedup/exclusion məntiqini müstəqil təkrarlamır, ona görə bu ölçmə təlimin
gördüyündən səssizcə sürüşə bilməz.

İşlətmə (`src.data.splits`-dən SONRA):
    python -m src.data.truncation --config configs/experiment.yaml

Çıxış:
    results/truncation.json
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from src.data.splits import read_jsonl
from src.utils import (base_argparser, canonical_transplant_tag, ensure_dir,
                       get_logger, load_config, resolve_bases, setup_logging,
                       transplant_dir, write_json)

log = get_logger(__name__)

MEASURED_MAX_LENGTHS = (128, 256)


def _token_lengths(tokenizer, texts: list[str], batch: int = 512) -> np.ndarray:
    """Hər mətnin xüsusi tokenlərlə birlikdə tam (kəsilməmiş) token sayı."""
    out: list[int] = []
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        enc = tokenizer(chunk, truncation=False, padding=False)
        out.extend(len(ids) for ids in enc["input_ids"])
    return np.asarray(out, dtype=np.int64)


def _length_stats(lengths: np.ndarray) -> dict:
    return {
        "tokens_mean": round(float(lengths.mean()), 3),
        "tokens_median": float(np.median(lengths)),
        "tokens_p90": float(np.percentile(lengths, 90)),
        "tokens_p99": float(np.percentile(lengths, 99)),
        "tokens_max": int(lengths.max()),
    }


def main() -> None:
    ap = base_argparser(__doc__)
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    datadir = Path(cfg.experiment.artifacts_dir) / "data"
    texts: list[str] = []
    for split in ("az_train", "az_val", "az_test"):
        path = datadir / f"{split}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} tapılmadı — əvvəlcə `python -m src.data.splits` işlədin.")
        texts.extend(r["text"] for r in read_jsonl(path))
    log.info("Ölçülən korpus: %d nümunə (train+val+test, splits.py-ın yazdığı)", len(texts))

    bases = {b.role: b for b in resolve_bases(cfg)}
    primary = bases.get("primary")
    contrast = bases.get("contrast")

    # Şərt 3/4-ün faktiki işlətdiyi tokenizator transplant artefaktındadır;
    # o hələ qurulmayıbsa donor modelin öz tokenizatoru eyni nəticəni verir
    # (transplant tokenizatoru DONORUNKUDUR — çəkilər dəyişir, bölmə yox).
    donor_tag = canonical_transplant_tag(cfg)
    donor_dir = transplant_dir(cfg, primary.short, donor_tag) if primary else None
    donor_source = (str(donor_dir) if donor_dir and donor_dir.exists()
                    else cfg.models.donor)

    setups: dict[str, str] = {}
    if primary:
        setups["base_original_xlm15"] = primary.hf_id
    setups["donor_transplanted"] = donor_source
    if contrast:
        setups["contrast_original_xlmr"] = contrast.hf_id

    lengths: dict[str, np.ndarray] = {}
    for name, model in setups.items():
        log.info("Tokenizator yüklənir (%s): %s", name, model)
        kwargs = {"revision": cfg.models.donor_revision} if model == cfg.models.donor else {}
        tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        lengths[name] = _token_lengths(tokenizer, texts)
        log.info("  orta=%.1f  median=%.0f  p99=%.0f  max=%d",
                 lengths[name].mean(), np.median(lengths[name]),
                 np.percentile(lengths[name], 99), lengths[name].max())

    by_max_length: dict[str, dict] = {}
    for cap in MEASURED_MAX_LENGTHS:
        per_setup = {}
        for name, arr in lengths.items():
            n_trunc = int((arr > cap).sum())
            per_setup[name] = {
                "max_length": cap,
                "n": int(len(arr)),
                "truncation_rate_pct": round(100.0 * n_trunc / max(len(arr), 1), 3),
                "n_truncated": n_trunc,
                "model": setups[name],
            }
        confound = {}
        if "base_original_xlm15" in per_setup and "donor_transplanted" in per_setup:
            confound["xlm15_original_vs_donor_truncation_delta_pct"] = round(
                per_setup["base_original_xlm15"]["truncation_rate_pct"] -
                per_setup["donor_transplanted"]["truncation_rate_pct"], 3)
        if "contrast_original_xlmr" in per_setup:
            confound["xlmr_original_truncation_pct"] = \
                per_setup["contrast_original_xlmr"]["truncation_rate_pct"]
        by_max_length[str(cap)] = {"setups": per_setup, "confound_note": confound}

    report = {
        "dataset": cfg.data.az.get("hf_name") or cfg.data.az.get("csv_path"),
        "exclude_labels": list(cfg.data.az.get("exclude_labels") or []),
        "n_examples": len(texts),
        "max_lengths_measured": list(MEASURED_MAX_LENGTHS),
        "configured_max_length": int(cfg.models.max_length),
        "by_max_length": by_max_length,
        "length_distribution": {name: _length_stats(arr) for name, arr in lengths.items()},
        "note": ("Konfound ölçüsü, `xlm15_original_vs_donor_truncation_delta_pct`, "
                 "1pp-dən aşağı olmalıdır; əks halda şərtlər arasındakı fərq "
                 "GÖRÜLƏN MƏTN MİQDARI ilə qarışır."),
    }
    path = ensure_dir(cfg.experiment.results_dir) / "truncation.json"
    write_json(report, path)

    configured = str(int(cfg.models.max_length))
    if configured in by_max_length:
        gap = by_max_length[configured]["confound_note"].get(
            "xlm15_original_vs_donor_truncation_delta_pct")
        log.info("max_length=%s → şərtlərarası kəsmə fərqi = %s pp", configured, gap)
        if gap is not None and gap > 1.0:
            log.warning("Fərq 1pp-dən böyükdür — məqalədə MƏHDUDİYYƏT kimi qeyd edin.")
    log.info("Yazıldı: %s", path)


if __name__ == "__main__":
    main()
