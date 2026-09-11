"""
C1c / top-1 DİAQNOSTİKASI ÜÇÜN QİYMƏTLƏNDİRMƏ MƏTNİ  —  TƏK mənbə

Niyə ayrıca modul: BPC (`controls.py`) və top-1 (`top1_accuracy.py`) YALNIZ
EYNİ mətn üzərində ölçüldükdə birlikdə oxuna bilər. İki modul mətni ayrı-ayrı
həll etsəydi (biri fayldan, digəri fallback-dan), "BPC düşdü amma top-1
sıfırdır" kimi bir nəticə iki FƏRQLİ korpusun artefaktı ola bilərdi.

Niyə lazım oldu: `--text-file` default-u `resources/az_eval_text.txt` idi,
amma bu fayl repoda HEÇ VAXT olmayıb. Ona görə hər iki diaqnostika səssizcə
6 cümləlik daxili FALLBACK_TEXT-ə düşürdü və top-1 CƏMİ 12 maskalanmış
mövqe üzərində ölçülürdü. 0/12 nəticəsi ilə həqiqi dəqiqliyi 15% olan bir
model arasında fərq qoymaq mümkün deyil (0.85^12 ≈ 0.14) — yəni transplantın
sağ olub-olmadığını təyin edən YEGANƏ nəzarət statistik olaraq kordur.

Mənbə sırası (hansının seçildiyi HƏMİŞƏ nəticəyə yazılır):
  1. açıq verilmiş `--text-file` (mövcuddursa)
  2. `artifacts/data/az_train.jsonl`-dan determinist nümunə  ← default
  3. daxili FALLBACK_TEXT (yalnız heç nə yoxdursa, xəbərdarlıqla)

(2) TRAIN bölgüsündəndir, val/test DEYİL. Bu, qəsdəndir: bu, model
seçiminə təsir edən bir qiymətləndirmə deyil, transplantın texniki
sağlamlıq yoxlamasıdır; train mətni modelin onsuz da görəcəyi mətndir, ona
görə burada işlədilməsi test-set toxunulmazlığını POZMUR (bax
docs/TEST_AUDIT.md).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils import get_logger

log = get_logger(__name__)

# Diaqnostikanın statistik gücü üçün hədəf cümlə sayı. 300 cümlə × ~30 token
# × 15% maskalama ≈ 1,300 maskalanmış mövqe — 12-nin əvəzinə.
DEFAULT_N_SENTENCES = 300
MIN_CHARS = 40  # çox qısa cümlələr maskalanacaq mövqe demək olar vermir

FALLBACK_TEXT = """
Azərbaycan dili türk dilləri ailəsinin oğuz qrupuna aiddir və ölkədə dövlət dilidir.
Uşaqlar məktəbdən gəlmişdilər və evdə valideynlərini gözləyirdilər.
Qanunvericilik sahəsindəki dəyişikliklər barədə rəsmi mənbələrdən məlumat almaq olar.
Kompüter elmləri fakültəsində süni intellekt üzərində tədqiqatlar aparılır.
Xəzər dənizinin ekoloji vəziyyəti ilə bağlı tədbirlər planı hazırlanmışdır.
Təyyarənin uçuşu meteoroloji şəraitə görə təxirə salınmış və sərnişinlərə məlumat verilmişdir.
"""


def resolve_eval_text(cfg, text_file: str | Path | None = None,
                      n_sentences: int = DEFAULT_N_SENTENCES,
                      seed: int = 0) -> tuple[str, dict]:
    """
    Qiymətləndirmə mətnini və ONUN PROVENANSINI qaytarır.

    Qaytarır: (mətn, {"source", "n_lines", "n_chars", ...}) — provenans
    nəticə JSON-una yazılmalıdır ki, iki ölçmənin eyni korpusdan gəldiyi
    sonradan YOXLANA bilsin, güman edilməsin.
    """
    if text_file is not None:
        path = Path(text_file)
        if path.exists():
            text = path.read_text(encoding="utf-8")
            lines = [ln for ln in text.strip().split("\n") if ln.strip()]
            return text, {"source": "file", "path": str(path),
                          "n_lines": len(lines), "n_chars": len(text)}

    train_path = Path(cfg.experiment.artifacts_dir) / "data" / "az_train.jsonl"
    if train_path.exists():
        from src.data.splits import read_jsonl

        records = read_jsonl(train_path)
        usable = [r["text"].replace("\n", " ").strip() for r in records
                  if len(r["text"].strip()) >= MIN_CHARS]
        if usable:
            rng = np.random.default_rng(seed)
            take = min(n_sentences, len(usable))
            idx = rng.choice(len(usable), size=take, replace=False)
            lines = [usable[i] for i in sorted(idx)]
            text = "\n".join(lines)
            return text, {
                "source": "az_train.jsonl",
                "path": str(train_path),
                "n_lines": len(lines),
                "n_chars": len(text),
                "sample_seed": seed,
                "note": ("TRAIN bölgüsündən determinist nümunə — val/test-ə "
                         "TOXUNULMUR (docs/TEST_AUDIT.md)."),
            }

    log.warning(
        "Nə --text-file, nə də %s tapıldı — daxili FALLBACK_TEXT (6 cümlə) "
        "işlədilir. Bu, C1c/top-1 diaqnostikasını statistik olaraq kor edir; "
        "əvvəlcə `python -m src.data.splits` işlədin.", train_path)
    lines = [ln for ln in FALLBACK_TEXT.strip().split("\n") if ln.strip()]
    return FALLBACK_TEXT, {
        "source": "FALLBACK_TEXT",
        "path": None,
        "n_lines": len(lines),
        "n_chars": len(FALLBACK_TEXT),
        "warning": ("Yalnız 6 cümlə — maskalanmış mövqe sayı ~12. Bu "
                    "ölçüdə sıfır top-1 'transplant dağılıb' SÜBUTU DEYİL."),
    }
