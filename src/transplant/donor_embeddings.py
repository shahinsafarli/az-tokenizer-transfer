"""
DONOR EMBEDDİNG MATRİSİNİN `trust_remote_code` OLMADAN OXUNMASI

`HPLT/hplt_bert_base_az` `AutoModelForMaskedLM.from_pretrained()` ilə
YÜKLƏNƏ BİLMİR — xüsusi arxitektura kodu tələb edir (`LtgbertForMaskedLM`,
`modeling_ltgbert.py`) və bunu icazəsiz İCRA ETMƏK arbitrary-kod-icrası
riskidir. Bizə isə tikinti prosesindən YALNIZ BİR ŞEY lazımdır: giriş
embedding matrisi (`get_input_embeddings().weight`). Bunun üçün arxitektura
kodunu İCRA ETMƏYƏ EHTİYAC YOXDUR — safetensors faylını birbaşa oxuyub
matrisi FORMASINA görə tapmaq kifayətdir.

TƏHLÜKƏSİZLİK QEYDLƏRİ:
  - `torch.load()` (pickle-based, `weights_only=True` olmadan) da ARBİTRARY
    KOD İCRASI riski daşıyır — `trust_remote_code`-dan YAXŞI DEYİL. Ona görə
    YALNIZ `safetensors` işlədilir (pickle DEYİL, icra riski yoxdur) və
    YALNIZ `model.safetensors` endirilir (`pytorch_model.bin` YOX).
  - Revizyon (commit SHA) PİNLƏNİR — brifin "bir-əmrlə reproduksiya" tələbi
    üçün, etibar məsələsindən TAMAMİLƏ ASILI OLMAYARAQ: donor repo sonradan
    dəyişsə belə, bu skript HƏMİŞƏ EYNİ çəkiləri oxuyacaq.

TAPILAN XÜSUSİYYƏT (bu modul YAZILARKƏN aşkar edildi, əvvəllər sənədləşməmişdi):
  - `tokenizer.get_vocab()` 32,770 giriş qaytarır, amma faktiki embedding
    matrisi YALNIZ 32,768 sətirdir (`tokenizer.vocab_size`). Fərq olan 2 id
    (32768=[BOS], 32769=[EOS]) embedding cədvəlinin ARXASINDADIR.
    ZƏRƏRSİZDİR: `build.py`-də bu iki id `align_special_tokens()` vasitəsilə
    ROL-ə görə BAZA modeldən köçürülür — E_donor-dan HEÇ VAXT oxunmur (bax
    `is_anchor[did]=True` xüsusi-token qolu). Ona görə bu modul YALNIZ
    32,768-sətirlik matrisi qaytarır, tokenizer-in TAM (32,770) uzunluğuna
    PADDING ETMİR.
  - Formaya görə axtarış İKİ namizəd tapır: `embedding.word_embedding.weight`
    VƏ `classifier.nonlinearity.5.weight` — HƏR İKİSİ [32768, 768]. Bu, forma-
    yalnız uyğunlaşmanın PRİNSİPCƏ qeyri-müəyyən qala biləcəyi REAL bir
    haldır (brifin nəzərdə tutduğu "ambiguous → raise" ssenarisinin özü).
    Aşağıdakı `select_embedding_tensor()` bunu AÇAR-SÖZ tie-break ilə (yalnız
    forma tək-başına ayırd edə bilmədikdə) həll edir və HANSI açarın
    seçildiyini LOQLAYIR — səssizcə seçmir.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.utils import get_logger

log = get_logger(__name__)

DEFAULT_WEIGHTS_FILE = "model.safetensors"

# Formanın TƏK-BAŞINA ayırd edə bilmədiyi HALDA (2+ namizəd) işlədilən
# açar-söz tie-break. BİRİNCİ SEÇİM MEXANİZMİ DEYİL — forma HƏMİŞƏ ilk filtrdir.
_PREFERRED_SUBSTRINGS = ("embed",)
_DEPRIORITIZED_SUBSTRINGS = ("classifier", "lm_head", "decoder", "predictions", "cls")


def select_embedding_tensor(state_dict: dict[str, Any], vocab_size: int) -> tuple[str, Any]:
    """
    `state_dict`-də formasının BİRİNCİ ölçüsü `vocab_size`-ə BƏRABƏR olan
    2-ölçülü tensoru tapır.

      0 namizəd  → ValueError, bütün 2-ölçülü tensorların siyahısı ilə.
      1 namizəd  → birbaşa qaytarılır.
      2+ namizəd → açar-söz tie-break sınanılır (yalnız BUNDA); tək
                   namizədə düşərsə seçim LOQLANIR (xəbərdarlıq kimi);
                   düşməzsə ValueError, BÜTÜN namizədlərin açar+forması ilə.

    Səhv matrisin SƏSSİZCƏ seçilməsinin qarşısını almaq — bu, hər tərplantı
    zəhərləyər — məqsədi budur.
    """
    def shape2d(v: Any):
        s = tuple(v.shape)
        return s if len(s) == 2 else None

    candidates = [(k, v) for k, v in state_dict.items()
                 if (s := shape2d(v)) is not None and s[0] == vocab_size]

    if not candidates:
        all_2d = [(k, shape2d(v)) for k, v in state_dict.items() if shape2d(v) is not None]
        raise ValueError(
            f"Heç bir tensor formasının birinci ölçüsü vocab_size={vocab_size}-ə BƏRABƏR deyil.\n"
            f"Mövcud 2-ölçülü tensorlar: {all_2d}"
        )
    if len(candidates) == 1:
        return candidates[0]

    preferred = [(k, v) for k, v in candidates
                if any(s in k.lower() for s in _PREFERRED_SUBSTRINGS)
                and not any(s in k.lower() for s in _DEPRIORITIZED_SUBSTRINGS)]
    if len(preferred) == 1:
        log.warning(
            "Forma-uyğunluğu %d namizəd verdi (%s) — açar-söz tie-break '%s'-i seçdi. "
            "Bu, transplant-dan ƏVVƏL YOXLANILMALIDIR (bax `resolved_embedding_key` "
            "diaqnostika sahəsi).",
            len(candidates), [k for k, _ in candidates], preferred[0][0],
        )
        return preferred[0]

    raise ValueError(
        f"{len(candidates)} tensor forması vocab_size={vocab_size} ilə UYĞUN GƏLİR, "
        f"açar-söz tie-break TƏK namizədə düşmədi: "
        f"{[(k, shape2d(v)) for k, v in candidates]}\n"
        "Səhv matrisin SƏSSİZCƏ seçilməsinin qarşısını almaq üçün DAYANDIRILDI."
    )


def load_donor_embeddings(donor_name: str, revision: str,
                          filename: str = DEFAULT_WEIGHTS_FILE
                          ) -> tuple[np.ndarray, str, dict]:
    """
    Donor modelinin GİRİŞ EMBEDDİNG matrisini `AutoModelForMaskedLM` (VƏ
    onunla birgə `trust_remote_code`) OLMADAN oxuyur.

    Qaytarır: (E_donor [n_rows, hidden] float32 numpy massivi,
               tapılan tensor açarı, diaqnostika dict-i).
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoTokenizer

    path = hf_hub_download(donor_name, filename, revision=revision)
    log.info("Donor çəkiləri endirildi (PİNLƏNMİŞ revizyon %s): %s", revision, path)

    tok = AutoTokenizer.from_pretrained(donor_name, revision=revision)
    vocab_size = tok.vocab_size          # BAZA vocab (added tokens XARİC) — embedding CƏDVƏLİNƏ uyğun olan budur
    n_get_vocab = len(tok.get_vocab())   # tokenizer-in TAM (added tokens DAXİL) uzunluğu

    with safe_open(path, framework="np") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()
                     if len(f.get_slice(k).get_shape()) == 2}
        key, tensor = select_embedding_tensor(state_dict, vocab_size)

    diag = {
        "donor_model": donor_name, "revision": revision, "weights_file": filename,
        "resolved_embedding_key": key, "embedding_shape": list(tensor.shape),
        "tokenizer_vocab_size_attr": vocab_size, "tokenizer_get_vocab_len": n_get_vocab,
    }
    if n_get_vocab != vocab_size:
        log.warning(
            "tokenizer.get_vocab() (%d) != tokenizer.vocab_size (%d) — %d id embedding "
            "CƏDVƏLİNİN XARİCİNDƏDİR (adətən BOS/EOS kimi əlavə tokenlər). Bunlar "
            "build.py-də `align_special_tokens()` ilə ROL-ə görə BAZA modeldən köçürülür, "
            "E_donor-dan HEÇ oxunmur — zərərsizdir, amma qeyd olunur.",
            n_get_vocab, vocab_size, n_get_vocab - vocab_size,
        )
    log.info("Embedding tapıldı: %s  forma=%s", key, tuple(tensor.shape))
    return np.asarray(tensor, dtype=np.float32), key, diag
