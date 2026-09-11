"""
ANCHOR (ortaq token) TAPILMASININ VAHİD MƏNBƏYİ  —  Tapşırıq T3

Problem: anchor tapılması (hansı donor tokeni birbaşa hansı baza tokeninin
embedding-ini miras ala bilər) indiyə qədər İKİ yerdə — `anchors.py` (GO/NO-GO
qapısı) və `build.py` (həqiqi transplant) — AYRI-AYRI yazılmışdı. Onlar
uyğunsuz düşərsə (biri bir konvensiya güman edir, digəri başqa), qapı
"GO" desə də, tikilən model səhv anchor cütləri ilə qurula bilər.

Bu modul TƏK funksiya ilə hər ikisinin İSTİFADƏ ETDİYİ məntiqi verir:

    compute_anchors(base_tok, donor_tok, mode) -> AnchorResult

Üç rejim, konvensiyadan asılılığın ARTAN dərəcədə aradan qaldırılması ilə:

  strict     — (çılpaq_sətir, söz_başlanğıcıdırmı) tuple uyğunlaşması
               (`canon.canon_vocab`). ƏN MÜHAFİZƏKAR ədəd; `xlmr`-də işləyir,
               amma fastBPE (`</w>`, söz-SONU marker) konvensiyalı bazalarda
               (`xlm15`) demək olar heç nə tapmır, çünki söz-başı/sonu
               ayrımı fastBPE üçün mənasızdır (canon.py-dəki qeydə bax).

  surface    — yalnız çılpaq sətir (sərhəd bayrağı atılır). fastBPE `</w>`
               indi düzgün kəsilir (`canon.strip_surface`). YAXINLAŞDIRMA:
               söz-sonu donor parçası ilə söz-başı baza parçasını səhvən
               uyğunlaşdıra bilər — nəticədə bu açıq bildirilir.

  functional — konvensiyadan TAM asılı olmayan üsul: hər donor tokeninin
               çılpaq sətrini BAZA TOKENİZATORUN ÖZÜNƏ verib kodlaşdırırıq.
               Baza tam BİR (xüsusi olmayan) token qaytararsa, həmin baza
               tokeni anchordur. Söz-başı/ortası qeyri-müəyyənliyini iki
               variant sınayaraq həll edirik: sətri TƏKBAŞINA və BOŞLUQLA
               əvvəllənmiş halda kodlaşdırırıq, hansının uyğun gəldiyini
               qeyd edirik.

İşlətmə (birbaşa deyil — `anchors.py` və `build.py` tərəfindən import olunur):
    from src.tokenization.anchor_map import compute_anchors
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from src.tokenization.canon import canon_vocab, detect_scheme, surface_vocab

MODES = ("strict", "surface", "functional")

# T3 qərarı: konvensiyadan asılı olmayan `functional` rejimi default və
# GO/NO-GO qapısının əsaslandığı rejimdir. `anchors.py` (qapı) və
# `build.py` (həqiqi transplant) HƏR İKİSİ bu sabiti işlədir ki, ikisi
# heç vaxt uyğunsuz düşməsin.
DEFAULT_MODE = "functional"


class VocabTokenizer(Protocol):
    """`compute_anchors`-in tələb etdiyi minimal interfeys (mock-lar üçün)."""

    def get_vocab(self) -> dict[str, int]: ...


@dataclass
class AnchorResult:
    """`donor_ids[i]` tokeninin embedding-i `base_ids[i]`-dən köçürülməlidir."""

    mode: str
    donor_ids: np.ndarray
    base_ids: np.ndarray
    base_scheme: str
    donor_scheme: str
    n_anchors: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


def compute_anchors(base_tok: VocabTokenizer, donor_tok: VocabTokenizer,
                     mode: str = DEFAULT_MODE) -> AnchorResult:
    """Donor→baza anchor xəritəsini verilmiş rejimdə hesablayır."""
    if mode not in MODES:
        raise ValueError(f"naməlum rejim: {mode!r} (mümkün: {MODES})")

    base_vocab = base_tok.get_vocab()
    donor_vocab = donor_tok.get_vocab()

    if mode == "strict":
        return _compute_strict(base_vocab, donor_vocab)
    if mode == "surface":
        return _compute_surface(base_vocab, donor_vocab)
    return _compute_functional(base_tok, donor_vocab)


# ---------------------------------------------------------------- strict
def _compute_strict(base_vocab: dict[str, int], donor_vocab: dict[str, int]) -> AnchorResult:
    """Köhnə davranış — `anchors.py`/`build.py`-də əvvəllər inline yazılmışdı."""
    base_scheme = detect_scheme(base_vocab.keys())
    donor_scheme = detect_scheme(donor_vocab.keys())
    base_canon, _ = canon_vocab(base_vocab, base_scheme)
    donor_canon, _ = canon_vocab(donor_vocab, donor_scheme)

    shared = sorted(set(base_canon) & set(donor_canon))
    donor_ids = np.array([donor_canon[k] for k in shared], dtype=np.int64)
    base_ids = np.array([base_canon[k] for k in shared], dtype=np.int64)

    return AnchorResult(
        mode="strict", donor_ids=donor_ids, base_ids=base_ids,
        base_scheme=base_scheme, donor_scheme=donor_scheme,
        n_anchors=len(shared),
        diagnostics={
            "note": "ən mühafizəkar ədəd — sərhəd bayrağı tam uyğun olmalıdır.",
        },
    )


# ---------------------------------------------------------------- surface
def _compute_surface(base_vocab: dict[str, int], donor_vocab: dict[str, int]) -> AnchorResult:
    """Çılpaq sətir uyğunlaşması — YAXINLAŞDIRMA, bax modul dosstring-i."""
    base_surf, base_scheme = surface_vocab(base_vocab)
    donor_surf, donor_scheme = surface_vocab(donor_vocab)

    shared = sorted(set(base_surf) & set(donor_surf))
    donor_ids = np.array([donor_surf[k] for k in shared], dtype=np.int64)
    base_ids = np.array([base_surf[k] for k in shared], dtype=np.int64)

    return AnchorResult(
        mode="surface", donor_ids=donor_ids, base_ids=base_ids,
        base_scheme=base_scheme, donor_scheme=donor_scheme,
        n_anchors=len(shared),
        diagnostics={
            "approximation": True,
            "note": ("Sərhəd (söz-başı/sonu) məlumatı atılıb — söz-sonu donor "
                     "parçası söz-başı baza parçası ilə səhvən uyğunlaşa bilər."),
        },
    )


# ---------------------------------------------------------------- functional
def _compute_functional(base_tok: Any, donor_vocab: dict[str, int]) -> AnchorResult:
    """
    Konvensiyadan asılı olmayan uyğunlaşma: hər donor tokeninin çılpaq
    sətrini baza tokenizatorunun ÖZÜ ilə kodlaşdırırıq.
    """
    donor_surf_by_id, donor_scheme = _decoded_surface_by_id(donor_vocab)
    base_scheme = detect_scheme(base_tok.get_vocab().keys())

    donor_ids_list: list[int] = []
    base_ids_list: list[int] = []
    n_standalone = n_leading_space = n_unmatched = 0

    for donor_id, surface in sorted(donor_surf_by_id.items()):
        if not surface:
            n_unmatched += 1
            continue
        base_id, variant = _encode_single_token(base_tok, surface)
        if base_id is None:
            n_unmatched += 1
            continue
        donor_ids_list.append(donor_id)
        base_ids_list.append(base_id)
        if variant == "standalone":
            n_standalone += 1
        else:
            n_leading_space += 1

    return AnchorResult(
        mode="functional",
        donor_ids=np.array(donor_ids_list, dtype=np.int64),
        base_ids=np.array(base_ids_list, dtype=np.int64),
        base_scheme=base_scheme, donor_scheme=donor_scheme,
        n_anchors=len(donor_ids_list),
        diagnostics={
            "n_matched_standalone": n_standalone,
            "n_matched_leading_space": n_leading_space,
            "n_unmatched": n_unmatched,
            "note": ("Hər donor tokeni baza tokenizatoru ilə kodlaşdırılıb; tam BİR "
                     "xüsusi-olmayan token qaytarırsa anchordur. Konvensiyadan asılı "
                     "deyil, ona görə ən düzgün say gözlənilir."),
        },
    )


def _decoded_surface_by_id(vocab: dict[str, int]) -> tuple[dict[int, str], str]:
    """{id: çılpaq_sətir} — byte-level açılıb, sərhəd markeri kəsilib."""
    from src.tokenization.canon import decode_vocab, strip_surface

    decoded, _ = decode_vocab(vocab)
    scheme = detect_scheme(decoded.keys())
    out: dict[int, str] = {}
    for tok, idx in decoded.items():
        out[idx] = strip_surface(tok, scheme)
    return out, scheme


def _encode_single_token(base_tok: Any, surface: str) -> tuple[int | None, str | None]:
    """
    `surface`-i baza tokenizatoru ilə kodlaşdırır. Tam BİR token qaytarırsa
    onu (id, variant) kimi verir — əvvəlcə TƏKBAŞINA, sonra BOŞLUQLA.
    """
    # DÜZƏLİŞ 2026-09-08 — NAMƏLUM/XÜSUSİ TOKEN RƏDD EDİLİR.
    # Əvvəl `len(ids) == 1` şərti KİFAYƏT sayılırdı. Amma baza tokenizatorunun
    # təmsil edə BİLMƏDİYİ donor səthi də dəqiq bir tokenə kodlaşır: <unk>.
    # O halda köhnə kod donor tokenini bazanın UNK sətri ilə cütləşdirib
    # "anchor" adlandırırdı. Nəticə: UNK vektoru rekonstruksiya lüğətinə
    # dəfələrlə daxil olur, həm anchor sayı şişir, həm də bütün
    # yenidənqurulmuş sətirlər UNK istiqamətinə çəkilir. Bu, transplantın
    # keyfiyyətini ölçən HƏR metrikanı (kosinus, BPC, top-1) korlayır.
    invalid = {i for i in (getattr(base_tok, "unk_token_id", None),) if i is not None}
    invalid |= {int(i) for i in (getattr(base_tok, "all_special_ids", None) or [])}

    for variant, text in (("standalone", surface), ("leading_space", " " + surface)):
        ids = base_tok.encode(text, add_special_tokens=False)
        if len(ids) == 1 and int(ids[0]) not in invalid:
            return int(ids[0]), variant
    return None, None
