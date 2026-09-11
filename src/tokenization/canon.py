"""
Tokenlərin KANONİK formaya salınması  [Nəzarət C2]

Problem: eyni məntiqi token fərqli konvensiyalarda fərqli yazılır.

    SentencePiece (XLM-R):   söz başlanğıcı "▁gəl",  davam "di"
    WordPiece     (BERT):    söz başlanğıcı "gəl",   davam "##di"

Sadə sətir müqayisəsi ("▁gəl" == "gəl"?) bunları uyğunlaşdırmır və ortaq
token sayı süni şəkildə ~0 çıxır. Həll: hər tokeni (sətir, söz_başıdır?)
cütünə çeviririk.

Bayrağı SAXLAYIRIQ, çünki söz başındakı "ev" ilə söz ortasındakı "ev"
fərqli təmsillərdir — birləşdirsək səhv anchor cütləri yaranar.
"""
from __future__ import annotations

from typing import Iterable, Literal

Scheme = Literal["sentencepiece", "wordpiece", "bytelevel", "fastbpe", "plain"]

SP_MARK = "▁"  # ▁
WP_MARK = "##"
BL_MARK = "Ġ"  # Ġ  (GPT-2 boşluq işarəsi)
FASTBPE_MARK = "</w>"  # fastBPE söz-SONU işarəsi (XLM-15 kimi modellər) — SP/WP-dən
                        # fərqli olaraq sözün BAŞLANĞICINI yox, SONUNU göstərir.
FASTBPE_THRESHOLD = 0.30  # bu nisbətdən çox token "</w>" ilə bitirsə → fastBPE


# ---------------------------------------------------------------- byte-level
def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 standart bayt→unicode xəritəsi."""
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


_U2B: dict[str, int] = {v: k for k, v in _bytes_to_unicode().items()}


def byte_decode(token: str) -> str | None:
    """
    Byte-level BPE tokenini əsl mətnə çevirir; uyğun deyilsə None.

    Bəzi tokenizatorlar (HPLT kimi) `▁` və `ə` hərflərini BAYT kimi kodlayır:
        'âĸģgÉĻlmiÅŁdi'  →  '▁gəlmişdi'
    Bu çevrilmə edilməsə, ortaq token axtarışı tamamilə pozulur.
    """
    try:
        return bytearray(_U2B[c] for c in token).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return None


def is_byte_level(vocab: Iterable[str], sample: int = 3000) -> bool:
    """
    Lüğət byte-level kodlanıbmı?

    İki şərt birlikdə: (a) tokenlərin əksəriyyəti bayt kimi açıla bilir,
    (b) açıldıqdan sonra bir hissəsi DƏYİŞİR. İkinci şərt vacibdir — saf ASCII
    lüğət özünə açılır, amma byte-level deyil.
    """
    toks = list(vocab)[:sample]
    if not toks:
        return False
    ok = changed = 0
    for t in toks:
        d = byte_decode(t)
        if d is not None:
            ok += 1
            if d != t:
                changed += 1
    n = len(toks)
    return (ok / n) >= 0.90 and (changed / n) >= 0.02


def decode_vocab(vocab: dict[str, int]) -> tuple[dict[str, int], bool]:
    """Byte-level olarsa lüğəti açır. Qaytarır: (lüğət, açılıbmı)."""
    if not is_byte_level(vocab.keys()):
        return vocab, False
    out: dict[str, int] = {}
    for tok, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
        d = byte_decode(tok)
        key = d if d is not None else tok
        if key not in out:
            out[key] = idx
    return out, True


# ---------------------------------------------------------------- sxem
def detect_scheme(vocab: Iterable[str]) -> Scheme:
    """
    Konvensiyanı təyin edir. Byte-level lüğətlər ƏVVƏLCƏ açılır — əks halda
    `▁` işarəsi 'âĸģ' kimi görünür və sxem səhv təyin olunur.

    fastBPE (XLM-15 kimi) `</w>` ilə söz SONUNU işarələyir — SP/WP-nin əksinə.
    Bunu aşkar etməmək "plain" sxeminə düşməyə səbəb olurdu (T3-ün blokeri):
    heç bir marker tanınmadığı üçün anchor uyğunlaşması demək olar sıfır idi.
    """
    vocab = list(vocab)
    if is_byte_level(vocab):
        decoded = [byte_decode(t) or t for t in vocab]
        # açıldıqdan sonra alt-konvensiyanı yoxla
        n = max(len(decoded), 1)
        if sum(1 for t in decoded if t.startswith(SP_MARK)) / n > 0.05:
            return "sentencepiece"
        if sum(1 for t in decoded if t.startswith(WP_MARK)) / n > 0.02:
            return "wordpiece"
        if sum(1 for t in decoded if t.endswith(FASTBPE_MARK)) / n > FASTBPE_THRESHOLD:
            return "fastbpe"
        return "bytelevel"

    n = max(len(vocab), 1)
    if sum(1 for t in vocab if t.startswith(SP_MARK)) / n > 0.05:
        return "sentencepiece"
    if sum(1 for t in vocab if t.startswith(BL_MARK)) / n > 0.05:
        return "bytelevel"
    if sum(1 for t in vocab if t.startswith(WP_MARK)) / n > 0.02:
        return "wordpiece"
    if sum(1 for t in vocab if t.endswith(FASTBPE_MARK)) / n > FASTBPE_THRESHOLD:
        return "fastbpe"
    return "plain"


def canon(token: str, scheme: Scheme) -> tuple[str, bool]:
    """
    Tokeni kanonik cütə çevirir: (təmiz_sətir, söz_başlanğıcıdırmı).

    >>> canon("▁gəl", "sentencepiece")
    ('gəl', True)
    >>> canon("gəl", "wordpiece")
    ('gəl', True)
    >>> canon("di", "sentencepiece")
    ('di', False)
    >>> canon("##di", "wordpiece")
    ('di', False)
    """
    if scheme == "sentencepiece":
        if token.startswith(SP_MARK):
            return token[len(SP_MARK):], True
        return token, False

    if scheme == "bytelevel":
        if token.startswith(BL_MARK):
            return token[len(BL_MARK):], True
        return token, False

    if scheme == "wordpiece":
        if token.startswith(WP_MARK):
            return token[len(WP_MARK):], False
        return token, True

    # "plain" VƏ QƏSDƏN "fastbpe" DƏ BURAYA DÜŞÜR (T3, `strict` rejimi).
    #
    # fastBPE `</w>` ilə söz SONUNU işarələyir (başlanğıcını yox), ona görə
    # (təmiz_sətir, söz_başlanğıcıdırmı) tuple-modeli üçün bunu SP/WP kimi
    # düzgün dekodlamaq mümkün deyil: "</w>"-siz token söz-başı VƏ ya
    # söz-ortası ola bilər (qeyri-müəyyən). Bunu burda "həll etməyə" çalışmaq
    # `strict` rejimini korlar — o, "ən mühafizəkar ədəd" olaraq TƏYİN
    # OLUNUB (bax anchor_map.py). Ona görə fastbpe qəsdən "plain" ilə EYNİ
    # (dəyişməz token, söz_başı=True) davranışına düşür. Konvensiyadan asılı
    # olmayan düzgün uyğunlaşma `surface`/`functional` rejimlərindədir —
    # bax `strip_surface()` və `src/tokenization/anchor_map.py`.
    return token, True


def canon_vocab(vocab: dict[str, int], scheme: Scheme | None = None
                ) -> tuple[dict[tuple[str, bool], int], Scheme]:
    """
    {token: id} → {(təmiz, söz_başı): id}

    Byte-level lüğətlər ƏVVƏLCƏ açılır (HPLT kimi tokenizatorlar üçün kritikdir).
    Kanonik forma toqquşarsa (nadir), ilk (ən kiçik id) saxlanılır.
    """
    vocab, _ = decode_vocab(vocab)
    if scheme is None:
        scheme = detect_scheme(vocab.keys())

    out: dict[tuple[str, bool], int] = {}
    for tok, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
        key = canon(tok, scheme)
        if key not in out:
            out[key] = idx
    return out, scheme


# ---------------------------------------------------------------- surface (T3, `surface` rejimi)
def strip_surface(token: str, scheme: Scheme) -> str:
    """
    Tokeni sərhəd (söz-başı/sonu) məlumatı OLMADAN çılpaq sətrə salır.

    `canon()`-dan fərqi: heç bir bool qaytarmır, sadəcə markeri kəsir.
    fastBPE üçün BURADA həqiqətən `</w>`-ni kəsirik (canon() qəsdən kəsmir —
    yuxarıdakı qeydə bax). Bu, `surface` rejiminin təsdiqlənmiş
    YAXINLAŞDIRMA (approximation) olmasının səbəbidir: söz-sonu donor
    parçası ilə söz-başı baza parçası eyni çılpaq sətirdə uyğunlaşa bilər.
    """
    if scheme == "sentencepiece" and token.startswith(SP_MARK):
        return token[len(SP_MARK):]
    if scheme == "bytelevel" and token.startswith(BL_MARK):
        return token[len(BL_MARK):]
    if scheme == "wordpiece" and token.startswith(WP_MARK):
        return token[len(WP_MARK):]
    if scheme == "fastbpe" and token.endswith(FASTBPE_MARK):
        return token[:-len(FASTBPE_MARK)]
    return token


def surface_vocab(vocab: dict[str, int], scheme: Scheme | None = None
                  ) -> tuple[dict[str, int], Scheme]:
    """
    {token: id} → {çılpaq_sətir: id} — sərhəd bayrağı atılır.

    Byte-level lüğətlər əvvəlcə açılır (HPLT). Toqquşma varsa (məs. söz-başı
    "▁ev" və söz-ortası "ev" eyni çılpaq sətrə düşür) ilk (ən kiçik id)
    saxlanılır — bu, `surface` rejiminin approximation olmasının bir
    hissəsidir və nəticədə aşkar edilib bildirilir.
    """
    vocab, _ = decode_vocab(vocab)
    if scheme is None:
        scheme = detect_scheme(vocab.keys())

    out: dict[str, int] = {}
    for tok, idx in sorted(vocab.items(), key=lambda kv: kv[1]):
        key = strip_surface(tok, scheme)
        if key not in out:
            out[key] = idx
    return out, scheme


# ---------------------------------------------------------------- xüsusi tokenlər
# Xüsusi tokenlər ROLA görə uyğunlaşdırılmalıdır, sətrə görə deyil:
# XLM-R "<s>"  ≡  BERT "[CLS]"  — eyni funksiya, fərqli ad.
SPECIAL_ROLES = ("bos", "eos", "cls", "sep", "pad", "unk", "mask")


def special_role_map(tokenizer) -> dict[str, str]:
    """Rol → həmin tokenizatordakı sətir (mövcud olanlar)."""
    m = {}
    for role, attr in [
        ("bos", "bos_token"), ("eos", "eos_token"), ("cls", "cls_token"),
        ("sep", "sep_token"), ("pad", "pad_token"), ("unk", "unk_token"),
        ("mask", "mask_token"),
    ]:
        val = getattr(tokenizer, attr, None)
        if val:
            m[role] = val
    return m


def align_special_tokens(base_tok, donor_tok) -> dict[int, int]:
    """
    donor_id → base_id  xəritəsi (yalnız xüsusi tokenlər üçün).

    Bu olmasa <pad>/<mask>/<cls> embedding-ləri səhv yerə düşür və model
    səssizcə pozulur — ən çox rast gəlinən gizli səhvlərdən biridir.
    """
    b = special_role_map(base_tok)
    d = special_role_map(donor_tok)
    mapping: dict[int, int] = {}
    for role in SPECIAL_ROLES:
        if role in b and role in d:
            bid = base_tok.convert_tokens_to_ids(b[role])
            did = donor_tok.convert_tokens_to_ids(d[role])
            if bid is not None and did is not None and bid >= 0 and did >= 0:
                mapping[did] = bid

    # BERT-də bos/eos yoxdur, XLM-R-də cls/sep yoxdur → çarpaz uyğunlaşdırma.
    # DÖRD İSTİQAMƏT DƏ lazımdır — YALNIZ "donor-da rol yoxdur" hallarını
    # yoxlamaq kifayət etmir. Real misal (T7-də tapıldı, HPLT←XLM-15):
    # XLM-15-in ÖZÜNDƏ "eos" rolu HEÇ YOXDUR (yalnız bos/cls/sep var, cls VƏ
    # sep hər ikisi "</s>"-ə düşür) — donor-un "eos"-u ("[EOS]") HEÇ bir baza
    # roluna uyğunlaşdırılmırdı, `unfamiliar_ids`-ə düşürdü, OMP onu E_donor-da
    # (əslində embedding cədvəlinin ÖZÜNDƏ) tapa bilmirdi → IndexError.
    if "cls" in d and "cls" not in b and "bos" in b:
        mapping[donor_tok.convert_tokens_to_ids(d["cls"])] = \
            base_tok.convert_tokens_to_ids(b["bos"])
    if "bos" in d and "bos" not in b and "cls" in b:
        mapping[donor_tok.convert_tokens_to_ids(d["bos"])] = \
            base_tok.convert_tokens_to_ids(b["cls"])
    if "sep" in d and "sep" not in b and "eos" in b:
        mapping[donor_tok.convert_tokens_to_ids(d["sep"])] = \
            base_tok.convert_tokens_to_ids(b["eos"])
    if "eos" in d and "eos" not in b and "sep" in b:
        mapping[donor_tok.convert_tokens_to_ids(d["eos"])] = \
            base_tok.convert_tokens_to_ids(b["sep"])
    return mapping
