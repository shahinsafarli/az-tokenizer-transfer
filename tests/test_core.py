"""
Vahid testləri — model yükləmədən, sürətli işləyir.
    pytest -q
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.aggregate import summarize_cell
from src.data.scramble import scramble_sentence
from src.tokenization.canon import canon, canon_vocab, detect_scheme
from src.transplant.omp import solve_coefficients, transplant_embeddings
from src.utils import Config


# ---------------------------------------------------------------- canon (C2)
def test_canon_sentencepiece():
    assert canon("▁gəl", "sentencepiece") == ("gəl", True)
    assert canon("di", "sentencepiece") == ("di", False)


def test_canon_wordpiece():
    assert canon("gəl", "wordpiece") == ("gəl", True)
    assert canon("##di", "wordpiece") == ("di", False)


def test_canon_aligns_across_schemes():
    """Əsas iddia: fərqli konvensiyalar kanonik formada uyğunlaşır."""
    sp = {"▁gəl": 0, "di": 1, "▁ev": 2}
    wp = {"gəl": 0, "##di": 1, "ev": 2}
    sp_c, _ = canon_vocab(sp, "sentencepiece")
    wp_c, _ = canon_vocab(wp, "wordpiece")

    assert len(set(sp) & set(wp)) == 0          # xam sətir: heç bir örtüşmə
    assert len(set(sp_c) & set(wp_c)) == 3      # kanonik: tam örtüşmə


def test_detect_scheme():
    assert detect_scheme(["▁a", "▁b", "c"] * 10) == "sentencepiece"
    assert detect_scheme(["a", "##b", "##c"] * 10) == "wordpiece"


# ---- align_special_tokens: baza-da rol TAMAMİLƏ yoxdursa (T7-də tapılan bug)
class _MockTok:
    """special_role_map/align_special_tokens-in tələb etdiyi minimum interfeys."""

    def __init__(self, **role_tokens: str):
        for role, tok in role_tokens.items():
            setattr(self, f"{role}_token", tok)
        self._ids = {tok: i for i, tok in enumerate(sorted(set(role_tokens.values())))}

    def convert_tokens_to_ids(self, tok):
        return self._ids.get(tok, -1)


def test_align_special_tokens_maps_donor_eos_when_base_has_no_eos_role():
    """
    Real hal (XLM-15, T7): baza tokenizatorda "eos" rolu HEÇ YOXDUR (yalnız
    bos/cls/sep var, cls VƏ sep hər ikisi eyni tokenə "</s>" düşür). Köhnə
    kod donor-un "eos"-unu heç bir baza roluna uyğunlaşdırmırdı → unfamiliar
    sayılıb OMP-yə göndərilirdi → IndexError (E_donor-da bu id üçün sətir
    yoxdur). Düzəliş: "eos" bazada yoxdursa, "sep"-ə uyğunlaşdırılsın.
    """
    from src.tokenization.canon import align_special_tokens
    base = _MockTok(bos="<s>", cls="</s>", sep="</s>", pad="<pad>", unk="<unk>", mask="<m>")
    donor = _MockTok(bos="[BOS]", eos="[EOS]", cls="[CLS]", sep="[SEP]",
                     pad="[PAD]", unk="[UNK]", mask="[MASK]")
    mapping = align_special_tokens(base, donor)
    donor_eos_id = donor.convert_tokens_to_ids("[EOS]")
    base_sep_id = base.convert_tokens_to_ids("</s>")
    assert donor_eos_id in mapping
    assert mapping[donor_eos_id] == base_sep_id


def test_align_special_tokens_maps_donor_bos_when_base_has_no_bos_role():
    """Simmetrik hal: baza-da "bos" yoxdur (yalnız cls var) → donor bos → baza cls."""
    from src.tokenization.canon import align_special_tokens
    base = _MockTok(cls="[CLS]", sep="[SEP]", pad="[PAD]", unk="[UNK]", mask="[MASK]")
    donor = _MockTok(bos="<s>", eos="</s>", pad="<pad>", unk="<unk>", mask="<m>")
    mapping = align_special_tokens(base, donor)
    donor_bos_id = donor.convert_tokens_to_ids("<s>")
    base_cls_id = base.convert_tokens_to_ids("[CLS]")
    assert donor_bos_id in mapping
    assert mapping[donor_bos_id] == base_cls_id


# ---------------------------------------------------------------- OMP
def test_omp_recovers_exact_combination():
    """
    Hədəf anchorların dəqiq kombinasiyasıdırsa, OMP onu tapmalıdır.

    Qeyd: ölçü (64) namizəd sayından (20) BÖYÜK olmalıdır — real setup-da
    da belədir (donor ölçüsü ~768–1024 > n_candidates=256). Əks halda lüğət
    overcomplete olur və greedy seçim səhv atom götürür.
    """
    rng = np.random.default_rng(0)
    anchors = rng.normal(size=(20, 64)).astype(np.float32)
    true_w = np.zeros(20, dtype=np.float32)
    true_w[[3, 7]] = [0.6, -0.4]
    target = true_w @ anchors

    w = solve_coefficients(target, anchors, k=2, method="omp")
    rec = w @ anchors
    assert np.allclose(rec, target, atol=1e-3)
    assert (np.abs(w) > 1e-6).sum() <= 2
    assert set(np.nonzero(np.abs(w) > 1e-6)[0]) == {3, 7}


def test_omp_degrades_when_overcomplete():
    """
    Sənədləşdirilmiş məhdudiyyət: namizəd sayı ölçüdən çox olduqda OMP
    dəqiq bərpa etmir. Buna görə transplant.n_candidates < hidden_size.
    """
    rng = np.random.default_rng(0)
    anchors = rng.normal(size=(20, 8)).astype(np.float32)   # 20 atom, 8 ölçü
    true_w = np.zeros(20, dtype=np.float32)
    true_w[[3, 7]] = [0.6, -0.4]
    target = true_w @ anchors

    w = solve_coefficients(target, anchors, k=2, method="omp")
    assert not np.allclose(w @ anchors, target, atol=1e-3)


def test_omp_sparsity_respected():
    rng = np.random.default_rng(1)
    anchors = rng.normal(size=(50, 16)).astype(np.float32)
    target = rng.normal(size=16).astype(np.float32)
    for k in (1, 4, 8):
        w = solve_coefficients(target, anchors, k=k, method="omp")
        assert (np.abs(w) > 1e-8).sum() <= k


def test_transplant_dimension_mismatch_ok():
    """Donor və baza fərqli hidden ölçüdə ola bilər — əmsallar köçürülür."""
    rng = np.random.default_rng(2)
    E_donor = rng.normal(size=(100, 32)).astype(np.float32)   # d_d = 32
    E_base = rng.normal(size=(80, 16)).astype(np.float32)     # d_b = 16
    anchor_d = np.arange(60)
    anchor_b = np.arange(60)
    unfamiliar = np.arange(60, 100)

    out = transplant_embeddings(
        E_donor, E_base, anchor_d, anchor_b, unfamiliar,
        k=8, n_candidates=32, method="omp", normalize=True, batch_size=16)

    assert out.shape == (40, 16)
    assert np.isfinite(out).all()


def test_transplant_embeddings_output_scale_matches_base_not_donor():
    """
    RÜCU TESTİ — T7-də tapılan real bug: köhnə kod nəticə vektorunu DONOR
    hədəfinin normuna yenidən miqyaslayırdı, halbuki vektor artıq BAZA
    fəzasındadır (ƏMSALLAR köçürülür, vektorlar yox — modulun öz dosstringi).
    Real hal: HPLT (donor) orta norm ≈13.9, XLM-15 (baza) orta norm ≈0.6 —
    bu qarışıqlıq nəticəni ~18× şişirdib MLM BPC-ni fəlakətə uğradırdı
    (8.7 → 161 bit/hərf, C1c nəzarəti tutdu). Sintetik olaraq eyni miqyas
    nisbətini yaradıb yoxlayırıq: nəticə BAZA miqyasında qalmalıdır.
    """
    rng = np.random.default_rng(0)
    d_d, d_b, n_anchor = 4, 3, 6
    E_donor = rng.normal(size=(20, d_d)).astype(np.float32) * 50.0   # donor: BÖYÜK miqyas
    E_base = rng.normal(size=(20, d_b)).astype(np.float32) * 0.5    # baza: KİÇİK miqyas

    anchor_donor_ids = np.arange(n_anchor)
    anchor_base_ids = np.arange(n_anchor)
    unfamiliar_ids = np.arange(n_anchor, n_anchor + 3)

    out = transplant_embeddings(
        E_donor, E_base, anchor_donor_ids, anchor_base_ids, unfamiliar_ids,
        k=2, n_candidates=n_anchor, method="mean", normalize=True, batch_size=8)

    base_anchor_norms = np.linalg.norm(E_base[anchor_base_ids], axis=1)
    donor_target_norms = np.linalg.norm(E_donor[unfamiliar_ids], axis=1)
    out_norms = np.linalg.norm(out, axis=1)

    assert out_norms.max() < 5 * base_anchor_norms.max()      # BAZA miqyasına yaxın
    assert out_norms.max() < 0.5 * donor_target_norms.min()   # DONOR miqyasına YAXIN DEYİL


# ---- embedding norm hesabatı — C1a-nın yanında tələb olunan yoxlama (T7)
def test_compute_norm_report_flags_unhealthy_ratio():
    from src.transplant.embedding_norms import compute_norm_report
    rng = np.random.default_rng(0)
    E_base = rng.normal(size=(10, 4)).astype(np.float32) * 0.6
    E_donor = rng.normal(size=(10, 4)).astype(np.float32) * 14.0
    E_new = E_base.copy()
    anchor_ids = np.array([0, 1, 2])
    unfamiliar_ids = np.array([3, 4, 5])
    E_new[unfamiliar_ids] *= 0.1   # "büzülmüş" (under-normed) yenidənqurmanı simulyasiya edir

    report = compute_norm_report(E_base, E_donor, E_new, anchor_ids, unfamiliar_ids)
    assert report["reconstructed_to_anchor_median_ratio"] < 0.85
    assert report["in_healthy_range"] is False


def test_compute_norm_report_passes_healthy_ratio():
    from src.transplant.embedding_norms import compute_norm_report
    rng = np.random.default_rng(0)
    E_base = rng.normal(size=(10, 4)).astype(np.float32) * 0.6
    E_donor = rng.normal(size=(10, 4)).astype(np.float32) * 14.0
    E_new = E_base.copy()   # anchor VƏ "reconstructed" eyni paylanmadan — sağlam nisbət
    anchor_ids = np.array([0, 1, 2])
    unfamiliar_ids = np.array([3, 4, 5])

    report = compute_norm_report(E_base, E_donor, E_new, anchor_ids, unfamiliar_ids)
    assert report["in_healthy_range"] is True


def test_random_coef_control_differs_from_omp():
    """[C1] Təsadüfi əmsallar OMP-dən fərqli nəticə verməlidir."""
    rng = np.random.default_rng(3)
    anchors = rng.normal(size=(30, 12)).astype(np.float32)
    target = anchors[5] * 0.9
    w_omp = solve_coefficients(target, anchors, k=4, method="omp")
    w_rnd = solve_coefficients(target, anchors, k=4, method="random_coef",
                               rng=np.random.default_rng(0))
    assert not np.allclose(w_omp, w_rnd)


# ---------------------------------------------------------------- scramble (C4)
def test_scramble_preserves_word_multiset():
    rng = np.random.default_rng(0)
    s = "bugün hava çok güzel ve sıcak"
    out = scramble_sentence(s, rng)
    assert sorted(out.split()) == sorted(s.split())


def test_scramble_short_sentence_unchanged():
    rng = np.random.default_rng(0)
    assert scramble_sentence("merhaba", rng) == "merhaba"


# ---------------------------------------------------------------- frozen dual metrics
def test_dual_metric_uses_only_escaped_f1_values():
    records = [
        {"escaped": False, "selected_validation_macro_f1": 0.333,
         "source_path": "collapsed.json"},
        {"escaped": True, "selected_validation_macro_f1": 0.577,
         "source_path": "escaped.json"},
    ]
    result = summarize_cell(records, iters=200)
    assert result["escape_rate"] == 0.5
    assert result["conditional_macro_f1"] == pytest.approx(0.577)
    assert result["n_escaped"] == 1


def test_zero_escape_is_valid_null_conditional_measurement():
    records = [
        {"escaped": False, "selected_validation_macro_f1": 0.333,
         "source_path": "a.json"},
        {"escaped": False, "selected_validation_macro_f1": 0.334,
         "source_path": "b.json"},
    ]
    result = summarize_cell(records, iters=200)
    assert result["status"] == "MEASURED"
    assert result["escape_rate_count"] == "0/2"
    assert result["conditional_macro_f1"] is None
    assert result["n_escaped"] == 0


# ---------------------------------------------------------------- byte-level (HPLT)
def test_byte_decode_real_hplt_tokens():
    """HPLT tokenizatorundan real nümunələr."""
    from src.tokenization.canon import byte_decode
    assert byte_decode("âĸģgÉĻlmiÅŁdi") == "▁gəlmişdi"
    assert byte_decode("lÉĻr") == "lər"
    assert byte_decode("âĸģqanunvericilik") == "▁qanunvericilik"


def test_is_byte_level_detects_hplt_style():
    from src.tokenization.canon import is_byte_level
    bl = ["âĸģgÉĻlmiÅŁdi", "lÉĻr", "âĸģqanunvericilik", "âĸģev", "dir"] * 200
    assert is_byte_level(bl) is True


def test_is_byte_level_rejects_plain_utf8():
    """Real UTF-8 azərbaycan tokenləri byte-level SAYILMAMALIDIR."""
    from src.tokenization.canon import is_byte_level
    assert is_byte_level(["▁gəlmişdi", "lər", "▁ev", "##dir"] * 200) is False


def test_is_byte_level_rejects_pure_ascii():
    from src.tokenization.canon import is_byte_level
    assert is_byte_level(["the", "##ing", "cat", "dog"] * 200) is False


def test_bytelevel_vocab_matches_sentencepiece_after_fix():
    """
    ƏSAS TEST: byte-level lüğət ilə SentencePiece lüğəti kanonik formada
    uyğunlaşmalıdır. Düzəlişdən əvvəl bu 0 örtüşmə verirdi (saxta NO_GO).
    """
    from src.tokenization.canon import canon_vocab
    sp = {"▁gəlmişdi": 0, "lər": 1, "▁ev": 2}
    bl = {"âĸģgÉĻlmiÅŁdi": 0, "lÉĻr": 1, "âĸģev": 2}
    sp_c, sp_s = canon_vocab(sp)
    bl_c, bl_s = canon_vocab(bl)
    assert sp_s == "sentencepiece"
    assert bl_s == "sentencepiece"          # açıldıqdan sonra düzgün təyin olunur
    assert len(set(sp_c) & set(bl_c)) == 3  # tam örtüşmə


# ================================================================
#  fastBPE / anchor_map (T3) — anchor uyğunlaşması konvensiyadan asılı olmasın
# ================================================================
def test_detect_scheme_fastbpe():
    """XLM-15 kimi lüğətlər: `</w>` söz-SONU markeri, ▁/## heç biri yoxdur."""
    from src.tokenization.canon import detect_scheme
    vocab = ["bu</w>", "san</w>", "ie", "str", "len", "jusqu'</w>"] * 50
    assert detect_scheme(vocab) == "fastbpe"


def test_strip_surface_fastbpe():
    """`</w>` kəsilir (canon() isə QƏSDƏN kəsmir — strict rejimi qorunsun deyə)."""
    from src.tokenization.canon import canon, strip_surface
    assert strip_surface("bu</w>", "fastbpe") == "bu"
    assert strip_surface("ie", "fastbpe") == "ie"          # marker yoxdursa dəyişməz
    # canon() strict üçün fastbpe-ni "plain" kimi saxlayır — token TOXUNULMAZ qalır
    assert canon("bu</w>", "fastbpe") == ("bu</w>", True)


def test_surface_vocab_bridges_fastbpe_and_sentencepiece():
    """
    `strict` (tuple) rejimi fastBPE/SP arasında 0 örtüşmə verir (sərhəd
    bayrağı uyğun gəlmir), amma `surface` (çılpaq sətir) tam uyğunlaşır.
    """
    from src.tokenization.canon import canon_vocab, surface_vocab
    donor_fastbpe = {"bu</w>": 0, "san": 1, "ev</w>": 2, "başqa</w>": 3,
                      "əlavə</w>": 4, "ekstra": 5, "digər</w>": 6, "sıra</w>": 7}
    base_sp = {"▁bu": 0, "san": 1, "▁ev": 2}

    strict_c, strict_s = canon_vocab(donor_fastbpe)
    assert strict_s == "fastbpe"
    base_c, _ = canon_vocab(base_sp)
    assert len(set(strict_c) & set(base_c)) == 0     # strict: uyğunlaşmır

    donor_surf, donor_s = surface_vocab(donor_fastbpe)
    assert donor_s == "fastbpe"
    base_surf, _ = surface_vocab(base_sp)
    assert set(donor_surf) & set(base_surf) == {"bu", "san", "ev"}


class _GreedyMockTokenizer:
    """
    Test üçün minimal tokenizator: `get_vocab()` + `encode()`. Lüğətindəki
    parçalarla ən-uzun-uyğunluq (greedy longest-match) prinsipi ilə
    kodlaşdırır — real BPE deyil, amma `compute_anchors`-in "tam BİR token"
    məntiqini yoxlamaq üçün kifayətdir. Aparıcı boşluğu SentencePiece kimi
    "▁"-yə çevirir ki, söz-başı/söz-ortası fərqi sınana bilsin.
    """

    def __init__(self, vocab: dict[str, int]):
        self._vocab = dict(vocab)
        self._pieces = sorted(self._vocab, key=len, reverse=True)

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text.startswith(" "):
            text = "▁" + text[1:]
        ids: list[int] = []
        i = 0
        while i < len(text):
            for piece in self._pieces:
                if text.startswith(piece, i):
                    ids.append(self._vocab[piece])
                    i += len(piece)
                    break
            else:
                ids.append(-1)   # naməlum simvol — "birdən çox token" halını yaradır
                i += 1
        return ids


def test_functional_matching_synthetic_mock():
    """
    T3 ƏSAS TEST: `functional` rejimi konvensiyadan asılı olmadan uyğunlaşdırır.

      donor "di"  (marker yox)      → baza "di" ilə TƏKBAŞINA kodlaşır  → anchor
      donor "ev</w>" (söz-sonu)     → çılpaq "ev" yalnız BOŞLUQLA "▁ev" kimi
                                       TƏK tokenə düşür                  → anchor (leading_space)
      donor "xyz</w>"               → baza lüğətində heç bir parça yoxdur → anchor DEYİL
    """
    from src.tokenization.anchor_map import compute_anchors

    donor_tok = _GreedyMockTokenizer({"di": 1, "ev</w>": 2, "xyz</w>": 3})
    base_tok = _GreedyMockTokenizer({"di": 12, "▁ev": 30})

    r = compute_anchors(base_tok, donor_tok, mode="functional")
    assert r.mode == "functional"
    assert r.n_anchors == 2
    mapping = dict(zip(r.donor_ids.tolist(), r.base_ids.tolist()))
    assert mapping == {1: 12, 2: 30}
    assert r.diagnostics["n_matched_standalone"] == 1     # "di"
    assert r.diagnostics["n_matched_leading_space"] == 1  # "ev</w>" → "▁ev"
    assert r.diagnostics["n_unmatched"] == 1              # "xyz</w>"


def test_anchor_modes_return_consistent_shapes():
    """Hər üç rejim eyni şəkilli (donor_ids, base_ids) massivləri qaytarmalıdır."""
    from src.tokenization.anchor_map import MODES, compute_anchors

    donor_tok = _GreedyMockTokenizer({"di": 1, "ev</w>": 2, "xyz</w>": 3})
    base_tok = _GreedyMockTokenizer({"di": 12, "▁ev": 30})

    for mode in MODES:
        r = compute_anchors(base_tok, donor_tok, mode=mode)
        assert r.mode == mode
        assert r.donor_ids.shape == r.base_ids.shape
        assert r.donor_ids.dtype == np.int64 and r.base_ids.dtype == np.int64
        assert r.n_anchors == len(r.donor_ids)


# ================================================================
#  İKİ BAZA MODEL — doza-cavab dizaynının regressiya testləri
# ================================================================
def _two_base_cfg():
    return Config({
        "models": {
            "primary": {"id": "org/xlm15", "short": "xlm15",
                        "az_in_pretraining": False},
            "contrast": {"id": "org/xlmr", "short": "xlmr",
                         "az_in_pretraining": True},
            "donor": "org/donor",
        },
        "experiment": {"artifacts_dir": "artifacts", "seeds": [13, 42, 1337],
                       "seeds_high_resource": [13]},
        "transplant": {"method": "omp", "k": 64},
        "data": {"train_sizes": [100, 500, 2000, 10000]},
        "analysis": {"bootstrap_iters": 400, "alpha": 0.05,
                     "reference_size": 2000},
        "training": {"max_steps": 2000},
        "run": {"bases": ["primary", "contrast"], "contrast_sizes": [2000],
                "priority_first": True, "skip_existing": True},
    })


def test_resolve_bases_reads_roles_and_flags():
    from src.utils import resolve_bases
    bases = resolve_bases(_two_base_cfg())
    assert [b.short for b in bases] == ["xlm15", "xlmr"]
    assert bases[0].az_in_pretraining is False
    assert bases[1].az_in_pretraining is True


def test_resolve_bases_unknown_role_is_explicit():
    from src.utils import resolve_bases
    with pytest.raises(ValueError, match="models.donorr"):
        resolve_bases(_two_base_cfg(), ["donorr"])


def test_runkey_filename_separates_bases():
    """İki baza modelin run-ları AYRI fayllara yazılmalıdır."""
    from src.utils import RunKey
    a = RunKey(base="xlm15", condition="baza", train_size=2000, seed=42).filename
    b = RunKey(base="xlmr", condition="baza", train_size=2000, seed=42).filename
    assert a != b
    assert a.startswith("base=xlm15__") and b.startswith("base=xlmr__")


def test_transplant_dir_is_base_specific():
    """
    Embedding fəzası baza modelə məxsusdur — iki baza eyni qovluğa yazsa,
    biri digərinin çəkilərini üstələyər. Bu test onun qarşısını alır.
    """
    from src.utils import default_tag, transplant_dir
    cfg = _two_base_cfg()
    tag = default_tag(cfg)
    d1 = transplant_dir(cfg, "xlm15", tag)
    d2 = transplant_dir(cfg, "xlmr", tag)
    assert d1 != d2
    assert "xlm15" in str(d1) and "xlmr" in str(d2)


def test_sizes_for_base_limits_contrast():
    from src.utils import resolve_bases, sizes_for_base
    cfg = _two_base_cfg()
    prim, cont = resolve_bases(cfg)
    assert sizes_for_base(cfg, prim) == [100, 500, 2000, 10000]
    assert sizes_for_base(cfg, cont) == [2000]


def test_queue_covers_both_bases_at_reference_size():
    from src.training.orchestrate import build_queue
    from src.utils import resolve_bases
    cfg = _two_base_cfg()
    cfg["conditions"] = [{"id": 1, "name": "baza"}, {"id": 3, "name": "tokenizator"}]
    cfg["experiment"]["seeds"] = [13, 42, 1337]
    cfg["experiment"]["seeds_high_resource"] = [13, 42, 1337]
    cfg["analysis"]["reference_size"] = 2000
    cfg["run"]["priority_first"] = True
    q = build_queue(cfg, resolve_bases(cfg), only_priority=True)
    # 2 baza × 3 seed × 2 şərt
    assert len(q) == 12
    assert {b.short for b, *_ in q} == {"xlm15", "xlmr"}
    assert all(n == 2000 for _, _, n, _ in q)


def test_queue_full_gives_contrast_only_reference_size():
    from src.training.orchestrate import build_queue
    from src.utils import resolve_bases
    cfg = _two_base_cfg()
    cfg["conditions"] = [{"id": 1, "name": "baza"}]
    cfg["experiment"]["seeds"] = [13]
    cfg["experiment"]["seeds_high_resource"] = [13]
    cfg["analysis"]["reference_size"] = 2000
    cfg["run"]["priority_first"] = True
    q = build_queue(cfg, resolve_bases(cfg))
    sizes = {b.short: sorted({n for bb, _, n, _ in q if bb.short == b.short})
             for b in resolve_bases(cfg)}
    assert sizes["xlm15"] == [100, 500, 2000, 10000]
    assert sizes["xlmr"] == [2000]          # compute qənaəti qorunur


def test_turkish_cache_key_keeps_model_tokenizer_data_and_seed_distinct():
    """AZ size is reusable; base, tokenizer arm, TR data arm, and seed are not."""
    from src.training.orchestrate import build_queue, tr_stage_cache_summary
    from src.utils import resolve_bases
    cfg = _two_base_cfg()
    cfg["conditions"] = [
        {"id": 1, "name": "baza", "tokenizer": "original", "turkish": "none"},
        {"id": 2, "name": "turk", "tokenizer": "original", "turkish": "real"},
        {"id": 3, "name": "tokenizator", "tokenizer": "omp", "turkish": "none"},
        {"id": 4, "name": "her_ikisi", "tokenizer": "omp", "turkish": "real"},
        {"id": 5, "name": "turk_qarisiq", "tokenizer": "original", "turkish": "scrambled"},
    ]
    cfg["experiment"]["seeds"] = [13, 42, 1337]
    cfg["experiment"]["seeds_high_resource"] = [13, 42, 1337]
    cfg["analysis"]["reference_size"] = 2000
    cfg["run"]["priority_first"] = True
    queue = build_queue(cfg, resolve_bases(cfg))
    # 45 requested TR stages. There are 18 valid cache entries:
    # 2 bases × 3 initialization/data arms × 3 seeds. Sharing real with
    # scrambled or original with OMP would change the experiment.
    assert tr_stage_cache_summary(queue) == (45, 18)


def _schema_complete_run_result():
    return {
        "run_result_schema_version": 4,
        "n_labels": 2,
        "step_history": [
            {
                "step": 65,
                "epoch": 1.0,
                "eval_loss": 0.8,
                "eval_macro_f1": 0.5,
                "train_loss": 0.9,
                "validation_prediction_counts": {"0": 1400, "1": 1391},
            }
        ],
        "escaped": True,
        "escape_step": 65,
        "selected_step": 65,
        "selected_validation_macro_f1": 0.5,
        "terminal_step": 65,
        "terminal_validation_macro_f1": 0.5,
        "escaped_but_terminal_collapsed": False,
        "test_comparison": {
            "eval_split": "val",
            "selection": "best_validation_checkpoint",
            "matched_step": 65,
            "validation_macro_f1_matched": 0.5,
            "validation_macro_f1_selected_max": 0.5,
            "validation_macro_f1_terminal": 0.5,
            "restored_validation_macro_f1": 0.5,
            "selected_step": 65,
            "terminal_step": 65,
            "test_macro_f1": None,
            "delta_test_minus_validation_matched": None,
            "validation_drift_selected_minus_terminal": 0.0,
            "escaped_on_validation": True,
            "escaped_but_terminal_collapsed": False,
            "estimand_note": "synthetic fixture",
        },
        "max_steps": 2000,
        "completed_steps": 2000,
        "steps_per_epoch": 63,
        "effective_epochs": 2000 / 63,
        "per_run_verdict": "ESCAPED",
        "eval_split": "val",
        "training_settings": {
            "max_steps": 2000,
            "eval_every_steps": 65,
            "metric_for_best_model": None,
            "early_stopping_patience": None,
            "load_best_model_at_end": False,
        },
    }


def test_run_result_writer_persists_complete_training_evidence(tmp_path):
    import json
    from src.training.finetune import write_run_result

    path = tmp_path / "run.json"
    expected = _schema_complete_run_result()
    write_run_result(expected, path)
    assert json.loads(path.read_text(encoding="utf-8")) == expected


def test_run_result_writer_rejects_incomplete_step_history(tmp_path):
    from src.training.finetune import write_run_result

    result = _schema_complete_run_result()
    del result["step_history"][0]["validation_prediction_counts"]
    with pytest.raises(AssertionError, match="validation_prediction_counts"):
        write_run_result(result, tmp_path / "must_not_exist.json")
    assert not (tmp_path / "must_not_exist.json").exists()


# ---- cross-base comparison must retain both quantities
def _cross_cells():
    rows = []
    values = {
        ("xlm15", "baza"): (0.6, 0.70),
        ("xlm15", "tokenizator"): (1.0, 0.76),
        ("xlmr", "baza"): (1.0, 0.72),
        ("xlmr", "tokenizator"): (0.8, 0.72),
    }
    for (base, condition), (escape, conditional) in values.items():
        rows.append({"base": base, "condition": condition, "train_size": 2000,
                     "status": "MEASURED", "escape_rate": escape,
                     "conditional_macro_f1": conditional, "n_escaped": 3,
                     "source_paths": [f"{base}-{condition}.json"]})
    return rows


def test_compare_bases_compares_escape_and_conditional_quantities():
    from src.analysis.decompose import compare_bases
    result = compare_bases(_cross_cells(), 2000, _two_base_cfg())
    assert result["cross_base"]["escape_rate_effect_difference"] == pytest.approx(0.6)
    assert result["cross_base"]["conditional_macro_f1_effect_difference"] == pytest.approx(0.06)
    assert len(result["source_paths"]) == 4


def test_compare_bases_reports_missing_without_inventing_numbers():
    from src.analysis.decompose import compare_bases
    result = compare_bases(_cross_cells()[:-1], 2000, _two_base_cfg())
    assert result["status"] == "NOT MEASURED"


# ---- dedup / leakage
def test_dedup_removes_repeats_and_conflicts():
    from src.data.splits import _dedup
    recs = [{"text": "salam dünya", "label": 1},
            {"text": "  Salam   Dünya ", "label": 1},   # normalizasiyadan sonra eyni
            {"text": "ziddiyyət", "label": 0},
            {"text": "ziddiyyət", "label": 1},          # eyni mətn, fərqli etiket
            {"text": "tək qeyd", "label": 2}]
    out, st = _dedup(recs)
    texts = {r["text"] for r in out}
    assert st["duplicates_removed"] == 3
    assert st["conflicting_texts"] == 1
    assert "ziddiyyət" not in texts
    assert len(out) == 2


def test_dedup_is_idempotent():
    from src.data.splits import _dedup
    recs = [{"text": f"cümlə {i}", "label": i % 3} for i in range(50)]
    out1, st1 = _dedup(recs)
    out2, st2 = _dedup(out1)
    assert st1["duplicates_removed"] == 0
    assert len(out2) == len(out1) == 50


# ---- T6 audit qərarı: neutral sinfinin atılması (data.az.exclude_labels)
def test_apply_exclude_labels_drops_only_named_class():
    from src.data.splits import apply_exclude_labels
    recs = [{"text": "a", "label": "positive"}, {"text": "b", "label": "neutral"},
            {"text": "c", "label": "negative"}, {"text": "d", "label": "neutral"}]
    out, stats = apply_exclude_labels(recs, Config({"exclude_labels": ["neutral"]}))
    assert {r["label"] for r in out} == {"positive", "negative"}
    assert stats == {"excluded_labels": ["neutral"], "n_before": 4, "n_after": 2, "n_excluded": 2}


def test_apply_exclude_labels_noop_when_absent():
    from src.data.splits import apply_exclude_labels
    recs = [{"text": "a", "label": "positive"}, {"text": "b", "label": "neutral"}]
    out, stats = apply_exclude_labels(recs, Config({}))
    assert out == recs
    assert stats["n_excluded"] == 0


# ================================================================
#  T6 · etiket auditi — kor vərəqlərdə gold etiket sızmaması
# ================================================================
def _audit_cfg(tmp_path):
    return Config({"experiment": {"results_dir": str(tmp_path)}})


def _write_key(tmp_path, gold_by_idx: dict[int, str]) -> None:
    import json
    key = {str(i): {"text": f"mətn {i}", "gold_label_name": lab}
           for i, lab in gold_by_idx.items()}
    (tmp_path / "label_audit_key.json").write_text(
        json.dumps(key, ensure_ascii=False), encoding="utf-8")


def _write_blind_csv(tmp_path, name: str, rows: list[dict]) -> None:
    import csv
    with open(tmp_path / name, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "text", "your_label"])
        w.writeheader()
        w.writerows(rows)


def test_verify_no_leak_passes_on_blank_sheets(tmp_path):
    from src.data.audit import RESULTS_ANN1, RESULTS_ANN2, verify_no_leak
    _write_key(tmp_path, {0: "positive", 1: "negative"})
    rows = [{"idx": 0, "text": "mətn 0", "your_label": ""},
            {"idx": 1, "text": "mətn 1", "your_label": ""}]
    _write_blind_csv(tmp_path, RESULTS_ANN1, rows)
    _write_blind_csv(tmp_path, RESULTS_ANN2, rows)
    assert verify_no_leak(_audit_cfg(tmp_path))["all_clean"] is True


def test_verify_no_leak_does_not_flag_correct_annotations(tmp_path):
    """
    RÜCU TESTİ: auditorun DÜZ tapması sızma SAYILMAMALIDIR. Əvvəlki (səhv)
    dizayn `your_label == gold` olan sətirləri "sızma" kimi bildirirdi —
    bu, hər düzgün cavabı yalançı-müsbət edərdi. Struktur-yalnız yoxlama bunu
    düzəldir: gold dəyərlə üst-üstə düşmə TƏMİZ qalmalıdır.
    """
    from src.data.audit import RESULTS_ANN1, RESULTS_ANN2, verify_no_leak
    _write_key(tmp_path, {0: "positive", 1: "negative"})
    # hər iki auditor 100% DÜZ (gold ilə eyni) cavab verib — sızma DEYİL
    rows = [{"idx": 0, "text": "mətn 0", "your_label": "positive"},
            {"idx": 1, "text": "mətn 1", "your_label": "negative"}]
    _write_blind_csv(tmp_path, RESULTS_ANN1, rows)
    _write_blind_csv(tmp_path, RESULTS_ANN2, rows)
    assert verify_no_leak(_audit_cfg(tmp_path))["all_clean"] is True


def test_verify_no_leak_detects_forbidden_column(tmp_path):
    import csv
    from src.data.audit import RESULTS_ANN1, RESULTS_ANN2, verify_no_leak
    _write_key(tmp_path, {0: "positive"})
    # sızma: gold etiket AYRICA sütunda saxlanılıb
    with open(tmp_path / RESULTS_ANN1, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "text", "your_label", "label"])
        w.writeheader()
        w.writerow({"idx": 0, "text": "mətn 0", "your_label": "", "label": "positive"})
    _write_blind_csv(tmp_path, RESULTS_ANN2,
                     [{"idx": 0, "text": "mətn 0", "your_label": ""}])
    check = verify_no_leak(_audit_cfg(tmp_path))
    assert check["all_clean"] is False
    assert "label" in check["files"][RESULTS_ANN1]["forbidden_columns_found"]


def test_assert_blind_rows_unfilled_catches_export_bug():
    from src.data.audit import _assert_blind_rows_unfilled
    with pytest.raises(AssertionError):
        _assert_blind_rows_unfilled(
            [{"idx": 0, "text": "x", "your_label": ""},
             {"idx": 1, "text": "y", "your_label": "positive"}])   # bug: əvvəlcədən doldurulub


# ---- Excel round-trip: utf-8-sig BOM sızmasız açılıb-saxlanmalıdır
def test_write_csv_round_trips_azerbaijani_text_via_utf8_sig(tmp_path):
    """
    Real annotator1/2.csv-nin başına gələn bug: BOM-suz UTF-8 Excel-də sistem
    lokalı (cp1254/cp1252) kimi açılıb-saxlanır, 'ə' kimi hərflər YOX OLUR.
    `_write_csv` indi `utf-8-sig` yazır — BOM Excel-ə faylın UTF-8 olduğunu
    açıq bildirir. Bu test yazıb-oxumanın (Excel-i simulyasiya etmədən, amma
    BOM-un mövcudluğunu və `_read_csv_robust`-un onu İLK NÖVBƏDƏ seçdiyini
    təsdiqləyərək) mətni korlanmadan saxladığını yoxlayır.
    """
    from src.data.audit import BLIND_COLUMNS, _read_csv_robust, _write_csv
    text = "Bir şey çox əcaib idi, ə/ş/ç/ö/ü/ğ/ı hamısı burda: gəlmişdilər."
    path = tmp_path / "roundtrip.csv"
    _write_csv(path, [{"idx": 0, "text": text, "your_label": ""}], BLIND_COLUMNS)

    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")   # UTF-8 BOM sizin faylda mövcuddur

    rows, enc = _read_csv_robust(path)
    assert enc == "utf-8-sig"
    assert rows[0]["text"] == text   # HEÇ bir hərf korlanmayıb


# ---- test-set toxunulmazlığı: audit nümunəsi val/test ilə ÜST-ÜSTƏ düşməməlidir
def test_assert_sample_disjoint_from_passes_when_disjoint():
    from src.data.audit import assert_sample_disjoint_from
    assert_sample_disjoint_from(["salam dünya"], ["tamamilə fərqli mətn"], "test")  # raise etməməlidir


def test_assert_sample_disjoint_from_raises_on_overlap():
    """Normalizasiyadan sonra eyni olan mətn (boşluq/böyük-kiçik hərf fərqi) də tutulmalıdır."""
    from src.data.audit import assert_sample_disjoint_from
    with pytest.raises(AssertionError):
        assert_sample_disjoint_from(["Salam Dünya"], ["  salam   dünya  "], "test")


# ---- score: unclear istisnası + İKİ-ÖLÇÜLÜ qərar qaydası
def _write_scored_fixture(tmp_path, gold: list[str], ann1: list[str], ann2: list[str]) -> None:
    from src.data.audit import RESULTS_ANN1, RESULTS_ANN2
    n = len(gold)
    _write_key(tmp_path, {i: gold[i] for i in range(n)})
    rows1 = [{"idx": i, "text": f"mətn {i}", "your_label": ann1[i]} for i in range(n)]
    rows2 = [{"idx": i, "text": f"mətn {i}", "your_label": ann2[i]} for i in range(n)]
    _write_blind_csv(tmp_path, RESULTS_ANN1, rows1)
    _write_blind_csv(tmp_path, RESULTS_ANN2, rows2)


def test_score_verdict_proceed_when_both_agree_and_match_gold(tmp_path):
    from src.data.audit import score_audit
    gold = ["positive", "negative"] * 5
    _write_scored_fixture(tmp_path, gold, list(gold), list(gold))
    out = score_audit(_audit_cfg(tmp_path))
    assert out["decision_gate"]["verdict"] == "PROCEED"


def test_score_verdict_task_too_subjective_when_hh_below_60(tmp_path):
    """
    HH < 60%-dirsə (auditorlar bir-biri ilə DƏ razılaşmır), qərar
    STOP_TASK_TOO_SUBJECTIVE olmalıdır — HD-nin nə olduğundan ASILI OLMAYARAQ.
    (DÜZƏLDİLMİŞ eşik: köhnə versiya 85% işlədirdi, amma bu, İSTƏNİLƏN
    3-sinifli duyğu tapşırığını rədd edərdi — bax HANDOFF §3.11/§3.12.)
    """
    from src.data.audit import score_audit
    gold = ["positive", "negative"] * 5           # 10 sətir, alternativ
    ann1 = list(gold)                              # gold ilə TAM üst-üstə (HD1=100%)
    ann2 = ["positive"] * 10                       # HD2=50%, VƏ ann1 ilə YALNIZ 50% üst-üstə (HH=50%)
    _write_scored_fixture(tmp_path, gold, ann1, ann2)
    out = score_audit(_audit_cfg(tmp_path))
    assert out["human_human_agreement_pct"] == pytest.approx(50.0, abs=0.1)
    assert out["decision_gate"]["verdict"] == "STOP_TASK_TOO_SUBJECTIVE"


def test_score_verdict_labels_unreliable_when_hd_far_below_hh(tmp_path):
    """
    HH yüksəkdir (auditorlar bir-biri ilə TAM razılaşır, tapşırıq yaxşı
    təyin olunub) AMMA hər ikisinin dataset-lə uzlaşması HH-dən 12pp-dən
    ÇOX aşağıdır — bu, 'dataset-in etiketləri etibarsızdır' deməkdir,
    STOP_TASK_TOO_SUBJECTIVE-dən FƏRQLİ diaqnoz və FƏRQLİ düzəliş yolu.
    """
    from src.data.audit import score_audit
    gold = ["positive", "negative"] * 5            # 10 sətir, alternativ
    ann1 = ["negative"] * 10                        # HD1=50%
    ann2 = ["negative"] * 10                        # HD2=50%, HH(ann1,ann2)=100%
    _write_scored_fixture(tmp_path, gold, ann1, ann2)
    out = score_audit(_audit_cfg(tmp_path))
    assert out["human_human_agreement_pct"] == pytest.approx(100.0, abs=0.1)
    # HD1/HD2 AYRI bildirilir (ortalanmır) — hər ikisi 50% olmalıdır
    assert out["annotator1_vs_gold_pct"] == pytest.approx(50.0, abs=0.1)
    assert out["annotator2_vs_gold_pct"] == pytest.approx(50.0, abs=0.1)
    # HD-HH = 50-100 = -50pp, hər ikisi üçün < -12pp
    assert out["decision_gate"]["tier_annotator1"] == "STOP_LABELS_UNRELIABLE"
    assert out["decision_gate"]["tier_annotator2"] == "STOP_LABELS_UNRELIABLE"
    assert out["decision_gate"]["verdict"] == "STOP_LABELS_UNRELIABLE"


def test_score_verdict_proceed_with_caveat_when_one_annotator_10pp_below_hh(tmp_path):
    """
    YEKUN (bağlayıcı) verdikt İKİ tier-dən PİS olanıdır — 'zəif həlqə
    bağlayıcıdır' prinsipi qorunur, sadəcə indi RELATİV (HH-ə nisbətdə)
    ölçülür, MÜTLƏQ hədlə yox. Konstruksiya: annotator1 HH-dən +20pp yuxarı
    (aydın PROCEED), annotator2 HH-dən −10pp aşağı (−12..−5pp zolağı,
    PROCEED_WITH_CAVEAT) — YEKUN nəticə PROCEED_WITH_CAVEAT olmalıdır,
    tək annotator1-ə baxıb PROCEED deyilməməlidir.
    """
    from src.data.audit import score_audit
    gold = ["a"] * 10 + ["b"] * 10                 # idx0-9="a", idx10-19="b"
    ann1 = list(gold)
    ann1[0] = ann1[1] = "b"                                             # HD1 = 18/20 = 90%
    ann2 = list(gold)
    for i in range(8):                                                  # idx0-7 → "b"
        ann2[i] = "b"                                                   # HD2 = 12/20 = 60%
    _write_scored_fixture(tmp_path, gold, ann1, ann2)
    out = score_audit(_audit_cfg(tmp_path))
    # HH: yalnız idx2-7 fərqlənir (ann1 unflipped "a", ann2 flipped "b") → 14/20 = 70%
    assert out["human_human_agreement_pct"] == pytest.approx(70.0, abs=0.1)
    assert out["annotator1_vs_gold_pct"] == pytest.approx(90.0, abs=0.1)
    assert out["annotator2_vs_gold_pct"] == pytest.approx(60.0, abs=0.1)
    assert out["decision_gate"]["tier_annotator1"] == "PROCEED"          # diff = 90-70 = +20pp
    assert out["decision_gate"]["tier_annotator2"] == "PROCEED_WITH_CAVEAT"  # diff = 60-70 = -10pp
    assert out["decision_gate"]["verdict"] == "PROCEED_WITH_CAVEAT"      # ikisindən PİS olan bağlayıcıdır


def test_score_excludes_unclear_from_both_denominators(tmp_path):
    from src.data.audit import score_audit
    gold = ["positive"] * 8 + ["negative"] * 2
    ann1 = ["positive"] * 8 + ["unclear", "unclear"]        # 2 unclear
    ann2 = ["positive"] * 8 + ["negative", "negative"]      # unclear yoxdur, hamısı düz
    _write_scored_fixture(tmp_path, gold, ann1, ann2)
    out = score_audit(_audit_cfg(tmp_path))
    # ann1: 2 sətir 'unclear' — məxrəcdən atılıb, qalan 8/8 düz
    assert out["annotator1_vs_gold"]["n_unclear_excluded"] == 2
    assert out["annotator1_vs_gold"]["n"] == 8
    assert out["annotator1_vs_gold"]["accuracy"] == pytest.approx(1.0)
    assert out["annotator1_vs_gold"]["unclear_rate_pct"] == pytest.approx(20.0, abs=0.1)
    assert out["annotator1_vs_gold"]["text_quality_concern"] is True   # 20% > 15% hədd
    # ann2: unclear yoxdur, tam məxrəc
    assert out["annotator2_vs_gold"]["n_unclear_excluded"] == 0
    assert out["annotator2_vs_gold"]["n"] == 10
    # auditor↔auditor: ann1-in 2 'unclear'-i olan cüt HH məxrəcindən də atılmalıdır
    assert out["annotator1_vs_annotator2"]["n_excluded_either_unclear"] == 2
    assert out["annotator1_vs_annotator2"]["n"] == 8


# ---- real vərəqlərin formatı: RƏQƏMLƏR (1/2/3) + 'bilmirəm' (ad/unclear yox)
def test_score_accepts_digit_labels_and_bilmirem(tmp_path):
    """
    Real doldurulmuş vərəqlər ad (positive/…) yox, RƏQƏM (1/2/3) və bir dənə
    'bilmirəm' istifadə edib. Scorer bunu ADLARLA EYNİ nəticəyə gətirməlidir —
    DIGIT_LABEL_MAP (1=positive, 2=neutral, 3=negative) vasitəsilə.
    """
    from src.data.audit import score_audit
    gold = ["positive", "negative", "neutral", "positive", "negative"]
    ann1_digits = ["1", "3", "2", "1", "3"]              # gold ilə TAM üst-üstə (rəqəmlə)
    ann2_digits = ["1", "3", "bilmirəm", "1", "1"]       # 1 abstention + 1 səhv (idx4: 1≠3)
    _write_scored_fixture(tmp_path, gold, ann1_digits, ann2_digits)
    out = score_audit(_audit_cfg(tmp_path))

    assert out["annotator1_vs_gold_pct"] == pytest.approx(100.0, abs=0.1)
    assert out["annotator2_vs_gold"]["n_unclear_excluded"] == 1     # idx2 'bilmirəm'
    assert out["annotator2_vs_gold"]["n"] == 4                     # 5-1 (unclear atılıb)
    assert out["annotator2_vs_gold"]["n_correct"] == 3              # idx0,1,3 düz; idx4 səhv (1≠3)
    assert out["annotator2_vs_gold_pct"] == pytest.approx(75.0, abs=0.1)


def test_is_unclear_catches_encoding_mangled_bilmirem():
    """
    RÜCU TESTİ (real annotator2.csv-də tapılan bug): Excel-in lossy
    kodlaşdırması "bilmirəm"-i "bilmir?m"-ə çevirir ('ə' xəritələnə bilmir,
    hərfi '?' olur). Bu, UNCLEAR_VALUES-də DƏQİQ uyğun gəlmir — ön-şəkilçi
    yoxlaması bunu tutmalıdır, əks halda bu sətir səhvən "cavab" kimi
    sayılıb gold ilə müqayisə edilərdi.
    """
    from src.data.audit import _is_unclear
    assert _is_unclear("bilmir?m") is True
    assert _is_unclear("Bilmir?m") is True
    assert _is_unclear("bilmirəm") is True
    assert _is_unclear("positive") is False
    assert _is_unclear("1") is False


# ---- T1: rəqəm↔ad xəritəsinin real (probe) datadan yoxlanılması
def test_build_label_mapping_verifies_against_real_probes(tmp_path):
    from src.data.audit import (LABEL_MAPPING_PROBES, RESULTS_ANN1, RESULTS_ANN2,
                                build_label_mapping)
    gold_by_idx = {p["idx"]: p["expected_gold"] for p in LABEL_MAPPING_PROBES}
    _write_key(tmp_path, gold_by_idx)
    rows = [{"idx": p["idx"], "text": f"mətn {p['idx']}", "your_label": p["expected_digit"]}
           for p in LABEL_MAPPING_PROBES]
    _write_blind_csv(tmp_path, RESULTS_ANN1, rows)
    _write_blind_csv(tmp_path, RESULTS_ANN2, rows)
    mapping = build_label_mapping(_audit_cfg(tmp_path))
    assert mapping["fully_verified_both_annotators"] is True
    assert all(p["gold_matches_expected"] for p in mapping["probes"])


def test_build_label_mapping_flags_missing_sheet():
    """ann2 boşdursa (real vəziyyətimizdə olduğu kimi), xəritə QİSMİ təsdiqlənməlidir."""
    import tempfile
    from pathlib import Path as _Path
    from src.data.audit import (LABEL_MAPPING_PROBES, RESULTS_ANN1, RESULTS_ANN2,
                                build_label_mapping)
    with tempfile.TemporaryDirectory() as d:
        tmp_path = _Path(d)
        gold_by_idx = {p["idx"]: p["expected_gold"] for p in LABEL_MAPPING_PROBES}
        _write_key(tmp_path, gold_by_idx)
        rows = [{"idx": p["idx"], "text": f"mətn {p['idx']}", "your_label": p["expected_digit"]}
               for p in LABEL_MAPPING_PROBES]
        _write_blind_csv(tmp_path, RESULTS_ANN1, rows)
        # ann2 YAZILMIR — real vəziyyəti simulyasiya edir (boş/mövcud deyil)
        empty_rows = [{"idx": p["idx"], "text": f"mətn {p['idx']}", "your_label": ""}
                     for p in LABEL_MAPPING_PROBES]
        _write_blind_csv(tmp_path, RESULTS_ANN2, empty_rows)
        mapping = build_label_mapping(_audit_cfg(tmp_path))
        assert mapping["fully_verified_both_annotators"] is False
        assert mapping["probes"][0]["annotators"][RESULTS_ANN2]["available"] is False


# ---- neutral-sərhəd vs qütb-dəyişməsi ayrışdırması (bookkeeping deyil, ölçü)
def test_decompose_disagreements_splits_neutral_boundary_vs_polarity_flip():
    from src.data.audit import _decompose_disagreements
    a = ["positive", "positive", "negative", "neutral", "positive"]
    b = ["positive", "neutral", "positive", "neutral", "negative"]
    #     match      neutral-b   polarity-f   match      polarity-f
    out = _decompose_disagreements(a, b)
    assert out["n_agreements"] == 2
    assert out["n_disagreements"] == 3
    assert out["neutral_boundary"] == 1
    assert out["polarity_flip"] == 2


# ================================================================
#  T5 · küy × şərt qarşılıqlı təsiri (Fisher dəqiq test) — saf funksiya
# ================================================================
def _mk_item(has_tr_specific: bool, agrees: bool) -> dict:
    return {"has_turkish_specific_token": has_tr_specific, "agrees_with_dataset": agrees}


def test_run_fisher_test_no_association_when_independent():
    """Türk-spesifik token varlığı razılaşma ilə ƏLAQƏSİZDİRSƏ, p ≥ 0.05 olmalıdır."""
    from src.analysis.noise_interaction import run_fisher_test
    items = (
        [_mk_item(True, True)] * 10 + [_mk_item(True, False)] * 10 +
        [_mk_item(False, True)] * 10 + [_mk_item(False, False)] * 10
    )  # 2x2 tam balanslı → əlaqə yoxdur
    out = run_fisher_test(items)
    assert out["association_found"] is False
    assert out["fisher_p_value"] >= 0.05


def test_run_fisher_test_detects_association_when_confounded():
    """Uyğunsuzluq demək olar TAMAMİLƏ türk-spesifik tokenli elementlərdə cəmləşibsə, p < 0.05."""
    from src.analysis.noise_interaction import run_fisher_test
    items = (
        [_mk_item(True, False)] * 18 + [_mk_item(False, False)] * 2 +      # uyğunsuzluq: 18 türk-spesifik, 2 yox
        [_mk_item(True, True)] * 2 + [_mk_item(False, True)] * 18          # uyğunluq: əksinə
    )
    out = run_fisher_test(items)
    assert out["association_found"] is True
    assert out["fisher_p_value"] < 0.05


# ================================================================
#  T7 · donor embedding seçimi — trust_remote_code OLMADAN, forma+tie-break
# ================================================================
def test_select_embedding_tensor_picks_unique_shape_match():
    from src.transplant.donor_embeddings import select_embedding_tensor
    state = {
        "embedding.word_embedding.weight": np.zeros((100, 16)),
        "transformer.layers.0.attention.out_proj.weight": np.zeros((16, 16)),
        "transformer.layers.0.mlp.mlp.1.weight": np.zeros((64, 16)),
    }
    key, tensor = select_embedding_tensor(state, vocab_size=100)
    assert key == "embedding.word_embedding.weight"
    assert tensor.shape == (100, 16)


def test_select_embedding_tensor_resolves_real_hplt_style_ambiguity():
    """
    Real hal (HPLT/hplt_bert_base_az-də tapıldı): İKİ tensor EYNİ forma
    (giriş embedding VƏ MLM çıxış klassifikatoru). Forma TƏK-BAŞINA ayırd
    edə bilmir — açar-söz tie-break ('embed' var, 'classifier' yoxdur)
    düzgün namizədi seçməlidir.
    """
    from src.transplant.donor_embeddings import select_embedding_tensor
    state = {
        "embedding.word_embedding.weight": np.ones((32768, 768)),
        "classifier.nonlinearity.5.weight": np.full((32768, 768), 2.0),   # eyni forma, DEKOY
        "transformer.layers.0.attention.out_proj.weight": np.zeros((768, 768)),
    }
    key, tensor = select_embedding_tensor(state, vocab_size=32768)
    assert key == "embedding.word_embedding.weight"
    assert (tensor == 1.0).all()   # DEKOY (dolu 2.0) DEYİL, düzgün tensor qaytarılıb


def test_select_embedding_tensor_raises_on_true_ambiguity():
    """Tie-break DƏ ayırd edə bilmirsə (heç birində 'embed'/'classifier' yoxdur), raise etməlidir."""
    from src.transplant.donor_embeddings import select_embedding_tensor
    state = {
        "foo.weight": np.zeros((100, 16)),
        "bar.weight": np.zeros((100, 16)),
    }
    with pytest.raises(ValueError, match="forma.*UYĞUN"):
        select_embedding_tensor(state, vocab_size=100)


def test_select_embedding_tensor_raises_when_no_match():
    from src.transplant.donor_embeddings import select_embedding_tensor
    state = {"transformer.layers.0.attention.out_proj.weight": np.zeros((16, 16))}
    with pytest.raises(ValueError, match="Heç bir tensor"):
        select_embedding_tensor(state, vocab_size=100)
