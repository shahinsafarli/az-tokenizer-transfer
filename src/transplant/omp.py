"""
OMP (Orthogonal Matching Pursuit) ilə embedding transplantı — NÜVƏ ALQORİTM

Məntiq (Petrov et al., arXiv:2506.06607):

  1. DONOR fəzasında: tanış olmayan tokenin embedding-ini ortaq (anchor)
     tokenlərin SEYRƏK xətti kombinasiyası kimi ifadə et:

         E_donor[t]  ≈  Σ  w_i · E_donor[anchor_i]      (||w||_0 ≤ k)

  2. Həmin ƏMSALLARI baza modelin fəzasına tətbiq et:

         E_base_new[t]  =  Σ  w_i · E_base[anchor_i]

  Köçürülən VEKTOR deyil, ƏMSALLARDIR — buna görə donor və baza modelin
  hidden ölçüləri (d_d ≠ d_b) fərqli ola bilər.

Praktik optimizasiya: bütün anchorlar üzərində OMP çox yavaşdır. Hər hədəf
üçün əvvəlcə kosinus yaxınlığa görə ən yaxın `n_candidates` anchor seçilir,
OMP yalnız onların üzərində işləyir.

Ablasiya üçün üç metod:
    omp          — əsas metod
    mean         — sadə orta (ədəbiyyatda "degrade edir" deyilən baza)
    random_coef  — eyni seyrəklikdə TƏSADÜFİ əmsallar  [Nəzarət C1]
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit

from src.utils import get_logger

log = get_logger(__name__)


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps), norms


def solve_coefficients(
    target: np.ndarray,          # (d_d,)
    candidates: np.ndarray,      # (n_cand, d_d)
    k: int,
    method: str = "omp",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Bir hədəf üçün (n_cand,) ölçülü seyrək əmsal vektoru qaytarır."""
    n_cand = candidates.shape[0]
    k_eff = int(min(k, n_cand, candidates.shape[1]))

    if method == "mean":
        w = np.zeros(n_cand, dtype=np.float32)
        w[:k_eff] = 1.0 / k_eff          # ən yaxın k namizədin sadə ortası
        return w

    if method == "random_coef":
        rng = rng or np.random.default_rng(0)
        w = np.zeros(n_cand, dtype=np.float32)
        idx = rng.choice(n_cand, size=k_eff, replace=False)
        vals = rng.normal(size=k_eff).astype(np.float32)
        w[idx] = vals / (np.abs(vals).sum() + 1e-8)
        return w

    if method == "omp":
        # sklearn: fit(X, y),  X: (n_samples, n_features)
        # bizdə n_samples = d_d (ölçü),  n_features = n_cand (anchorlar)
        model = OrthogonalMatchingPursuit(
            n_nonzero_coefs=k_eff, fit_intercept=False
        )
        model.fit(candidates.T.astype(np.float64), target.astype(np.float64))
        return model.coef_.astype(np.float32)

    raise ValueError(f"naməlum metod: {method}")


def transplant_embeddings(
    E_donor: np.ndarray,          # (V_d, d_d)  donor embedding matrisi
    E_base: np.ndarray,           # (V_b, d_b)  baza embedding matrisi
    anchor_donor_ids: np.ndarray,  # (n_anchor,) donor tərəfdəki id-lər
    anchor_base_ids: np.ndarray,   # (n_anchor,) HƏMİN anchorların baza id-ləri
    unfamiliar_donor_ids: np.ndarray,  # (n_unf,) donorda var, bazada yox
    k: int = 64,
    n_candidates: int = 256,
    method: str = "omp",
    normalize: bool = True,
    batch_size: int = 256,
    seed: int = 0,
) -> np.ndarray:
    """
    Tanış olmayan tokenlər üçün (n_unf, d_b) ölçülü yeni embedding matrisi.
    """
    assert len(anchor_donor_ids) == len(anchor_base_ids), \
        "anchor cütləri eyni uzunluqda olmalıdır"
    rng = np.random.default_rng(seed)

    A_d = E_donor[anchor_donor_ids].astype(np.float32)   # (n_anchor, d_d)
    A_b = E_base[anchor_base_ids].astype(np.float32)     # (n_anchor, d_b)
    n_anchor, d_b = A_d.shape[0], A_b.shape[1]
    n_unf = len(unfamiliar_donor_ids)

    log.info("Transplant: metod=%s  k=%d  anchor=%d  tanış_olmayan=%d",
             method, k, n_anchor, n_unf)

    # Normalizasiya: miqyas fərqi əmsalları pozur.
    if normalize:
        A_d_n, _ = _l2_normalize(A_d)
        A_d_search = A_d_n
    else:
        A_d_n = A_d
        A_d_search = A_d

    out = np.zeros((n_unf, d_b), dtype=np.float32)
    n_cand = int(min(n_candidates, n_anchor))

    # DİQQƏT: OMP yalnız namizəd sayı < embedding ölçüsü olduqda etibarlı işləyir.
    # Əks halda lüğət "overcomplete" olur, greedy seçim səhv atom götürür və
    # yenidənqurma pozulur (bu, testlə də təsdiqlənib).
    d_d = A_d.shape[1]
    if n_cand >= d_d:
        log.warning(
            "n_candidates (%d) >= donor ölçüsü (%d) — OMP overcomplete rejimdədir "
            "və dəqiqliyi düşür. transplant.n_candidates dəyərini %d-dən kiçildin.",
            n_cand, d_d, d_d)

    for start in range(0, n_unf, batch_size):
        stop = min(start + batch_size, n_unf)
        ids = unfamiliar_donor_ids[start:stop]
        T = E_donor[ids].astype(np.float32)               # (b, d_d)

        # `normalize` YALNIZ ən-yaxın-namizəd axtarışını (kosinus) və OMP-nin
        # DONOR fəzasındakı sabitliyini yaxşılaşdırır — nəticə vektorunun
        # miqyasına aid deyil (bax aşağıdakı əsas qeyd).
        T_n = _l2_normalize(T)[0] if normalize else T

        # Ən yaxın namizədlər (kosinus) — hesablamanı kəskin azaldır
        sims = T_n @ A_d_search.T                          # (b, n_anchor)
        top = np.argpartition(-sims, kth=n_cand - 1, axis=1)[:, :n_cand]

        # ---- DÜZƏLİŞ 2026-09-08 · namizədlər SIRALANIR -------------------
        # `np.argpartition` YALNIZ "bu n_cand element ən yaxın n_cand-dır"
        # zəmanəti verir — ARALARINDAKI SIRA İSƏ İXTİYARİDİR. `omp` və
        # `random_coef` üçün bu əhəmiyyətsizdir (biri bütün namizəd dəsti
        # üzərində həll edir, digəri dəstdən təsadüfi seçir), amma `mean`
        # metodu `w[:k] = 1/k` edir — yəni SIRALANMAMIŞ 256-lığın İLK 64-nü
        # götürür. Nəticədə "ən yaxın 64 anchorun ortası" adlanan kontrol
        # əslində "ən yaxın 256-dan ixtiyari 64-nün ortası" olurdu, və
        # sənədləşdirilmiş tərifi ödəmirdi. Bir sətirlik sıralama bunu
        # düzəldir; OMP/random nəticəsini DƏYİŞMİR (sıra onlara təsir etmir).
        order = np.argsort(-np.take_along_axis(sims, top, axis=1), axis=1)
        top = np.take_along_axis(top, order, axis=1)

        for j in range(T.shape[0]):
            cand_idx = top[j]
            w = solve_coefficients(
                target=T_n[j], candidates=A_d_n[cand_idx],
                k=k, method=method, rng=rng,
            )
            # ƏMSAL KÖÇÜRMƏSİ — `w` DONOR fəzasında (vahid-normallaşdırılmış
            # anchor-larla) həll edilib, İNDİ isə EYNİ əmsallar BAZA
            # anchor-larına (A_b, ÖZ təbii miqyasında, NORMALLAŞDIRILMAMIŞ)
            # tətbiq olunur. Nəticə artıq BAZA fəzasının təbii miqyasındadır —
            # ƏLAVƏ miqyaslamaya EHTİYAC YOXDUR.
            #
            # DÜZƏLDİLMİŞ SƏHV (T7-də tapıldı): əvvəlki kod bu vektoru
            # DONOR hədəfinin normuna (`T_norms[j]`) yenidən miqyaslayırdı —
            # amma bu, YANLIŞ fəzanın normudur. Donor (HPLT) embedding
            # normları ~13.9 (orta), baza (XLM-15) isə ~0.6 — TAMAMILƏ FƏRQLİ
            # miqyaslar. Bu səhv nəticə matrisinin normunu ~18× ŞİŞİRDİRDİ,
            # transformerin dondurulmuş qatları GÖZLƏMƏDİYİ miqyasda giriş
            # alıb, MLM BPC-ni fəlakətli dərəcədə pozurdu (8.7 → 161 bit/hərf,
            # C1c nəzarəti bunu TUTDU — bax HANDOFF.md). `normalize=True`
            # YALNIZ OMP-nin özünün (donor fəzasında) daha sabit işləməsi
            # üçündür — nəticə vektorunun miqyasına AİD DEYİL.
            vec = w @ A_b[cand_idx]                        # (d_b,) — artıq baza miqyasında
            out[start + j] = vec

        if (start // batch_size) % 20 == 0:
            log.info("  ... %d/%d", stop, n_unf)

    return out


def reconstruction_error(
    E_donor: np.ndarray, anchor_donor_ids: np.ndarray,
    unfamiliar_donor_ids: np.ndarray, k: int, n_candidates: int,
    sample: int = 500, seed: int = 0,
) -> dict:
    """
    Diaqnostika: DONOR fəzasında yenidənqurma xətası nə qədərdir?
    Yüksəkdirsə, OMP hədəfləri yaxşı ifadə edə bilmir → k və ya n_candidates artırın.
    """
    rng = np.random.default_rng(seed)
    A_d = E_donor[anchor_donor_ids].astype(np.float32)
    A_d_n, _ = _l2_normalize(A_d)
    pick = rng.choice(len(unfamiliar_donor_ids),
                      size=int(min(sample, len(unfamiliar_donor_ids))), replace=False)
    T = E_donor[unfamiliar_donor_ids[pick]].astype(np.float32)
    T_n, _ = _l2_normalize(T)
    n_cand = int(min(n_candidates, A_d.shape[0]))

    errs, coss = [], []
    for j in range(T_n.shape[0]):
        sims = T_n[j] @ A_d_n.T
        cand_idx = np.argpartition(-sims, kth=n_cand - 1)[:n_cand]
        w = solve_coefficients(T_n[j], A_d_n[cand_idx], k, "omp")
        rec = w @ A_d_n[cand_idx]
        errs.append(float(np.linalg.norm(rec - T_n[j])))
        denom = np.linalg.norm(rec) * np.linalg.norm(T_n[j]) + 1e-8
        coss.append(float(rec @ T_n[j] / denom))
    return {
        "n_sampled": len(errs),
        "mean_l2_error": round(float(np.mean(errs)), 4),
        "mean_cosine": round(float(np.mean(coss)), 4),
        "p10_cosine": round(float(np.percentile(coss, 10)), 4),
    }
