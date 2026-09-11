"""
EMBEDDİNG NORM HESABATI  —  C1a-nın yanında TƏLƏB OLUNAN yoxlama

Niyə lazımdır (T7-də tapılan real hadisə, bax HANDOFF.md §3.16): `omp.py`-də
bir miqyas-qarışdırma bug-u xlm15-in transplant olunmuş modelinin BPC-sini
8.7-dən 161-ə (60× pisləşmə) apardı. NƏ reconstruction cosine (donor
fəzasında ölçülür, miqyasdan ASILI DEYİL), NƏ DƏ C1a (donor==baza olanda
iki fəzanın miqyası ZATƏN eynidir — bu bug sinfini HEÇ VAXT TUTA BİLMƏZ)
bunu aşkar edə bilmədi. YALNIZ C1c (BPC) VƏ bu modul (normların BİRBAŞA
müqayisəsi) tuta bildi/bilər.

Nəticə: reconstruction cosine transplant keyfiyyətini yoxlamaq üçün ZƏRURİDİR,
AMMA KİFAYƏT DEYİL — miqyas ayrıca yoxlanılmalıdır, hər dəfə.

İşlətmə (build.py-nin ÖZÜNDƏN, avtomatik) VƏ ya post-hoc:
    python -m src.transplant.check_embedding_norms --config configs/experiment.yaml
"""
from __future__ import annotations

import numpy as np


def _row_norm_stats(x: np.ndarray) -> dict:
    n = np.linalg.norm(x, axis=1)
    return {
        "n": int(len(n)),
        "median": round(float(np.median(n)), 4),
        "mean": round(float(n.mean()), 4),
        "std": round(float(n.std()), 4),
        "min": round(float(n.min()), 4),
        "max": round(float(n.max()), 4),
    }


def compute_norm_report(E_base: np.ndarray, E_donor: np.ndarray, E_new: np.ndarray,
                        anchor_donor_ids: np.ndarray, unfamiliar_ids: np.ndarray,
                        healthy_ratio_range: tuple[float, float] = (0.85, 1.15),
                        ) -> dict:
    """
    Baza/donor/yeni matrislərin sətir-norm paylanmasını müqayisə edir, VƏ
    yeni matrisin ÖZÜNDƏ anchor (birbaşa köçürülmüş) sətirləri ilə OMP-
    reconstruction olunmuş sətirləri ayırır. `reconstructed_to_anchor_
    median_ratio` ~1.0-dan çox kənarlaşarsa (aşağı — pis fit sıfıra doğru
    "büzülür"; yuxarı — köhnə miqyas-bug-u kimi şişmə), bu, kobud miqyas
    problemini erkən aşkar edir.
    """
    new_anchor = _row_norm_stats(E_new[anchor_donor_ids])
    new_recon = _row_norm_stats(E_new[unfamiliar_ids])
    ratio = new_recon["median"] / max(new_anchor["median"], 1e-8)
    lo, hi = healthy_ratio_range

    return {
        "base_all_rows": _row_norm_stats(E_base),
        "donor_all_rows": _row_norm_stats(E_donor),
        "new_anchor_rows": new_anchor,
        "new_reconstructed_rows": new_recon,
        "reconstructed_to_anchor_median_ratio": round(float(ratio), 4),
        "healthy_ratio_range": list(healthy_ratio_range),
        "in_healthy_range": bool(lo <= ratio <= hi),
    }
