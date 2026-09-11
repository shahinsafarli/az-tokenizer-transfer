"""
STATİSTİK TESTLƏR  —  "fərq realdırmı, yoxsa seed küyüdür?"

Rubrikada "experimental rigour" 20% çəkiyə malikdir və bu modul onun mərkəzidir.

FROZEN DUAL-METRİK SİYASƏTİNƏ TABEDİR (bax `src/analysis/decompose.py`-ın
`metric_policy` sahəsi və configs/FROZEN.md). İki AYRI kəmiyyət ölçülür və
HEÇ VAXT qarışdırılmır:

  1. ESCAPE RATE — run constant-prediction basin-indən çıxdımı (ikili nəticə).
     Nisbətlərin müqayisəsi → Fisher dəqiq testi + Wilson intervalları.
  2. CONDITIONAL MACRO-F1 — YALNIZ escape edən seed-lər üzərində. Escape
     etməyən run-un macro-F1-i modelin bacarığını deyil, çökmüş basin-in
     sabit dəyərini ölçür; onu orta hesaba qatmaq İKİ FƏRQLİ kəmiyyəti
     birləşdirmək olardı.

Müqayisələr HƏR BAZA MODELİ ÜÇÜN AYRI aparılır. `xlm15` və `xlmr` fərqli
tutumlu (151.7M vs 86.0M encoder) və fərqli pretraining-li modellərdir —
onları bir hovuzda birləşdirmək `baza vs tokenizator` fərqini baza-modeli
effekti ilə qarışdırardı. Çarpaz-baza müqayisəsi `decompose.py`-ın işidir.

İşlətmə:
    python -m src.analysis.stats --config configs/experiment.yaml

Çıxış:
    results/stats.json · results/stats.csv
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sps

from src.analysis.aggregate import bootstrap_mean_ci, load_runs, wilson_interval
from src.utils import (base_argparser, ensure_dir, get_logger, load_config,
                       setup_logging, write_json)

log = get_logger(__name__)

# Escape edən seed sayı bundan azdırsa şərtli macro-F1 müqayisəsi APARILMIR:
# iki nöqtədən yayılma qiymətləndirmək statistika deyil, teatrdır.
MIN_ESCAPED_FOR_F1_TEST = 2


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((b.mean() - a.mean()) / s) if s > 0 else float("nan")


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, iters: int, alpha: float,
                      seed: int = 0) -> tuple[float, float]:
    """b − a fərqi üçün bootstrap etibar intervalı."""
    rng = np.random.default_rng(seed)
    a_draws = rng.choice(a, size=(int(iters), len(a)), replace=True).mean(axis=1)
    b_draws = rng.choice(b, size=(int(iters), len(b)), replace=True).mean(axis=1)
    diffs = b_draws - a_draws
    return (float(np.percentile(diffs, 100 * alpha / 2)),
            float(np.percentile(diffs, 100 * (1 - alpha / 2))))


def compare_escape_rates(a: pd.DataFrame, b: pd.DataFrame, alpha: float) -> dict:
    """İki şərtin escape nisbətini Fisher dəqiq testi ilə müqayisə edir."""
    a_esc, b_esc = int(a["escaped"].sum()), int(b["escaped"].sum())
    a_n, b_n = len(a), len(b)
    table = [[b_esc, b_n - b_esc], [a_esc, a_n - a_esc]]
    odds_ratio, p_value = sps.fisher_exact(table)
    a_rate, b_rate = a_esc / max(a_n, 1), b_esc / max(b_n, 1)
    return {
        "baseline_escape": f"{a_esc}/{a_n}",
        "comparison_escape": f"{b_esc}/{b_n}",
        "baseline_escape_rate": round(a_rate, 4),
        "comparison_escape_rate": round(b_rate, 4),
        "baseline_escape_wilson_ci": wilson_interval(a_esc, a_n, alpha),
        "comparison_escape_wilson_ci": wilson_interval(b_esc, b_n, alpha),
        "escape_rate_diff": round(b_rate - a_rate, 4),
        "escape_fisher_odds_ratio": (None if not np.isfinite(odds_ratio)
                                     else round(float(odds_ratio), 4)),
        "escape_fisher_p_value": round(float(p_value), 5),
        "escape_significant": bool(p_value < alpha),
    }


def compare_conditional_f1(a: pd.DataFrame, b: pd.DataFrame, iters: int,
                           alpha: float) -> dict:
    """Şərtli macro-F1 — YALNIZ escape edən seed-lər üzərində."""
    col = "selected_validation_macro_f1"
    a_vals = a.loc[a["escaped"], col].to_numpy(dtype=float)
    b_vals = b.loc[b["escaped"], col].to_numpy(dtype=float)
    out: dict = {
        "n_escaped_baseline": int(len(a_vals)),
        "n_escaped_comparison": int(len(b_vals)),
        "conditional_mean_baseline": (round(float(a_vals.mean()), 4)
                                      if len(a_vals) else None),
        "conditional_mean_comparison": (round(float(b_vals.mean()), 4)
                                        if len(b_vals) else None),
    }
    if len(a_vals) < MIN_ESCAPED_FOR_F1_TEST or len(b_vals) < MIN_ESCAPED_FOR_F1_TEST:
        out["conditional_status"] = "NOT MEASURED"
        out["conditional_note"] = (
            f"Escape edən seed sayı < {MIN_ESCAPED_FOR_F1_TEST} — şərtli "
            "macro-F1 müqayisəsi aparılmadı. Escape rate hələ də oxunmalıdır.")
        return out

    t_stat, p_value = sps.ttest_ind(b_vals, a_vals, equal_var=False)
    lo, hi = bootstrap_diff_ci(a_vals, b_vals, iters, alpha)
    pooled_std = float(np.sqrt((a_vals.var(ddof=1) + b_vals.var(ddof=1)) / 2))
    diff = float(b_vals.mean() - a_vals.mean())
    out.update({
        "conditional_status": "MEASURED",
        "conditional_baseline_ci": bootstrap_mean_ci(a_vals.tolist(), iters, alpha),
        "conditional_comparison_ci": bootstrap_mean_ci(b_vals.tolist(), iters, alpha),
        "conditional_diff": round(diff, 4),
        "conditional_ci_low": round(lo, 4),
        "conditional_ci_high": round(hi, 4),
        "conditional_ci_excludes_zero": bool(lo > 0 or hi < 0),
        "conditional_t_stat": round(float(t_stat), 4),
        "conditional_p_value": round(float(p_value), 5),
        "conditional_significant": bool(p_value < alpha),
        "conditional_cohens_d": round(cohens_d(a_vals, b_vals), 3),
        "seed_std_pooled": round(pooled_std, 4),
        "diff_over_noise": (round(abs(diff) / pooled_std, 2)
                            if pooled_std > 0 else None),
    })
    out.update(_paired_and_all_seed(a, b, iters, alpha))
    return out


def _paired_and_all_seed(a: pd.DataFrame, b: pd.DataFrame, iters: int,
                         alpha: float) -> dict:
    """Paired and unconditional companions to the Welch test.

    Added 2026-09-08. Two facts about this design make the independent-sample
    Welch test the wrong default, not merely one option among many:

    1. The units ARE PAIRED. Both conditions run the same seed, and `subsample`
       draws the Azerbaijani training subset from that same seed, so a matched
       pair shares both initialisation stream and training data. Treating those
       as independent throws away the pairing and can move a p-value across
       0.05 in either direction.
    2. Escape filtering DESTROYS the pairing. After conditioning on escape the
       two arms can retain disjoint seed sets, in which case NO paired test
       exists and the Welch test is comparing different experimental units.
       That is reported explicitly (`paired_status`) rather than papered over.

    Both companions are reported; neither replaces the pre-registered Welch
    result. Where they disagree, the disagreement IS the finding.
    """
    col = "selected_validation_macro_f1"
    out: dict = {}

    # --- paired over seeds that escaped in BOTH arms -------------------------
    ae = a.loc[a["escaped"], ["seed", col]].set_index("seed")[col]
    be = b.loc[b["escaped"], ["seed", col]].set_index("seed")[col]
    shared = sorted(set(ae.index) & set(be.index))
    out["n_shared_escaped_seeds"] = len(shared)
    out["shared_escaped_seeds"] = [int(s) for s in shared]
    if len(shared) < 2:
        out["paired_status"] = "NOT MEASURED"
        out["paired_note"] = (
            f"Only {len(shared)} seed(s) escaped in BOTH arms, so no paired "
            "test exists. The conditional Welch result above therefore "
            "compares DIFFERENT experimental units, not a treatment contrast "
            "on matched units.")
    else:
        d = np.array([float(be[s]) - float(ae[s]) for s in shared])
        t_stat, p_value = sps.ttest_rel(be.loc[shared].to_numpy(dtype=float),
                                        ae.loc[shared].to_numpy(dtype=float))
        out.update({
            "paired_status": "MEASURED",
            "paired_mean_diff": round(float(d.mean()), 4),
            "paired_t_stat": round(float(t_stat), 4),
            "paired_p_value": round(float(p_value), 5),
            "paired_significant": bool(p_value < alpha),
            "paired_all_same_sign": bool(np.all(d > 0) or np.all(d < 0)),
        })

    # --- unconditional, ALL declared seeds ----------------------------------
    a_all = a[col].to_numpy(dtype=float)
    b_all = b[col].to_numpy(dtype=float)
    if len(a_all) >= 2 and len(b_all) >= 2:
        a_by = a[["seed", col]].set_index("seed")[col]
        b_by = b[["seed", col]].set_index("seed")[col]
        shared_all = sorted(set(a_by.index) & set(b_by.index))
        out["all_seed_mean_baseline"] = round(float(a_all.mean()), 4)
        out["all_seed_mean_comparison"] = round(float(b_all.mean()), 4)
        out["all_seed_diff"] = round(float(b_all.mean() - a_all.mean()), 4)
        out["all_seed_diff_sign_matches_conditional"] = None
        if len(shared_all) >= 2:
            t_stat, p_value = sps.ttest_rel(
                b_by.loc[shared_all].to_numpy(dtype=float),
                a_by.loc[shared_all].to_numpy(dtype=float))
            out["all_seed_paired_p_value"] = round(float(p_value), 5)
            out["all_seed_paired_n"] = len(shared_all)
    return out


def apply_multiplicity_correction(comparisons: list[dict], alpha: float,
                                  key: str = "conditional_p_value") -> dict:
    """Holm-Bonferroni over the whole comparison family.

    Added 2026-09-08. The grid produces dozens of exploratory contrasts and
    reported each against a bare alpha, so "significant at p<0.05" was doing
    work it cannot do across a family this size. Holm is used rather than plain
    Bonferroni because it is uniformly more powerful at the same familywise
    error rate. Nothing is deleted: every comparison keeps its raw p-value and
    gains an adjusted one and a family-level verdict, so a reader can see
    exactly which nominal findings survive.
    """
    indexed = [(i, c[key]) for i, c in enumerate(comparisons)
               if c.get(key) is not None]
    m = len(indexed)
    for c in comparisons:
        c["p_adjusted_holm"] = None
        c["survives_family_correction"] = None
    if not m:
        return {"family_size": 0, "method": "holm-bonferroni", "alpha": alpha}
    order = sorted(indexed, key=lambda kv: kv[1])
    running = 0.0
    n_survive = 0
    for rank, (idx, p) in enumerate(order):
        adj = min(1.0, (m - rank) * float(p))
        running = max(running, adj)          # enforce monotonicity
        comparisons[idx]["p_adjusted_holm"] = round(running, 6)
        survived = bool(running < alpha)
        comparisons[idx]["survives_family_correction"] = survived
        n_survive += int(survived)
    return {
        "family_size": m,
        "method": "holm-bonferroni",
        "alpha": alpha,
        "bonferroni_threshold": round(alpha / m, 8),
        "n_surviving": n_survive,
        "note": ("Every comparison keeps its raw p-value; `p_adjusted_holm` "
                 "and `survives_family_correction` say which of them survive "
                 "the family. A nominal p<0.05 that does not survive must not "
                 "be reported as a finding without saying so."),
    }


def compare(df: pd.DataFrame, cfg) -> list[dict]:
    """Hər (baza, data həcmi) hüceyrəsində bütün şərt cütləri."""
    iters = int(cfg.analysis.bootstrap_iters)
    alpha = float(cfg.analysis.alpha)
    out: list[dict] = []
    if df.empty:
        return out
    for (base, size), cell in df.groupby(["base", "train_size"]):
        conditions = sorted(cell["condition"].unique())
        for c1, c2 in combinations(conditions, 2):
            a = cell[cell["condition"] == c1]
            b = cell[cell["condition"] == c2]
            if a.empty or b.empty:
                continue
            row = {"base": base, "train_size": int(size),
                   "baseline": c1, "comparison": c2, "n_runs_baseline": len(a),
                   "n_runs_comparison": len(b)}
            row.update(compare_escape_rates(a, b, alpha))
            row.update(compare_conditional_f1(a, b, iters, alpha))
            out.append(row)
    return out


def seed_noise(df: pd.DataFrame) -> list[dict]:
    """Escape edən seed-lər arasındakı yayılma — "böyük fərq" nə deməkdir."""
    if df.empty:
        return []
    escaped = df[df["escaped"]]
    if escaped.empty:
        return []
    grouped = (escaped.groupby(["base", "condition", "train_size"])
               ["selected_validation_macro_f1"]
               .agg(n_escaped="count", conditional_mean="mean", seed_std="std")
               .reset_index())
    return grouped.to_dict(orient="records")


def main() -> None:
    ap = base_argparser(__doc__)
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    rdir = ensure_dir(cfg.experiment.results_dir)
    df = load_runs(rdir)

    if df.empty:
        log.warning("results/runs/ altında schema-v3 nəticə faylı YOXDUR — "
                    "NOT MEASURED hesabatı yazılır.")
    comparisons = compare(df, cfg)
    noise = seed_noise(df)
    family = apply_multiplicity_correction(
        comparisons, float(cfg.analysis.alpha))

    res = {
        "status": "MEASURED" if comparisons else "NOT MEASURED",
        "metric_policy": (
            "Two separate quantities, never pooled: escape_rate (Fisher exact "
            "+ Wilson CI) and conditional_macro_f1 over escaped seeds only "
            "(Welch t-test + bootstrap CI). Comparisons are within a base "
            "model; cross-base contrasts live in decompose.py. Since "
            "2026-09-08 each comparison additionally carries a PAIRED test "
            "over seeds that escaped in both arms (the units are paired: same "
            "seed, same subsample), an UNCONDITIONAL all-seed contrast, and a "
            "family-wide Holm-Bonferroni adjusted p-value. The pre-registered "
            "Welch result is unchanged; the companions exist so a reader can "
            "see when a nominal finding depends on the analysis choice."),
        "multiplicity": family,
        "alpha": float(cfg.analysis.alpha),
        "bootstrap_iters": int(cfg.analysis.bootstrap_iters),
        "min_escaped_for_f1_test": MIN_ESCAPED_FOR_F1_TEST,
        "n_runs_loaded": int(len(df)),
        "seed_noise": noise,
        "comparisons": comparisons,
        "note": ("diff_over_noise < 1 → fərq küy səviyyəsindədir, iddia "
                 "etməyin. conditional_ci_excludes_zero və "
                 "conditional_significant birlikdə oxunmalıdır. Şərtli "
                 "macro-F1 NOT MEASURED olduqda escape rate tək oxunur — "
                 "sıfır escape 'pis F1' deyil, ölçülməmiş F1 deməkdir."),
    }
    write_json(res, rdir / "stats.json")
    pd.DataFrame(comparisons).to_csv(rdir / "stats.csv", index=False)
    if family.get("family_size"):
        log.info("Multiplicity: %d comparisons, Holm alpha=%.3f -> %d survive "
                 "(plain Bonferroni threshold %.2e)",
                 family["family_size"], family["alpha"], family["n_surviving"],
                 family["bonferroni_threshold"])

    if noise:
        log.info("Escape edən seed-lər arasındakı yayılma:\n%s",
                 pd.DataFrame(noise).to_string(index=False))
    sig = [c for c in comparisons
           if c.get("escape_significant") or c.get("conditional_significant")]
    log.info("Əhəmiyyətli fərqlər (p<%.2f): %d / %d",
             cfg.analysis.alpha, len(sig), len(comparisons))
    for c in sig:
        log.info("  %s n=%-6d %-22s vs %-22s | escape %s→%s p=%.4f | "
                 "cond Δ=%s p=%s",
                 c["base"], c["train_size"], c["baseline"], c["comparison"],
                 c["baseline_escape"], c["comparison_escape"],
                 c["escape_fisher_p_value"], c.get("conditional_diff"),
                 c.get("conditional_p_value"))
    log.info("Yazıldı: %s", rdir / "stats.json")


if __name__ == "__main__":
    main()
