"""Coverage for the analysis statistics module.

`src/analysis/stats.py` previously read a `test_macro_f1` column that
`load_runs()` has never produced, so `python -m src.analysis.stats` — a step
in `run_all.sh` — raised `KeyError` on every invocation. No test touched the
module, so nothing caught it. These tests pin the contract.
"""
from __future__ import annotations


import pandas as pd
import pytest

from src.analysis.stats import (compare, compare_conditional_f1,
                                compare_escape_rates, seed_noise)
from src.utils import Config


def _cfg(iters: int = 200):
    return Config({"analysis": {"bootstrap_iters": iters, "alpha": 0.05}})


def _runs(base: str, condition: str, size: int, f1_by_seed: dict[int, float | None]):
    """One row per seed; a None F1 means the run never escaped the basin."""
    rows = []
    for seed, f1 in f1_by_seed.items():
        rows.append({
            "base": base, "condition": condition, "train_size": size, "seed": seed,
            "escaped": f1 is not None,
            "selected_validation_macro_f1": 0.333 if f1 is None else f1,
        })
    return rows


def test_compare_runs_on_real_load_runs_schema_without_test_column():
    """Regression: the module must not require a `test_*` column."""
    df = pd.DataFrame(
        _runs("xlm15", "baza", 2000, {42: None, 1337: None, 13: 0.61, 7: 0.60, 2024: None})
        + _runs("xlm15", "tokenizator", 2000,
                {42: 0.70, 1337: 0.71, 13: 0.69, 7: 0.72, 2024: 0.70}))
    assert "test_macro_f1" not in df.columns
    out = compare(df, _cfg())
    assert len(out) == 1
    row = out[0]
    assert row["baseline"] == "baza" and row["comparison"] == "tokenizator"
    assert row["baseline_escape"] == "2/5"
    assert row["comparison_escape"] == "5/5"
    assert row["conditional_status"] == "MEASURED"
    assert row["conditional_diff"] == pytest.approx(0.099, abs=0.002)


def test_comparisons_never_pool_across_base_models():
    """xlm15 and xlmr differ in capacity and pre-training; pooling them would
    confound the condition effect with the base-model effect."""
    df = pd.DataFrame(
        _runs("xlm15", "baza", 2000, {42: 0.60, 1337: 0.61})
        + _runs("xlm15", "tokenizator", 2000, {42: 0.70, 1337: 0.71})
        + _runs("xlmr", "baza", 2000, {42: 0.80, 1337: 0.81})
        + _runs("xlmr", "tokenizator", 2000, {42: 0.82, 1337: 0.83}))
    out = compare(df, _cfg())
    assert {r["base"] for r in out} == {"xlm15", "xlmr"}
    assert len(out) == 2  # one within-base pair each, never a cross-base pair


def test_conditional_f1_is_not_measured_when_too_few_seeds_escape():
    """A collapsed arm has no conditional F1 to report — not a low one."""
    a = pd.DataFrame(_runs("xlm15", "baza", 2000, {42: None, 1337: None, 13: None}))
    b = pd.DataFrame(_runs("xlm15", "tokenizator", 2000, {42: 0.7, 1337: 0.71, 13: 0.69}))
    out = compare_conditional_f1(a, b, iters=200, alpha=0.05)
    assert out["conditional_status"] == "NOT MEASURED"
    assert out["n_escaped_baseline"] == 0
    assert "conditional_diff" not in out


def test_conditional_f1_excludes_collapsed_seeds_from_the_mean():
    """The 0.333 collapse value must never enter the conditional mean."""
    a = pd.DataFrame(_runs("xlm15", "baza", 2000, {42: 0.60, 1337: 0.62, 13: None}))
    b = pd.DataFrame(_runs("xlm15", "tokenizator", 2000, {42: 0.70, 1337: 0.72, 13: None}))
    out = compare_conditional_f1(a, b, iters=200, alpha=0.05)
    assert out["conditional_mean_baseline"] == pytest.approx(0.61)
    assert out["n_escaped_baseline"] == 2


def test_escape_rate_comparison_uses_fisher_exact():
    a = pd.DataFrame(_runs("xlm15", "baza", 2000,
                           {1: None, 2: None, 3: None, 4: None, 5: None}))
    b = pd.DataFrame(_runs("xlm15", "tokenizator", 2000,
                           {1: 0.7, 2: 0.7, 3: 0.7, 4: 0.7, 5: 0.7}))
    out = compare_escape_rates(a, b, alpha=0.05)
    assert out["escape_rate_diff"] == 1.0
    assert out["escape_significant"] is True
    assert out["escape_fisher_p_value"] < 0.05


def test_empty_results_produce_no_comparisons_instead_of_raising():
    assert compare(pd.DataFrame(), _cfg()) == []
    assert seed_noise(pd.DataFrame()) == []
