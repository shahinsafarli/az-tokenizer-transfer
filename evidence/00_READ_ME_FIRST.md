# Report evidence pack — 164-run grid, DLE-AI-202 Track 1

Everything the report can cite, and nothing that is not backed by a file.
Assembled 2026-09-09 from `results_full.zip`, `run.log` and `env_capture.zip`.
Rule for the whole pack: **if a number is not in one of these files, it is
`NOT MEASURED` and must be written that way.**

---

## 1. The tables you will actually cite

| File | What it holds | Status |
|---|---|---|
| `results/paper_tables.md` | 36 cells: escape rate (Wilson CI) + conditional macro-F1 (bootstrap CI), with the backing run file for every number | **Incomplete** — no test column, no all-seed column, no degeneracy column. The generator was never updated for Amendment 8. |
| `tables/test_tables.md` | Per-cell **test** macro-F1 (conditional and all-seed, bootstrap CI), val-vs-test contrasts, sign-agreement summary, generalisation gap | Fills the gap above. Generated from the 164 run files. |
| `results/summary.csv` | Per-cell, machine-readable: adds `all_seed_macro_f1`, `escape_rate_sustained`, `n_escaped_then_collapsed`, `n_tr_stage_degenerate`, `tr_stage_macro_f1_mean` | Complete |
| `results/stats.csv` / `stats.json` | 86 contrasts: Fisher (escape), Welch, **paired t** on shared escaped seeds, all-seed contrast, `p_adjusted_holm`, `survives_family_correction` | Complete — 27 of 86 survive Holm |
| `results/aggregate.csv` | One row per run, 164 rows, including `test_macro_f1` and `delta_test_minus_validation_matched` | Complete |

**Use `stats.csv` for every claim of significance.** Quote `p_adjusted_holm`, not
the raw p-value. The family is 86 tests and the Bonferroni threshold is 5.81e-04.

## 2. Raw results

- `results/runs/` — 164 run records, schema v4. Each carries `eval_split=test`,
  `selected_step`, `selected_validation_macro_f1`, `restored_validation_macro_f1`,
  the full `test` block, `test_comparison` (with `matched_step` and
  `delta_test_minus_validation_matched`), `step_history` (10 rows),
  `task_head_reset`, and `escaped_but_terminal_collapsed`.
- `results/logs/` — one log per run.
- `results/launcher_state.json` — the queue: 164 requested, 164 completed.
- `execution/run.log` — the full session log, 8.5 MB.

## 3. Machinery evidence — cite these when a reviewer asks "how do you know?"

| Claim | File |
|---|---|
| The transplant write/reload is exact | `results/controls__xlm15.json`, `results/controls__xlmr.json` — gate and end-to-end both true, round-trip bit-identical, `max_abs_logit_delta = 0.0` on both bases |
| Rescaling did not alter the matrices | the six `results/controls__*__*_rescaled.json`, all `ratio = 1.000` |
| Anchors | `results/anchors.json` — xlm15 8,047, xlmr 9,776 (UNK/special ids rejected) |
| Intrinsic transplant quality | `results/cross_base_transplant_quality.json`, `results/top1_accuracy.json` — OMP best on both bases on both measures |
| Tokenizer efficiency | `results/fertility.json` |
| Vocabulary overlap (M3) | `results/overlap_control.json` — Latin 58.2%, Turkish 66.7%, +8.5 pt |
| Data integrity | `results/splits.json` — dedup, leakage checks all 0, content manifest (test `ca19c0bb…`, n=4,186) |
| Robustness probes | `results/scramble.json`, `results/truncation.json` |
| Error analysis | `results/errors.json`, `results/errors_manual_sample.csv` |
| Decomposition | `results/decompose.json` |

## 4. Figures

`figures/conditional_macro_f1.png`, `figures/escape_rate.png`,
`figures/fig1_overlap_en.png`, plus `figures/source_manifest.json` naming the
files behind each one. Three figures only — anything else must be drawn from the
CSVs above.

## 5. Pre-registration — the reason the results are credible

`preregistration/experiment.yaml`, `FROZEN.md` (Amendments 1–8),
`CONFIG_HASHES.lock`. The escape threshold of 0.40 was fixed from the pilots and
hash-locked **before** the grid; the report should say so and point at these
files, because it is the single strongest methodological claim in the project.

## 6. Environment and compute

`environment/ENVIRONMENT_AS_RUN.md` — hardware table, compute cost, and the
pin-file discrepancy. `requirements.txt` and `requirements.lock` regenerated from
the post-run freeze; `env_freeze_as_run.txt` and `env_hardware_as_run.txt` are
the untouched captures.

1 × H100 NVL (95,830 MiB), 8.4 process-hours, ≈9.5 h wall-clock, ≈9.5 GPU-hours,
1 stream. **Peak GPU memory is NOT MEASURED** — the sampler never ran.

## 7. Disclosures

`disclosures/TEST_AUDIT.md` and `results/test_evaluation_ledger.jsonl`.

Five things must appear in the report:

1. **22 pre-guard test evaluations** happened before the guard existed. The
   ledger is in this pack; disclose it.
2. **Peak GPU memory unmeasured.** Report wall-clock, process-hours and
   device-hours separately; do not estimate memory.
3. **Retained-label agreement is 44/67 = 65.67% unconditioned**, not only the
   conditioned 88.10%. See `carried_forward_pre_grid/PROVENANCE.md`.
4. **Licences**: dataset CC-BY-NC-SA-4.0, models CC-BY-NC-4.0.
5. **Title must claim what the design identifies** — "comparing", not
   "separating". The three mechanisms are compared, not cleanly separated.

Plus, from the pin files: the shipped `requirements.lock` did not describe the
run environment, and `sacremoses` and `protobuf` were undeclared dependencies.

## 8. Carried forward from before the grid — read its own README first

`carried_forward_pre_grid/` holds the human label audit and four dataset
analyses that the grid does not produce. **The split assignment there differs
from the grid's by one example and has no content manifest.** See
`carried_forward_pre_grid/PROVENANCE.md` before citing any of it.

---

## What is NOT in this pack, and why

| Missing | Why | Cost |
|---|---|---|
| `artifacts/tr_stage_cache` — 30 Turkish-stage checkpoints | Tens of GB; deliberately not downloaded | ≈2.5 H100-hours to rebuild, only needed for extra experiments |
| Per-run model weights | `save_total_limit=1`, `save_only_model=True`; not retained after the grid | Full re-run to recover |
| `gpu_memory_trace.csv` / `gpu_memory_summary.json` | Sampler never ran | Unrecoverable; reported as `NOT MEASURED` |
| `data.tgz` (the data snapshot) | Downloaded separately, kept outside this pack | Already on the author's machine — keep it, `hf_revision` is unpinned so it is the only exact copy |

## Known weaknesses to state, not hide

- `paper_tables.md` is behind its own data (see §1); `tables/test_tables.md` is
  the companion, and the two must be cited together.
- 14 of 30 Turkish source stages were degenerate, all on xlm15, none on xlmr.
  Any claim about Turkish stabilising xlm15 must carry this number.
- The Turkish low-resource effect on xlmr (Δ = +0.0169 at n=500, 5.6× the n=2000
  effect, paired p = 0.0021) does **not** survive Holm (0.196). Directional
  evidence, not an established result.
- Inference is on validation. `tables/test_tables.md` reports test descriptively
  with no hypothesis test, because a test-side family chosen after seeing the
  numbers would not be pre-registered.
