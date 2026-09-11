# Frozen experimental specification

Date frozen: **2026-09-04** (Asia/Baku)

This document is the preregistration for all post-amendment runs. None of the
values, queue rules, selection rules, or analysis rules below will change after
results are seen.

## Locked values

| Field | Frozen value |
|---|---|
| Azerbaijani update budget | `max_steps: 2000` |
| Validation interval | `eval_every_steps: 200` (optimizer steps, plus terminal step 2000) — amended 2026-09-04, was `65`; see "Amendments" below |
| Train sizes | `500, 2000, 10000, 20914` |
| Seeds at n <= 2000 | `42, 1337, 13, 7, 2024` |
| Seeds at n > 2000 | `42, 1337, 13` |
| Active condition order | no transplant; real OMP transplant; mean placebo; random-coefficient placebo; Turkish; OMP+Turkish; scrambled Turkish |
| OMP sparsity | `k: 64` |
| OMP candidate anchors | `n_candidates: 256` |
| Maximum token length | `max_length: 256` |
| Effective training batch | **32** (pre-registered; realised as `batch_size: 32` x `grad_accum: 1` — the 2026-09-04 accumulation amendment was reverted 2026-09-05, see below) |
| Early stopping | disabled (`null`) |
| Trainer-side best metric | ~~disabled (`null`)~~ — **SUPERSEDED, see Amendment 7.1**; it is `eval_macro_f1` |
| Load best model at end | `true`, `metric_for_best_model: eval_macro_f1` — amended 2026-09-07 (was `false` / `null`); see Amendment 6 |
| Evaluation split | validation by default; test requires explicit opt-in and ledger entry |

Turkish-bearing conditions run at n=500 and n=2000; `xlmr` runs at n=500 and
n=2000 (both amended 2026-09-07, see Amendments).
The single derived queue contains 164 runs and is ordered reference-size-first,
primary base before contrast base, then remaining primary sizes; within a cell,
condition order is the table order above and seed order is the applicable frozen
seed list.

## Model-selection rule

Selection occurs in analysis, never in the trainer: select the validation point
with the highest macro-F1; break exact ties by the lower optimizer-step index.
The selected validation value is not a test measurement.

## Why the update budget is fixed

At batch 32, a 20-epoch budget would give approximately 63, 313, 1,250, and
6,250 optimizer steps over the relevant data-size range, nearly a 100-fold
spread. Basin escape was measured at 250–630 steps. Under an epoch budget the
small cells could therefore be prevented from escaping by construction, and a
data curve would confound data quantity with update quantity. Two thousand
optimizer steps yields approximately 125 passes at n=500, 32 at n=2000, 6 at
n=10000, and 3 at n=20914 while holding update opportunity constant. Validation
every 65 steps yields 31 comparable points including the terminal step.

## Why these sizes and seeds are fixed

- n=100 is excluded because a binary task gives only about 50 examples per
  class, inadequate statistical power for a mechanism comparison; under 2,000
  fixed steps it would also expose the model to the same data roughly 400 times,
  shifting the measurement from transfer to memorization. This reason does not
  depend on observed collapse.
- n=20914 is included to establish the ceiling using all available training data.
- n=5000 is rejected as too close to n=10000 to add separate information.
- n=15000 is rejected as arbitrary.
- Five seeds are used through n=2000 because this is the low-resource region
  where transfer effects are expected and escape instability was measured.
  Three seeds are used above n=2000.

## Complete configuration diff

```diff
 experiment:
-  seeds: [13, 42, 1337]
+  seeds: [42, 1337, 13, 7, 2024]
+  seeds_high_resource: [42, 1337, 13]

 data:
-  train_sizes: [100, 500, 2000, 10000]
+  train_sizes: [500, 2000, 10000, 20914]

 training:
   lr: 2.0e-5
-  epochs_az: 10
+  max_steps: 2000
   epochs_tr: 2
   batch_size: 32
   grad_accum: 1
   warmup_ratio: 0.1
   weight_decay: 0.01
   fp16: true
-  early_stopping_patience: 3
-  eval_strategy: epoch
+  early_stopping_patience: null
+  eval_every_steps: 65
+  eval_strategy: steps
   metric: macro_f1
-  metric_for_best_model: eval_loss
-  greater_is_better: false
+  metric_for_best_model: null
+  load_best_model_at_end: false

 conditions:
-  - {id: 1, name: baza, tokenizer: original, turkish: none}
-  - {id: 2, name: turk, tokenizer: original, turkish: real}
-  - {id: 3, name: tokenizator, tokenizer: omp, turkish: none}
-  - {id: 4, name: her_ikisi, tokenizer: omp, turkish: real}
-  - {id: 5, name: turk_qarisiq, tokenizer: original, turkish: scrambled}
+  - {id: 1, name: baza, label: "Condition 1 — no transplant", tokenizer: original, turkish: none, transplant: none, sizes: all}
+  - {id: 3, name: tokenizator, label: "Real OMP transplant", tokenizer: omp, turkish: none, transplant: omp, transplant_tag: omp_k64_rescaled, sizes: all}
+  - {id: 6, name: transplant_mean, label: "mean placebo", tokenizer: omp, turkish: none, transplant: mean, transplant_tag: mean_k64_rescaled, sizes: all}
+  - {id: 7, name: transplant_random_coef, label: "random_coef placebo", tokenizer: omp, turkish: none, transplant: random_coef, transplant_tag: random_coef_k64_rescaled, sizes: all}
+  - {id: 2, name: turk, tokenizer: original, turkish: real, transplant: none, sizes: [2000]}
+  - {id: 4, name: her_ikisi, tokenizer: omp, turkish: real, transplant: omp, transplant_tag: omp_k64_rescaled, sizes: [2000]}
+  - {id: 5, name: turk_qarisiq, tokenizer: original, turkish: scrambled, transplant: none, sizes: [2000]}
```

The optimizer settings (`lr`, warmup, scheduler behavior, batch size, gradient
accumulation, weight decay, and FP16 policy) are intentionally unchanged.

## Amendments (post-freeze, non-scientific parameters only)

### 2026-09-04 — `eval_every_steps: 65` → `200`

Context: a launched run showed no completed validation interval after 41
minutes; diagnosis (see `HANDOFF.md §12`) found the GPU was stalled, not
compute-bound, and ruled out mixed precision, sequence padding, gradient
checkpointing, and the laptop power cap as causes. Reducing validation
frequency is an orthogonal, independent change made at the same time to cut
per-run wall-clock, not a fix for the stall itself.

`eval_every_steps=65` produced ~31 validation passes over the 2,789-example
validation split — roughly 2,700 extra forward-only batches per run,
comparable to the entire training cost — for temporal resolution the
analysis does not use, since basin escape was previously observed between
steps 250 and 630 (well inside a 200-step grid). `eval_every_steps=200` gives
~10 uniform points per run (steps 200, 400, …, 2000), still bracketing the
observed escape window on both sides.

This is an **observation-frequency parameter, not a training parameter**: it
does not change `max_steps`, `batch_size`, `max_length`, `k`, `n_candidates`,
the seed lists, or `train_sizes`, and it does not change what the optimizer
does at any step — only how often the (frozen) validation split is scored.
The model-selection rule (§ above, "highest macro-F1, ties broken by lower
step") is unaffected; it now simply selects among ~10 points instead of ~31.

### 2026-09-04 — `training.eval_batch_size: 8` (new, not previously a field)

Also part of the same diagnostic pass (`HANDOFF.md §12`). Not a pre-registered
scientific parameter — `per_device_eval_batch_size` governs only how many
validation examples are batched together during a forward-only eval pass; it
has no effect on the trained model, its gradients, or its optimizer
trajectory. `training.batch_size=32` (the *train*-time batch size) is
unchanged. Reduces peak VRAM during the ~10 validation passes per run.

**Superseded same day, see the A100-migration amendments immediately below —
this value was specific to the RTX 4060's memory wall and does not apply to
the A100 target.**

### 2026-09-04 — A100 migration: two amendments

Verdict on the RTX 4060 diagnostic (`HANDOFF.md §12`, Steps 1–4 of the
diagnostic conversation): disqualified as the training GPU. Peak reserved
VRAM of 8,657 MB against 8,585 MB total capacity — measured on a genuine
30-step benchmark at the locked `batch_size=32`/`max_length=256` — is a
capacity wall, not a tuning problem: even the pre-degradation portion of
that benchmark (0.3–0.65 s/step) projects the full grid (262,000 steps:
228,000 Azerbaijani across 114 runs + 33,780 Turkish across 30 distinct
checkpoints) to roughly 33 hours on that card, before accounting for the
degradation the benchmark also showed. Target: NVIDIA A100-SXM4-40GB, MIG
partition `3g.20gb` (20 GB VRAM slice).

**`training.eval_batch_size: 8 → 64`.** `8` was adopted specifically to fit
validation passes inside the 4060's memory wall; at `eval_batch_size=8` a
2,789-example validation split takes 349 forward batches per pass, ~10
passes per run, 114 runs — needless overhead that does not exist on a 20 GB
slice. `64` is the original default (`training.batch_size * 2`) from before
the 4060 diagnostic. Still not a pre-registered scientific parameter — it
governs only how validation batches are grouped, not the trained model.

**`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — kept as-is.** The
4060 diagnostic (Step 4 of the stall investigation) found this is a no-op on
Windows (`UserWarning: expandable_segments not supported on this platform`)
— it is, however, supported on Linux CUDA, which the A100 host runs. It
addresses exactly the mechanism measured on the 4060: batches of varying
natural token length (mean 29.6, p95 79, out of a `max_length=256` ceiling —
see `HANDOFF.md §12`) produce a different tensor shape almost every step,
which fragments a non-expandable caching allocator. This is an additional,
independent argument for the Linux move, not only the raw capacity wall.

**Gradient accumulation: implemented, default off.** 20 GB comfortably
holds the measured ~8.6 GB peak from the 4060 diagnostic, so
`gradient_accumulation_steps` stays at its pre-registered value of `1` and
is not enabled by default on the A100. It remains available as a fallback
only. If it is ever needed, the only authorized form is per-device batch `8`
with `gradient_accumulation_steps=4` — effective batch exactly `32`,
matching the pre-registered `training.batch_size`. This preserves the
pre-registered effective batch size exactly; `training.max_steps` counts
*optimizer* steps (post-accumulation), so total run length is unchanged
whether or not accumulation is used. If it is ever turned on, it must be
turned on identically on every machine and every run in the grid — no
per-run or per-machine exceptions.

### 2026-09-04 — gradient accumulation enabled: per-device 8 x accum 4

**This amendment uses the escape hatch this document already authorised**, in
exactly the form it authorised: "the only authorized form is per-device batch
`8` with `gradient_accumulation_steps=4` — effective batch exactly `32`,
matching the pre-registered `training.batch_size`."

**Why the earlier "default off" reasoning was wrong.** It read: "20 GB
comfortably holds the measured ~8.6 GB peak from the 4060 diagnostic, so
`gradient_accumulation_steps` stays at 1." That compares the run's footprint
against the **A100 MIG slice's physical capacity**. The binding constraint is
not the card — it is the brief's allocation: *"peak ~10–12 GB of GPU memory
**per team**"*, with "Exceeding your compute allocation" listed under
Automatic deductions. Those are different quantities, and the argument
substituted one for the other.

Two consequences follow:

1. A single run at 8.6 GB is already 72–86% of the team allowance. There is
   no headroom for anything concurrent.
2. The staged run plan requires **2–3 concurrent streams** to bring the
   projected grid (21–49 h serial) inside the ~2-day window. At 8.6 GB per
   process that is ~17–26 GB — over the allowance, on a shared machine,
   which the brief also flags under shared-machine etiquette ("Do not exceed
   your VRAM allocation — it starves other teams").

Accumulation is therefore a **compliance requirement**, not a throughput
measure. Per-process peak drops to roughly 3–4 GB, so three streams fit
inside the 10 GB ceiling now enforced by `run.vram_ceiling_gb`.

**What is unchanged.** Effective batch (32), `max_steps` (2000 — it counts
optimizer steps, which are post-accumulation), `lr`, warmup, weight decay,
`max_length`, `k`, `n_candidates`, the seed lists, and `train_sizes`. The
experimental design is untouched.

**One honest consequence, recorded before the runs.** Computing a batch as
4x8 draws from the RNG in a different order than 32-in-one-pass, so dropout
masks — and therefore results — are **not bit-identical** to a
non-accumulated run. This is neither better nor worse. Comparisons remain
valid because the setting is uniform across every reported run, and every
pre-accumulation result is already quarantined (schema-v2, excluded by
`load_runs()`). It must be applied identically on every machine and every
run in the grid — no per-run or per-machine exceptions.

**`training.eval_batch_size: 64 -> 32`** (not pre-registered; forward-only,
cannot affect the trained model). Held below 64 so a validation pass does not
silently exceed the per-process peak that accumulation was adopted to obtain.
The memory ceiling has to hold across the whole run, not only its training
steps.

### 2026-09-04 — two-phase execution and parallel streams

Non-scientific execution machinery; changes no training parameter. Phase 1
builds all 30 Turkish checkpoints **serially** and writes
`artifacts/tr_stage_cache/phase1_manifest.json`; Phase 2 verifies that
manifest and opens the cache **read-only**, refusing to start otherwise. This
removes — rather than mitigates — the concurrent cache-corruption failure
mode, in which two streams building the same uncached checkpoint interleave
writes to one path and produce a weights file that still loads, with
silently wrong values.

New `run` keys: `parallel_streams`, `vram_ceiling_gb`, `vram_per_stream_gb`.
Every result file records `streams_active`, because a per-run wall-clock
measured under contention is not that run's isolated cost and must never be
the timing figure reported in the paper.

### 2026-09-05 — gradient accumulation REVERTED to the pre-registered form

**This restores the original pre-registration**, it does not deviate from it:
`training.batch_size: 32`, `training.grad_accum: 1`. The 8x4 form remains
authorised as a documented fallback and is unchanged in its wording above.

**Why the 2026-09-04 amendment is being undone.** That amendment adopted 8x4
to create memory margin so that **2–3 concurrent streams** would fit the
brief's ~10–12 GB per-team allocation. Two things have since been established:

1. **At one stream there is no margin to buy.** A single run peaks at 8.6 GB
   (measured, `HANDOFF.md` §12), which is already **under** the 10 GB figure.
   Accumulation only matters if streams are stacked on the shared workstation
   — and per `docs/COMPUTE_ESTIMATES.md` §2 the per-process floor (~4.45 GB of
   batch-size-independent AdamW state) means two streams cannot fit 10 GB in
   either configuration. The premise of the amendment does not hold.
2. **Accumulation costs ~2.1x wall-clock here.** The workload is
   kernel-launch-bound (Azerbaijani sequences average 29.6 tokens), so
   splitting one batch of 32 into four micro-batches multiplies the overhead
   that already dominates: ~0.52 s/optimizer-step versus ~0.25 at batch 32.
   Against a ~2-day window that is the difference between ~45 h and ~23 h
   serially.

**Nothing is quarantined by this revert.** Zero schema-v3 result files exist;
no run has ever completed under the 8x4 setting. The window in which this
change is free closes the moment the first real run completes.

**Consequence for reproducibility, in our favour.** The dropout-mask RNG-order
caveat recorded on 2026-09-04 no longer applies: at `grad_accum: 1` a batch is
consumed in a single pass, which is the ordering the original pre-registration
assumed.

**`training.eval_batch_size: 32 -> 64`** (not pre-registered; forward-only).
Restored to the `batch_size * 2` default now that the training peak is 8.6 GB
again — a no-grad pass at 64 stays under that, so it does not raise the
per-process figure the stream ceiling is computed from.

**Venue does not change the science, only `run.parallel_streams`.** Batch 32
is compliant on the shared workstation (8.6 GB < 10 GB) at **one** stream, and
is also the fastest configuration on a rented GPU where the team allocation
does not bind and several streams fit. One configuration therefore serves both
venues; only the stream count — execution machinery, not a training parameter
— differs, and it is recorded per run in `streams_active`.

### 2026-09-07 — Turkish comparison at n=500, and the held-out test split

**Written before the run it authorises.** No result file from the grid this
amendment describes existed when this section was committed. The previous grid
(114 runs, validation only, H100, 2026-09-06) is **retained unmodified** as a
separate artifact; it is not quarantined, not deleted, and not pooled with the
new grid. Two grids are two samples, and the analysis of the new grid must not
be conditioned on which of them looks better.

**Amendment 1 — `conditions[2,4,5].sizes: [2000] -> [500, 2000]`.**
Kardes-NLU's claim is specifically that Turkish intermediate fine-tuning helps
**when Azerbaijani supervision is scarce**. Running the Turkish conditions at
n=2000 only left that claim untested at the point where it should be strongest,
which the previous report had to carry as an explicit limitation ("Turkish null
at one data size"). This varies the **Azerbaijani** size only — an axis already
pre-registered in `data.train_sizes` — and needs no new machinery. Turkish
DOSE-response (varying the Turkish corpus itself) remains frozen out.

**Amendment 2 — `run.contrast_sizes: [2000] -> [500, 2000]`.**
Mechanical consequence of Amendment 1, not an independent choice. The size gate
is per-BASE and is applied *before* a condition's own `sizes` scope
(`orchestrate.build_queue` -> `utils.sizes_for_base`, then
`utils.condition_applies`), so with `contrast_sizes: [2000]` the n=500 Turkish
comparison could not reach the contrast base at all and Amendment 1 would have
been asymmetric across bases. It also gives the contrast base a two-point data
curve, so its near-zero transplant effect stops being a single-size
observation.

**Amendment 3 — the held-out test split is spent, once, on this grid.**
`--eval-split test --allow-test-eval` is now forwarded by
`src.training.run_grid` to both the serial and the parallel Phase-2 path
(previously the launcher hard-coded `eval_split="val"`). What this does and
does not change:

- **Training is byte-for-byte the same procedure.** `_train_stage` always
  passes `az_val` as the Trainer's `eval_dataset`, in both settings. Periodic
  evaluation, the escape criterion, `escape_step`, `selected_step` and
  `selected_validation_macro_f1` are therefore all still VALIDATION evidence,
  computed exactly as before.
- **What is added** is one terminal evaluation of the **final-step** model on
  `az_test`, its predictions, and one append to
  `results/test_evaluation_ledger.jsonl` per run.
- **Estimands differ and must not be compared naively.** The validation figure
  is a MAXIMUM over the evaluation points; the test figure is the FINAL-STEP
  model, because `save_strategy: "no"` means the best-step weights are never
  retained. The comparable validation number is therefore the **terminal row of
  `step_history`**, which every run already records. Any val-vs-test statement
  in the paper must use that row, not `selected_validation_macro_f1`.
- **No selection is made from the test split.** `metric_for_best_model: null`
  and `load_best_model_at_end: false` are unchanged, so no model, step,
  hyperparameter or condition can be chosen using a test number.

**Not changed by this amendment.** `lr`, `max_steps`, `warmup_ratio`,
`weight_decay`, `batch_size`, `grad_accum`, `fp16`, `eval_every_steps`,
`max_length`, `seeds`, `seeds_high_resource`, `data.train_sizes`,
`transplant.k`, `transplant.n_candidates`, `rescale_reconstructed`, the split
seed, and every condition definition other than the three `sizes` fields named
above. `configs/CONFIG_HASHES.lock` is regenerated for this amendment and for
this amendment only.

**Amendment 4 — run-result schema v3 -> v4 (bookkeeping, not science).**
Two fields are now REQUIRED in every run file, so the interpretable comparison
cannot be omitted and the misleading one cannot be mistaken for it:

- `terminal_step` / `terminal_validation_macro_f1` — the validation figure for
  the same weights a test evaluation sees.
- `test_comparison` — carries `validation_macro_f1_terminal`,
  `validation_macro_f1_selected_max`, `test_macro_f1`,
  `delta_test_minus_validation_matched` (the reportable one) and
  `delta_test_minus_validation_selected_DO_NOT_REPORT` (retained only to show
  how much of an apparent drop is the estimand artefact).
- `escaped_but_terminal_collapsed` — true when a seed crossed the 0.40
  threshold mid-training and fell back into the collapse basin by the terminal
  step. Such a run is legitimately ESCAPED while its test figure is a collapsed
  one; the flag makes that countable instead of appearing as an unexplained
  inconsistency between escape rate and test F1.

`assert_run_result_schema` verifies each of these against `step_history` and
recomputes the matched delta, so a hand-edited or mis-derived number fails at
write time. `src.analysis.aggregate.load_runs` accepts v4 EXACTLY (not ">=")
and prints a count of any other-version files it refused to pool, so pointing
the analysis at a directory holding both grids produces a visible refusal
rather than a quiet average. No training parameter is affected.

**Amendment 5 — per-method C1c provenance (bug fix, no science changed).**
`src.transplant.controls` wrote only `results/controls__<base>.json`, a fixed
name, so probing omp -> mean -> random_coef left one file holding whichever
method ran last; `src.transplant.cross_base_quality` then copied
`C1c_delta_bpc` out of it and labelled the result with the canonical tag
WITHOUT reading the `transplanted_tag` / `is_canonical_artifact` fields that
`controls.py` writes for precisely that purpose. A placebo's delta BPC could
therefore be reported as the OMP transplant's. Fixed three ways:

1. `controls.py` now writes `results/controls__<base>__<tag>.json` per method;
   the base-only filename is reserved for the canonical artifact.
2. `cross_base_quality.py` prefers the method-tagged file, verifies the
   measured tag, and on mismatch emits `error_controls_tag_mismatch` and
   WITHHOLDS the BPC fields rather than reporting a plausible wrong number.
3. It now also carries the top-1 counts from the same masked positions
   alongside every delta BPC, with a note that a lower delta BPC at near-zero
   top-1 is a CALIBRATION effect, not recovered lexical knowledge — the
   distinction on which the "mean beats OMP" reading turns.

`run_all.sh` gained the placebo C1c probes (forward passes only, minutes) so
all three methods have clean, self-identifying records in one prep pass.

**Amendment 6 — best-validation checkpoint selection (AZ stage only).**
`training.metric_for_best_model: null -> eval_macro_f1`,
`training.load_best_model_at_end: false -> true`. The Azerbaijani stage now
saves at each evaluation point with `save_total_limit=1` and
`save_only_model=True`, and restores the selected checkpoint at the end.

**Why the original `false` no longer applies.** It was chosen so the collapse
diagnostics would see a full, uninterrupted 2000-step history — the worry being
early stopping. Best-checkpoint restore is NOT early stopping:
`early_stopping_patience` stays `null`, every run still trains all 2000 steps,
and `step_history` still records all ten evaluation points. The setting changes
only which weights remain in memory after `trainer.train()` returns.

**Why it is needed.** Standard protocol is to select on validation and report
the selected checkpoint's held-out test score. With no checkpoint retained, the
only weights a test evaluation could see were the FINAL-step ones, while the
reported validation figure was the MAXIMUM over evaluation points — two
different models. Amendment 4 handled that by recording a matched terminal
figure; this amendment removes the mismatch instead of documenting it. The
matched pair is now (selected step validation, selected step test).

**Nothing scientific is redefined.** Training is unchanged, so escape rate,
`escape_step`, `selected_step` and `selected_validation_macro_f1` are produced
exactly as before and remain directly comparable with the previous grid. Only
the test evaluation moves — from weights nobody would have shipped to the
weights the selection rule actually names.

**Verified, not assumed.** A silent no-op restore would make every test number
in the grid describe the wrong weights with nothing in the output to show it.
So after training the restored model is re-scored on validation and checked
against the recorded maximum (`restored_validation_macro_f1`, tolerance 1e-4);
a mismatch aborts the run instead of writing a plausible wrong result.
`assert_run_result_schema` re-checks this at write time.

**Turkish stage unchanged.** It keeps `save_strategy: "no"` and evaluates per
epoch: its product is the END of Turkish training, which is what
`_tr_cache_spec` records, so the 30 cached checkpoints and their keys are
untouched.

**Disk.** Transient only: at most two model-only checkpoints per concurrent
stream, in a per-run temp directory removed when the run ends. The pre-flight
guard adds `2 x parallel_streams x per_checkpoint` as headroom rather than a
per-run cost, since it never accumulates.

### 2026-09-08 — Amendment 7: reconciliation of superseded text

**This amendment changes no parameter and no code.** It exists because
Amendment 6 (best-validation checkpoint selection) falsified text written
earlier in this document, and the honest repair for a pre-registration is a
dated amendment saying which text is dead — not a silent rewrite of the
paragraphs that recorded the earlier decision. Amendments 1–6 are left exactly
as written. Where they conflict with this section, **this section governs**,
and each conflict is named so the reader never has to guess which is live.

Written **before** the 164-run grid was launched.

**7.1 — "Locked values" row `Trainer-side best metric | disabled (null)`.**
SUPERSEDED: it is `metric_for_best_model: eval_macro_f1` per Amendment 6. The
row two lines below already records this; the older row was not removed when
Amendment 6 was written. Early stopping is a separate setting and remains
`null` — every run still trains the full `max_steps: 2000`.

**7.2 — "Model-selection rule": "Selection occurs in analysis, never in the
trainer".** SUPERSEDED. Selection now occurs in BOTH places and they are
required to agree: the trainer restores the highest-`eval_macro_f1`
checkpoint; the analysis independently recomputes that argmax from
`step_history`; and `run_single` re-scores the restored model on validation and
refuses to write the run unless the two agree within 1e-4
(`restored_validation_macro_f1`). The tie-break is unchanged and still correct —
the lower optimizer-step index wins, because `transformers` compares with a
strict `>`, matching `min(step where f1 == max)`.

The remaining sentence, "The selected validation value is not a test
measurement", **stands and matters more, not less**: it is a MAXIMUM over ten
evaluation points and upward-biased by construction; the test figure is the
single unbiased measurement.

**7.3 — Amendment 3's "final-step model" bullets.** SUPERSEDED by Amendment 6,
which retains the best-step weights and evaluates test on them. The estimand
mismatch those bullets documented **no longer exists**: the matched pair is now
(`selected_validation_macro_f1`, `test_macro_f1`), both describing the same
weights, and `test_comparison.matched_step` equals `selected_step`.
Consequently Amendment 3's instruction to use the terminal row for any
val-vs-test statement is **reversed**: use
`test_comparison.validation_macro_f1_matched`.
`validation_macro_f1_terminal` is context only — the gap between it and the
selected figure is post-peak drift on validation
(`validation_drift_selected_minus_terminal`), not a generalisation gap.
Amendment 3's other assertions stand: training is the same procedure, the
Trainer's `eval_dataset` is always `az_val`, escape evidence is
validation-only, and no selection is made from the test split.

**7.4 — Amendment 4's field `delta_test_minus_validation_selected_DO_NOT_REPORT`.**
Does not exist in schema v4 as shipped; it was replaced during implementation
by `validation_drift_selected_minus_terminal`. The authoritative field list is
`RUN_RESULT_REQUIRED_KEYS` in `src/training/finetune.py` plus the keys
`run_single` writes into `test_comparison`. Where this document and the code
disagree about a field name, the code is authoritative and this line records
the divergence.

**7.5 — "n=20914 … all available training data".** The Azerbaijani training
split holds **20,937** examples, so the largest cell trains on 99.89% of it.
The number is NOT re-frozen — changing a pre-registered `train_sizes` entry to
make a sentence true is the wrong direction of fix. The paper says
"essentially all (20,914 of 20,937) training examples". Amendment 8.6 adds a
content manifest so this figure can no longer drift unnoticed.

**7.6 — Disk pre-flight scope.** The headroom term
`2 x parallel_streams x per_checkpoint` in
`orchestrate.estimate_grid_disk_bytes` is enforced only on the SERIAL path
(`execute_queue`). `phases.run_phase2`, the parallel path used at
`--streams > 1`, performs no disk pre-flight, and transient checkpoints live
under `$TMPDIR`, a filesystem `_disk_free_bytes` does not inspect. A KNOWN GAP,
not a claim of protection: handled operationally (provision 150 GB and
`export TMPDIR` onto the same mount), not by patching the launcher on launch
day. The failure mode is bounded — `write_json` publishes atomically and
`skip_existing` resumes, so the cost is a stall, not data loss.

**7.7 — Environment pinning.** `requirements.lock` does not resolve
(`fsspec==2026.7.0` against `datasets==2.21.0`, which requires
`fsspec[http]<=2024.6.1`), and its `torch==2.4.1+cu121` pin is older than the
stack the 2026-09-06 grid ran on. No launcher installs it any more; a
`transformers` major-version guard refuses a 5.x image; and
`results/environment_provenance.json` records what actually ran. The pinned
files are to be regenerated from the completed run's `pip freeze` so the
declared environment is the one that produced the numbers.

### 2026-09-08 — Amendment 8: forensic-audit repairs, before the final grid

**Written before the run it authorises.** No result file from the 164-run grid
existed when this section was committed. An independent forensic audit of the
2026-09-06 grid identified defects in the *intervention machinery* and in the
*analysis estimand*, distinct from the protocol defects Amendments 3–7 handled.
Six of them are repaired here. Each changes what the experiment measures, so
this is the last moment they can be fixed without quarantining results — and
each is a case where the code did not do what this document and the paper said
it did.

**8.1 — `mean` placebo now averages the NEAREST k anchors.** `omp.py` selected
candidates with `np.argpartition`, which guarantees only that the returned
`n_candidates` are the nearest ones — **their order among themselves is
arbitrary**. `solve_coefficients(method="mean")` then took `w[:k] = 1/k`, i.e.
an arbitrary 64 of the nearest 256. The control described everywhere as "the
mean of the k nearest anchors" was therefore not that. The candidate set is now
sorted by descending similarity before coefficients are solved. `omp` and
`random_coef` are provably unaffected (OMP solves over the whole candidate set;
`random_coef` samples from it uniformly), so **only the mean artifact changes** —
and the C3−C6 contrast, "does the fitted solve beat the crudest weighting",
becomes the comparison it was always claimed to be.

**8.2 — functional anchors reject unknown and special tokens.**
`_encode_single_token` accepted any surface that encoded to exactly one token.
A donor surface the base tokenizer cannot represent encodes to exactly one
token — `<unk>` — so such pairs were admitted as anchors, binding donor tokens
to the base UNK row. Every contaminated anchor injects the UNK vector into the
reconstruction dictionary, inflates the anchor count, and pulls reconstructed
rows toward UNK, which corrupts reconstruction cosine, ΔBPC and top-1 alike.
`unk_token_id` and `all_special_ids` are now rejected. **Consequence: the anchor
counts change.** The previously reported 8,079 (xlm15) and 9,776 (xlmr) are
superseded by whatever this run measures, and every downstream transplant
artifact is rebuilt.

**8.3 — the task head is now genuinely discarded between stages.**
`ignore_mismatched_sizes=True` re-initialises only SHAPE-mismatched parameters.
`XLMRobertaClassificationHead` is `dense` (hidden→hidden, shape independent of
`num_labels`) then `out_proj` (hidden→`num_labels`). Going from the 3-label
Turkish stage to the 2-label Azerbaijani stage resets `out_proj` and **silently
carries `dense` over** — verified directly: marking `dense`, saving a 3-label
checkpoint and reloading with `num_labels=2` leaves the marker intact, and
transformers' own warning names only `out_proj`. A Turkish-trained task
projection was thus leaking into the Azerbaijani stage of exactly the arms the
M1/M2 comparison rests on, contradicting this document, the README and the
paper. `finetune.reset_task_head()` now re-initialises every top-level module
outside `base_model_prefix` with the model's own initialiser, for EVERY
condition so no arm gets a differently-aged head, and records what it reset in
`task_head_reset`.

**8.4 — degenerate source stages are countable.** Several primary-base Turkish
stages in the previous grid ended predicting one class for every Turkish
validation example (macro-F1 0.1595 over three classes). The metric was
recorded but nothing carried it downstream, so a FAILED manipulation was
indistinguishable in aggregate from a delivered one — and "Turkish warm-up did
not help" cannot be concluded from a run where Turkish warm-up did not happen.
`tr_stage_macro_f1`, `tr_stage_constant_prediction` and `tr_stage_degenerate`
(threshold 0.40, or constant prediction) are now written into every
Turkish-bearing run and aggregated per cell as `n_tr_stage_degenerate`. It is
deliberately **not a gate**: excluding failed source stages would be
outcome-based exclusion. It is reported.

**8.5 — the estimand is reported both ways, and the tests match the design.**
`conditional_macro_f1` averages only escaped runs, so two conditions can be
compared on disjoint survivor sets — selection on outcome, and the previous
grid contained cells where baseline and OMP shared **no** escaped seed. Added,
alongside the unchanged pre-registered quantities:
`all_seed_macro_f1` (every declared seed, failures at their actual score) with
its bootstrap CI and its difference from the conditional mean;
`escape_rate_sustained` and `n_escaped_then_collapsed`, since `escaped` is
"crossed 0.40 at any point" and a seed can fall back;
a **paired** t-test over seeds escaping in both arms, with an explicit
`paired_status: NOT MEASURED` when fewer than two seeds are shared — which is
itself the finding that the Welch comparison has no matched units;
an unconditional all-seed paired contrast; and a family-wide
**Holm-Bonferroni** adjustment (`p_adjusted_holm`,
`survives_family_correction`, plus the plain Bonferroni threshold) over every
comparison, because a bare alpha cannot carry a family of this size. No
pre-registered statistic is removed or replaced; where a companion disagrees
with it, the disagreement is the result.

**8.6 — provenance and measurement gaps closed.**
`write_json` now emits valid JSON: non-finite floats become `null` with an
explicit `n_nonfinite_values_nulled` count, and `allow_nan=False` makes any
survivor a loud error. The previous grid wrote 60 bare `Infinity` tokens across
48 run files, which strict parsers reject, so a third party could not read
those results with standard tooling.
`results/splits.json` now carries `az_content_manifest`: an order-invariant
SHA-256 over the sorted normalised text+label of each split. Two artifacts of
this project record 20,936/2,791/4,187 and 20,937/2,791/4,186 for the same
nominal split — a one-example drift with no code change to explain it, because
`data.az.hf_name` is not revision-pinned while the donor model is. Counts
cannot distinguish "same holdout" from "different holdout"; the hash can.
`data.az.hf_revision` exists as a config field (null, with a warning when
unset).
`transplant.seed` is explicit (0, the value the existing artifacts used) and
recorded in every transplant manifest, so the paper can state plainly that
`random_coef` is ONE fixed realisation and that variance across model seeds is
fine-tuning noise, not transplant-draw noise.
The C1a identity control gains a real end-to-end leg — anchor self-mapping, an
embedding write plus `save_pretrained`/`from_pretrained` round trip, and a
forward-pass logit comparison — because the original compared `E` with
`E.copy()` after reassigning rows from `E` and therefore could not fail. It is
recorded (`C1a_identity.end_to_end`) and logged loudly but **does not gate**:
it is new code that could not be executed against the real base models before
this run, and `main()`'s `SystemExit` on `passed` runs under
`set -euo pipefail`. Promotion into the gate is a one-line change documented in
the function, to be made once the first prep run shows it passing.
`scripts/vastai_run_full.sh` samples device-total GPU memory every 10 s for the
whole of Phase 2 and writes `results/gpu_memory_trace.csv` plus
`gpu_memory_summary.json` with the device peak and Phase-2 wall-clock, so the
per-team memory figure is measured rather than inferred from a per-process one.
`run_all.sh` now invokes `src.tokenization.overlap_control`, whose raw export a
paper figure depends on and which no launcher previously ran.

Dataset label reliability, the identification limits of the shuffled-Turkish
and cross-base contrasts, the missing shuffled+donor interaction arm, and the
scope of the paper's title are scientific matters for the write-up. This
amendment does not touch `lr`, `max_steps`, `warmup_ratio`, `weight_decay`,
`batch_size`, `grad_accum`, `fp16`, `eval_every_steps`, `max_length`, the seed
lists, `train_sizes`, `transplant.k`, `n_candidates`,
`rescale_reconstructed`, the split seed, or any condition definition.
