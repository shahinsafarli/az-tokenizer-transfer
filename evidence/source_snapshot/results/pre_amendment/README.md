# Pre-amendment result quarantine

Quarantined on **2026-09-04**.

**Label:** incomplete logging + pre-amendment hyperparameters.

These files are retained as the immutable record that motivated the direct
diagnostic experiment. They must never be deleted, overwritten, included in a
post-amendment aggregate, or compared in the same results table as runs made
under the amendment. The historical run JSON files did not persist complete
per-epoch validation histories, validation prediction counts, or unambiguous
model-selection metadata.

## Exact configuration that produced the quarantined results

```yaml
models:
  max_length: 256
transplant:
  k: 64
  n_candidates: 256
training:
  lr: 2.0e-5
  epochs_az: 5
  epochs_tr: 2
  batch_size: 32
  grad_accum: 1
  warmup_ratio: 0.1
  weight_decay: 0.01
  fp16: true
  early_stopping_patience: 2
  eval_strategy: epoch
  metric: macro_f1
```

The code in effect supplied `metric_for_best_model=macro_f1`,
`greater_is_better=true`, and `load_best_model_at_end=true`. The five runs at
issue all used base `xlm15`, Condition 3 (`tokenizator`), Azerbaijani training
size 2,000, and the base-anchor-median-rescaled control artifacts:

- `mean_k64_rescaled`: seeds 13, 42, and 1337
- `random_coef_k64_rescaled`: seeds 13 and 42


## Corrected verdict record (2026-09-04)

This correction was supplied by the investigator; it does not identify an
implementation error.

- `mean`, seed 1337 is **UNRESOLVED**, not `UNSTABLE`. Loss-based early
  stopping terminated it at epoch 5 while it remained at the majority baseline;
  sibling runs escaped at epochs 4 and 10. A run stopped inside that observed
  escape window cannot be classified as a completed non-escape.
- `UNSTABLE` is a condition-level description across seeds and is not in the
  per-run verdict set. Post-freeze per-run verdicts are `ESCAPED`,
  `NOT_ESCAPED`, and `UNRESOLVED` only.
- No cross-seed spread is reported for the current `random_coef` evidence,
  because only two current seeds exist. The older three-seed files remain
  historical records but are not pooled with the current diagnostic files.
