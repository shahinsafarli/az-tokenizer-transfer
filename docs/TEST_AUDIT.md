# Test-set evaluation audit

Audit date: 2026-09-04.

The sole live model-evaluation call site is
`src.training.finetune.run_single`. It defaults to `eval_split="val"`. Test
evaluation requires both `eval_split="test"` and `allow_test_eval=True`; a
successful call appends timestamp, base, condition, size, seed, and call site to
the persistent ledger.

Other test references are not evaluation call sites:

- `src.analysis.errors` reads predictions already present in result files.
- `src.data.audit` reads test text only to prove an annotation sample is disjoint.
- `src.data.truncation` profiles split lengths.

The frozen launcher calls `run_single(..., eval_split="val")` explicitly.
