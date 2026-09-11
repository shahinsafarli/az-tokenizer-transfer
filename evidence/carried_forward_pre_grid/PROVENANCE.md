# Carried forward from BEFORE the 164-run grid — read this before citing anything here

These files were **not** produced by the 164-run grid. No launcher invokes the
scripts that make them, so they are absent from the grid's `results/`. They are
copied here from the pre-grid package because the report needs them and, in the
case of the label audit, they cannot be regenerated at all.

## The split assignment is NOT the same

| | Pre-grid (`splits_PRE_GRID.json`) | 164-run grid (`results/splits.json`) |
|---|---:|---:|
| AZ train | 20,936 | 20,937 |
| AZ val | 2,791 | 2,791 |
| AZ test | 4,187 | 4,186 |
| Content manifest | **absent** | present (test `ca19c0bb…`) |

One example moved between train and test. The cause is that
`data.az.hf_revision` is unpinned (`null`), so the upstream dataset was fetched
without a revision lock; the order-invariant content manifest was added
afterwards precisely so that this can be detected rather than assumed away.
Because the pre-grid package has no manifest, **it cannot be proven that the
underlying data is identical** — only that the totals differ by one example.

Consequence for the report: describe every file in this folder as
characterising **the dataset**, not a specific split, and never state that these
analyses were run on the same holdout as the results.

## label_audit.* — human double annotation, 100 sampled examples

Irreplaceable: two human annotators, and the sample was drawn under the pre-grid
split. Verified numbers, recomputed from the raw CSVs and the key:

| Quantity | Value |
|---|---|
| Three-class human–human agreement | 66/99 = **66.67%**, κ = 0.491 |
| Human–human on gold-retained (positive/negative) rows, **unconditioned** | 44/67 = **65.67%** |
| Human–human conditioned on all three sources committing to a polarity | 37/42 = **88.10%**, κ = 0.746 |
| Annotator 1 vs gold, retained rows | 46/67 = 68.66% |
| Annotator 2 vs gold, retained rows | 40/67 = 59.70% |
| Task ceiling (working, conservative) | 85.0% |
| Decision gate verdict | `PROCEED_WITH_CAVEAT` |

**Required disclosure.** The 88.10% figure is *conditioned* — it keeps only the
rows where the gold label and both annotators all committed to a polarity, which
selects for easy examples and is optimistic by construction. The report must give
the unconditioned retained-label figure, 44/67 = 65.67%, alongside it. Reporting
only 88.10% would overstate label quality by more than 22 percentage points.

Also note the annotator spread: 68.66% vs 59.70% against gold on the same rows.
The two annotators are not interchangeable, and the audit's own rule is to take
the worse of the two as binding — which is what produced `PROCEED_WITH_CAVEAT`
rather than `PROCEED`.

## The other five files

`diacritics.json`, `noise_interaction.json`, `corpus_fertility.json`,
`dataset_candidates.json`, `dataset_candidates_tr.json`.

All five are CPU-only analyses and **can be regenerated** against the grid's
split if the report cites them quantitatively. Until they are, cite them as
pre-grid dataset characterisation only. `corpus_fertility.json` in particular
overlaps with the grid's own `results/fertility.json` — prefer the grid's file,
and do not mix numbers from the two.
