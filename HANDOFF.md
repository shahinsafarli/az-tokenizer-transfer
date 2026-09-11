# PROJECT HANDOFF — Turkish→Azerbaijani Transfer: Lexical or Syntactic?

> **Read this file first.** It contains the full project context, everything measured so far,
> the exact code that exists, and the ordered list of what to do next.
> Written 30 August 2026. Deadline: **7 September 2026, 23:59** (~8 days).

---

## 1. What this project is

**Course:** Deep Learning Final Project · Track 1 (Pure Research) · DLE-AI-202, AI Academy / NAIC
**Deliverables:** GitHub repo (tag `v1.0-final`) + IEEE two-column paper + slides + oral defense
**Compute budget:** ~2 days of scheduled A100 time, ~10–12 GB peak VRAM

### Research question (REVISED TWICE — see §3.4 for the current version)

> Kardeş-NLU (EACL 2024) showed that fine-tuning on Turkish before Azerbaijani improves
> Azerbaijani task performance. It never explained **why**.
>
> **Where does the benefit come from — tokenization (how the words are cut up),
> lexical similarity (shared subwords warmed up), or syntax (grammar transferred)?**

We separate these three with a 5-condition design run across **two base models chosen to differ
in exactly one property: whether Azerbaijani was in pretraining.** That contrast is what turns a
measured effect into a causal claim about the mechanism.

### Why it matters

- Fills a documented gap: dozens of papers show "transfer from a high-resource relative helps",
  none separate *relatedness* from *shared subword inventory*.
- Generalizes: the method transfers to any low-resource language with a high-resource relative
  (Uzbek–Turkish, Belarusian–Russian, Catalan–Spanish).
- Cheap practical payoff: if the benefit is lexical, practitioners may skip an entire
  intermediate fine-tuning stage.

---

## 2. Current status

| Item | State |
|---|---|
| Tokenizer analysis (XLM-R) | ✅ **DONE** — real measurements, see §3 |
| **Two-base-model refactor** | ✅ **DONE** — every module is base-aware; 58 unit tests pass |
| Code base | ✅ **BUILT** — 58 unit tests passing |
| Analysis chain (incl. cross-base) | ✅ **TESTED end-to-end** on synthetic run data for both bases |
| Dedup-before-split + leakage assert | ✅ **DONE** |
| Figure 1 | ✅ **DONE** — `fig1_overlap_en.png` |
| Diacritic confound (T1/T2) | ✅ **DONE** — Option C confirmed, §3.6 |
| **XLM-15 fertility + anchor gates** | ✅ **DONE** — fertility PASSED, anchor `GO_WITH_CARE` via `functional` mode (T3), §3.7 |
| Dataset selection (AZ + TR) | ✅ **DONE** — LocalDoc (AZ) + TRSAv1 (TR) confirmed, config fully filled, §3.8 |
| Truncation confound | ✅ **DONE** — `max_length` 128→256, re-measured on binary corpus, §3.9/§3.14 |
| Licence stack + headroom gate | ✅ **RECORDED** — §3.10, sharpened per-base in §3.15 |
| **Label quality audit** | ✅ **DONE** — both sheets scored, `PROCEED_WITH_CAVEAT` (labels ~human-quality, HH=66.7% ceiling) → **binary adopted for measurement power**, §3.12 |
| Neutral class dropped, binary re-split | ✅ **DONE** — train 20,936/val 2,791/test 4,187, §3.13 |
| Noise×condition interaction test | ✅ **DONE** — no association found (Fisher p=1.0), §3.12 |
| Cross-base transplant-quality gate | ❌ not run — needs T7 (build+controls) for both bases |
| Transplant build (T7) | ❌ **NEXT** — no longer blocked behind the audit; needs torch |
| Fine-tuning runs (T8) | ❌ not started — smoke test allowed, full runs paused pending GPU-budget go-ahead |
| Paper / slides | ❌ not started |

### Environment (Windows machine)

```
Repo root : C:\Users\Lenovo\Desktop\az-tokenizer-transfer\az-tokenizer-transfer
venv      : C:\Users\Lenovo\Desktop\az-tokenizer-transfer\az-tokenizer-transfer\.venv
            (INSIDE the inner folder — beside the code, not one level up)
Run with  : C:\Users\Lenovo\Desktop\az-tokenizer-transfer\az-tokenizer-transfer\.venv\Scripts\python.exe -m src.<module> ...
```

⚠️ There is a **nested folder** (`az-tokenizer-transfer\az-tokenizer-transfer`). Both the venv and
the code live in the INNER folder. (An earlier version of this document said the venv was one
level up — that was wrong and cost time.) Activating the venv in a new terminal has failed
before — using the full path to `python.exe` always works, from the inner folder as cwd.

Installed: `transformers`, `tokenizers`, `sentencepiece`, `datasets`, `torch 2.4.1`, `numpy`,
`scikit-learn`, `scipy`, `pandas`, `matplotlib`, `pyyaml`, `pytest`.

---

## 3. What we measured on day 1 (and what it changed)

### 3.1 The original hypothesis — TESTED AND REJECTED

We believed XLM-R shreds Azerbaijani words because Azerbaijani has a tiny share of a
100-language vocabulary, and that Turkish fine-tuning indirectly compensates.

**Measured on 5,000 AZ + 5,000 TR Wikipedia sentences (102,726 words):**

| Model | AZ fertility | TR fertility | AZ/TR |
|---|---|---|---|
| xlm-roberta-base | 1.904 | 1.821 | **1.046** |
| HPLT/hplt_bert_base_az | 1.758 | 2.529 | 0.695 |

XLM-R penalises Azerbaijani by only **4.6%**. A dedicated Azerbaijani tokenizer improves
fertility by only **7.6%**. **There is no tokenization deficit to fix.** Hypothesis rejected.

Command that produced this: `python -m src.tokenization.corpus_fertility --config configs/experiment.yaml --n-sentences 5000`
Raw output: `results/corpus_fertility.json`

### 3.2 What we found instead — token overlap is driven by SCRIPT, not language family

Azerbaijani token types that also appear in each control language (alphabetic tokens only,
5,000 Wikipedia sentences per language):

| | Turkish (related) | English (control) | Finnish (control) | Russian (Cyrillic) |
|---|---|---|---|---|
| **XLM-R** | **46.4%** | 34.1% | 33.6% | **9.0%** |
| **HPLT** | **32.1%** | 19.3% | 19.1% | **7.7%** |

**Reading:**
- English ≈ Finnish (34.1 vs 33.6, spread **0.5 points**) — neither is related to Azerbaijani,
  yet they give the same number → that level is a **script / international-vocabulary baseline**,
  not relatedness. This also rules out a topic confound.
- Russian collapses to 9% — the only difference is the writing system → confirms the script effect.
- Turkish sits **+12.6 points** above the Latin baseline (≈**1,264 Turkish-specific token types**
  out of 10,054 AZ alphabetic types). Signal-to-noise = 12.6 / 0.5 = **25×**.
- The lift is nearly identical across two completely different tokenizers (+12.6 and +12.9)
  → it is a fact about the *languages*, not an artifact of one tokenizer.

Command: `python -m src.tokenization.overlap_control --config configs/experiment.yaml --n-sentences 5000 --langs tr,en,ru,fi`
Raw output: `results/overlap_control.json`

### 3.3 Anchor analysis (GO/NO-GO gate for the tokenizer transplant)

| Metric | Value |
|---|---|
| Naive string intersection | 2,856 |
| **Canonical shared anchors** | **10,662** (32.6% of donor vocab) |
| Unfamiliar tokens (to be reconstructed by OMP) | 22,079 |
| Gain from canonical normalisation | **3.7×** |
| Verdict | `GO_WITH_CARE` |

⚠️ **The 3.7× gain is a methodological contribution — mention it in the paper.** HPLT's tokenizer
is byte-level encoded (`▁` is stored as `âĸģ`, `ə` as `ÉĻ`). Without byte-decoding + canonical
matching, the anchor count reads as 2,856 and the project would have falsely aborted with `NO_GO`.

Raw output: `results/anchors.json`

### 3.4 The pivot: TWO base models instead of one  ← READ THIS FIRST

The day-1 measurement above did not kill the project — it killed the **choice of base model**.
XLM-R was only ever chosen because Kardeş-NLU used it. But our research question is about the
**mechanism** (does understanding come from tokenization or from lexical similarity?), and
nothing obliges us to inherit their model. So we changed it.

**Second base model added: `FacebookAI/xlm-mlm-tlm-xnli15-1024` (XLM-15).**
15 pretraining languages — **Turkish is in, Azerbaijani is out.** That is exactly the
configuration the original hypothesis assumed, and XLM-R never satisfied it.

| | `xlm15` (**primary**) | `xlmr` (**contrast**) |
|---|---|---|
| HF id | `FacebookAI/xlm-mlm-tlm-xnli15-1024` | `xlm-roberta-base` |
| Pretraining languages | 15 | 100 |
| Turkish present | ✅ yes | ✅ yes |
| **Azerbaijani present** | ❌ **no** | ✅ yes |
| AZ/TR fertility | **to be measured (gate: expect > 1.3)** | 1.046 (measured) |
| Tokenizer knob | has range → segmentation can matter | stuck → segmentation cannot matter |
| Role in the design | where the effect should appear | **zero-effect control** |
| Licence | CC-BY-NC-4.0 (research use — fine) | MIT |

**Why this is better than replacing XLM-R.** Keeping both turns the day-1 rejection from a
setback into a necessary component: XLM-R is now a *negative control*. The claim we can make
with two models is far stronger than anything one model allows:

```
tokenizer effect(xlm15) ≫ tokenizer effect(xlmr) ≈ 0
    → the effect tracks the tokenization deficit  → MECHANISM CONFIRMED

tokenizer effect(xlm15) ≈ tokenizer effect(xlmr)
    → the effect is not segmentation; it is embedding re-initialisation
    → MECHANISM REJECTED (still a publishable finding)
```

A single model can only ever show *that* something happened. Two models with different
Azerbaijani coverage show **why**. This is implemented as
`decompose.compare_bases()` → `results/decompose.json → cross_base_comparison.verdict`,
with verdicts `MECHANISM_SUPPORTED` / `MECHANISM_PARTIAL` / `MECHANISM_REJECTED` / `UNEXPECTED`,
and plotted as Figure 5.

**Cost control:** `primary` runs all 4 training sizes; `contrast` runs only the reference size
(2,000). That is 60 + 15 = **75 runs**, not 120. Configured via `run.contrast_sizes`.

**Accepted trade-off:** Kardeş-NLU is no longer a setup we extend — it becomes the work that
*motivates* the question. Be ready to say this out loud in the defence; it is a deliberate
choice, not an oversight.

### 3.5 Consequence for the project

**The work does not change — the framing does.** Same 5 conditions, same controls, same code,
same budget. What changes:

- Main question: "is the benefit tokenization?" → "**is the benefit lexical or syntactic?**"
- Primary conditions: 1, 3 → **1, 2, 5** (none of which touch the tokenizer)
- Conditions 3, 4 become a **secondary but valuable check**: we predicted a small effect from
  fertility, and confirming that downstream answers an open literature question —
  *does better tokenization translate into better downstream performance?*
- The fertility/overlap measurement is no longer motivation; it is the **paper's first result**.

**Revised H1:** Most of the Turkish transfer benefit comes from embedding warm-up on ~1,264
Turkish-specific shared subword types, not from grammatical transfer.
**H2:** The benefit shrinks as Azerbaijani training data grows.
**H3:** The effect is stronger on technical/formal text than colloquial text
(Turkish language reform split the technical vocabulary: `bilgisayar`/`kompüter`, `uçak`/`təyyarə`).

### 3.6 Diacritic confound — measured (T1) and resolved (T2)

Swapping XLM-15's tokenizer for the donor's does not change *only* segmentation. XLM-15 uses
`do_lowercase_and_remove_accent=True` (the XLM default), so five of Azerbaijani's nine special
letters are destroyed at the input stage, before segmentation even happens. Any Condition-3
improvement is therefore a **compound treatment**: better segmentation *and* restored diacritics,
mixed together. This had to be quantified before touching the anchor logic (T3), because an
examiner will ask about it.

**Measured** (`python -m src.tokenization.diacritics --config configs/experiment.yaml
--n-sentences 5000` → `results/diacritics.json`):

| | flag=True (actual training config) | flag=False |
|---|---|---|
| XLM-15 AZ fertility | 3.2064 | 3.7278 |
| XLM-15 TR fertility | 1.7676 | 2.6871 |
| AZ/TR ratio | **1.814** | 1.387 |

Turning the flag off makes fertility *worse* for both languages (AZ +0.52, TR +0.92
tokens/word) — it is not a usable fix. Vocab letter counts (95,000 XLM-15 tokens; HPLT is
byte-level and was decoded via `canon.decode_vocab` before counting, or its diacritics would
have read as zero too — the same class of bug as gotcha #1):

| | ş | ç | ö | ü | ğ | ə | ı | İ |
|---|---|---|---|---|---|---|---|---|
| xlm15 | 0 | 0 | 0 | 0 | 0 | 2 | 1240 | 0 |
| xlmr | 891 | 631 | 1422 | 1189 | 407 | 1366 | 1404 | 144 |
| HPLT (donor) | 2253 | 967 | 1069 | 2094 | 928 | 8814 | 4241 | 981 |

**Reading:** `ş ç ö ü ğ` are **structurally absent** from XLM-15's vocab — erased during
pretraining, not just at tokenization time — so the flag cannot recover them; it only produces
UNK/lone-byte tokens (hence fertility gets worse, not better, with the flag off). `ə` survives
only as 2 vocab entries, i.e. essentially always emitted as an isolated single-character token.
`ı`/`İ` are untouched because they carry no Unicode combining accent to strip, and were already
common from Turkish.

**Conclusion:** the confound is real and **cannot be turned off with a tokenizer-load flag** —
fixing it requires normalizing the *text*, not the config.

**Decision (confirmed): Option C.** Report Condition 3 throughout as "tokenizer replacement
(segmentation + orthography)" and state the confound in Limitations (Option A), **plus** one
extra ablation at the reference size (2,000): run Condition 3 a second time on accent-stripped
AZ/TR text (lowercase + strip `ş ç ö ü ğ`, same rule for both the base and donor tokenizer paths
so only *segmentation* differs) and diff it against the raw-text Condition 3. That difference is
the measured orthographic component. Rejected: Option B (pre-normalize everywhere) — it would
make every condition non-comparable with prior work and throw away information the donor
tokenizer can legitimately use; a confound we *measure* is worth more than one we *design away*.

**Cost:** ~3 extra runs at `reference_size` only (not the full data-size curve) — cheap, per the
brief's "cleanliness of controls > breadth of experiments" rule.

**Not yet implemented** — per the brief, T2 is decision-only. The accent-stripping ablation runs
are scheduled for **Step E/F** (transplant + smoke-test stage). The `mean` and `random_coef`
Condition-3 controls are now promoted to a separate, mandatory pre-full-run gate (see §3.18);
the `k=8`/`k=128` breadth ablations remain later work.

### 3.7 Anchor logic fixed (T3) — `functional` mode, symmetric across both bases

The `NO_GO` on XLM-15 was a measurement artifact: `canon.py` mislabeled its fastBPE vocabulary
(`</w>` end-of-word marker) as `plain` and found almost no anchors. Fixed in
`src/tokenization/anchor_map.py` (`compute_anchors(base_tok, donor_tok, mode)`), the single
function now called by **both** `anchors.py` (the gate) and `build.py` (the real transplant) —
they can never diverge again. Three modes, `strict`/`surface`/`functional` (see code docstrings
for the exact algorithm); GO/NO-GO and the real transplant both key off `functional`.

| Base | scheme (now correct) | strict | surface | **functional (used)** |
|---|---|---|---|---|
| xlm15 ← HPLT | `fastbpe` (was `plain`) | 1,637 | 3,938 | **8,079** (24.65%) → `GO_WITH_CARE` |
| xlmr ← HPLT | `sentencepiece` | 10,662 (regression-protected) | 9,539 | 9,776 (29.83%) |

**Decision: `functional` for BOTH bases, always — methodological symmetry, not convenience.**
`strict` happens to be slightly higher for xlmr (10,662 vs 9,776 — xlmr "loses" 886 anchors under
`functional`), but using `strict` for xlmr and `functional` for xlm15 would make the two arms of
the cross-base comparison (§3.4, the paper's whole causal argument) differ in *method* as well as
in *model* — the same class of defect as the T2 orthography confound. Paying 886 anchors on the
xlmr side buys a comparison where the only thing that differs between arms is the base model, not
also the anchor-finding algorithm. State this explicitly in the methods section.

**`union(strict, functional)` is logged as a diagnostic only — never used to build a transplant.**
For xlmr: union = 11,505 (strict-only 1,729, functional-only 843, both cite HPLT). For xlm15:
union = 8,534 (strict-only 455, functional-only 6,897). `strict`'s xlm15 anchors come from
fastBPE falling through to the `plain` catch-all in `canon.py` (deliberately — see T3 code
comments) — those pairs match a donor word-start token against a base token whose boundary status
is literally unknown. A few hundred extra atoms aren't worth mixing that ambiguity into the
embedding matrix itself. `results/anchors.json → bases.<base>.union_strict_functional_diagnostic_only`
carries the numbers so a reviewer can see it was checked.

**Cross-base transplant-quality gate — required before T8's full runs, not yet run (needs T7).**
Anchor *share* of donor vocab differs: 24.65% (xlm15) vs 29.83% (xlmr) — OMP reconstructs xlm15's
tokens from a sparser dictionary. Before the full runs, compare for both bases:
- `results/transplant__<base>__<tag>.json → reconstruction.mean_cosine`
- `results/controls__<base>.json → C1c_delta_bpc`, always reported together with the matching
  top-1 counts from `results/top1_accuracy.json` (per-base file — see §3.16, was a flat
  `controls.json` that would have overwritten between bases; fixed)

If xlm15's reconstruction is materially worse, its Condition 3 could underperform for
*transplant-quality* reasons rather than *segmentation* reasons — this would bias **against**
finding the hypothesized mechanism (a conservative failure mode, not a fatal one), but it must be
measured and reported regardless of which way it comes out. **Do not proceed to T8's full runs
without both numbers in hand for both bases.**

**`transplant.k` stays identical across both bases in the main runs** — tuning it per base would
reintroduce the exact asymmetry just ruled out above. Confirmed no drift: `n_candidates=256 <
768` (xlmr hidden) and `256 < 1024` (xlm15 hidden), `k=64 ≤ 256` — config unchanged, both bases
share the single `transplant:` block. The `k ∈ {8, 32, 64, 128}` ablation (already required, §T7)
carries the sensitivity analysis instead of per-base tuning.

### 3.8 Datasets chosen (T4, AZ + TR) — CONFIRMED, config now fully filled

`src/data/discover.py` (the generic Hub-search script this document originally planned) was
**never actually written** — this document was stale on that point, and the human confirmed the
correction (the brief that spawned this project has been fixed to match). Since candidates were
already named (AZ) or found by direct Hub search (TR), `src/data/profile_candidates.py` profiles
each against the brief's criteria plus a README provenance check. Reproduce:
`python -m src.data.profile_candidates --config configs/experiment.yaml --lang all` →
`results/dataset_candidates.json` (AZ) and `results/dataset_candidates_tr.json` (TR).

**AZ — `LocalDoc/sentiments_dataset_azerbaijani`. CONFIRMED, locked into `configs/experiment.yaml`
(`data.az.hf_name`/`text_column`/`label_column`).**

| Criterion | `LocalDoc/sentiments_dataset_azerbaijani` | `hajili/…tweet_emotion_classification` |
|---|---|---|
| Rows after dedup | 41,723 (dup 0.66%, 29 conflicts) | 139,034 (dup 7.31%, 24 conflicts) |
| Gated / licence | not gated / `cc-by-nc-sa-4.0` | not gated / `mit` |
| Classes / majority | 3 / 33.4% | 3 / 35.9% |
| Headroom proxy (TF-IDF+LogReg macro-F1, **not final** — see §3.9) | 0.578 | 0.433 |

Not a criteria-count win alone — `readme_provenance` caught that hajili's own README states its
3 labels come from **an emoji-based rule classifier**, and that emojis were **then stripped from
the text** in cleaning. The signal that produced the label is gone from what any model would
ever see — disqualifying, confirmed by the human as the right call ("labels derived from a
signal that was then deleted from the input are not labels, they're noise with structure").

**TR — `maydogan/Turkish_SentimentAnalysis_TRSAv1`. CONFIRMED, locked into
`configs/experiment.yaml` (`data.tr.hf_name`/`text_column`/`label_column`).** Found via
`HfApi().list_datasets(search=...)` (71 unique hits across 5 search terms), profiled the top 3:

| Criterion | **TRSAv1 (chosen)** | `winvoker/turkish-sentiment…` | `fthbrmnby/turkish_product_reviews` |
|---|---|---|---|
| Rows after dedup | 138,640 (dup 7.6%, 937 conflicts) | 487,385 (dup 0.5%, 97 conflicts) | 233,370 (dup 0.8%, 76 conflicts) |
| Licence | **none on HF card** (see caveat) | `cc-by-sa-4.0` | `unknown` |
| Classes / majority | 3 / 34.6% | 3 / 53.5% | 2 / **93.7%** (fails) |
| Headroom proxy (TF-IDF+LogReg) | **0.788 — in 0.60–0.85 band** | 0.842 — in band, but see below | 0.538 — out of band |
| Provenance | peer-reviewed: Aydoğan & Kocaman, *J. Information Science*, Feb 2022, doi:10.1177/01655515221074328 — single domain (Turkish e-commerce reviews) | crowdsourced patchwork of 5 sources | "found" data, no methodology given |

`winvoker` scores 6/6 criteria on paper, but `source_composition_by_label` (a generic
per-label × metadata-column cross-tab the profiler now computes whenever a candidate declares a
source/provenance column) caught that its **`Notr` (neutral) label is 99.7% sourced from Turkish
Wikipedia dump text**, not from actual neutral-sentiment reviews — a domain-leakage /
construct-validity defect: a classifier could hit that headroom by learning "is this Wikipedia
prose" rather than "is this sentiment-neutral," which is very plausibly why its proxy score is
the highest of the three. `fthbrmnby` fails outright on class balance (93.7% positive). **TRSAv1
is the only candidate that is (a) in the headroom band, (b) free of the domain-leakage pattern,
and (c) traceable to a peer-reviewed source** — chosen despite lacking an explicit SPDX licence
tag on its HF card (flagged, not swept under the rug; academic-benchmark norm is citation, not
an OSI licence, but this means the repo cannot claim a clean redistribution licence for this
dataset specifically — cite the paper, do not imply a permissive licence it doesn't declare).

Per the human's instruction, class-count parity with AZ was **not** required for TR (`min_classes
= 2`, not 3) — the classification head is discarded between stages ([C3]), so the label space
need not match. TRSAv1 happens to be 3-class anyway.

`configs/experiment.yaml` has **no remaining `FILL` fields** — `load_config(require_complete=True)`
passes.

### 3.9 Truncation confound (T5) — `max_length` raised to 256, CONFIRMED

Initial measurement at 128 showed Condition 3 (donor tokenizer) truncating 2.94pp less than
Condition 1/2/5 within xlm15 — flagged as a real but modest confound, with the cost of fixing it
via `max_length` left as an open question (the brief lists that as a human-call decision).

**The human's correction**: `finetune.py` uses `DataCollatorWithPadding(tokenizer)` with no
`padding="max_length"` anywhere in the pipeline (verified directly in the code before acting) —
batches pad dynamically to the **longest example in that batch**, not to the configured ceiling.
`max_length` is a truncation ceiling only; it does not set per-batch compute cost. Verified this
before implementing, per instruction. So raising 128→256 costs real compute only in the ~1–5% of
batches that contain a long example, not across the board — the "~2× cost" read from the first
report was wrong.

**Decision: `models.max_length: 128 → 256`, confirmed in `configs/experiment.yaml`.** Re-measured
on the full 41,723-row deduped LocalDoc corpus, both ceilings kept side by side in
`results/truncation.json → by_max_length`:

| Setup | truncation @ 128 | truncation @ 256 |
|---|---|---|
| xlm15 original (Cond 1/2/5) | 4.58% (1,913/41,723) | **1.16%** (485/41,723) |
| donor/transplanted (Cond 3/4) | 1.64% (686/41,723) | **0.34%** (140/41,723) |
| xlmr original (contrast) | 2.09% (872/41,723) | **0.44%** (184/41,723) |
| **cross-condition gap** (xlm15-original − donor) | 2.94pp | **0.83pp** |

Matches the human's prediction closely (predicted "~1%" and "under 1pp"; measured 1.16% and
0.83pp). Reporting the 2.94pp gap as a stated limitation at 128 would have been the lazy choice
given how cheap the fix was — raising the ceiling was correct. `training.batch_size: 32` is now
locked across machines and conditions. **If OOM occurs during the T8 smoke test at 256, stop and
tell the human; do not silently lower either batch size or `max_length`.**

---

### 3.10 Licence stack and the headroom gate (recorded now, while still predictions)

**Licence stack.** `LocalDoc/sentiments_dataset_azerbaijani` is `CC-BY-NC-SA-4.0`; the primary
base model `FacebookAI/xlm-mlm-tlm-xnli15-1024` (XLM-15) is `CC-BY-NC-4.0`. Non-commercial is
fine for coursework, but **share-alike propagates**: any transplanted checkpoint or fine-tuned
model built from LocalDoc's data and published publicly inherits the SA (share-alike) obligation
from the dataset licence on top of the NC (non-commercial) restriction from both the dataset and
the base model. **Do not publish a checkpoint/artifact from this pipeline without carrying both
restrictions in its own licence/model card.** `maydogan/Turkish_SentimentAnalysis_TRSAv1` (TR
stage) has no declared licence at all — cite the paper, do not claim a redistribution licence for
it. State all of this explicitly in the paper's ethics/licence line.

**Headroom revisit trigger — three concrete thresholds, written down now as predictions, not
rationalisations.** The 0.578 TF-IDF proxy is a known-pessimistic lower bound and does **not**
block proceeding. But it becomes a real gate at the first real number: after the first xlm15
Condition 1 (`baza`) run at `n=2000` (T7/T8), check the baseline `test_macro_f1`:

| Real Condition-1 macro-F1 | Action |
|---|---|
| **0.60 – 0.85** | Proceed — headroom confirmed, as hoped |
| **> 0.90** | **STOP, tell the human before burning the GPU window.** Task is saturated; Condition 3 (tokenizer swap) has nowhere to move — the whole design loses its ability to show an effect |
| **< 0.45** | **STOP, tell the human.** Suspect label noise (T6 will have already run by this point — if the audit passed at ≥85% agreement but the model still can't learn the task, that itself is worth reporting) |
| 0.45 – 0.60 or 0.85 – 0.90 | Borderline — proceed but flag in the paper as marginal headroom, do not silently treat as clean |

Write the observed value and the action taken into `results/` when this check actually runs —
do not skip recording it even if it lands safely in-band.

### 3.11 T6 audit kit built — ready to hand to the two native speakers

`src/data/audit.py`, three subcommands: `export`, `verify`, `score`.

**Deliberately diverges from the brief's original plan.** The brief's Step C sketched a single
CSV with `label` and a blank `verdict` column — but showing the annotator the gold label anchors
their judgement toward confirming it, which would inflate the measured agreement rate and defeat
the point of an *independent* audit. Instead:

- **Two separate, fully blind CSVs** (`label_audit_annotator1.csv`, `label_audit_annotator2.csv`)
  — columns `idx, text, your_label` only, `your_label` empty, **no gold label anywhere**, so the
  two annotators can't see each other's sheet or the dataset's own label.
- Gold labels live only in `label_audit_key.json` — **private, not for the annotators**.
- `label_audit_INSTRUCTIONS.md` gives the label schema (`negative / neutral / positive`) without
  gold answers.

**Confirmed programmatically, not asserted** — `verify_no_leak()` checks (a) no forbidden column
name (`label`, `gold`, `emotion`, …) appears in either blind CSV, (b) the column set is exactly
`idx, text, your_label`. Runs automatically after `export` and is safe to re-run any time,
including after the sheets are filled in. Guarded by 4 new tests (42 total now) — including a
regression test for a real bug caught and fixed during development: an earlier version flagged
`your_label == gold` as a "leak," which would have falsely flagged *every correct annotation* —
fixed to check structure only, not values, and verified the fix by simulating a 100%-correct
annotator and confirming `verify` still reports clean.

**Two more fixes made before any human saw a sheet — both irreversible if caught late.**

1. **Sampling source.** The first version sampled from the full deduped LocalDoc corpus (41,723
   rows) — but the test split is 15% of that, so ~15 of 100 examples would have been test-set
   rows, putting two humans' eyes on test data and letting test content influence a
   dataset-level decision. Non-negotiable per §3 of the project brief ("test set touched exactly
   once"). Fixed: `_load_az_train()` now reads **only** `artifacts/data/az_train.jsonl`. Before
   writing anything, `assert_sample_disjoint_from()` normalizes and checks the sampled texts
   against both `az_val.jsonl` and `az_test.jsonl` and raises if any overlap is found — this ran
   clean on the actual export (`TƏSDİQLƏNDİ: audit nümunəsi ... ÜST-ÜSTƏ DÜŞMÜR`). 2 new tests
   cover the checker itself (disjoint passes, overlap raises after normalization). Extracted
   `normalize_text()` as a shared function in `splits.py` so `_dedup` and this check can never
   drift apart on what counts as "the same text."
2. **Forced-choice schema inflates agreement.** `negative/neutral/positive` with no escape hatch
   means an annotator who genuinely can't tell will guess — two guessers agree ~33% of the time
   by chance, pushing measured agreement up and hiding exactly the label noise being tested for.
   Added a 4th value, `unclear` (`?` also accepted), to both sheets and `INSTRUCTIONS.md`.
   `score` excludes `unclear` rows from **both** denominators (annotator↔dataset per annotator,
   and annotator↔annotator — excluded if *either* side said unclear) and reports the exclusion
   count and rate per annotator, flagging `text_quality_concern: true` above 15%. Also added the
   six borderline rules to `INSTRUCTIONS.md` (mixed sentiment → stronger side, tie → unclear;
   sarcasm → intended meaning; sentiment-free question → neutral; bare factual statement →
   neutral unless dissatisfaction implied; truncated/garbled/non-Azerbaijani → unclear; profanity
   alone isn't a label) — without these, disagreement between the two annotators would measure
   the gap in guidance, not the gap in the data.

**Verdict logic is two-dimensional, not a single threshold** — confirmed. Human↔human agreement
(HH) and human↔dataset agreement (HD) measure different things and point to opposite fixes.

⚠️ **SUPERSEDED 2025-09-02 — kept for the record, do not use.** The table below judged HD against
an *absolute* floor. Problem, caught by the human after the real audit ran: **HH itself doesn't
reach 85% on any three-class sentiment task** — the floor would reject a dataset regardless of
the dataset's own quality, testing "is this task easy for humans" rather than "are these labels
good." See §3.12 for the corrected, relative-to-HH table that replaced it.

| | | |
|---|---|---|
| HH < 85% | *(regardless of HD)* | ~~`STOP_TASK_AMBIGUOUS`~~ |
| HH ≥ 85%, HD < 85% | | ~~`STOP_LABELS_UNRELIABLE`~~ |
| HH ≥ 85%, HD ≥ 90% | | ~~`PROCEED`~~ |
| HH ≥ 85%, 85% ≤ HD < 90% | | ~~`PROCEED_WITH_CAVEAT`~~ |

The original synthetic tests for this table were rewritten for the corrected version in §3.12
(same branch coverage, corrected thresholds and fixture numbers). 59 tests total, all passing.

**Re-generated (2025-09-01, seed 4242) after both fixes**: 100 examples from `az_train.jsonl`
only, stratified 33/34/33 across neutral/positive/negative, disjointness from val/test confirmed
programmatically, gold-leak absence re-verified.

### 3.12 Audit interpreted — binary task adopted — CONFIRMED against real data (both sheets)

**`annotator2.csv` was empty in this repo when this section was first written (2025-09-01). Once
the real completed sheet was placed here, `python -m src.data.audit score` reproduced every
number below from committed code** — nothing here was hand-typed. `results/label_audit.json` is
the artifact of record.

```
Agreement, n=100, abstentions excluded (1 in annotator2 — see the encoding note below):
  Annotator 1 vs dataset     71.00%   κ=0.5659   95% CI [62.0, 79.0]
  Annotator 2 vs dataset     58.59%   κ=0.3800   95% CI [49.5, 67.7]
  Annotator 1 vs Annotator 2 66.67%   κ=0.4911   95% CI [57.6, 75.8]

Disagreement decomposition:
  A1 vs A2:      33 disagreements — 28 neutral-boundary (84.85%), 5 polarity flips (15.15%)
  Dataset vs A1: 29 disagreements — 19 neutral-boundary (65.52%), 10 polarity flips (34.48%)
  Dataset vs A2: 41 disagreements — 35 neutral-boundary (85.37%),  6 polarity flips (14.63%)

Polarity-only (dataset+A1+A2 all committed, n=42): HH = 88.10%, κ=0.7458, CI [78.6, 97.6] —
conditions on commitment (optimistic). task_ceiling_pct recorded as 85.0 (working estimate),
not the 88.1% point estimate quoted bare.
```

Matches the human's hand-relayed figures exactly (71.0/58.6/66.7, κ≈0.566/0.380/0.491, decomposition
84.85%≈85%, 65.52%≈66%, polarity-only 88.1%/CI[78.6,97.6]).

**Encoding bug caught and fixed — corrected framing (2025-09-02): the annotators read clean
Azerbaijani; only a saved *copy* was affected, and it never touched scoring validity.** The
exported sheets are written in UTF-8 and were unmangled when annotators opened and filled them
in. The corruption happened **afterward**, when Excel *saved the file back*: on that write it
used the Windows local codepage (cp1254 here), which has no mapping for `ə` — so annotator2's
`bilmirəm` landed in the saved copy as literal `bilmir?m`. This is harmless for two independent
reasons: (1) scoring only ever reads `idx` and `your_label` (ASCII digits, untouched by any
codepage), never the sheet's `text` column — the canonical text lives in `az_train.jsonl`; (2) it
was still a real bug worth fixing, because `bilmir?m` isn't in `UNCLEAR_VALUES` and would have
been scored as a wrong answer instead of excluded as an abstention. Fixed with a prefix check
(`UNCLEAR_PREFIXES = ("bilmir",)`, catches any mangling regardless of what the corrupted character
becomes) and regression-tested (`test_is_unclear_catches_encoding_mangled_bilmirem`). **HD1 and
HD2 above are valid, real figures — not lower bounds, not degraded by anything an annotator saw.**
Prevented from recurring: sheets now export as `utf-8-sig` (BOM-tagged UTF-8), which Excel opens
*and saves back* as UTF-8 without guessing at a codepage; round-trip-tested
(`test_write_csv_round_trips_azerbaijani_text_via_utf8_sig`). No re-audit, no re-annotation needed.

**Verdict — corrected 2025-09-02, this is the one correction that changes an actual number.** The
original table (marked superseded above) judged HD against an absolute floor (85%/90%) — but HH
itself doesn't reach 85% on *any* three-class sentiment task, so that branch would reject every
dataset regardless of quality. Corrected: judge HD **relative to HH**, since HH is the achievable
task ceiling, not a fixed target.

| Condition | Verdict |
|---|---|
| HH < 60% | `STOP_TASK_TOO_SUBJECTIVE` |
| HD ≥ HH − 5pp | `PROCEED` — labels at human quality |
| HD ≥ HH − 12pp | `PROCEED_WITH_CAVEAT` |
| HD < HH − 12pp | `STOP_LABELS_UNRELIABLE` |

Applied per annotator (HD1, HD2 — still never averaged), the worse of the two tiers is binding
(same "weakest link" principle as before, now relative rather than absolute):

```
HH = 66.67% (≥ 60% floor, task is not hopelessly subjective)
Annotator 1: HD1=71.00%, diff = +4.33pp → PROCEED
Annotator 2: HD2=58.59%, diff = −8.08pp → PROCEED_WITH_CAVEAT   (between −12pp and −5pp)
Binding (worse of the two): PROCEED_WITH_CAVEAT
```

Annotator 1's number alone reproduces the human's own worked example exactly (HH=66.7%,
HD1=71.0% → PROCEED, "the dataset agrees with annotator 1 better than the annotators agree with
each other"). The **binding/overall** verdict accounting for annotator 2 as well is
`PROCEED_WITH_CAVEAT`, not bare `PROCEED` — flagged here rather than quietly matched to the
single-annotator example, since silently picking the more favorable of two annotators is exactly
the kind of averaging-over-the-spread this project has repeatedly corrected against. Both
readings agree on the substance: **the dataset's labels are approximately human quality, not
unusable.** `results/label_audit.json → decision_gate` carries `tier_annotator1`,
`tier_annotator2`, and the binding `verdict` all separately, so a reader can see both.

**Framing correction, applies wherever "why binary" comes up**: three-class did **not** fail an
unusable-dataset gate. **HH = 66.7% is the honest human ceiling for three-class sentiment on this
data — a real, load-bearing property of the task, not a defect.** The move to binary is about
*measurement power*, not escaping bad labels: collapsing neutral into the polarity-only judgement
lifts the ceiling to ~85–88% (§3.15's noise-attenuation math) and recovers roughly a third of the
lost statistical power, on labels that were already approximately human-quality. State it this
way in the paper — "moved to binary to improve measurement power on labels at human quality," not
"the three-class dataset was unusable." Implemented in T3 (§3.13), pre-registered before any
model was trained.

**T1 — label mapping `1=positive, 2=neutral, 3=negative` — fully verified against both
annotators.** `build_label_mapping()` confirms all three probes (idx 38/48/57) against gold
**and both annotators' actual digits** — `fully_verified_both_annotators: true`.

**T2 — scorer** (`src/data/audit.py::score_audit`) writes `results/label_audit.json`:
`label_mapping`, **HD1/HD2 reported separately, never averaged** (`hd_spread_pct: 12.41`, flagged
— the two annotators are measurably not applying the same threshold, and both readings are
equally valid, see the encoding note above), three κ values each with a bootstrap 95% CI, the
neutral-boundary/polarity-flip decomposition for all three pairs, the polarity-only subset,
per-annotator abstention counts (0 / 1), `task_ceiling_pct`, and the corrected relative-to-HH
verdict (`tier_annotator1`, `tier_annotator2`, binding `verdict`).

**T5 — noise×condition interaction test, run for real** (`src/analysis/noise_interaction.py` →
`results/noise_interaction.json`). Turkish-specific shared subword types computed fresh as
`AZ∩TR − AZ∩(EN∪FI)` (5,000 Wikipedia sentences/language, `xlm-roberta-base`): **1,816 types**.
**This is a different quantity from the ~1,264 figure in §3.2 — neither supersedes the other.**
1,264 is the *relatedness-attributable excess* (the +12.6-point lift over the 33.8% Latin-script
baseline, applied to 10,054 AZ alphabetic types — an existing paper result). 1,816 is a plain
set-difference count (AZ∩TR minus AZ∩(EN∪FI)) built specifically for this interaction test's
per-item token check — a different, cruder construction serving a different purpose. Both stand.

```
Consensus items (both annotators agree, non-abstaining): 66 of 100 (34 excluded — 33 A1/A2
  disagreements + 1 abstention)
Contingency: disagree+has-token=15, disagree+no-token=1, agree+has-token=48, agree+no-token=2
Rate(has Turkish-specific token | disagreement) = 93.75%
Rate(has Turkish-specific token | agreement)    = 96.00%
Fisher exact: OR=0.625, p=1.0000 → NO ASSOCIATION
```

**No association found — the common-mode noise defence holds, one sentence for Limitations,
objection closed per the pre-registered rule.** Honest caveat this session is adding, not in the
pre-registered rule but worth stating plainly: at n=66 with two contingency cells at 1 and 2, this
test has limited power — the near-identical 93.75%/96.00% rates come from Turkish-specific tokens
being extremely common in short informal AZ text generally (found in the large majority of items
regardless of agreement), leaving little room for a differential signal either way. The null
result is real and the Fisher test is exact regardless of small counts, but it should be read as
"no association *detected at this sample size and base rate*," not as strong evidence of a large
true effect being ruled out. State this alongside the one-sentence Limitations note rather than
dropping the caveat.

---

### 3.13 T3 — neutral class dropped, binary corpus re-split. Done, numbers below.

`configs/experiment.yaml → data.az.exclude_labels: [neutral]`. Implemented as
`src/data/splits.py::apply_exclude_labels()` — called after `_load_records`, before `_dedup`/
`_encode_labels` (not hardcoded in the loader), so label indices stay contiguous (0/1, not 0/_/2)
and dedup statistics describe the **final** (binary) corpus, not the pre-exclusion one. 2 new
unit tests (drops only the named class; no-ops when absent).

Reproduce: `python -m src.data.splits --config configs/experiment.yaml` then
`python -m src.data.scramble --config configs/experiment.yaml`.

```
42,000 → 28,000 after excluding neutral (14,000 dropped) → 27,914 after dedup
  duplicate_pct: 0.31% (was 0.66% pre-exclusion — neutral carried more of the duplicate mass)
  conflicting_texts: 0 (was 29 — all resolved by dropping neutral)
train 20,936 / val 2,791 / test 4,187   (label-balanced: train 10,464/10,472, test 2,093/2,094)
leakage_check: train_test=0, train_val=0, val_test=0  (clean)
```

`train=20,936 ≥ 10,000` — no stop triggered. (Human's rough estimate was ~20,914/2,789/4,182;
actual is within ~0.1% — fine, consistent with "roughly.") TR side (`scramble.py`) re-ran clean:
20,000 sentences, word-multiset preserved 100%, 19,567 reordered.

### 3.14 T4 — truncation re-measured on the binary corpus. 256 remains the right ceiling.

`src/data/truncation.py` rewritten to read `artifacts/data/az_{train,val,test}.jsonl` directly
(the files `splits.py` itself just wrote) instead of re-deriving dedup/exclusion independently —
so this measurement can never silently drift from what training actually sees. Reproduce:
`python -m src.data.truncation --config configs/experiment.yaml`.

```
                          @128 truncation      @256 truncation
xlm15 original (1/2/5)   5.16% (1,441/27,914)  1.26% (351/27,914)
donor/transplanted (3/4) 1.81% (504/27,914)    0.35% (98/27,914)
xlmr original (contrast) 2.31% (644/27,914)    0.47% (132/27,914)
cross-condition gap      3.36pp                0.91pp
```

**256 confirmed as the right ceiling** — gap stays under 1pp, consistent with the T5 finding on
the 3-class corpus. One honest surprise: mean/p90 length went slightly *up* (47.4→49.0 for
xlm15-original), not down — the "long formal texts concentrated in neutral" hypothesis predicted
the opposite direction. Small effect (both gap figures moved by <0.5pp versus the 3-class
measurement), reported as measured rather than forced to match the prediction.

### 3.15 Two rules recorded before T8's runs (per the human, before running — not after)

**Rule 1 — the saturation gate is per base model, not global.** A character n-gram TF-IDF
baseline on the real (binary) splits gives 0.792; a fine-tuned transformer will land higher —
but that projection does not account for XLM-15's 3.21 tokens/word, its absent Azerbaijani
pretraining, or its five destroyed letters (§3.6). **XLM-15 will score below XLM-R on the same
task.** Consequence: evaluate the `>0.90` STOP separately for **each** base model at T8, not
once globally — a global check could pass on xlmr's number while xlm15 is actually saturated (or
vice versa) and the two would be silently conflated. In the results, report each base's
Condition-1 baseline distance from the ~85–89% human ceiling (§3.12) **alongside** the effect
sizes — unequal headroom between the two arms is itself a confound in the cross-base comparison
(§3.7) and must be shown, not buried, regardless of which way the mechanism finding goes.

**Rule 2 — the GPU fallback is the priority block, not a condition subset.** If the compute
window slips, the guaranteed minimum is `--only-priority` with **both** bases: 5 conditions × 2
bases × 3 seeds at `n=2,000` = 30 runs, ≈4.5h on a T4. Only the data-size curve (the other three
`train_sizes`) is sacrificed. **Do not** fall back to a three-condition subset — dropping
Conditions 3 and 4 deletes the M3 arm *and* the cross-base contrast, which is the half of the
design that survives even if the Turkish benefit (Conditions 2/5) fails to replicate. `orchestrate.py`'s
existing `priority_first` + `--only-priority` already implements this; the instruction here is
never to override it with a hand-picked condition list under time pressure.

**Noise-attenuation figures — record now, they belong in Limitations regardless of outcome.**
For symmetric label noise η over k classes, an observed difference shrinks by a factor of
`1 − ηk/(k−1)`:

```
3-class, η≈0.24:  1 − 0.24×3/2 = 1 − 0.36 = 0.64×   (observed diff is 64% of the true diff)
binary,  η≈0.12:  1 − 0.12×2/1 = 1 − 0.24 = 0.76×   (observed diff is 76% of the true diff)
```

Binary recovers roughly a third of the lost statistical power relative to 3-class (0.76 vs 0.64
— a ~19% relative improvement in the attenuation factor). This matters directly for the
`diff_over_noise < 1` check in `stats.py` (§7 gotcha, non-negotiable #5): a weaker attenuation
factor means real effects survive the noise filter more often, so switching to binary isn't just
about clearing the label-audit gate — it also makes the downstream significance test itself more
sensitive.

**Then proceed to T7**: transplant builds for both bases, the C1 controls, and the cross-base
transplant-quality comparison (`reconstruction.mean_cosine`, `C1c_delta_bpc`, and top-1 counts
per base, §3.7)
— required before any full run, not after. **Complete — see §3.16.**

### 3.16 T7 — transplant build. THREE real bugs found and fixed — the third is the important one.

**Donor model loaded without `trust_remote_code` — by design, per explicit instruction.**
`AutoModelForMaskedLM.from_pretrained("HPLT/hplt_bert_base_az")` requires `trust_remote_code=True`
(custom `LtgbertForMaskedLM` architecture). Rather than execute remote code, `build.py` now reads
only what it actually needs — the input embedding matrix — directly from `model.safetensors` via
new `src/transplant/donor_embeddings.py`. Revision-pinned
(`a126552c30733333cc95425b4eefd3d85b39d878`, now `configs/experiment.yaml → models.donor_revision`,
also written into every `transplant__<base>__<tag>.json`) — reproducibility, independent of trust.

Hit the exact ambiguity anticipated: two identically-shaped `[32768, 768]` tensors —
`embedding.word_embedding.weight` (real) and `classifier.nonlinearity.5.weight` (untied MLM output
head, decoy). Shape alone cannot disambiguate these; resolved with a **disclosed, logged**
keyword tie-break (prefers `embed`, deprioritizes `classifier`/`lm_head`/`decoder`/`predictions`/`cls`),
which raises rather than guesses if the tie-break itself doesn't resolve to exactly one candidate.
Tested against both this real case and a genuinely-unresolvable synthetic case (must raise).

**Also discovered, undocumented until now: `tokenizer.get_vocab()` (32,770) ≠ the actual embedding
table (32,768 rows, = `tokenizer.vocab_size`).** The 2 missing ids are `[BOS]`=32768, `[EOS]`=32769
— outside the checkpoint's embedding matrix entirely. Harmless *if* `align_special_tokens()`
correctly routes them to the base model instead of leaving them for OMP to reconstruct from a
donor row that doesn't exist.

**It didn't, for one of the two — a real, previously-unexercised bug in `align_special_tokens()`
(`canon.py`), not something this session introduced.** `build.py` had never been run to
completion before this session; the bug was latent. XLM-15's own tokenizer has **no `eos` role at
all** — only `bos`/`cls`/`sep`, with `cls` and `sep` both pointing at the same string (`</s>`).
The existing cross-mapping fallbacks only handled "donor lacks a role" (BERT-style donor missing
bos/eos), never the mirror case ("base lacks a role") — so donor's `[EOS]` (id 32769) had nowhere
to map to, fell into `unfamiliar_ids`, and OMP crashed trying to read donor row 32769 from a
32768-row table (`IndexError`). **First real run of `build.py` in this project's history**,
caught on the first attempt. Fixed: added the two missing mirror fallbacks (`eos→sep` and
`bos→cls`, symmetric to the existing `cls→bos`/`sep→eos` ones). 4 new regression tests (2 for the
embedding-tensor selector's real ambiguity, 2 for the mirrored special-token fallback, using mock
tokenizers reproducing XLM-15's exact role structure — no network needed). 65 tests total.

**Also fixed while preparing to run both bases: `controls.py` wrote a single flat
`results/controls.json`, no base suffix — running it for xlm15 then xlmr would have silently
overwritten the first result.** Same class of bug as gotcha #4 ("every run file must carry
`base=`"), just never triggered because controls.py had never been run for both bases either. Now
writes `results/controls__<base>.json`.

New `src/transplant/cross_base_quality.py` reads both bases' `transplant__*.json` and
`controls__*.json` and writes `results/cross_base_transplant_quality.json` with an explicit
`reconstruction_mean_cosine` / `C1c_delta_bpc` comparison, paired in reporting with top-1 masked-
token accuracy — a worse-reconstructing base is framed
as a conservative confound (can only hide the mechanism effect, never manufacture one), not a
disqualifying one, but must be shown regardless of which way it comes out.

**Third bug — the one that actually mattered, and exactly what C1c exists to catch.** First real
build completed successfully (both bases), but `controls.py`'s C1c check immediately flagged
something catastrophic: xlm15's transplanted-model BPC came back **161.12**, versus **8.73** for
the untouched base — Δ = **+152.4**, not a "the transplant made things somewhat worse" number but
a "something is structurally broken" number. Per-masked-token losses (checked directly, not just
the aggregate) were uniformly catastrophic (230–1088 nats across every single token, all 6 eval
lines) — ruling out a small-sample outlier artifact and pointing at a systematic bug.

Root cause, confirmed by direct measurement: **`transplant_embeddings()` in `omp.py` rescaled
each reconstructed vector to match the norm of its *donor-space* target — but the vector at that
point already lives in *base* space.** Donor (HPLT) embedding row-norms average **13.93**; base
(XLM-15) row-norms average **0.598** — a ~23× scale difference between the two spaces. The old
code multiplied a base-space vector by a donor-space norm, inflating the reconstructed portion of
the embedding matrix to ~18× its correct scale (measured: new-matrix mean row-norm 10.57 vs the
untouched base's 0.598). Feeding XLM-15's frozen deeper layers embeddings 18× larger than
anything they were trained on is sufficient on its own to produce exactly this kind of
catastrophic breakdown — no need to invoke any other explanation.

This directly contradicts the method's own stated design, quoted from `omp.py`'s own docstring:
*"Köçürülən VEKTOR deyil, ƏMSALLARDIR"* — coefficients transfer, not vectors. Applying donor-space
coefficients to (unnormalized, naturally-scaled) base anchors already yields a vector at the
correct base scale; rescaling it again by an unrelated donor-space quantity was never correct.
**Fix: removed the rescale entirely** — `normalize=True` now only governs candidate search and
OMP's own numerical stability in donor space, never the output vector's final scale. One new
regression test constructs a synthetic donor/base pair with the same ~23× scale ratio actually
observed (HPLT vs XLM-15) and asserts the output lands at base scale, not donor scale — this
would have failed under the old code. 66 tests total.

**What's actually invalidated, precisely — not everything.** `reconstruction.mean_cosine`/
`mean_l2_error` (0.6619 xlm15, 0.7024 xlmr) come from `reconstruction_error()`, which measures
OMP's fit **entirely in donor space** (reconstructed vs true target, both donor-normalized) — that
code path never touched the buggy rescale and is unaffected; those two numbers should reproduce
identically on re-run. What **is** invalid: every saved embedding matrix/model artifact (their
final base-space scale was wrong) and every downstream number computed from them — `C1c_delta_bpc`
above all. Deleted: `artifacts/transplanted__{xlm15,xlmr}__omp_k64`,
`results/transplant__{xlm15,xlmr}__omp_k64.json`, `results/controls__xlm15.json`. Re-running
`build` for both bases now with the fix to regenerate the artifacts and get a valid `C1c_delta_bpc`.

**Status: rebuild complete, both bases. Final numbers below.**

**The lesson this bug actually teaches — record for the paper.** Reconstruction cosine measured
**0.6619 (xlm15) / 0.7024 (xlmr) both before and after the fix, bit-for-bit identical.** The
standard transplantation quality metric was entirely blind to a scale bug that made the model
**60× worse than random** — bits-per-character caught it on the very first measurement, cosine
never moved. Two consequences, both now binding on how this project validates transplants:

1. **Reconstruction cosine is necessary but not sufficient for validating a transplant.** It
   verifies *directional* fit in donor space; it has no way to see a uniform scale error applied
   afterward, in a different space, because cosine similarity is scale-invariant by construction.
2. **C1a (identity transplant) cannot catch this class of bug, structurally, not just in this
   instance.** C1a sets donor = base, so the donor-space and base-space scales coincide by
   construction — there is no scale gap for it to expose. A bug that only manifests when donor and
   base *disagree* in scale is invisible to a control built around donor == base.

**Required going forward: compare embedding norm distributions across base, donor, and new — as
a check alongside C1a, not a substitute for it.** This is now implemented
(`src/transplant/embedding_norms.py::compute_norm_report`, wired into `build.py` automatically and
available post-hoc via `check_embedding_norms.py`) and is the only one of the three checks (C1a,
reconstruction cosine, norm report) that would have caught this bug directly, rather than via its
downstream BPC symptom.

**Final per-base numbers** (`results/controls__<base>.json`, `results/embedding_norm_report__
<base>.json`, `results/cross_base_transplant_quality.json`):

| | xlm15 (primary) | xlmr (contrast) |
|---|---|---|
| `C1a_identity.passed` | true | true |
| `reconstruction.mean_cosine` | 0.6619 | 0.7024 |
| `C1c_bpc_base` → `C1c_bpc_transplanted` | 8.7287 → 4.9947 | 0.4138 → 17.0796 |
| `C1c_delta_bpc` | **−3.734** (improved) | **+16.6658** (worsened) |
| top-1, base → canonical transplanted | 0/37 → 0/12 | 13/17 → 0/12 |
| anchor-vs-reconstructed median norm ratio | 1.1330 (healthy, 0.85–1.15) | **1.9091 (unhealthy — inflated)** |

**xlmr's norm ratio falls outside the healthy range, on the inflation side — no ad-hoc fix
applied, per pre-registration.** The pre-registered contingency in this pass covered only
*shrinkage* (reconstructed rows falling below ~0.7× anchor rows, the failure mode the earlier
rescale bug's fix was meant to guard against); it did not cover inflation, and touching `k` /
`n_candidates` or inventing a new rescale-toward-anchor-median variant now would be exactly the
ad-hoc, non-pre-registered tuning ruled out for this pass. Reporting the number as measured, not
patching it: xlmr's OMP-reconstructed rows land at ~1.9× its own anchor rows' median norm — a
real, structural difference between the two bases' behavior under the same coefficients, method,
and `k`, most plausibly reflecting different anisotropy/conditioning of the two base embedding
spaces (xlm-roberta-base's anchor-row norms cluster far more tightly around the donor's own scale
than XLM-15's do), not a bug in the fixed code path. Candidate explanation only — not tested
further in this pass, since doing so would mean touching pre-registered parameters. Flagged here
as a finding for the `k`-ablation phase (§ planned `k ∈ {8,32,64,128}`) rather than resolved now.

**Cross-base comparison — CORRECTED, an earlier pass here had this backwards.** The first version
of this section called xlm15 the "worse-reconstructing" base because its reconstruction cosine is
lower (0.6619 vs xlmr's 0.7024) and framed that as a conservative confound. **That verdict was
decided on the one metric this very section already establishes is blind to scale problems.** The
two metrics that are *not* scale-blind say the opposite:

| | xlm15 | xlmr |
|---|---|---|
| `C1c_delta_bpc` | −3.73 (improved) | **+16.67 (broken)** |
| top-1, base → canonical transplanted | 0/37 → 0/12 | 13/17 → 0/12 |
| anchor : reconstructed norm ratio | 1.13 (healthy) | **1.91 (inflated)** |

**xlmr's transplant is the broken one, not xlm15's.** This matters far more than a mislabeled
adjective: xlmr is the zero-effect control (AZ already in its pretraining). A broken control
scores badly on Condition 3 for reasons that have nothing to do with segmentation, and
`compare_bases()`-style logic reading "large effect in xlm15, negative effect in xlmr" would call
that `MECHANISM_SUPPORTED` — **a false positive on the headline claim**, not a conservative bias
that only hides an effect. Getting this backwards would have inverted the paper's central result.

**Required going forward, added to the required-checks list: every BPC result must be printed
beside its top-1 counts. The cross-base transplant-quality verdict uses `C1c_delta_bpc` and the
norm ratio, never reconstruction cosine, but BPC must not be described as usable MLM evidence
when top-1 is zero.**
`cross_base_quality.py`'s `worse_reconstruction` field (cosine-based) is retained only as a
reported-but-non-authoritative diagnostic; the authoritative verdict field is now BPC/norm-ratio
based (see code).

For xlm15, **cross-entropy fell 43% (8.7287 → 4.9947 bits/character) while top-1 accuracy
remained 0/37 for the base and 0/12 for the transplanted model**. The transplant therefore
improves the output distribution's calibration but does **not** restore usable MLM ability.
This is a diagnostic result, not independent evidence for M3, and the BPC and top-1 numbers must
always be reported together.

The anchor-share asymmetry (xlm15 24.66% vs xlmr 29.84% of donor vocab) remains a real, measured
confound and is still reported — it is orthogonal to which base's transplant is "broken" in the
BPC/norm sense above, and does not change the corrected verdict.

**The symmetric rescale-variant investigation this correction triggered — resolved, both bases,
pre-registered.** The pre-registered principle was never "shrinkage below 0.7× is bad" specifically
— it was that reconstructed rows must live on the *base model's own scale*. 1.91× violates that
exactly as 0.7× would; the earlier framing of the contingency as one-sided was a mistake, corrected
here. Built a variant (`src/transplant/build_rescale_variant.py`, no OMP re-run needed — same
coefficients, only the final row norm changes) that rescales each reconstructed row to the
**median L2 norm of the base's own anchor rows** (the original idea, this time with the correct
target — not the donor's norm, which was the original bug). Applied identically to both bases,
`k`/`n_candidates` untouched. Four builds compared:

| | xlm15 no-rescale | xlm15 rescaled | xlmr no-rescale | xlmr rescaled |
|---|---|---|---|---|
| `C1c_delta_bpc` | −3.734 | **−4.6445** | +16.6658 | **+4.3532** |
| top-1, base → transplanted | 0/37 → not measured | 0/37 → 0/12 | 13/17 → not measured | 13/17 → 0/12 |
| anchor : reconstructed norm ratio | 1.1330 | **1.0000** | 1.9091 | **1.0000** |

**Rescaling wins on both scale-sensitive diagnostics for both bases** — lower (better) BPC delta and a healthy norm
ratio (exactly 1.0 by construction) in every case, not just the base that was unhealthy before.
Top-1 remains 0/12 for both canonical rescaled transplants, so this does not establish usable MLM.
xlmr's transplant is still worse than xlm15's (still a positive Δ, i.e. still degraded relative to
its own strong native baseline) but far less broken (+4.35 vs +16.67). Symmetry outranks per-base
optimality here exactly as it did for anchor mode (functional anchors used for both bases even
though it wasn't independently optimal for each) — but in this case symmetric application also
happens to be strictly best for both, so there was no tension to resolve.

**Decision, recorded as pre-registered in `configs/experiment.yaml → transplant.rescale_
reconstructed: true`, before any full run**: adopt the base-anchor-median rescale for both bases.
Canonical artifacts going forward: `artifacts/transplanted__<base>__omp_k64_rescaled` (the
no-rescale builds are retained on disk as the ablation baseline that motivated this decision, not
deleted — both sets of four numbers above must remain reproducible).

**Required check, added because deciding this on the wrong metric would have been catastrophic:
the cross-base transplant-quality verdict must be decided on `C1c_delta_bpc` and the norm ratio,
never on reconstruction cosine; its report must pair BPC with top-1 accuracy.**
`cross_base_quality.py` (`worse_transplant` field) now decides
this way; cosine is retained only as a reported, non-authoritative diagnostic. Cosine ranked
xlm15 as "worse" while BPC/norm ratio show xlm15 improved and xlmr collapsed — deciding on cosine
would have inverted the paper's central claim (xlmr is the zero-effect control; a broken control
scoring badly on Condition 3 reads as `MECHANISM_SUPPORTED`, a false positive, not the conservative
confound cosine-based framing called it).

