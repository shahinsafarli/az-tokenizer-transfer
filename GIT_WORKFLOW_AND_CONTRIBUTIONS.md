# Git Workflow and Contribution Division

**Project** — Turkish→Azerbaijani transfer: language knowledge, or tokenisation?  
**Course** — DLE-AI-202, Cohort I 2026 · Track 1 · 5 members  
**Plan ID** — `balanced-edited-v4-4-commits` · audited 10 September 2026  
**Commit budget** — 4 authored main-branch commits per member; 20 total  
**Execution** — “do PR#” resolves to the full card in §6 and `.github/pr-plan.yml`.  
**Scope** — Fresh-history workflow design. This package has not created, changed or published a GitHub repository.

---

## 0. What this version updates

This is the **4-commit-per-member alternative for the edited code ZIP**. It supersedes the earlier 4-commit package, while the companion 6-commit alternative offers a different PR size. Choose one registry; do not mix plan IDs or PR numbers. Existing merged history must be inspected before migration.

The edited archive changes 33 paths. It removes historical test fields/ledger entries and related prose, updates the frozen-document lock hash and removes six Python comment lines. Executable Python and notebook code remain unchanged. Source owner assignments and necessity weights remain unchanged; physical line counts, evidence bytes, archive origins and hashes are recomputed.

The separately supplied full project still provides 164 schema-4 final runs, 164 logs and 36 cells. These are retained as earlier as-run evidence, not newly rerun results. Its 164-entry ledger and older disclosure documents are distinct from the edited snapshot's empty ledger. The plans do not infer zero previous test access from deleted stored entries. See §8 and docs/CODEBASE_UPDATE.md for the exact test-access behavior and provenance requirements.

## 1. Members and essential ownership

| ID | Member | GitHub | Essential responsibility |
| --- | --- | --- | --- |
| SS | Safarli, Shahin | @shahinsafarli | Token/anchor correctness and grid scheduling |
| FA | Alibabayeva, Fatima | @FatimaAlibabayeva | Data identity, leakage, label audit and control validation |
| EJ | Jafarov, Emil | @Emil-Jafarov-06 | OMP transplant, reconstruction validity and report generation |
| FB | Baghirova, Fidan | @Fidan6557 | Fine-tuning execution and representation confounds |
| SA | Allahverdiyev, Suleyman | @SuleymanAllahverdiyev | Statistical inference, decomposition and shared foundations |


The original package-only division could not give close sizes: training contained much more substantive code. This version uses explicit whole-file partitions **within `src/`**, with some adjacent cross-package duties. Files remain in their original locations. Each member owns behavior essential to the experiment; nobody receives only plotting, documentation or generated outputs. Paper authors are ordered Alibabayeva, Allahverdiyev, Baghirova, Jafarov, Safarli.

## 2. Measured balanced import

| Member | src text lines | src substantive lines | All existing code | Importance score | Runs + logs | Evidence bytes |
| --- | --- | --- | --- | --- | --- | --- |
| SS | 1783 | 1156 | 1754 | 5121 | 32 + 32 | 4326661 |
| FA | 1747 | 1186 | 1901 | 5301 | 33 + 33 | 4393516 |
| EJ | 1812 | 1145 | 1737 | 4905 | 33 + 33 | 4394205 |
| FB | 1659 | 1204 | 1798 | 5372 | 33 + 33 | 4325678 |
| SA | 1678 | 1155 | 1753 | 4988 | 33 + 33 | 4387740 |


Max/min ratios: **source text 1.092; substantive source 1.052; all existing code 1.094; importance 1.095; evidence bytes 1.015842**. Every ratio is below 1.10. Complete non-evidence snapshot text: SS 3892, FA 3768, EJ 3797, FB 3995, SA 3895 lines; max/min 1.060.

Substantive Python lines are token-bearing physical lines excluding comments, blanks and docstrings; shell and notebook code uses nonblank/non-comment source lines. Notebook JSON/output, generated results and prose receive zero programming-line credit. “All existing code” includes tests/scripts/notebook source; it does **not** claim to measure not-yet-written modules, CI, the paper or this supplied AI workflow tooling.

Importance weights are explicit role judgements: 5 core training/validity/inference/shared foundations, 4 representation/evaluation correctness, 3 diagnostics, 2 rendering/CLI, 0 empty package markers. Multiply each production file's substantive lines by its weight. This is a planning proxy, not objectively measured scientific value or proof of equal effort. Critical ownership is also assessed qualitatively in §1. Do not adjust weights merely to make a table look equal.

### Exact `src/` partitions

#### SS — Token/anchor correctness and grid scheduling

| File | Physical lines | Substantive lines | Importance weight |
| --- | --- | --- | --- |
| `src/tokenization/__init__.py` | 0 | 0 | 0 |
| `src/tokenization/anchor_map.py` | 212 | 112 | 5 |
| `src/tokenization/anchors.py` | 209 | 138 | 4 |
| `src/tokenization/canon.py` | 295 | 152 | 5 |
| `src/training/orchestrate.py` | 302 | 225 | 5 |
| `src/training/phases.py` | 485 | 337 | 5 |
| `src/training/run_grid.py` | 194 | 137 | 2 |
| `src/transplant/check_embedding_norms.py` | 86 | 55 | 3 |

#### FA — Data identity, leakage, label audit and control validation

| File | Physical lines | Substantive lines | Importance weight |
| --- | --- | --- | --- |
| `src/data/__init__.py` | 0 | 0 | 0 |
| `src/data/audit.py` | 607 | 417 | 5 |
| `src/data/profile_candidates.py` | 249 | 173 | 3 |
| `src/data/scramble.py` | 120 | 61 | 5 |
| `src/data/splits.py` | 395 | 252 | 5 |
| `src/data/truncation.py` | 156 | 105 | 4 |
| `src/training/stability_diagnostics.py` | 125 | 102 | 4 |
| `src/training/transplant_controls.py` | 95 | 76 | 4 |

#### EJ — OMP transplant, reconstruction validity and report generation

| File | Physical lines | Substantive lines | Importance weight |
| --- | --- | --- | --- |
| `src/analysis/report.py` | 99 | 83 | 2 |
| `src/tokenization/overlap_control.py` | 260 | 184 | 4 |
| `src/transplant/__init__.py` | 0 | 0 | 0 |
| `src/transplant/build.py` | 220 | 141 | 5 |
| `src/transplant/build_rescale_variant.py` | 163 | 97 | 4 |
| `src/transplant/controls.py` | 360 | 231 | 5 |
| `src/transplant/cross_base_quality.py` | 242 | 162 | 4 |
| `src/transplant/donor_embeddings.py` | 146 | 69 | 4 |
| `src/transplant/eval_text.py` | 105 | 59 | 4 |
| `src/transplant/omp.py` | 217 | 119 | 5 |

#### FB — Fine-tuning execution and representation confounds

| File | Physical lines | Substantive lines | Importance weight |
| --- | --- | --- | --- |
| `src/tokenization/corpus_fertility.py` | 188 | 135 | 3 |
| `src/tokenization/diacritics.py` | 210 | 138 | 4 |
| `src/tokenization/fertility.py` | 168 | 120 | 3 |
| `src/training/__init__.py` | 0 | 0 | 0 |
| `src/training/finetune.py` | 1093 | 811 | 5 |

#### SA — Statistical inference, decomposition and shared foundations

| File | Physical lines | Substantive lines | Importance weight |
| --- | --- | --- | --- |
| `src/__init__.py` | 0 | 0 | 0 |
| `src/analysis/__init__.py` | 0 | 0 | 0 |
| `src/analysis/aggregate.py` | 209 | 161 | 5 |
| `src/analysis/decompose.py` | 98 | 82 | 5 |
| `src/analysis/errors.py` | 176 | 131 | 3 |
| `src/analysis/figures.py` | 71 | 59 | 2 |
| `src/analysis/noise_interaction.py` | 190 | 116 | 4 |
| `src/analysis/stats.py` | 351 | 255 | 5 |
| `src/transplant/embedding_norms.py` | 60 | 29 | 3 |
| `src/transplant/top1_accuracy.py` | 138 | 87 | 3 |
| `src/utils.py` | 385 | 235 | 5 |

The complete test/script/config/document allocations are in the exact import cards below. The large core-test file goes to FA because it includes substantial data/audit coverage; reviewers from other domains cover its other tests. Tests are not artificially cut into five files to manufacture equal file counts.

One historical prose file is imported in two natural, ordered portions: FA introduces `HANDOFF.md` lines 1–933; SA appends lines 934–1811 beginning at section 3.17. Both portions are copied byte-for-byte; the completed file matches its original hash. This is import credit, not newly written prose. No production file is split or reformatted.

## 3. Four commits per member

This alternative has **19 PRs plus one bootstrap commit**, yielding exactly **20 planned main-branch author commits, four per member**. This is the four-commit alternative for the edited ZIP. Owners, substantive code size, importance weights and final run/log pairs are retained; changed input hashes, physical lines and evidence bytes are updated.

| Member | Commit 1 | Commit 2 | Commit 3 | Commit 4 | Total |
| --- | --- | --- | --- | --- | --- |
| SS | Bootstrap | PR05 source | PR10 evidence | PR15 combined delivery | 4 |
| FA | PR02 governance | PR06 source | PR11 evidence | PR16 combined delivery | 4 |
| EJ | PR01 CI | PR07 source | PR12 evidence | PR17 combined delivery | 4 |
| FB | PR03 tooling | PR08 source | PR13 evidence | PR18 combined delivery | 4 |
| SA | PR04 registry/skeletons | PR09 source | PR14 evidence | PR19 combined delivery | 4 |

The first three contributions are the member's infrastructure/bootstrap, source partition and evidence partition. The fourth combines new verification code, the member's two paper sections/four slides, and completion duties. Use one delivery branch and one final squash commit for that combined contribution. Necessary local revisions may be WIP commits before squashing; this count concerns the permanent authored commits on main.

Fewer commits means larger final delivery PRs. The underlying work is retained, not replaced by empty sign-offs or omitted verification. The measured existing source/code/evidence balance still holds. Future code and writing effort must be assessed when the actual diffs exist; the measured import metrics do not certify future LOC. Genuine post-merge fixes, if needed, are recorded honestly rather than concealed to maintain a count.

SS's bootstrap is counted as one of the four. Therefore SS has three PRs plus bootstrap; each other member has four PRs. Tags, reviews and uploading an already built release create no additional source commit. The workflow does not create a fifth administrative or final-PDF commit per person.

## 4. New code, paper, slides and completion work

| Final PR | Owner | New code / verification | Writing and completion |
| --- | --- | --- | --- |
| PR15 | SS | Anchor/grid verifier and tests | Abstract, Discussion/Limitations; slides 1–4; README/navigation; own contribution row |
| PR16 | FA | Split/provenance verifier and tests | Introduction, Data Setup; slides 5–8; actual data/provenance audit; own row |
| PR17 | EJ | Model/environment verification; CI failure checks | Method, Tools/Acknowledgements; slides 9–12; staged reproduction audit; own row |
| PR18 | FB | Test-table/report generator; release-builder tests | Training Setup, Conclusion; slides 13–16; table-policy reconciliation; packaging instructions; own row |
| PR19 | SA | Numeric-claim checker and tests | Related Work, Results; slides 17–20; integrate all five rows; build final PDFs/manifest |

Every member supplies their own claim/reference fragments, captions/evidence and factual contribution row. SA integrates completed contributions and generated outputs in PR19; that does not grant SA authorship of other members' work. All four other members review/acknowledge the final contribution report. EJ attaches a final independent clean-checkout log to PR19, and FB checks/executes release packaging. These are real reviews recorded on the PR, not extra sign-off commits.

PR04 creates buildable section, bibliography and claim stubs. The report loads each member's bibliography fragment directly, avoiding a late shared-file integration commit. PR15–18 build the then-current draft and run their own tests. PR19 runs the global strict numeric-claim checks and validates the finished documents after all other members' deliveries are present.

## 5. Git workflow and the meaning of “do PR#”

Use `DO_PR.md` and this version's `.github/pr-plan.yml`. “do PR3” and “do PR03” resolve to the same exact card. The AI checks the real repository, authenticated owner and merged dependencies; creates the named branch; imports/implements only listed paths; runs declared validation; stages named files; commits and opens the PR with its real reviewers. Opening the PR ends the instruction. Merging and release publishing require separate instructions.

The edited-source imports remain PR05–09, merged in order. Evidence imports remain PR10–14, with the original balanced run/log assignments. The combined deliveries merge **PR15 → PR16 → PR17 → PR18 → PR19**. Prepare drafts in parallel where useful, then update and validate them against their dependencies before merge. Do not introduce dependencies on the retired PR20–29.

Readiness gates: scope/hygiene for infrastructure; complete source tests after PR09; full 164-run evidence integrity after all PR10–14; member-specific tests and draft PDF builds on PR15–18; global strict claims, final documents, contribution report and release readiness on PR19. Earlier PRs must not call a checker that has not yet been delivered or claim that missing final inputs passed. Real failed checks still block the affected PR.

One final author commit per PR, with squash merge. Two real non-author approvals are required on PR19; all five members acknowledge contribution wording. Per-path source-owner review is retained for cross-owner amendments. Do not use the legacy account-switching/token files, manufacture approvals or change author metadata to simulate team participation.

This is a fresh-history plan. If prior-version commits already exist on GitHub, inspect them before applying it; the new count does not authorize deleting the repo, resetting history or renumbering previous contributions. Planning PR IDs and actual GitHub numbers can differ; record their mapping.

The helper prepares existing archive files locally and checks scope/hashes. It does not create future modules, log in, commit, push, open or merge PRs. The package is external AI context until PR04 installs it. Dataset archives, model weights, session authentication files and the large execution log are not source-tree imports; documented release assets are uploaded separately after final verification.

The source-import cap remains 4,000 human-maintained lines per import PR, with separate 5,000,000-byte evidence batches. Combined delivery PR15–19 have multiple logical review sections rather than an artificial promise to fit the old source-import line cap. Generated JSON/manifests/PDFs and imported historical material receive no newly authored programming credit. GitHub squash and review behavior is documented in [pull request merges](https://docs.github.com/en/pull-requests/reference/pull-request-merges), [CODEOWNERS](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners), and [required checks](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks).

## 6. Binding PR roadmap

Bootstrap: SS imports README.md and .gitignore from the edited ZIP, plus byte-preserving .gitattributes, as one `chore(repo): initialize repository and byte-preserving import policy` commit. PR00 is a helper identifier, not a pull request. Check all named input archive hashes first. The package stays external context until PR04 installs it.

| PR | Owner | Scope | Dependencies |
| --- | --- | --- | --- |
| PR01 | EJ | PR01 · ci(repo): ci foundation | bootstrap |
| PR02 | FA | PR02 · docs(repo): governance | bootstrap |
| PR03 | FB | PR03 · build(repo): local verification | PR01 |
| PR04 | SA | PR04 · chore(repo): registry and writing skeletons | PR01, PR02, PR03 |
| PR05 | SS | PR05 · feat(src): ss source partition | PR04 |
| PR06 | FA | PR06 · feat(src): fa source partition | PR05 |
| PR07 | EJ | PR07 · feat(src): ej source partition | PR06 |
| PR08 | FB | PR08 · feat(src): fb source partition | PR07 |
| PR09 | SA | PR09 · feat(src): sa source partition | PR08 |
| PR10 | SS | PR10 · feat(results): ss evidence partition | PR09 |
| PR11 | FA | PR11 · feat(results): fa evidence partition | PR09 |
| PR12 | EJ | PR12 · feat(results): ej evidence partition | PR09 |
| PR13 | FB | PR13 · feat(results): fb evidence partition | PR09 |
| PR14 | SA | PR14 · feat(results): sa evidence partition | PR09 |
| PR15 | SS | PR15 · feat(delivery): anchors paper and artifact guide | PR10, PR11, PR12, PR13, PR14 |
| PR16 | FA | PR16 · feat(delivery): data verification paper and provenance | PR15 |
| PR17 | EJ | PR17 · feat(delivery): model verification paper and ci | PR16 |
| PR18 | FB | PR18 · feat(delivery): test tables paper and release tooling | PR17 |
| PR19 | SA | PR19 · feat(delivery): claims results and final integration | PR18 |

### PR01 · ci(repo): ci foundation

**Owner:** EJ · **Reviewers:** FA, FB · **Approvals:** 1  
**Branch:** `ci/pr01-ej-ci-foundation` → `main`  
**Dependencies:** bootstrap  
**Final commit:** `ci(repo): ci foundation`

Create phased CI, size/scope guards and CI dependency declarations. Add reviewed LICENSE/NOTICE texts after checking project and third-party notices; do not change the frozen scientific source.

**Submit exactly these paths:**

- `.github/scripts/pr_hygiene.py`
- `.github/workflows/ci.yml`
- `LICENSE`
- `NOTICE`
- `requirements-ci.txt`

**Validation:**

- workflow_context: `python scripts/pr_plan.py scope PR01 --repo ${REPO}`
- repository: `python -m py_compile .github/scripts/pr_hygiene.py`

**Review/acceptance:**

- Initial CI checks only available infrastructure. Full project tests, complete-grid integrity and PDFs become required at their actual readiness gates.
- Do not reuse token-switching scripts or private authentication files from _repo_setup.
- PR checks use fixtures and existing evidence; no full training wrapper or real held-out test evaluation runs in CI.
- All declared checks pass, required real non-author review is recorded, and the diff stays in the registered paths. Opening the PR is the end of "do PR#"; merging is a separate instruction.

### PR02 · docs(repo): governance

**Owner:** FA · **Reviewers:** SS, EJ · **Approvals:** 1  
**Branch:** `docs/pr02-fa-governance` → `main`  
**Dependencies:** bootstrap  
**Final commit:** `docs(repo): governance`

Create CODEOWNERS, templates and contribution rules from the new exact file ownership. Primary and alternate reviewers are on the same CODEOWNERS pattern line.

**Submit exactly these paths:**

- `.github/CODEOWNERS`
- `.github/ISSUE_TEMPLATE/bug.yml`
- `.github/ISSUE_TEMPLATE/config.yml`
- `.github/ISSUE_TEMPLATE/finding.yml`
- `.github/labels.yml`
- `.github/pull_request_template.md`
- `CONTRIBUTING.md`

**Validation:**

- workflow_context: `python scripts/pr_plan.py scope PR02 --repo ${REPO}`
- workflow_context: `python scripts/pr_plan.py verify-plan`

**Review/acceptance:**

- Every source path has its assigned owner and a different alternate; specific rules follow broad rules.
- No single-owner self-review deadlock. One code-owner approval does not imply two independent approvals.
- All declared checks pass, required real non-author review is recorded, and the diff stays in the registered paths. Opening the PR is the end of "do PR#"; merging is a separate instruction.

### PR03 · build(repo): local verification

**Owner:** FB · **Reviewers:** SA, SS · **Approvals:** 1  
**Branch:** `build/pr03-fb-local-verification` → `main`  
**Dependencies:** PR01  
**Final commit:** `build(repo): local verification`

Create Makefile, lint configuration and phased verification entry point. Preserve frozen code; lint new files explicitly.

**Submit exactly these paths:**

- `Makefile`
- `ruff.toml`
- `scripts/verify_repo.py`

**Validation:**

- workflow_context: `python scripts/pr_plan.py scope PR03 --repo ${REPO}`
- repository: `python -m py_compile scripts/verify_repo.py`
- repository: `make help`

**Review/acceptance:**

- Verification distinguishes source, historical provenance, final evidence and paper readiness.
- Do not hard-code successful placeholder checks.
- All declared checks pass, required real non-author review is recorded, and the diff stays in the registered paths. Opening the PR is the end of "do PR#"; merging is a separate instruction.

### PR04 · chore(repo): registry and writing skeletons

**Owner:** SA · **Reviewers:** EJ, FA · **Approvals:** 1  
**Branch:** `chore/pr04-sa-registry-and-writing-skeletons` → `main`  
**Dependencies:** PR01, PR02, PR03  
**Final commit:** `chore(repo): registry and writing skeletons`

Install the supplied plan, manifest, local helper and command contract. Create buildable report/slide skeletons with exactly two report section stubs and four slide slots for each member. Integrate repository path aliases for preserved evidence. Reserve the five bibliography and five claim fragments, and load the bibliography fragments directly from the report build so later member deliveries do not need another integration commit. Install the supplied edited-source provenance note at docs/CODEBASE_UPDATE.md.

**Submit exactly these paths:**

- `.github/import-manifest.json`
- `.github/pr-plan.yml`
- `DO_PR.md`
- `GIT_WORKFLOW_AND_CONTRIBUTIONS.md`
- `docs/CODEBASE_UPDATE.md`
- `evidence/README_REPOSITORY_PATHS.md`
- `presentation/README.md`
- `presentation/beamerthemeAIA.sty`
- `presentation/presentation.tex`
- `presentation/sections/ej.tex`
- `presentation/sections/fa.tex`
- `presentation/sections/fb.tex`
- `presentation/sections/sa.tex`
- `presentation/sections/ss.tex`
- `report/README.md`
- `report/claims/ej.yaml`
- `report/claims/fa.yaml`
- `report/claims/fb.yaml`
- `report/claims/sa.yaml`
- `report/claims/ss.yaml`
- `report/latexmkrc`
- `report/references/ej.bib`
- `report/references/fa.bib`
- `report/references/fb.bib`
- `report/references/sa.bib`
- `report/references/ss.bib`
- `report/refs.bib`
- `report/report.tex`
- `report/sections/00_abstract.tex`
- `report/sections/01_introduction.tex`
- `report/sections/02_related_work.tex`
- `report/sections/03_method.tex`
- `report/sections/04_setup_data.tex`
- `report/sections/05_setup_training.tex`
- `report/sections/06_results.tex`
- `report/sections/07_discussion_limitations.tex`
- `report/sections/08_conclusion.tex`
- `report/sections/09_tools_ack.tex`
- `scripts/pr_plan.py`

**Validation:**

- workflow_context: `python scripts/pr_plan.py scope PR04 --repo ${REPO}`
- workflow_context: `python scripts/pr_plan.py verify-plan`
- repository: `make pdf-report`
- repository: `make pdf-slides`

**Review/acceptance:**

- Committed registry exactly matches plan_id balanced-edited-v4-4-commits; old 19-PR registry is not retained as another authority.
- Skeleton PDFs may be labelled drafts; release validation must reject unfilled stubs.
- Preserved evidence documents use root-relative aliases; execution/run.log resolves to a release asset, not a missing Git file.
- The report loads the five member bibliography fragments directly, with empty stubs valid until their owners fill them.
- Member claim fragments exist as explicit drafts; strict numeric-claim validation starts only when the resolver is delivered in PR19.
- Makefile provides the declared per-member test/build targets and the final contribution/release targets; readiness is not represented by fake success.
- Exactly one current commit-budget plan is active; original source ZIP is not an accepted fallback. Preserve the explicit distinction between edited snapshot and earlier as-run evidence.
- All declared checks pass, required real non-author review is recorded, and the diff stays in the registered paths. Opening the PR is the end of "do PR#"; merging is a separate instruction.

### PR05 · feat(src): ss source partition

**Owner:** SS · **Reviewers:** FB, SA · **Approvals:** 1  
**Branch:** `feat/pr05-ss-ss-source-partition` → `main`  
**Dependencies:** PR04  
**Final commit:** `feat(src): ss source partition`

Import canonical token alignment plus grid orchestration, assigned test and runtime support files. This owner carries both the representation gate and scheduling correctness.

**Submit exactly these paths:**

- `docs/A100_SETUP.md`
- `docs/A100_TRANSFER_MANIFEST.md`
- `docs/VASTAI_STEP_BY_STEP.md`
- `notebooks/vastai_h100.ipynb`
- `scripts/a100_bootstrap.sh`
- `scripts/vastai_stage1.sh`
- `scripts/vastai_stage2.sh`
- `src/tokenization/__init__.py`
- `src/tokenization/anchor_map.py`
- `src/tokenization/anchors.py`
- `src/tokenization/canon.py`
- `src/training/orchestrate.py`
- `src/training/phases.py`
- `src/training/run_grid.py`
- `src/transplant/check_embedding_norms.py`
- `tests/test_frozen_pipeline.py`

**Validation:**

- workflow_context: `python scripts/pr_plan.py scope PR05 --repo ${REPO}`
- workflow_context: `python scripts/pr_plan.py audit --repo ${REPO} --pr PR05`

**Review/acceptance:**

- Only these exact files/part are imported; no src/** bulk-add and no source reformatting.
- Match source bytes and hashes; partial imports are not advertised as runnable.
- The complete source test gate runs after PR09; earlier imports run path/hash checks.
- Review default validation in Python entry points and the unchanged full shell wrappers that default to test and pass opt-in flags; do not run a full wrapper during import.
- All declared checks pass, required real non-author review is recorded, and the diff stays in the registered paths. Opening the PR is the end of "do PR#"; merging is a separate instruction.

### PR06 · feat(src): fa source partition

**Owner:** FA · **Reviewers:** SS, EJ · **Approvals:** 1  
**Branch:** `feat/pr06-fa-fa-source-partition` → `main`  
**Dependencies:** PR05  
**Final commit:** `feat(src): fa source partition`

Import data selection, splits, leakage/label audit, control diagnostics and the shared core-test file. Introduce HANDOFF.md lines 1–933 verbatim.

**Submit exactly these paths:**

- `HANDOFF.md` — lines 1–933 only
- `docs/TEST_AUDIT.md`
- `src/data/__init__.py`
- `src/data/audit.py`
- `src/data/profile_candidates.py`
- `src/data/scramble.py`
- `src/data/splits.py`
- `src/data/truncation.py`
- `src/training/stability_diagnostics.py`
- `src/training/transplant_controls.py`
- `tests/test_core.py`

**Validation:**

- workflow_context: `python scripts/pr_plan.py scope PR06 --repo ${REPO}`
- workflow_context: `python scripts/pr_plan.py audit --repo ${REPO} --pr PR06`

**Review/acceptance:**

- Only these exact files/part are imported; no src/** bulk-add and no source reformatting.
- Match source bytes and hashes; partial imports are not advertised as runnable.
- The complete source test gate runs after PR09; earlier imports run path/hash checks.
- Import the revised docs/TEST_AUDIT.md exactly. Its shortened text is not evidence of zero previous test accesses. HANDOFF prefix remains lines 1–933.
- All declared checks pass, required real non-author review is recorded, and the diff stays in the registered paths. Opening the PR is the end of "do PR#"; merging is a separate instruction.

### PR07 · feat(src): ej source partition

**Owner:** EJ · **Reviewers:** FA, FB · **Approvals:** 1  
**Branch:** `feat/pr07-ej-ej-source-partition` → `main`  
**Dependencies:** PR06  
**Final commit:** `feat(src): ej source partition`

Import OMP transplant and quality controls, overlap control, table/report generation and the tiny-model end-to-end test.

**Submit exactly these paths:**

- `configs/CONFIG_HASHES.lock`
- `configs/FROZEN.md`
- `docs/PAPER_SKELETON.md`
- `docs/SCOPE_FROZEN.md`
- `docs/a100_transfer_checksums.sha256`
- `notebooks/colab_t4.ipynb`
- `requirements.lock`
- `requirements.txt`
- `scripts/a100_benchmark.sh`
- `scripts/verify_amendment_20260907.py`
- `src/analysis/report.py`
- `src/tokenization/overlap_control.py`
- `src/transplant/__init__.py`
- `src/transplant/build.py`
- `src/transplant/build_rescale_variant.py`
- `src/transplant/controls.py`
- `src/transplant/cross_base_quality.py`
- `src/transplant/donor_embeddings.py`
- `src/transplant/eval_text.py`
- `src/transplant/omp.py`
- `tests/test_end_to_end.py`

**Validation:**

- workflow_context: `python scripts/pr_plan.py scope PR07 --repo ${REPO}`
- workflow_context: `python scripts/pr_plan.py audit --repo ${REPO} --pr PR07`

**Review/acceptance:**

- Only these exact files/part are imported; no src/** bulk-add and no source reformatting.
- Match source bytes and hashes; partial imports are not advertised as runnable.
- The complete source test gate runs after PR09; earlier imports run path/hash checks.
- Import revised FROZEN.md and CONFIG_HASHES.lock together. Verify all locked file hashes without changing experiment parameters; keep as-run preregistration evidence separate.
- All declared checks pass, required real non-author review is recorded, and the diff stays in the registered paths. Opening the PR is the end of "do PR#"; merging is a separate instruction.

### PR08 · feat(src): fb source partition

**Owner:** FB · **Reviewers:** SA, SS · **Approvals:** 1  
**Branch:** `feat/pr08-fb-fb-source-partition` → `main`  
**Dependencies:** PR07  
**Final commit:** `feat(src): fb source partition`

Import the actual fine-tuning engine, token efficiency/diacritic confound checks, parallel-execution tests and assigned launch/configuration files.

**Submit exactly these paths:**

- `configs/experiment.yaml`
- `configs/pilot_t4.yaml`
- `docs/COMPUTE_ESTIMATES.md`
- `docs/RUNBOOK.md`
- `notebooks/colab_t4_pilot.ipynb`
- `scripts/vastai_run_full.sh`
- `src/tokenization/corpus_fertility.py`
- `src/tokenization/diacritics.py`
- `src/tokenization/fertility.py`
- `src/training/__init__.py`
- `src/training/finetune.py`
- `tests/test_parallel_execution.py`

**Validation:**

- workflow_context: `python scripts/pr_plan.py scope PR08 --repo ${REPO}`
- workflow_context: `python scripts/pr_plan.py audit --repo ${REPO} --pr PR08`

**Review/acceptance:**

- Only these exact files/part are imported; no src/** bulk-add and no source reformatting.
- Match source bytes and hashes; partial imports are not advertised as runnable.
- The complete source test gate runs after PR09; earlier imports run path/hash checks.
- The fine-tune runtime guard and ledger append behavior are unchanged. Model test evaluation remains outside normal import/PR checks.
- All declared checks pass, required real non-author review is recorded, and the diff stays in the registered paths. Opening the PR is the end of "do PR#"; merging is a separate instruction.

### PR09 · feat(src): sa source partition

**Owner:** SA · **Reviewers:** EJ, FA · **Approvals:** 1  
**Branch:** `feat/pr09-sa-sa-source-partition` → `main`  
**Dependencies:** PR08  
**Final commit:** `feat(src): sa source partition`

Import inference/statistics/decomposition, shared utilities, norm/top-1 diagnostics and assigned analysis support. Append HANDOFF.md lines 934–1811 verbatim.

**Submit exactly these paths:**

- `HANDOFF.md` — append lines 934–1811 to the verified prefix
- `notebooks/START_HERE.md`
- `notebooks/a100.ipynb`
- `pytest.ini`
- `resources/README.txt`
- `resources/domain_keywords.txt`
- `resources/false_friends.csv`
- `run_all.sh`
- `scripts/a100_micro_benchmark.py`
- `src/__init__.py`
- `src/analysis/__init__.py`
- `src/analysis/aggregate.py`
- `src/analysis/decompose.py`
- `src/analysis/errors.py`
- `src/analysis/figures.py`
- `src/analysis/noise_interaction.py`
- `src/analysis/stats.py`
- `src/transplant/embedding_norms.py`
- `src/transplant/top1_accuracy.py`
- `src/utils.py`
- `tests/test_stats.py`

**Validation:**

- workflow_context: `python scripts/pr_plan.py scope PR09 --repo ${REPO}`
- workflow_context: `python scripts/pr_plan.py audit --repo ${REPO} --pr PR09`
- workflow_context: `python scripts/pr_plan.py audit --repo ${REPO} --phase source`
- repository: `python -m pytest -q -m not slow`
- workflow_context: `python scripts/pr_plan.py audit-test-access --repo ${REPO} --stage source`

**Review/acceptance:**

- Only these exact files/part are imported; no src/** bulk-add and no source reformatting.
- Match source bytes and hashes; partial imports are not advertised as runnable.
- The complete source test gate runs after PR09; earlier imports run path/hash checks.
- This completes the frozen source-import milestone; PR15–19 later deliver new code plus member writing/close-out work in one commit each.
- Import errors.py with six removed comment lines and unchanged schema filter. Append HANDOFF lines 934–1811; its shortened document needs the new full/prefix hashes.
- All declared checks pass, required real non-author review is recorded, and the diff stays in the registered paths. Opening the PR is the end of "do PR#"; merging is a separate instruction.

### PR10 · feat(results): ss evidence partition

**Owner:** SS · **Reviewers:** FB, SA · **Approvals:** 1  
**Branch:** `feat/pr10-ss-ss-evidence-partition` → `main`  
**Dependencies:** PR09  
**Final commit:** `feat(results): ss evidence partition`

Import exactly this manifest batch: 32 final schema-4 runs, their 32 logs, and assigned reports/provenance. Each final run/log pair stays together; this is verification ownership, not experiment execution credit.

**Submit exactly these paths:**

- `evidence/carried_forward_pre_grid/PROVENANCE.md`
- `evidence/carried_forward_pre_grid/label_audit.json`
- `evidence/carried_forward_pre_grid/noise_interaction.json`
- `evidence/source_snapshot/figures/source_manifest.json`
- `evidence/source_snapshot/results/anchors.json`
- `evidence/source_snapshot/results/controls__xlm15.json`
- `evidence/source_snapshot/results/cross_base_transplant_quality.json`
- `evidence/source_snapshot/results/decompose.json`
- `evidence/source_snapshot/results/pre_amendment/runs/base=xlm15__cond=tokenizator__n=100__seed=13.json`
- `evidence/source_snapshot/results/runs/base=xlm15__cond=tokenizator__n=2000__seed=13__variant=mean_k64_rescaled.json`
- `evidence/source_snapshot/results/runs/base=xlm15__cond=tokenizator__n=2000__seed=13__variant=random_coef_k64_rescaled.json`
- `evidence/source_snapshot/results/transplant__xlm15__omp_k64.json`
- `evidence/source_snapshot/results/transplant__xlm15__omp_k64_rescaled.json`
- `evidence/source_snapshot/results/transplant__xlmr__mean_k64.json`
- `evidence/source_snapshot/results/transplant_controls/runs/base=xlm15__method=mean__cond=tokenizator__n=2000__seed=13.json`
- `evidence/source_snapshot/results/transplant_controls/runs/base=xlm15__method=random_coef__cond=tokenizator__n=2000__seed=13.json`
- `evidence/supplemental/test_tables_received.md`
- `results/controls__xlm15__random_coef_k64_rescaled.json`
- `results/controls__xlmr__mean_k64_rescaled.json`
- `results/logs/base=xlm15__cond=baza__n=2000__seed=1337__20260908T193948Z.log`
- `results/logs/base=xlm15__cond=baza__n=20914__seed=42__20260909T014103Z.log`
- `results/logs/base=xlm15__cond=baza__n=500__seed=7__20260908T231925Z.log`
- `results/logs/base=xlm15__cond=her_ikisi__n=2000__seed=42__20260908T205432Z.log`
- `results/logs/base=xlm15__cond=her_ikisi__n=500__seed=2024__20260909T004219Z.log`
- `results/logs/base=xlm15__cond=tokenizator__n=2000__seed=42__20260908T195301Z.log`
- `results/logs/base=xlm15__cond=tokenizator__n=20914__seed=13__20260909T015759Z.log`
- `results/logs/base=xlm15__cond=tokenizator__n=500__seed=42__20260908T232612Z.log`
- `results/logs/base=xlm15__cond=transplant_mean__n=10000__seed=1337__20260909T012530Z.log`
- `results/logs/base=xlm15__cond=transplant_mean__n=2000__seed=1337__20260908T201104Z.log`
- `results/logs/base=xlm15__cond=transplant_mean__n=500__seed=1337__20260908T234448Z.log`
- `results/logs/base=xlm15__cond=transplant_random_coef__n=10000__seed=42__20260909T013145Z.log`
- `results/logs/base=xlm15__cond=transplant_random_coef__n=2000__seed=42__20260908T202301Z.log`
- `results/logs/base=xlm15__cond=transplant_random_coef__n=500__seed=1337__20260909T000016Z.log`
- `results/logs/base=xlm15__cond=turk__n=2000__seed=42__20260908T203800Z.log`
- `results/logs/base=xlm15__cond=turk__n=500__seed=42__20260909T001238Z.log`
- `results/logs/base=xlm15__cond=turk_qarisiq__n=2000__seed=1337__20260908T211252Z.log`
- `results/logs/base=xlm15__cond=turk_qarisiq__n=500__seed=42__20260909T004527Z.log`
- `results/logs/base=xlmr__cond=baza__n=2000__seed=1337__20260908T213256Z.log`
- `results/logs/base=xlmr__cond=baza__n=500__seed=1337__20260909T022342Z.log`
- `results/logs/base=xlmr__cond=her_ikisi__n=2000__seed=2024__20260908T225229Z.log`
- `results/logs/base=xlmr__cond=her_ikisi__n=500__seed=1337__20260909T033347Z.log`
- `results/logs/base=xlmr__cond=tokenizator__n=2000__seed=2024__20260908T215912Z.log`
- `results/logs/base=xlmr__cond=tokenizator__n=500__seed=1337__20260909T023923Z.log`
- `results/logs/base=xlmr__cond=transplant_mean__n=2000__seed=2024__20260908T221226Z.log`
- `results/logs/base=xlmr__cond=transplant_mean__n=500__seed=2024__20260909T030054Z.log`
- `results/logs/base=xlmr__cond=transplant_random_coef__n=2000__seed=2024__20260908T222514Z.log`
- `results/logs/base=xlmr__cond=transplant_random_coef__n=500__seed=2024__20260909T031411Z.log`
- `results/logs/base=xlmr__cond=turk__n=2000__seed=2024__20260908T223903Z.log`
- `results/logs/base=xlmr__cond=turk__n=500__seed=2024__20260909T032818Z.log`
- `results/logs/base=xlmr__cond=turk_qarisiq__n=2000__seed=2024__20260908T230619Z.log`
- `results/logs/base=xlmr__cond=turk_qarisiq__n=500__seed=2024__20260909T035545Z.log`
- `results/overlap_control.json`
- `results/paper_tables.md`
- `results/runs/base=xlm15__cond=baza__n=2000__seed=1337.json`
- `results/runs/base=xlm15__cond=baza__n=20914__seed=42.json`
- `results/runs/base=xlm15__cond=baza__n=500__seed=7.json`
- `results/runs/base=xlm15__cond=her_ikisi__n=2000__seed=42.json`
- `results/runs/base=xlm15__cond=her_ikisi__n=500__seed=2024.json`
- `results/runs/base=xlm15__cond=tokenizator__n=2000__seed=42.json`
- `results/runs/base=xlm15__cond=tokenizator__n=20914__seed=13.json`
- `results/runs/base=xlm15__cond=tokenizator__n=500__seed=42.json`
- `results/runs/base=xlm15__cond=transplant_mean__n=10000__seed=1337.json`
- `results/runs/base=xlm15__cond=transplant_mean__n=2000__seed=1337.json`
- `results/runs/base=xlm15__cond=transplant_mean__n=500__seed=1337.json`
- `results/runs/base=xlm15__cond=transplant_random_coef__n=10000__seed=42.json`
- `results/runs/base=xlm15__cond=transplant_random_coef__n=2000__seed=42.json`
- `results/runs/base=xlm15__cond=transplant_random_coef__n=500__seed=1337.json`
- `results/runs/base=xlm15__cond=turk__n=2000__seed=42.json`
- `results/runs/base=xlm15__cond=turk__n=500__seed=42.json`
- `results/runs/base=xlm15__cond=turk_qarisiq__n=2000__seed=1337.json`
- `results/runs/base=xlm15__cond=turk_qarisiq__n=500__seed=42.json`
- `results/runs/base=xlmr__cond=baza__n=2000__seed=1337.json`
- `results/runs/base=xlmr__cond=baza__n=500__seed=1337.json`
- `results/runs/base=xlmr__cond=her_ikisi__n=2000__seed=2024.json`
- `results/runs/base=xlmr__cond=her_ikisi__n=500__seed=1337.json`
- `results/runs/base=xlmr__cond=tokenizator__n=2000__seed=2024.json`
- `results/runs/base=xlmr__cond=tokenizator__n=500__seed=1337.json`
- `results/runs/base=xlmr__cond=transplant_mean__n=2000__seed=2024.json`
- `results/runs/base=xlmr__cond=transplant_mean__n=500__seed=2024.json`
- `results/runs/base=xlmr__cond=transplant_random_coef__n=2000__seed=2024.json`
- `results/runs/base=xlmr__cond=transplant_random_coef__n=500__seed=2024.json`
- `results/runs/base=xlmr__cond=turk__n=2000__seed=2024.json`
- `results/runs/base=xlmr__cond=turk__n=500__seed=2024.json`
- `results/runs/base=xlmr__cond=turk_qarisiq__n=2000__seed=2024.json`
- `results/runs/base=xlmr__cond=turk_qarisiq__n=500__seed=2024.json`
- `results/scramble.json`
- `results/stats.json`
- `results/summary.json`
- `results/transplant__xlm15__mean_k64.json`
- `results/transplant__xlm15__random_coef_k64.json`
- `results/transplant__xlmr__random_coef_k64_rescaled.json`

**Validation:**

- workflow_context: `python scripts/pr_plan.py scope PR10 --repo ${REPO}`
- workflow_context: `python scripts/pr_plan.py audit --repo ${REPO} --pr PR10`

**Review/acceptance:**

- Record schema, unique run key, matching log, source hash, metric consistency and provenance status in the PR review.
- Results-full is authoritative at results/; pre-grid records stay under evidence/, never results/runs/.
- AMENDMENT: initial import of final evidence and explicitly separated prior evidence.
- After all five evidence PRs merge, run the full import audit and the actual grid integrity check before PR15–19.
- Use the edited ZIP for evidence/source_snapshot origins, including its reduced test fields and empty ledger. Keep results_full.zip final-run bytes and its separate 164-entry ledger unchanged; never fill removed fields from the old source ZIP.
- Include the registered `AMENDMENT:` explanation for result/table writes.
- All declared checks pass, required real non-author review is recorded, and the diff stays in the registered paths. Opening the PR is the end of "do PR#"; merging is a separate instruction.

### PR11 · feat(results): fa evidence partition

**Owner:** FA · **Reviewers:** SS, EJ · **Approvals:** 1  
**Branch:** `feat/pr11-fa-fa-evidence-partition` → `main`  
**Dependencies:** PR09  
**Final commit:** `feat(results): fa evidence partition`

Import exactly this manifest batch: 33 final schema-4 runs, their 33 logs, and assigned reports/provenance. Each final run/log pair stays together; this is verification ownership, not experiment execution credit.

**Submit exactly these paths:**

- `evidence/carried_forward_pre_grid/dataset_candidates.json`
- `evidence/carried_forward_pre_grid/label_audit_annotator2.csv`
- `evidence/disclosures/TEST_AUDIT.md`
- `evidence/environment/ENVIRONMENT_AS_RUN.md`
- `evidence/environment/env_freeze_as_run.txt`
- `evidence/source_snapshot/results/embedding_norm_report__xlmr.json`
- `evidence/source_snapshot/results/pre_amendment/README.md`
- `evidence/source_snapshot/results/pre_amendment/runs/base=xlmr__cond=tokenizator__n=100__seed=13.json`
- `evidence/source_snapshot/results/runs/base=xlm15__cond=tokenizator__n=2000__seed=1337__variant=mean_k64_rescaled.json`
- `evidence/source_snapshot/results/stats.json`
- `evidence/source_snapshot/results/transplant__xlmr__mean_k64_rescaled.json`
- `evidence/source_snapshot/results/transplant_controls/runs/base=xlmr__method=mean__cond=tokenizator__n=2000__seed=42.json`
- `evidence/source_snapshot/results/transplant_controls/runs/base=xlmr__method=random_coef__cond=tokenizator__n=2000__seed=13.json`
- `figures/conditional_macro_f1.png`
- `figures/source_manifest.json`
- `results/aggregate.csv`
- `results/controls__xlm15__mean_k64_rescaled.json`
- `results/cross_base_transplant_quality.json`
- `results/embedding_norm_report__xlm15.json`
- `results/fertility.json`
- `results/logs/base=xlm15__cond=baza__n=10000__seed=1337__20260909T010604Z.log`
- `results/logs/base=xlm15__cond=baza__n=2000__seed=2024__20260908T194941Z.log`
- `results/logs/base=xlm15__cond=baza__n=500__seed=13__20260908T231601Z.log`
- `results/logs/base=xlm15__cond=her_ikisi__n=2000__seed=2024__20260908T210632Z.log`
- `results/logs/base=xlm15__cond=her_ikisi__n=500__seed=1337__20260909T003259Z.log`
- `results/logs/base=xlm15__cond=tokenizator__n=10000__seed=42__20260909T011302Z.log`
- `results/logs/base=xlm15__cond=tokenizator__n=2000__seed=1337__20260908T195601Z.log`
- `results/logs/base=xlm15__cond=tokenizator__n=500__seed=1337__20260908T232919Z.log`
- `results/logs/base=xlm15__cond=transplant_mean__n=10000__seed=13__20260909T012837Z.log`
- `results/logs/base=xlm15__cond=transplant_mean__n=2000__seed=42__20260908T200805Z.log`
- `results/logs/base=xlm15__cond=transplant_mean__n=20914__seed=42__20260909T020109Z.log`
- `results/logs/base=xlm15__cond=transplant_mean__n=500__seed=42__20260908T234144Z.log`
- `results/logs/base=xlm15__cond=transplant_random_coef__n=2000__seed=1337__20260908T202601Z.log`
- `results/logs/base=xlm15__cond=transplant_random_coef__n=20914__seed=13__20260909T021646Z.log`
- `results/logs/base=xlm15__cond=transplant_random_coef__n=500__seed=42__20260908T235712Z.log`
- `results/logs/base=xlm15__cond=turk__n=2000__seed=1337__20260908T204120Z.log`
- `results/logs/base=xlm15__cond=turk__n=500__seed=1337__20260909T001602Z.log`
- `results/logs/base=xlm15__cond=turk_qarisiq__n=2000__seed=42__20260908T210934Z.log`
- `results/logs/base=xlm15__cond=turk_qarisiq__n=500__seed=1337__20260909T004852Z.log`
- `results/logs/base=xlmr__cond=baza__n=2000__seed=42__20260908T212610Z.log`
- `results/logs/base=xlmr__cond=baza__n=500__seed=42__20260909T021953Z.log`
- `results/logs/base=xlmr__cond=her_ikisi__n=2000__seed=42__20260908T224155Z.log`
- `results/logs/base=xlmr__cond=her_ikisi__n=500__seed=42__20260909T033107Z.log`
- `results/logs/base=xlmr__cond=tokenizator__n=2000__seed=42__20260908T214710Z.log`
- `results/logs/base=xlmr__cond=tokenizator__n=500__seed=42__20260909T023634Z.log`
- `results/logs/base=xlmr__cond=transplant_mean__n=2000__seed=42__20260908T220202Z.log`
- `results/logs/base=xlmr__cond=transplant_mean__n=500__seed=42__20260909T025020Z.log`
- `results/logs/base=xlmr__cond=transplant_random_coef__n=2000__seed=42__20260908T221502Z.log`
- `results/logs/base=xlmr__cond=transplant_random_coef__n=500__seed=42__20260909T030333Z.log`
- `results/logs/base=xlmr__cond=turk__n=2000__seed=42__20260908T222751Z.log`
- `results/logs/base=xlmr__cond=turk__n=500__seed=42__20260909T031649Z.log`
- `results/logs/base=xlmr__cond=turk_qarisiq__n=2000__seed=42__20260908T225503Z.log`
- `results/logs/base=xlmr__cond=turk_qarisiq__n=500__seed=42__20260909T034420Z.log`
- `results/runs/base=xlm15__cond=baza__n=10000__seed=1337.json`
- `results/runs/base=xlm15__cond=baza__n=2000__seed=2024.json`
- `results/runs/base=xlm15__cond=baza__n=500__seed=13.json`
- `results/runs/base=xlm15__cond=her_ikisi__n=2000__seed=2024.json`
- `results/runs/base=xlm15__cond=her_ikisi__n=500__seed=1337.json`
- `results/runs/base=xlm15__cond=tokenizator__n=10000__seed=42.json`
- `results/runs/base=xlm15__cond=tokenizator__n=2000__seed=1337.json`
- `results/runs/base=xlm15__cond=tokenizator__n=500__seed=1337.json`
- `results/runs/base=xlm15__cond=transplant_mean__n=10000__seed=13.json`
- `results/runs/base=xlm15__cond=transplant_mean__n=2000__seed=42.json`
- `results/runs/base=xlm15__cond=transplant_mean__n=20914__seed=42.json`
- `results/runs/base=xlm15__cond=transplant_mean__n=500__seed=42.json`
- `results/runs/base=xlm15__cond=transplant_random_coef__n=2000__seed=1337.json`
- `results/runs/base=xlm15__cond=transplant_random_coef__n=20914__seed=13.json`
- `results/runs/base=xlm15__cond=transplant_random_coef__n=500__seed=42.json`
- `results/runs/base=xlm15__cond=turk__n=2000__seed=1337.json`
- `results/runs/base=xlm15__cond=turk__n=500__seed=1337.json`
- `results/runs/base=xlm15__cond=turk_qarisiq__n=2000__seed=42.json`
- `results/runs/base=xlm15__cond=turk_qarisiq__n=500__seed=1337.json`
- `results/runs/base=xlmr__cond=baza__n=2000__seed=42.json`
- `results/runs/base=xlmr__cond=baza__n=500__seed=42.json`
- `results/runs/base=xlmr__cond=her_ikisi__n=2000__seed=42.json`
- `results/runs/base=xlmr__cond=her_ikisi__n=500__seed=42.json`
- `results/runs/base=xlmr__cond=tokenizator__n=2000__seed=42.json`
- `results/runs/base=xlmr__cond=tokenizator__n=500__seed=42.json`
- `results/runs/base=xlmr__cond=transplant_mean__n=2000__seed=42.json`
- `results/runs/base=xlmr__cond=transplant_mean__n=500__seed=42.json`
- `results/runs/base=xlmr__cond=transplant_random_coef__n=2000__seed=42.json`
- `results/runs/base=xlmr__cond=transplant_random_coef__n=500__seed=42.json`
- `results/runs/base=xlmr__cond=turk__n=2000__seed=42.json`
- `results/runs/base=xlmr__cond=turk__n=500__seed=42.json`
- `results/runs/base=xlmr__cond=turk_qarisiq__n=2000__seed=42.json`
- `results/runs/base=xlmr__cond=turk_qarisiq__n=500__seed=42.json`
- `results/splits.json`
- `results/test_evaluation_ledger.jsonl`
- `results/top1_accuracy.json`
- `results/transplant__xlm15__omp_k64.json`
- `results/transplant__xlmr__omp_k64_rescaled.json`

**Validation:**

- workflow_context: `python scripts/pr_plan.py scope PR11 --repo ${REPO}`
- workflow_context: `python scripts/pr_plan.py audit --repo ${REPO} --pr PR11`

**Review/acceptance:**

- Record schema, unique run key, matching log, source hash, metric consistency and provenance status in the PR review.
- Results-full is authoritative at results/; pre-grid records stay under evidence/, never results/runs/.
- AMENDMENT: initial import of final evidence and explicitly separated prior evidence.
- After all five evidence PRs merge, run the full import audit and the actual grid integrity check before PR15–19.
- Use the edited ZIP for evidence/source_snapshot origins, including its reduced test fields and empty ledger. Keep results_full.zip final-run bytes and its separate 164-entry ledger unchanged; never fill removed fields from the old source ZIP.
- Include the registered `AMENDMENT:` explanation for result/table writes.
- All declared checks pass, required real non-author review is recorded, and the diff stays in the registered paths. Opening the PR is the end of "do PR#"; merging is a separate instruction.

### PR12 · feat(results): ej evidence partition

**Owner:** EJ · **Reviewers:** FA, FB · **Approvals:** 1  
**Branch:** `feat/pr12-ej-ej-evidence-partition` → `main`  
**Dependencies:** PR09  
**Final commit:** `feat(results): ej evidence partition`

Import exactly this manifest batch: 33 final schema-4 runs, their 33 logs, and assigned reports/provenance. Each final run/log pair stays together; this is verification ownership, not experiment execution credit.

**Submit exactly these paths:**

- `evidence/carried_forward_pre_grid/diacritics.json`
- `evidence/carried_forward_pre_grid/label_audit_annotator1.csv`
- `evidence/carried_forward_pre_grid/label_audit_key.json`
- `evidence/carried_forward_pre_grid/splits_PRE_GRID.json`
- `evidence/environment/requirements.lock`
- `evidence/preregistration/CONFIG_HASHES.lock`
- `evidence/preregistration/FROZEN.md`
- `evidence/source_snapshot/results/controls__xlmr__rescaled.json`
- `evidence/source_snapshot/results/pre_amendment/runs/base=xlmr__cond=baza__n=100__seed=13.json`
- `evidence/source_snapshot/results/transplant__xlm15__mean_k64.json`
- `evidence/source_snapshot/results/transplant__xlm15__random_coef_k64.json`
- `evidence/source_snapshot/results/transplant__xlmr__random_coef_k64_rescaled.json`
- `evidence/source_snapshot/results/transplant_controls/runs/base=xlm15__method=random_coef__cond=tokenizator__n=2000__seed=1337.json`
- `evidence/source_snapshot/results/transplant_controls/runs/base=xlmr__method=mean__cond=tokenizator__n=2000__seed=1337.json`
- `evidence/source_snapshot/results/transplant_controls/runs/base=xlmr__method=random_coef__cond=tokenizator__n=2000__seed=42.json`
- `figures/fig1_overlap_en.png`
- `results/anchors.json`
- `results/controls__xlmr__random_coef_k64_rescaled.json`
- `results/embedding_norm_report__xlmr.json`
- `results/errors.json`
- `results/errors_manual_sample.csv`
- `results/launcher_state.json`
- `results/logs/base=xlm15__cond=baza__n=10000__seed=13__20260909T010934Z.log`
- `results/logs/base=xlm15__cond=baza__n=2000__seed=7__20260908T194623Z.log`
- `results/logs/base=xlm15__cond=baza__n=20914__seed=1337__20260909T014434Z.log`
- `results/logs/base=xlm15__cond=baza__n=500__seed=42__20260908T230911Z.log`
- `results/logs/base=xlm15__cond=her_ikisi__n=2000__seed=7__20260908T210329Z.log`
- `results/logs/base=xlm15__cond=her_ikisi__n=500__seed=7__20260909T003912Z.log`
- `results/logs/base=xlm15__cond=tokenizator__n=2000__seed=13__20260908T195902Z.log`
- `results/logs/base=xlm15__cond=tokenizator__n=20914__seed=42__20260909T015144Z.log`
- `results/logs/base=xlm15__cond=tokenizator__n=500__seed=7__20260908T233531Z.log`
- `results/logs/base=xlm15__cond=transplant_mean__n=2000__seed=7__20260908T201700Z.log`
- `results/logs/base=xlm15__cond=transplant_mean__n=20914__seed=13__20260909T020721Z.log`
- `results/logs/base=xlm15__cond=transplant_mean__n=500__seed=13__20260908T234754Z.log`
- `results/logs/base=xlm15__cond=transplant_random_coef__n=10000__seed=1337__20260909T013449Z.log`
- `results/logs/base=xlm15__cond=transplant_random_coef__n=2000__seed=7__20260908T203201Z.log`
- `results/logs/base=xlm15__cond=transplant_random_coef__n=500__seed=7__20260909T000629Z.log`
- `results/logs/base=xlm15__cond=turk__n=2000__seed=7__20260908T204753Z.log`
- `results/logs/base=xlm15__cond=turk__n=500__seed=7__20260909T002255Z.log`
- `results/logs/base=xlm15__cond=turk_qarisiq__n=2000__seed=7__20260908T211930Z.log`
- `results/logs/base=xlm15__cond=turk_qarisiq__n=500__seed=7__20260909T005542Z.log`
- `results/logs/base=xlmr__cond=baza__n=2000__seed=7__20260908T214022Z.log`
- `results/logs/base=xlmr__cond=baza__n=500__seed=7__20260909T023028Z.log`
- `results/logs/base=xlmr__cond=her_ikisi__n=2000__seed=7__20260908T224951Z.log`
- `results/logs/base=xlmr__cond=her_ikisi__n=500__seed=7__20260909T033907Z.log`
- `results/logs/base=xlmr__cond=tokenizator__n=2000__seed=7__20260908T215611Z.log`
- `results/logs/base=xlmr__cond=tokenizator__n=500__seed=7__20260909T024501Z.log`
- `results/logs/base=xlmr__cond=transplant_mean__n=2000__seed=7__20260908T220948Z.log`
- `results/logs/base=xlmr__cond=transplant_mean__n=500__seed=7__20260909T025814Z.log`
- `results/logs/base=xlmr__cond=transplant_random_coef__n=2000__seed=7__20260908T222240Z.log`
- `results/logs/base=xlmr__cond=transplant_random_coef__n=500__seed=7__20260909T031131Z.log`
- `results/logs/base=xlmr__cond=turk__n=2000__seed=7__20260908T223615Z.log`
- `results/logs/base=xlmr__cond=turk__n=500__seed=7__20260909T032527Z.log`
- `results/logs/base=xlmr__cond=turk_qarisiq__n=2000__seed=7__20260908T230325Z.log`
- `results/logs/base=xlmr__cond=turk_qarisiq__n=500__seed=7__20260909T035254Z.log`
- `results/runs/base=xlm15__cond=baza__n=10000__seed=13.json`
- `results/runs/base=xlm15__cond=baza__n=2000__seed=7.json`
- `results/runs/base=xlm15__cond=baza__n=20914__seed=1337.json`
- `results/runs/base=xlm15__cond=baza__n=500__seed=42.json`
- `results/runs/base=xlm15__cond=her_ikisi__n=2000__seed=7.json`
- `results/runs/base=xlm15__cond=her_ikisi__n=500__seed=7.json`
- `results/runs/base=xlm15__cond=tokenizator__n=2000__seed=13.json`
- `results/runs/base=xlm15__cond=tokenizator__n=20914__seed=42.json`
- `results/runs/base=xlm15__cond=tokenizator__n=500__seed=7.json`
- `results/runs/base=xlm15__cond=transplant_mean__n=2000__seed=7.json`
- `results/runs/base=xlm15__cond=transplant_mean__n=20914__seed=13.json`
- `results/runs/base=xlm15__cond=transplant_mean__n=500__seed=13.json`
- `results/runs/base=xlm15__cond=transplant_random_coef__n=10000__seed=1337.json`
- `results/runs/base=xlm15__cond=transplant_random_coef__n=2000__seed=7.json`
- `results/runs/base=xlm15__cond=transplant_random_coef__n=500__seed=7.json`
- `results/runs/base=xlm15__cond=turk__n=2000__seed=7.json`
- `results/runs/base=xlm15__cond=turk__n=500__seed=7.json`
- `results/runs/base=xlm15__cond=turk_qarisiq__n=2000__seed=7.json`
- `results/runs/base=xlm15__cond=turk_qarisiq__n=500__seed=7.json`
- `results/runs/base=xlmr__cond=baza__n=2000__seed=7.json`
- `results/runs/base=xlmr__cond=baza__n=500__seed=7.json`
- `results/runs/base=xlmr__cond=her_ikisi__n=2000__seed=7.json`
- `results/runs/base=xlmr__cond=her_ikisi__n=500__seed=7.json`
- `results/runs/base=xlmr__cond=tokenizator__n=2000__seed=7.json`
- `results/runs/base=xlmr__cond=tokenizator__n=500__seed=7.json`
- `results/runs/base=xlmr__cond=transplant_mean__n=2000__seed=7.json`
- `results/runs/base=xlmr__cond=transplant_mean__n=500__seed=7.json`
- `results/runs/base=xlmr__cond=transplant_random_coef__n=2000__seed=7.json`
- `results/runs/base=xlmr__cond=transplant_random_coef__n=500__seed=7.json`
- `results/runs/base=xlmr__cond=turk__n=2000__seed=7.json`
- `results/runs/base=xlmr__cond=turk__n=500__seed=7.json`
- `results/runs/base=xlmr__cond=turk_qarisiq__n=2000__seed=7.json`
- `results/runs/base=xlmr__cond=turk_qarisiq__n=500__seed=7.json`
- `results/transplant__xlm15__mean_k64_rescaled.json`
- `results/transplant__xlm15__omp_k64_rescaled.json`
- `results/transplant__xlmr__omp_k64.json`

**Validation:**

- workflow_context: `python scripts/pr_plan.py scope PR12 --repo ${REPO}`
- workflow_context: `python scripts/pr_plan.py audit --repo ${REPO} --pr PR12`

**Review/acceptance:**

- Record schema, unique run key, matching log, source hash, metric consistency and provenance status in the PR review.
- Results-full is authoritative at results/; pre-grid records stay under evidence/, never results/runs/.
- AMENDMENT: initial import of final evidence and explicitly separated prior evidence.
- After all five evidence PRs merge, run the full import audit and the actual grid integrity check before PR15–19.
- Use the edited ZIP for evidence/source_snapshot origins, including its reduced test fields and empty ledger. Keep results_full.zip final-run bytes and its separate 164-entry ledger unchanged; never fill removed fields from the old source ZIP.
- Include the registered `AMENDMENT:` explanation for result/table writes.
- All declared checks pass, required real non-author review is recorded, and the diff stays in the registered paths. Opening the PR is the end of "do PR#"; merging is a separate instruction.

### PR13 · feat(results): fb evidence partition

**Owner:** FB · **Reviewers:** SA, SS · **Approvals:** 1  
**Branch:** `feat/pr13-fb-fb-evidence-partition` → `main`  
**Dependencies:** PR09  
**Final commit:** `feat(results): fb evidence partition`

Import exactly this manifest batch: 33 final schema-4 runs, their 33 logs, and assigned reports/provenance. Each final run/log pair stays together; this is verification ownership, not experiment execution credit.

**Submit exactly these paths:**

- `evidence/00_READ_ME_FIRST.md`
- `evidence/carried_forward_pre_grid/corpus_fertility.json`
- `evidence/carried_forward_pre_grid/dataset_candidates_tr.json`
- `evidence/environment/env_hardware_as_run.txt`
- `evidence/environment/requirements.txt`
- `evidence/source_snapshot/results/controls__xlmr.json`
- `evidence/source_snapshot/results/paper_report.json`
- `evidence/source_snapshot/results/paper_tables.md`
- `evidence/source_snapshot/results/pre_amendment/condition3_transplant_controls.json`
- `evidence/source_snapshot/results/pre_amendment/runs/base=xlm15__cond=baza__n=100__seed=13.json`
- `evidence/source_snapshot/results/runs/base=xlm15__cond=tokenizator__n=2000__seed=42__variant=random_coef_k64_rescaled.json`
- `evidence/source_snapshot/results/stability_diagnostics/runs/step1_mean_collapsed_seed42.json`
- `evidence/source_snapshot/results/transplant__xlm15__mean_k64_rescaled.json`
- `evidence/source_snapshot/results/transplant__xlm15__random_coef_k64_rescaled.json`
- `evidence/source_snapshot/results/transplant__xlmr__omp_k64.json`
- `evidence/source_snapshot/results/transplant__xlmr__random_coef_k64.json`
- `evidence/source_snapshot/results/transplant_controls/runs/base=xlm15__method=mean__cond=tokenizator__n=2000__seed=42.json`
- `evidence/source_snapshot/results/transplant_controls/runs/base=xlmr__method=random_coef__cond=tokenizator__n=2000__seed=1337.json`
- `evidence/source_snapshot/results/truncation.json`
- `results/controls__xlm15__omp_k64_rescaled.json`
- `results/controls__xlmr__omp_k64_rescaled.json`
- `results/logs/base=xlm15__cond=baza__n=2000__seed=42__20260908T193630Z.log`
- `results/logs/base=xlm15__cond=baza__n=20914__seed=13__20260909T014810Z.log`
- `results/logs/base=xlm15__cond=baza__n=500__seed=1337__20260908T231236Z.log`
- `results/logs/base=xlm15__cond=her_ikisi__n=2000__seed=1337__20260908T205732Z.log`
- `results/logs/base=xlm15__cond=her_ikisi__n=500__seed=42__20260909T002951Z.log`
- `results/logs/base=xlm15__cond=tokenizator__n=10000__seed=1337__20260909T011611Z.log`
- `results/logs/base=xlm15__cond=tokenizator__n=2000__seed=2024__20260908T200503Z.log`
- `results/logs/base=xlm15__cond=tokenizator__n=500__seed=2024__20260908T233838Z.log`
- `results/logs/base=xlm15__cond=transplant_mean__n=10000__seed=42__20260909T012224Z.log`
- `results/logs/base=xlm15__cond=transplant_mean__n=2000__seed=2024__20260908T202000Z.log`
- `results/logs/base=xlm15__cond=transplant_mean__n=500__seed=2024__20260908T235405Z.log`
- `results/logs/base=xlm15__cond=transplant_random_coef__n=10000__seed=13__20260909T013756Z.log`
- `results/logs/base=xlm15__cond=transplant_random_coef__n=2000__seed=2024__20260908T203502Z.log`
- `results/logs/base=xlm15__cond=transplant_random_coef__n=20914__seed=1337__20260909T021338Z.log`
- `results/logs/base=xlm15__cond=transplant_random_coef__n=500__seed=2024__20260909T000934Z.log`
- `results/logs/base=xlm15__cond=turk__n=2000__seed=2024__20260908T205114Z.log`
- `results/logs/base=xlm15__cond=turk__n=500__seed=2024__20260909T002624Z.log`
- `results/logs/base=xlm15__cond=turk_qarisiq__n=2000__seed=2024__20260908T212252Z.log`
- `results/logs/base=xlm15__cond=turk_qarisiq__n=500__seed=2024__20260909T005908Z.log`
- `results/logs/base=xlmr__cond=baza__n=2000__seed=2024__20260908T214354Z.log`
- `results/logs/base=xlmr__cond=baza__n=500__seed=2024__20260909T023333Z.log`
- `results/logs/base=xlmr__cond=her_ikisi__n=2000__seed=1337__20260908T224433Z.log`
- `results/logs/base=xlmr__cond=her_ikisi__n=500__seed=2024__20260909T034143Z.log`
- `results/logs/base=xlmr__cond=tokenizator__n=2000__seed=1337__20260908T215035Z.log`
- `results/logs/base=xlmr__cond=tokenizator__n=500__seed=2024__20260909T024741Z.log`
- `results/logs/base=xlmr__cond=transplant_mean__n=2000__seed=1337__20260908T220434Z.log`
- `results/logs/base=xlmr__cond=transplant_mean__n=500__seed=1337__20260909T025257Z.log`
- `results/logs/base=xlmr__cond=transplant_random_coef__n=2000__seed=1337__20260908T221734Z.log`
- `results/logs/base=xlmr__cond=transplant_random_coef__n=500__seed=1337__20260909T030612Z.log`
- `results/logs/base=xlmr__cond=turk__n=2000__seed=1337__20260908T223041Z.log`
- `results/logs/base=xlmr__cond=turk__n=500__seed=1337__20260909T031942Z.log`
- `results/logs/base=xlmr__cond=turk_qarisiq__n=2000__seed=1337__20260908T225751Z.log`
- `results/logs/base=xlmr__cond=turk_qarisiq__n=500__seed=1337__20260909T034712Z.log`
- `results/paper_report.json`
- `results/runs/base=xlm15__cond=baza__n=2000__seed=42.json`
- `results/runs/base=xlm15__cond=baza__n=20914__seed=13.json`
- `results/runs/base=xlm15__cond=baza__n=500__seed=1337.json`
- `results/runs/base=xlm15__cond=her_ikisi__n=2000__seed=1337.json`
- `results/runs/base=xlm15__cond=her_ikisi__n=500__seed=42.json`
- `results/runs/base=xlm15__cond=tokenizator__n=10000__seed=1337.json`
- `results/runs/base=xlm15__cond=tokenizator__n=2000__seed=2024.json`
- `results/runs/base=xlm15__cond=tokenizator__n=500__seed=2024.json`
- `results/runs/base=xlm15__cond=transplant_mean__n=10000__seed=42.json`
- `results/runs/base=xlm15__cond=transplant_mean__n=2000__seed=2024.json`
- `results/runs/base=xlm15__cond=transplant_mean__n=500__seed=2024.json`
- `results/runs/base=xlm15__cond=transplant_random_coef__n=10000__seed=13.json`
- `results/runs/base=xlm15__cond=transplant_random_coef__n=2000__seed=2024.json`
- `results/runs/base=xlm15__cond=transplant_random_coef__n=20914__seed=1337.json`
- `results/runs/base=xlm15__cond=transplant_random_coef__n=500__seed=2024.json`
- `results/runs/base=xlm15__cond=turk__n=2000__seed=2024.json`
- `results/runs/base=xlm15__cond=turk__n=500__seed=2024.json`
- `results/runs/base=xlm15__cond=turk_qarisiq__n=2000__seed=2024.json`
- `results/runs/base=xlm15__cond=turk_qarisiq__n=500__seed=2024.json`
- `results/runs/base=xlmr__cond=baza__n=2000__seed=2024.json`
- `results/runs/base=xlmr__cond=baza__n=500__seed=2024.json`
- `results/runs/base=xlmr__cond=her_ikisi__n=2000__seed=1337.json`
- `results/runs/base=xlmr__cond=her_ikisi__n=500__seed=2024.json`
- `results/runs/base=xlmr__cond=tokenizator__n=2000__seed=1337.json`
- `results/runs/base=xlmr__cond=tokenizator__n=500__seed=2024.json`
- `results/runs/base=xlmr__cond=transplant_mean__n=2000__seed=1337.json`
- `results/runs/base=xlmr__cond=transplant_mean__n=500__seed=1337.json`
- `results/runs/base=xlmr__cond=transplant_random_coef__n=2000__seed=1337.json`
- `results/runs/base=xlmr__cond=transplant_random_coef__n=500__seed=1337.json`
- `results/runs/base=xlmr__cond=turk__n=2000__seed=1337.json`
- `results/runs/base=xlmr__cond=turk__n=500__seed=1337.json`
- `results/runs/base=xlmr__cond=turk_qarisiq__n=2000__seed=1337.json`
- `results/runs/base=xlmr__cond=turk_qarisiq__n=500__seed=1337.json`
- `results/stats.csv`
- `results/summary.csv`
- `results/test_tables.md`
- `results/transplant__xlmr__mean_k64_rescaled.json`

**Validation:**

- workflow_context: `python scripts/pr_plan.py scope PR13 --repo ${REPO}`
- workflow_context: `python scripts/pr_plan.py audit --repo ${REPO} --pr PR13`

**Review/acceptance:**

- Record schema, unique run key, matching log, source hash, metric consistency and provenance status in the PR review.
- Results-full is authoritative at results/; pre-grid records stay under evidence/, never results/runs/.
- AMENDMENT: initial import of final evidence and explicitly separated prior evidence.
- After all five evidence PRs merge, run the full import audit and the actual grid integrity check before PR15–19.
- Use the edited ZIP for evidence/source_snapshot origins, including its reduced test fields and empty ledger. Keep results_full.zip final-run bytes and its separate 164-entry ledger unchanged; never fill removed fields from the old source ZIP.
- Include the registered `AMENDMENT:` explanation for result/table writes.
- All declared checks pass, required real non-author review is recorded, and the diff stays in the registered paths. Opening the PR is the end of "do PR#"; merging is a separate instruction.

### PR14 · feat(results): sa evidence partition

**Owner:** SA · **Reviewers:** EJ, FA · **Approvals:** 1  
**Branch:** `feat/pr14-sa-sa-evidence-partition` → `main`  
**Dependencies:** PR09  
**Final commit:** `feat(results): sa evidence partition`

Import exactly this manifest batch: 33 final schema-4 runs, their 33 logs, and assigned reports/provenance. Each final run/log pair stays together; this is verification ownership, not experiment execution credit.

**Submit exactly these paths:**

- `evidence/PAPER_SKELETON.md`
- `evidence/carried_forward_pre_grid/label_audit_INSTRUCTIONS.md`
- `evidence/preregistration/experiment.yaml`
- `evidence/source_snapshot/figures/conditional_macro_f1.png`
- `evidence/source_snapshot/results/aggregate.csv`
- `evidence/source_snapshot/results/controls__xlm15__rescaled.json`
- `evidence/source_snapshot/results/embedding_norm_report__xlm15.json`
- `evidence/source_snapshot/results/runs/base=xlm15__cond=tokenizator__n=2000__seed=42__variant=mean_k64_rescaled.json`
- `evidence/source_snapshot/results/scramble.json`
- `evidence/source_snapshot/results/summary.csv`
- `evidence/source_snapshot/results/summary.json`
- `evidence/source_snapshot/results/test_evaluation_ledger.jsonl`
- `evidence/source_snapshot/results/top1_accuracy.json`
- `evidence/source_snapshot/results/transplant__xlmr__omp_k64_rescaled.json`
- `evidence/source_snapshot/results/transplant_controls/runs/base=xlm15__method=mean__cond=tokenizator__n=2000__seed=1337.json`
- `evidence/source_snapshot/results/transplant_controls/runs/base=xlm15__method=random_coef__cond=tokenizator__n=2000__seed=42.json`
- `evidence/source_snapshot/results/transplant_controls/runs/base=xlmr__method=mean__cond=tokenizator__n=2000__seed=13.json`
- `figures/escape_rate.png`
- `results/controls__xlm15.json`
- `results/controls__xlmr.json`
- `results/decompose.json`
- `results/logs/base=xlm15__cond=baza__n=10000__seed=42__20260909T010235Z.log`
- `results/logs/base=xlm15__cond=baza__n=2000__seed=13__20260908T194304Z.log`
- `results/logs/base=xlm15__cond=baza__n=500__seed=2024__20260908T232249Z.log`
- `results/logs/base=xlm15__cond=her_ikisi__n=2000__seed=13__20260908T210029Z.log`
- `results/logs/base=xlm15__cond=her_ikisi__n=500__seed=13__20260909T003607Z.log`
- `results/logs/base=xlm15__cond=tokenizator__n=10000__seed=13__20260909T011917Z.log`
- `results/logs/base=xlm15__cond=tokenizator__n=2000__seed=7__20260908T200201Z.log`
- `results/logs/base=xlm15__cond=tokenizator__n=20914__seed=1337__20260909T015450Z.log`
- `results/logs/base=xlm15__cond=tokenizator__n=500__seed=13__20260908T233227Z.log`
- `results/logs/base=xlm15__cond=transplant_mean__n=2000__seed=13__20260908T201402Z.log`
- `results/logs/base=xlm15__cond=transplant_mean__n=20914__seed=1337__20260909T020416Z.log`
- `results/logs/base=xlm15__cond=transplant_mean__n=500__seed=7__20260908T235100Z.log`
- `results/logs/base=xlm15__cond=transplant_random_coef__n=2000__seed=13__20260908T202903Z.log`
- `results/logs/base=xlm15__cond=transplant_random_coef__n=20914__seed=42__20260909T021029Z.log`
- `results/logs/base=xlm15__cond=transplant_random_coef__n=500__seed=13__20260909T000324Z.log`
- `results/logs/base=xlm15__cond=turk__n=2000__seed=13__20260908T204436Z.log`
- `results/logs/base=xlm15__cond=turk__n=500__seed=13__20260909T001928Z.log`
- `results/logs/base=xlm15__cond=turk_qarisiq__n=2000__seed=13__20260908T211612Z.log`
- `results/logs/base=xlm15__cond=turk_qarisiq__n=500__seed=13__20260909T005219Z.log`
- `results/logs/base=xlmr__cond=baza__n=2000__seed=13__20260908T213656Z.log`
- `results/logs/base=xlmr__cond=baza__n=500__seed=13__20260909T022714Z.log`
- `results/logs/base=xlmr__cond=her_ikisi__n=2000__seed=13__20260908T224715Z.log`
- `results/logs/base=xlmr__cond=her_ikisi__n=500__seed=13__20260909T033627Z.log`
- `results/logs/base=xlmr__cond=tokenizator__n=2000__seed=13__20260908T215328Z.log`
- `results/logs/base=xlmr__cond=tokenizator__n=500__seed=13__20260909T024213Z.log`
- `results/logs/base=xlmr__cond=transplant_mean__n=2000__seed=13__20260908T220711Z.log`
- `results/logs/base=xlmr__cond=transplant_mean__n=500__seed=13__20260909T025536Z.log`
- `results/logs/base=xlmr__cond=transplant_random_coef__n=2000__seed=13__20260908T222008Z.log`
- `results/logs/base=xlmr__cond=transplant_random_coef__n=500__seed=13__20260909T030852Z.log`
- `results/logs/base=xlmr__cond=turk__n=2000__seed=13__20260908T223328Z.log`
- `results/logs/base=xlmr__cond=turk__n=500__seed=13__20260909T032233Z.log`
- `results/logs/base=xlmr__cond=turk_qarisiq__n=2000__seed=13__20260908T230037Z.log`
- `results/logs/base=xlmr__cond=turk_qarisiq__n=500__seed=13__20260909T035001Z.log`
- `results/runs/base=xlm15__cond=baza__n=10000__seed=42.json`
- `results/runs/base=xlm15__cond=baza__n=2000__seed=13.json`
- `results/runs/base=xlm15__cond=baza__n=500__seed=2024.json`
- `results/runs/base=xlm15__cond=her_ikisi__n=2000__seed=13.json`
- `results/runs/base=xlm15__cond=her_ikisi__n=500__seed=13.json`
- `results/runs/base=xlm15__cond=tokenizator__n=10000__seed=13.json`
- `results/runs/base=xlm15__cond=tokenizator__n=2000__seed=7.json`
- `results/runs/base=xlm15__cond=tokenizator__n=20914__seed=1337.json`
- `results/runs/base=xlm15__cond=tokenizator__n=500__seed=13.json`
- `results/runs/base=xlm15__cond=transplant_mean__n=2000__seed=13.json`
- `results/runs/base=xlm15__cond=transplant_mean__n=20914__seed=1337.json`
- `results/runs/base=xlm15__cond=transplant_mean__n=500__seed=7.json`
- `results/runs/base=xlm15__cond=transplant_random_coef__n=2000__seed=13.json`
- `results/runs/base=xlm15__cond=transplant_random_coef__n=20914__seed=42.json`
- `results/runs/base=xlm15__cond=transplant_random_coef__n=500__seed=13.json`
- `results/runs/base=xlm15__cond=turk__n=2000__seed=13.json`
- `results/runs/base=xlm15__cond=turk__n=500__seed=13.json`
- `results/runs/base=xlm15__cond=turk_qarisiq__n=2000__seed=13.json`
- `results/runs/base=xlm15__cond=turk_qarisiq__n=500__seed=13.json`
- `results/runs/base=xlmr__cond=baza__n=2000__seed=13.json`
- `results/runs/base=xlmr__cond=baza__n=500__seed=13.json`
- `results/runs/base=xlmr__cond=her_ikisi__n=2000__seed=13.json`
- `results/runs/base=xlmr__cond=her_ikisi__n=500__seed=13.json`
- `results/runs/base=xlmr__cond=tokenizator__n=2000__seed=13.json`
- `results/runs/base=xlmr__cond=tokenizator__n=500__seed=13.json`
- `results/runs/base=xlmr__cond=transplant_mean__n=2000__seed=13.json`
- `results/runs/base=xlmr__cond=transplant_mean__n=500__seed=13.json`
- `results/runs/base=xlmr__cond=transplant_random_coef__n=2000__seed=13.json`
- `results/runs/base=xlmr__cond=transplant_random_coef__n=500__seed=13.json`
- `results/runs/base=xlmr__cond=turk__n=2000__seed=13.json`
- `results/runs/base=xlmr__cond=turk__n=500__seed=13.json`
- `results/runs/base=xlmr__cond=turk_qarisiq__n=2000__seed=13.json`
- `results/runs/base=xlmr__cond=turk_qarisiq__n=500__seed=13.json`
- `results/transplant__xlm15__random_coef_k64_rescaled.json`
- `results/transplant__xlmr__mean_k64.json`
- `results/transplant__xlmr__random_coef_k64.json`
- `results/truncation.json`
- `tables/test_tables.md`

**Validation:**

- workflow_context: `python scripts/pr_plan.py scope PR14 --repo ${REPO}`
- workflow_context: `python scripts/pr_plan.py audit --repo ${REPO} --pr PR14`

**Review/acceptance:**

- Record schema, unique run key, matching log, source hash, metric consistency and provenance status in the PR review.
- Results-full is authoritative at results/; pre-grid records stay under evidence/, never results/runs/.
- AMENDMENT: initial import of final evidence and explicitly separated prior evidence.
- After all five evidence PRs merge, run the full import audit and the actual grid integrity check before PR15–19.
- Use the edited ZIP for evidence/source_snapshot origins, including its reduced test fields and empty ledger. Keep results_full.zip final-run bytes and its separate 164-entry ledger unchanged; never fill removed fields from the old source ZIP.
- Include the registered `AMENDMENT:` explanation for result/table writes.
- All declared checks pass, required real non-author review is recorded, and the diff stays in the registered paths. Opening the PR is the end of "do PR#"; merging is a separate instruction.

### PR15 · feat(delivery): anchors paper and artifact guide

**Owner:** SS · **Reviewers:** FB, SA · **Approvals:** 1  
**Branch:** `feat/pr15-ss-anchors-paper-and-artifact-guide` → `main`  
**Dependencies:** PR10, PR11, PR12, PR13, PR14  
**Final commit:** `feat(delivery): anchors paper and artifact guide`

Deliver the anchor/grid verifier and focused tests; write Abstract, Discussion/Limitations and slides 1–4; complete the SS claim/reference fragments and truthful contribution row; update README and artifact navigation. Use the shared skeleton and directly loaded bibliography fragments from PR04. Do not incorporate another member's unfinished source or section prose into this commit. Add synthetic entry-point regression tests documenting Python val defaults, refusal of test without opt-in and current shell-wrapper test defaults; do not execute training or the real test split.

**Submit exactly these paths:**

- `README.md`
- `docs/ANCHOR_VERIFICATION.md`
- `docs/ARTIFACT_NAVIGATION.md`
- `docs/contributions/ss.md`
- `presentation/sections/ss.tex`
- `report/claims/ss.yaml`
- `report/references/ss.bib`
- `report/sections/00_abstract.tex`
- `report/sections/07_discussion_limitations.tex`
- `results/anchor_report.json`
- `src/tokenization/anchor_report.py`
- `tests/test_anchor_report.py`
- `tests/test_test_access_entrypoints.py`

**Validation:**

- workflow_context: `python scripts/pr_plan.py scope PR15 --repo ${REPO}`
- workflow_context: `python scripts/pr_plan.py audit --repo ${REPO}`
- workflow_context: `python scripts/pr_plan.py audit-test-access --repo ${REPO} --stage evidence`
- repository: `python -m pytest -q tests/test_anchor_report.py`
- repository: `python -m src.tokenization.anchor_report --config configs/experiment.yaml`
- repository: `make pdf-report`
- repository: `make pdf-slides`
- repository: `python -m pytest -q tests/test_test_access_entrypoints.py`

**Review/acceptance:**

- Anchor values are derived from actual evidence; test invalid/special anchors and unknown bases.
- Own numeric claims are manually traced to source files now; the automatic global strict check is a PR19 gate, not an unavailable PR15 command.
- The two owned report sections, four slides, README and contribution row are complete; other member sections may still be draft stubs.
- Do not regenerate a shared refs.bib snapshot that would require another commit after later fragments arrive.
- Any final navigation review after PR19 is recorded as PR review/comment if no tracked-file fix is needed.
- Full frozen import and ledger audits run before implementing new modules. Entry-point tests mock subprocess/model/data access; the empty snapshot ledger never resets access history.
- Include the registered `AMENDMENT:` explanation for result/table writes.
- All declared checks pass and real non-author review is recorded on this head. Commit this complete delivery once. Opening the PR ends "do PR#"; merging or publishing is a separate instruction.

### PR16 · feat(delivery): data verification paper and provenance

**Owner:** FA · **Reviewers:** SS, EJ · **Approvals:** 1  
**Branch:** `feat/pr16-fa-data-verification-paper-and-provenance` → `main`  
**Dependencies:** PR15  
**Final commit:** `feat(delivery): data verification paper and provenance`

Deliver the split-manifest verifier, focused provenance tests, DATA.md and actual data-audit evidence. Write Introduction, Data Setup and slides 5–8; complete FA claim/reference fragments and truthful contribution row. Verify the received dataset rather than copying an existing success statement. Verify the edited-snapshot versus as-run ledger/configuration provenance using fixtures and supplied file hashes, and describe their different histories in DATA.md.

**Submit exactly these paths:**

- `docs/DATA.md`
- `docs/DATA_PROVENANCE_AUDIT.md`
- `docs/contributions/fa.md`
- `presentation/sections/fa.tex`
- `report/claims/fa.yaml`
- `report/references/fa.bib`
- `report/sections/01_introduction.tex`
- `report/sections/04_setup_data.tex`
- `results/data_integrity_report.json`
- `results/provenance_verification.json`
- `src/data/verify_splits.py`
- `tests/test_evidence_provenance.py`
- `tests/test_verify_splits.py`

**Validation:**

- workflow_context: `python scripts/pr_plan.py scope PR16 --repo ${REPO}`
- repository: `python -m pytest -q tests/test_verify_splits.py tests/test_evidence_provenance.py`
- repository: `python -m src.data.verify_splits --config configs/experiment.yaml --data-root ${DATA_ROOT}`
- repository: `make pdf-report`
- repository: `make pdf-slides`

**Review/acceptance:**

- Reproduce the exact train/validation/test hashes and 20937/2791/4186 counts; verify zero normalized-text overlap.
- Keep the dataset outside Git and describe the one-example pre-grid difference and unpinned revision accurately.
- Validate the two own paper sections and four slides; full global claims/finished-paper validation occurs in PR19.
- The provenance report records measurements actually repeated by the contributor.
- Data identity checks may hash the supplied test split but must not evaluate a model or use its labels for tuning. The empty edited snapshot ledger and 164-entry as-run ledger are distinct sources, not proof of zero lifetime test access.
- Include the registered `AMENDMENT:` explanation for result/table writes.
- All declared checks pass and real non-author review is recorded on this head. Commit this complete delivery once. Opening the PR ends "do PR#"; merging or publishing is a separate instruction.

### PR17 · feat(delivery): model verification paper and ci

**Owner:** EJ · **Reviewers:** FA, FB · **Approvals:** 1  
**Branch:** `feat/pr17-ej-model-verification-paper-and-ci` → `main`  
**Dependencies:** PR16  
**Final commit:** `feat(delivery): model verification paper and ci`

Deliver the model/parameter and environment verification, focused release-integrity failure tests and CI readiness logic. Write Method, Tools/Acknowledgements and slides 9–12; complete EJ claim/reference fragments and contribution row. Document the clean checkout and stages actually tested at this point. Define the final validation commands for PR19; do not claim to have tested not-yet-delivered final files. Record the edited FROZEN.md/lock hashes alongside retained as-run configuration provenance; no claim that old results were rerun on this edited ZIP.

**Submit exactly these paths:**

- `.github/scripts/pr_hygiene.py`
- `.github/workflows/ci.yml`
- `docs/CLEAN_CLONE_AUDIT.md`
- `docs/ENVIRONMENT_AS_RUN.md`
- `docs/MODEL_AND_CONTROL_VERIFICATION.md`
- `docs/contributions/ej.md`
- `presentation/sections/ej.tex`
- `report/claims/ej.yaml`
- `report/references/ej.bib`
- `report/sections/03_method.tex`
- `report/sections/09_tools_ack.tex`
- `requirements-as-run.txt`
- `results/model_table.json`
- `scripts/verify_repo.py`
- `src/analysis/model_table.py`
- `tests/test_model_table.py`
- `tests/test_release_integrity.py`

**Validation:**

- workflow_context: `python scripts/pr_plan.py scope PR17 --repo ${REPO}`
- repository: `python -m pytest -q tests/test_model_table.py tests/test_release_integrity.py`
- repository: `python -m src.analysis.model_table --config configs/experiment.yaml`
- repository: `make pdf-report`
- repository: `make pdf-slides`

**Review/acceptance:**

- Parameter-count definitions and as-run/pinned environment differences are explicit; do not silently change historical pins.
- Failure fixtures cover missing runs, duplicate identities, stale claims and missing PDFs without pretending the absent final deliverables passed.
- CI enforces stage-appropriate checks using the registered plan ID; final gates genuinely run on PR19 rather than requiring impossible inputs on PR17.
- EJ performs the final independent clean-checkout review on PR19 and attaches its log/artifact to that PR; no separate sign-off commit is needed when no tracked changes are required.
- Compare each configuration capture to its own archive hash. A different historical FROZEN.md is disclosed provenance, not a reason to overwrite it or regenerate metrics.
- Include the registered `AMENDMENT:` explanation for result/table writes.
- All declared checks pass and real non-author review is recorded on this head. Commit this complete delivery once. Opening the PR ends "do PR#"; merging or publishing is a separate instruction.

### PR18 · feat(delivery): test tables paper and release tooling

**Owner:** FB · **Reviewers:** SA, SS, EJ · **Approvals:** 1  
**Branch:** `feat/pr18-fb-test-tables-paper-and-release-tooling` → `main`  
**Dependencies:** PR17  
**Final commit:** `feat(delivery): test tables paper and release tooling`

Deliver the test-table generator, focused tests, Amendment-8 report columns and the existing analysis-only wiring. Resolve and document the two test-table versions' inference-policy difference. Write Training Setup, Conclusion and slides 13–16; complete FB claim/reference fragments and contribution row. Add release-building tooling and its tests now, so the complete PDFs and manifests can be produced after PR19 fills the final sections. Do not commit prematurely labelled final PDFs or a stale final release manifest. Add synthetic fine-tune guard and ledger regression tests; cover validation without test-file access, refused test without opt-in, authorized synthetic test with an appended ledger record, and analysis of records lacking test fields. Extend the existing analysis-only route; it already exists in this source version.

**Submit exactly these paths:**

- `docs/REPRODUCE_RELEASE.md`
- `docs/TEST_TABLES_POLICY.md`
- `docs/contributions/fb.md`
- `presentation/sections/fb.tex`
- `report/claims/fb.yaml`
- `report/references/fb.bib`
- `report/sections/05_setup_training.tex`
- `report/sections/08_conclusion.tex`
- `results/paper_tables.md`
- `run_all.sh`
- `scripts/build_release.py`
- `src/analysis/report.py`
- `src/analysis/test_tables.py`
- `tables/test_tables.md`
- `tests/test_release_package.py`
- `tests/test_test_access_runtime.py`
- `tests/test_test_tables.py`

**Validation:**

- workflow_context: `python scripts/pr_plan.py scope PR18 --repo ${REPO}`
- repository: `python -m pytest -q tests/test_test_tables.py tests/test_release_package.py`
- disposable_analysis_checkout: `bash run_all.sh --analysis-only`
- disposable_analysis_checkout: `git diff --exit-code -- tables/test_tables.md`
- repository: `make pdf-report`
- repository: `make pdf-slides`
- repository: `python -m pytest -q tests/test_test_access_runtime.py`

**Review/acceptance:**

- Canonical T1–T4 come from the evidence pack; the standalone T5/T6 supplement remains separately preserved and its inference status is explicitly reviewed.
- Resolve bootstrap ordering/seed from evidence. If exact regeneration is not defensible, record and review a deterministic amendment instead of hard-coding output.
- Existing report.py and run_all.sh changes have an AMENDMENT explanation and source-owner review.
- build_release.py works on a complete input tree, tests missing/changed artifacts, and writes checksums deterministically; integration with the complete final tree is checked in PR19.
- FB retains packaging/publication responsibility after PR19. A release tag and asset upload do not create another code commit; publishing requires a separate explicit instruction.
- All test-access regression fixtures are synthetic or mocked; no real model evaluation of the held-out split occurs.
- Run analysis-only regeneration in a disposable verification checkout containing the reviewed PR code and retained final evidence. Check snapshot/result hashes before and after; allow writes only to documented derived outputs there. Stage only the registered report/table outputs back in the PR checkout. Never run the full wrapper as a shortcut.
- Reduced historical records may lack test fields. Final tables admit schema-4 records only and must not fabricate predictions or treat absent metrics as zero. Keep the original final ledger unchanged.
- Include the registered `AMENDMENT:` explanation for result/table writes.
- All declared checks pass and real non-author review is recorded on this head. Commit this complete delivery once. Opening the PR ends "do PR#"; merging or publishing is a separate instruction.

### PR19 · feat(delivery): claims results and final integration

**Owner:** SA · **Reviewers:** EJ, FA, FB, SS · **Approvals:** 2  
**Branch:** `feat/pr19-sa-claims-results-and-final-integration` → `main`  
**Dependencies:** PR18  
**Final commit:** `feat(delivery): claims results and final integration`

Deliver the numeric-claim checker/tests and own Related Work, Results and slides 17–20. Complete the SA reference/claim/contribution fragments, integrate all five factual contribution rows, and build the complete report/deck/contribution PDFs. Run the actual full clean-tree verification and release builder from PR18; commit the resulting final artifacts, changelog and release manifest in this one delivery. Other members provide their real final review acknowledgements and reproduction logs on this PR, not separate sign-off commits. Resolve test-access/history claims against the edited source note and separately retained as-run evidence; reject unsupported claims that an empty edited ledger proves no prior test access.

**Submit exactly these paths:**

- `CHANGELOG.md`
- `contribution_report.pdf`
- `contribution_report.tex`
- `docs/CONTRIBUTIONS.md`
- `docs/RELEASE_MANIFEST.json`
- `docs/contributions/ej.md`
- `docs/contributions/fa.md`
- `docs/contributions/fb.md`
- `docs/contributions/sa.md`
- `docs/contributions/ss.md`
- `presentation/presentation.pdf`
- `presentation/sections/sa.tex`
- `report/claims/ej.yaml`
- `report/claims/fa.yaml`
- `report/claims/fb.yaml`
- `report/claims/index.yaml`
- `report/claims/sa.yaml`
- `report/claims/ss.yaml`
- `report/references/sa.bib`
- `report/report.pdf`
- `report/sections/02_related_work.tex`
- `report/sections/06_results.tex`
- `scripts/verify_paper_claims.py`
- `tests/test_verify_paper_claims.py`

**Validation:**

- workflow_context: `python scripts/pr_plan.py scope PR19 --repo ${REPO}`
- repository: `python -m pytest -q tests/test_verify_paper_claims.py`
- repository: `python scripts/verify_paper_claims.py --strict`
- repository: `make pdf-contribution`
- repository: `make pdf`
- repository: `make verify-all`
- repository: `python scripts/build_release.py --output-manifest docs/RELEASE_MANIFEST.json`

**Review/acceptance:**

- All 164 final records, 164 logs, every final numeric claim, dataset identity, final PDF and contribution row pass the actual declared checks.
- Per-member claim fragments from PR15–18 are respected. A correction in another member's fragment must be explicit, reviewed by that member and credited as a correction.
- SA integrates the other members' contribution rows and generated outputs; this does not transfer authorship of their code, prose, execution or review to SA.
- Every member acknowledges the contribution wording; EJ attaches independent clean-checkout verification and FB checks release packaging.
- Require two real non-author approvals on the final head. Do not merge, tag or publish on the basis of "do PR19" alone.
- Final release assets include data.tgz, run.log.txt and the evidence pack, separately from the code tree.
- Every test metric points to an unchanged schema-4 final record. Missing historical test fields are absence of stored data, not zero scores or evidence that access never happened.
- All declared checks pass and real non-author review is recorded on this head. Commit this complete delivery once. Opening the PR ends "do PR#"; merging or publishing is a separate instruction.

## 7. Measured balance and final accounting

| Member | src physical lines | src substantive lines | All existing code | Importance score | Runs + logs |
| --- | --- | --- | --- | --- | --- |
| SS | 1783 | 1156 | 1754 | 5121 | 32 + 32 |
| FA | 1747 | 1186 | 1901 | 5301 | 33 + 33 |
| EJ | 1812 | 1145 | 1737 | 4905 | 33 + 33 |
| FB | 1659 | 1204 | 1798 | 5372 | 33 + 33 |
| SA | 1678 | 1155 | 1753 | 4988 | 33 + 33 |

| Member | Commit 1 | Commit 2 | Commit 3 | Commit 4 | Total |
| --- | --- | --- | --- | --- | --- |
| SS | Bootstrap | PR05 source | PR10 evidence | PR15 combined delivery | 4 |
| FA | PR02 governance | PR06 source | PR11 evidence | PR16 combined delivery | 4 |
| EJ | PR01 CI | PR07 source | PR12 evidence | PR17 combined delivery | 4 |
| FB | PR03 tooling | PR08 source | PR13 evidence | PR18 combined delivery | 4 |
| SA | PR04 registry/skeletons | PR09 source | PR14 evidence | PR19 combined delivery | 4 |

All 539 canonical paths and owners are retained. Edited-file hashes and archive origins are updated. The source, code and importance max/min ratios remain below 1.10. Each five-seed cell gives one run to each member, with three-seed cells rotated. The evidence remains 32/33/33/33/33 paired runs/logs and 4.326–4.394 MB per member.

Record new code, imported code, evidence verification, review and actual experiment execution separately. The four-commit plan balances permanent main history; it does not redefine historical authorship or prove identical future intellectual effort. Any necessary additional change after merge must be visible and honestly credited.

## 8. Current source and retained experiment evidence

The source of every snapshot import is **aztokenizertransfer_FINAL_v3_edited.zip**. The old source ZIP is not required and must never be used as a fallback. Earlier results_full.zip, report_evidence_pack.zip, env_capture.zip and the standalone test_tables.md remain the inputs for their separately registered evidence paths. These archives were not replaced by the edited code ZIP.

The edited snapshot empties its historical test ledger and removes test/test_predictions/test_gold from 22 older run records. The workflow imports those reduced bytes exactly under evidence/source_snapshot; it does not restore the removed fields. An empty edited ledger does not establish that no previous test access occurred. Earlier supplied audit documents still describe 22 pre-guard accesses, and the unchanged final-run ledger has 164 records. Treat these as different provenance layers, not interchangeable counts or proof of an untouched holdout. Final paper wording must reflect the available history and uncertainty.

The current run_single/run_grid Python interfaces default to validation and require explicit test opt-in. **run_all.sh and scripts/vastai_run_full.sh still default their full-run wrappers to EVAL_SPLIT=test and supply the opt-in flags.** Those wrappers are unchanged in the edited ZIP. Ordinary PR verification must not execute a full wrapper or a real test evaluation. Use synthetic fixtures for guard tests; use the existing analysis-only route for stored-result tables in a disposable verification checkout. Audit dataset identity without training, prediction or choosing models from test scores.

The 164 final schema-4 results and 164 logs are retained byte-for-byte from results_full.zip; they were not rerun on the edited snapshot. The current configs/FROZEN.md hash differs from the retained as-run preregistration copy. Verify each version against its own origin and record the distinction; do not relabel old results as newly generated. PR16 covers provenance, PR17 environment/configuration comparison, PR18 table policy and runtime guard regression tests, and PR19 numeric claim interpretation. Each owner also covers their assigned source responsibilities.

Canonical figures and T1–T4 tables come from the evidence pack. The standalone test_tables.md adds T5/T6 analyses with a test-inference policy conflict; PR18 preserves both received versions and documents their status. No policy conflict is resolved merely by emptying a ledger. Data counts/hashes remain 20,937 / 2,791 / 4,186 with previously verified zero normalized-text overlap; PR16 repeats the identity checks. Environment, GPU-memory, dataset-revision, annotation and mechanism limitations still require evidence-based wording. The original course PDF was not provided; this workflow does not certify rules that are absent from the supplied materials.

See docs/CODEBASE_UPDATE.md for exact changed paths, hash inputs and PR responsibilities. Read it with the current registry; earlier source instructions are historical context.

## 9. Contribution report and publication

Each member commits their own docs/contributions/<initials>.md row in PR15–19. PR19 compiles the report, gets all acknowledgements, incorporates final PDFs and the release manifest, and requires two real non-author approvals. Corrections to another person's row or claim fragment require their review; integration is not original authorship.

After PR19 is merged, a separate explicit release instruction lets FB tag that existing reviewed commit and upload the prepared assets. Do not create a separate fifth release/cleanup/PDF commit merely for ceremony. If a real defect needs a fix, preserve an honest history and explain the count change.

## 10. Definition of done

- [ ] The helper validates this plan version, PR01–PR19, the dependency graph and four planned authored main commits per member including SS bootstrap.
- [ ] All source and evidence imports pass exact-path/hash checks; the recomputed balance metrics remain within 10%.
- [ ] PR15–19 include all assigned new code/tests, two paper sections, four slides and completion duties; nothing required was dropped to reduce commit count.
- [ ] PR19 genuinely passes complete-run integrity, strict claims, data identity, final PDF, contribution-report and release-builder checks on the final head.
- [ ] All five contribution rows are factual and acknowledged; independent reviews/reproduction logs are attached without artificial sign-off commits.
- [ ] Bootstrap plus 19 squash commits gives 20 planned main commits, four per member, or any unavoidable extra fix is transparently documented.
- [ ] Tagging/uploading occurs only after the reviewed final commit and a separate release instruction.
