# The meaning of “do PR#” — balanced-edited-v4-4-commits

Read this file and `.github/pr-plan.yml` from this workflow package. Until PR04 installs the package in the target repo, keep this directory as the external workflow context. `PR_PLAN.yml` is an identical portability copy, not a second editable authority. After installation, `.github/pr-plan.yml` is authoritative. The registry uses JSON syntax, which is valid YAML and can also be read with Python's standard library.

“do PR3”, “do PR03” and “do PR03 from balanced-edited-v4-4-commits” resolve to the exact PR03 entry. Never resolve a number against the attached legacy 19-PR Claude registry, remembered conversation, or a different plan version. If the live repository already has a different PR plan/history, inspect it and reconcile before executing this fresh-history plan; never erase, force-reset or renumber existing contributions implicitly. Actual GitHub issue/PR numbers can differ from plan IDs: record the mapping by exact title/branch/plan_id, not numeric coincidence.

The user invoking “do PR#” requests the complete branch, implementation/import, validation, truthful commit, push and opening of that registered PR. It does not request approving another member's work, merging, changing branch protections mid-task or publishing a release. A separate merge/release instruction handles those actions. This document itself requests no GitHub action.

## Standard lifecycle

1. Run `python scripts/pr_plan.py show PR03` in this workflow directory (replace the number with the requested ID). Read objective, owner, exact paths, dependencies, commit message, reviewers, validation and acceptance. Do not invent scope or branch names.
2. Read `docs/CODEBASE_UPDATE.md`. Run `python scripts/pr_plan.py check-inputs --input-dir <INPUT_DIR>` before imports. Resolve `INPUT_DIR`, `REPO` and, when required, `DATA_ROOT` to actual absolute paths. Validate target remote, a clean working tree, correct plan version and the merged state of every dependency using actual GitHub PR metadata/merge SHAs. Do not detect dependencies by a substring in `git log`. Do not consume legacy PAT files, device codes or account-switching helpers from the supplied archive.
3. Check the authenticated GitHub account and real author identity. The assigned member must genuinely own/perform the contribution. If the required account is unavailable, prepare the reviewable local work and report that login is needed. Never change names/emails or act through other people's stored tokens to simulate participation. Never manufacture reviewer approvals.
4. Fetch and synchronize `main`, then create the registry's exact branch. For import PRs, run `materialize` against the extracted **DL Final Project** input folder containing the edited code ZIP plus the earlier evidence inputs listed in docs/CODEBASE_UPDATE.md. It copies only that PR's registered members after verifying source hashes; it does not execute imported files or run Git. For new-work PRs, implement the registered outputs, keeping existing baseline bytes intact except explicitly listed amendments.
5. Inspect the full diff, including staged, unstaged and untracked files. Run `scope` plus every validation entry, using its declared working directory. `${REPO}` and `${DATA_ROOT}` are argument substitutions, not text to pass literally. Do not skip a missing prerequisite and describe the check as passed. Source import PR05–08 intentionally use hashes/scope; the full project test gate starts after PR09. PR15–18 use their available member-specific tests and draft builds; global strict claims, complete-document and final release checks run on PR19.
6. Stage **only named paths**, using a literal pathspec file or safely quoted arguments. One final author commit per planned PR with the exact `commits[0]` message; necessary local WIP commits may be squashed. Distinguish imported existing code from new implementation. Record shared authorship only where real.
7. Push only the feature branch; open the PR with exact title, real owner and requested reviewers. Include plan ID, path manifest, source hashes, tests actually run, omissions/limitations, new-versus-imported contribution and the required `AMENDMENT:` line. Requesting review does not constitute approval. Stop here and return the PR link/status.
8. On a later merge instruction, verify actual independent approvals and all applicable checks on the latest head, then squash-merge once. Preserve the actual author and record the merge SHA. Any extra necessary fix is a real additional contribution; update the plan honestly instead of padding or rewriting history to retain equal counts.

## Local commands

```powershell
python scripts/pr_plan.py verify-plan
python scripts/pr_plan.py show PR05
python scripts/pr_plan.py materialize PR05 --input-dir 'C:\path\DL Final Project' --output 'C:\path\repo'
python scripts/pr_plan.py scope PR05 --repo 'C:\path\repo' --base origin/main
python scripts/pr_plan.py audit --repo 'C:\path\repo' --pr PR05
```

The helper materializes existing imports only. It does not implement future modules, create PDFs, configure CI, commit, log in, push, open or merge PRs. PR06 introduces the exact HANDOFF prefix; PR09 requires that prefix and appends the remainder. Full source audits apply at the source-import milestone; the full 539-file audit applies after the five evidence imports and before intentional amendments to generated tables/source. Later amendment verification uses its declared original/new hashes and tests, not a demand that an intentionally modified file still equal its original hash.

## Four-commit accounting

This version has **19 PRs plus one bootstrap commit = 20 main-branch commits, four per member**. SS has bootstrap + PR05 + PR10 + PR15. FA has PR02 + PR06 + PR11 + PR16. EJ has PR01 + PR07 + PR12 + PR17. FB has PR03 + PR08 + PR13 + PR18. SA has PR04 + PR09 + PR14 + PR19.

Each final PR15–19 combines its owner's new verification code, two paper sections, four slides and completion duties. Keep the combined change on that member's one delivery branch; local revision/WIP commits may be consolidated before the single squash merge. Do not open the retired PR20–29 or a separate sign-off/packaging commit. Full final integration runs on PR19 after PR15–18 have merged. FB's later authorized tagging and uploading of already prepared assets does not add another code commit.

All validation and real review still applies. If a real defect after merge requires a tracked change, record the necessary fix honestly and explain the departure from the planned count. Four planned commits is not permission to hide changes, forge identity, rewrite merged history or pretend failed validation passed.

## Edited source and test access

Use the exact edited ZIP filename and hash in the manifest. Original source ZIP bytes are not an alternative. Preserve the edited historical record omissions and empty snapshot ledger, and preserve separately supplied as-run evidence at its own paths. Ordinary PR work never authorizes a new real held-out evaluation or clearing a ledger. Full shell wrappers still default to test evaluation. Synthetic guard tests and stored-result analysis are the registered checks.

`audit-test-access --repo <REPO> --stage source` runs at PR09. `--stage evidence` runs after all PR10–14 and before PR15 changes the baseline. These are frozen-import milestone checks, not post-amendment hash expectations. Later final checks use the PR15/PR18 regression tests and recorded amendment hashes.

For PR18, the `disposable_analysis_checkout` validation context means ANALYSIS_REPO, a separate disposable checkout of this PR's reviewed code with the retained evidence. Record hashes before/after analysis; inspect any differences. Copy back only declared report/table outputs. A mismatch against reference tables is a finding for the registered deterministic-amendment process, never permission to run new test evaluations or fabricate matching output.
