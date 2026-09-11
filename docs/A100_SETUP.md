# A100 setup — one reproducible sequence

Target: NVIDIA A100-SXM4-40GB, MIG partition `3g.20gb` (20 GB VRAM slice).
Written 2026-09-04 after the RTX 4060 diagnostic (`HANDOFF.md §12`,
`configs/FROZEN.md` "Amendments"). Follow this in order; every step must
succeed before the next one runs. **This document verifies the environment
only — it does not launch the preflight or the grid.**

**Automated form of steps 1–7 below:** `bash scripts/a100_bootstrap.sh`,
run from the repo root after cloning and after the manual transfer in
step 1.5. It fails loudly (non-zero exit) on the first broken step and
prints `docs/TEST_AUDIT.md` at the end. The manual steps below are the
reference for *why* each check exists — read them at least once even if
you run the script.

## 1. Clone

```bash
git clone <this-repo-url> az-tokenizer-transfer
cd az-tokenizer-transfer
git log --oneline -1
# expect: 5ca0aa3 Record parameter decomposition + base-model capacity-mismatch limitation
# (or a later commit on main, if this doc has been updated since — check
# `git log --oneline` for the tip, not this exact hash, if it doesn't match)
```

If the commit hash printed differs from the one above, stop — you have not
cloned the state this document describes.

## 1.5. Manual transfer (not part of the clone)

`artifacts/` is gitignored — the transplanted model weights and the frozen
AZ/TR data splits do not travel with `git clone`. See
`docs/A100_TRANSFER_MANIFEST.md` for the exact file list (38 files, ~3.57 GB
for what the frozen queue actually needs) and transfer them now, then
verify:
```bash
sha256sum -c docs/a100_transfer_checksums.sha256
```
Every line must read `OK`. None of steps 2–7 below actually touch
`artifacts/` (the test suite is fully mocked, no real data files), but
`scripts/a100_benchmark.sh` and the real grid both need it — do the
transfer now so it's not a surprise later.

## 2. Create the venv and install from the lock file

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu121
```

> **2026-09-08 — do not run the line above as written.** `requirements.lock` does not resolve (`fsspec==2026.7.0` vs `datasets==2.21.0`, which needs `fsspec[http]<=2024.6.1`). See that file's own header for what the grid actually ran on.

Do **not** `pip install -r requirements.txt` here — that file resolves
versions fresh and is not what produced the diagnostic. `requirements.lock`
is authoritative for this migration; see its header for why `--no-deps` is
deliberately *not* used.

## 3. Verify the CUDA stack

```bash
python -c "
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
print('device name', torch.cuda.get_device_name(0))
print('device count', torch.cuda.device_count())
props = torch.cuda.get_device_properties(0)
print('total VRAM (GB)', round(props.total_memory / 1e9, 2))
print('cudnn', torch.backends.cudnn.version())
"
```

Expect: `torch 2.4.1+cu121`, `cuda available True`, device name containing
`A100`, and **total VRAM close to 20 GB** (the MIG `3g.20gb` slice — not
40 GB; seeing 40 GB means the MIG partition isn't actually in effect and
this document's Step 5 VRAM-vs-slice numbers won't apply). `cudnn` should
read `90100` to match the 4060 dev machine (see `requirements.lock`); a
different cuDNN build number is not necessarily wrong but is worth noting
before comparing timing numbers across machines.

```bash
nvidia-smi -L
```

**Expected and NOT an error** on a MIG guest: `Failed to display GPU
instances: Insufficient Permissions`. The assigned MIG device is still
visible and usable — this message means the guest can't enumerate sibling
instances on the physical card, which is normal MIG guest behavior, not a
broken setup. Confirm usability with the `torch.cuda` check above instead;
that's the real signal.

## 4. Pull the donor model at its pinned revision

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForMaskedLM
donor = 'HPLT/hplt_bert_base_az'
revision = 'a126552c30733333cc95425b4eefd3d85b39d878'
tok = AutoTokenizer.from_pretrained(donor, revision=revision)
model = AutoModelForMaskedLM.from_pretrained(donor, revision=revision)
print('donor pulled OK at pinned revision', revision)
print('vocab size', tok.vocab_size)
"
```

This mirrors exactly how `src/transplant/donor_embeddings.py` loads the
donor (`revision=` pinned, no `trust_remote_code`) — see that file's
docstring for why. If this succeeds, the transplant build step
(`python -m src.transplant.build`) has what it needs; artifacts themselves
are gitignored and are not part of this clone (rebuild them, don't expect
them to appear from git).

## 5. Verify the frozen configs hash-match the committed versions

This is enforced automatically now — every real entry point
(`src.training.orchestrate.main`, `src.training.finetune.main`) calls
`src.utils.verify_config_hashes()` immediately after loading the config,
against `configs/CONFIG_HASHES.lock`, and **exits loudly** on any mismatch
(including in `--dry-run`). You don't need a separate manual step for this —
Step 7's dry-run below already exercises it — but to check explicitly:

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.utils import verify_config_hashes
verify_config_hashes()
print('config hashes OK')
"
```

If this raises `SystemExit`, stop. Nothing scientific may differ between the
4060 dev machine and this one; a hash mismatch means the config on disk here
is not the one the diagnostic and the frozen preregistration describe, and
must be fixed (re-clone, or check for local edits) before anything else runs.

## 6. Run the test suite

```bash
python -m pytest tests/ -q
```

Expect `68 passed` (the count on the 4060 dev machine as of this commit). A
different count means either the clone is incomplete or something in the
environment behaves differently — resolve before proceeding, don't skip.

## 7. Dry-run the queue

```bash
python -m src.training.orchestrate --config configs/experiment.yaml --dry-run
```

Expect exactly:
- **114 total runs** listed, in the frozen priority/condition/seed order
  (reference size 2000 first for each base, `baza` → `tokenizator` →
  `transplant_mean` → `transplant_random_coef` → `turk` → `her_ikisi` →
  `turk_qarisiq`, primary base before contrast base).
- `TURKISH STAGES: 30 requests, 30 distinct checkpoints` on the last line.

If either number differs, stop — the queue construction (`configs/
experiment.yaml`, `src/utils.py`'s `sizes_for_base`/`seeds_for_size`, or
`src/training/orchestrate.py`'s `build_queue`) is not the version this
document was written against, even if Step 5's hash check somehow passed.

## Done

Once steps 1–7 all pass, the environment is verified equivalent to the 4060
dev machine's for everything that can affect results (dependency versions,
CUDA/cuDNN, donor model content, frozen config content, queue construction).

Next: `bash scripts/a100_benchmark.sh` (two 30-step benchmarks — Azerbaijani
and Turkish stages — not a training run; see the script's own docstring).
Report its numbers, including whether `expandable_segments` produced a
`UserWarning` on Linux and whether per-step time is stable or degrading
across the 30 steps, before anyone decides to launch the preflight or the
grid — **this document and both scripts stop here and launch neither.**
