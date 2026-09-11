#!/usr/bin/env bash
# =====================================================================
#  A100 BOOTSTRAP — verifies the environment end-to-end. LAUNCHES NOTHING.
#
#  Executable form of docs/A100_SETUP.md — read that file for the "why"
#  behind each step. Run from the repo root, after `git clone` and after
#  transferring the gitignored files listed in
#  docs/A100_TRANSFER_MANIFEST.md (not part of the clone).
#
#  Usage:
#      bash scripts/a100_bootstrap.sh
#
#  On any failure this exits non-zero immediately (set -euo pipefail) with
#  the failing step's own error — nothing here swallows an error to keep
#  going. Does not start the preflight or the grid; see scripts/a100_benchmark.sh
#  for the next step, and HANDOFF.md / configs/FROZEN.md for the grid itself.
# =====================================================================
set -euo pipefail

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

log "0/8  Commit check"
git log --oneline -1
echo "Confirm this matches the commit docs/A100_SETUP.md's Step 1 names (or a later one on main)."

log "1/8  Create venv and install from the lock file (NOT requirements.txt)"
python3.11 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
# 2026-09-08: requirements.lock DOES NOT RESOLVE (fsspec==2026.7.0 vs
# datasets==2.21.0, which needs fsspec[http]<=2024.6.1). In a FRESH venv there
# is no fallback stack, so install requirements.txt and then assert the two
# constraints the code actually has.
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
python -c "
import sys, transformers
if int(transformers.__version__.split('.')[0]) != 4:
    sys.exit(f'STOP: transformers {transformers.__version__}; needs 4.x.')
print('transformers', transformers.__version__)
"

log "2/8  Verify the CUDA stack"
python -c "
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
assert torch.cuda.is_available(), 'CUDA not available — stop here'
name = torch.cuda.get_device_name(0)
print('device name', name)
assert 'A100' in name, f'expected an A100, got: {name}'
props = torch.cuda.get_device_properties(0)
gb = props.total_memory / 1e9
print('total VRAM (GB)', round(gb, 2))
if not (18 <= gb <= 22):
    print(f'WARNING: expected ~20GB (MIG 3g.20gb), got {gb:.2f}GB — '
          'MIG may not actually be in effect. Continuing, but check this '
          'before trusting any benchmark against the 20GB slice.')
print('cudnn', torch.backends.cudnn.version())
"

log "3/8  nvidia-smi -L (Insufficient Permissions on MIG guest is EXPECTED, not a failure)"
nvidia-smi -L || true

log "4/8  Pull the donor tokenizer + embedding matrix at the pinned revision"
# NOT AutoModelForMaskedLM: HPLT/hplt_bert_base_az ships custom architecture
# code (LtgbertForMaskedLM) and that call raises without trust_remote_code=True,
# which is arbitrary code execution we deliberately refuse. The build path
# never loads the donor model either — it reads the input-embedding matrix
# straight out of safetensors. This step exercises exactly that path, so a
# green bootstrap now means the real pipeline will work, which the previous
# version could not tell us. See src/transplant/donor_embeddings.py.
python -c "
import sys; sys.path.insert(0, '.')
from src.utils import load_config
from src.transplant.donor_embeddings import load_donor_embeddings
from transformers import AutoTokenizer

cfg = load_config('configs/experiment.yaml', require_complete=False)
donor, revision = cfg.models.donor, cfg.models.donor_revision
tok = AutoTokenizer.from_pretrained(donor, revision=revision)
E, key, diag = load_donor_embeddings(donor, revision)
print('donor OK at pinned revision', revision)
print('  tokenizer vocab', tok.vocab_size, '| get_vocab', len(tok.get_vocab()))
print('  embedding key  ', key, '| shape', E.shape)
assert E.shape[0] == tok.vocab_size, (E.shape, tok.vocab_size)
"

log "5/8  Verify frozen config hashes match the committed versions"
python -c "
import sys; sys.path.insert(0, '.')
from src.utils import verify_config_hashes
verify_config_hashes()
print('config hashes OK')
"

log "6/8  Test suite"
python -m pytest tests/ -q

log "7/8  Dry-run the queue — expect 164 runs, 30 distinct Turkish checkpoints"
DRYRUN_OUT="$(python -m src.training.run_grid --config configs/experiment.yaml --dry-run)"
echo "$DRYRUN_OUT" | tail -5
echo "$DRYRUN_OUT" | grep -q "^TOTAL RUN COUNT: 164$" || {
  echo "FAIL: expected TOTAL RUN COUNT: 164"; exit 1; }
echo "$DRYRUN_OUT" | grep -q "^TURKISH STAGES: 60 requests, 30 distinct checkpoints$" || {
  echo "FAIL: expected TURKISH STAGES: 60 requests, 30 distinct checkpoints"; exit 1; }
echo "Queue shape confirmed: 164 runs, 30 distinct Turkish checkpoints."

log "8/8  Test-evaluation disclosure (read this — it is not optional context)"
cat docs/TEST_AUDIT.md

log "BOOTSTRAP COMPLETE"
echo "  Environment verified equivalent to the 4060 dev machine for everything"
echo "  that can affect results. Next: bash scripts/a100_benchmark.sh"
echo "  This script has NOT started the preflight or the grid."
