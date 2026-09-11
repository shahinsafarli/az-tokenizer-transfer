#!/usr/bin/env bash
# =====================================================================
#  A100 MICRO-BENCHMARK — two 30-step benchmarks. NOT a training run,
#  NOT the preflight, NOT the grid. Run scripts/a100_bootstrap.sh FIRST
#  and confirm it passed (test suite + --dry-run both green) before
#  running this.
#
#  Usage:
#      bash scripts/a100_benchmark.sh
# =====================================================================
set -euo pipefail

# shellcheck disable=SC1091
[ -f .venv/bin/activate ] && source .venv/bin/activate

export PYTHONUNBUFFERED=1
# Already set as a default inside src/training/finetune.py; exported here
# too so it's visible in process listings / logs, matching run_all.sh.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Re-verifying config hashes before benchmarking (cheap, catches drift early)..."
python -c "
import sys; sys.path.insert(0, '.')
from src.utils import verify_config_hashes
verify_config_hashes()
print('config hashes OK')
"

python scripts/a100_micro_benchmark.py

echo
echo "Benchmark complete. This script has NOT started the preflight or the grid."
echo "Report the numbers above before launching anything further."
