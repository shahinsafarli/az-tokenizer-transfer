#!/usr/bin/env bash
# =====================================================================
#  VAST.AI STAGE 1 — prep, health check, benchmark, Tranche A. STOPS THERE.
#
#  ~2.5-3 h on an H100 SXM. Launches NOTHING beyond Tranche A: the point is
#  to reach the pre-registered decision gate (run plan §8) before committing
#  the remaining ~7 h. Read the output at the end before running stage 2.
#
#  Usage (inside tmux!):
#      bash scripts/vastai_stage1.sh
#
#  See docs/VASTAI_STEP_BY_STEP.md.
# =====================================================================
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CONFIG="${CONFIG:-configs/experiment.yaml}"
STAGE_T0=$(date +%s)

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m!!! %s\033[0m\n' "$*"; }

if [ -z "${TMUX:-}" ]; then
  warn "You are NOT inside tmux. An SSH drop will kill this run."
  warn "Ctrl-C now, then:  tmux new -s grid   and re-run this script."
  sleep 10
fi

# ---------------------------------------------------------------- 0. hardware
log "0/8  Hardware"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
echo "vCPU: $(nproc)   free disk: $(df -BG --output=avail / | tail -1 | tr -d ' ')"
python - <<'PY'
import subprocess, sys
name = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
                      capture_output=True, text=True).stdout.strip()
if any(b in name.upper() for b in ("B200", "B300")):
    sys.exit(f"\nSTOP: {name} is Blackwell (sm_100). The pinned torch 2.4.1+cu121 "
             "has no kernels for it. Destroy this instance and rent an H100 SXM.")
print(f"GPU OK: {name}")
PY

# ---------------------------------------------------------------- 1. install
log "1/8  Pinned environment (requirements.lock)"
# ------------------------------------------------------------------ DEPS
# DISABLED 2026-09-08. `requirements.lock` DOES NOT RESOLVE: it pins
# fsspec==2026.7.0 while datasets==2.21.0 requires fsspec[http]<=2024.6.1.
# pip exits with ResolutionImpossible and `set -euo pipefail` kills this
# script before the GPU is ever touched. BOTH lines are commented because the
# first ends in a backslash and a bash COMMENT ending in a backslash does NOT
# continue — commenting only the first would leave `--extra-index-url ...`
# running as a command, the same fatal error by another route.
# The 2026-09-06 grid ran on the CONTAINER IMAGE's stack, which is also the
# scientifically preferable choice: this lock pins torch 2.4.1+cu121, OLDER
# than the image's 2.11.0+cu128.
#   pip install -q -r requirements.lock \
#       --extra-index-url https://download.pytorch.org/whl/cu121
python - <<'PY'
import sys, transformers
if int(transformers.__version__.split(".")[0]) != 4:
    sys.exit(f"STOP: transformers {transformers.__version__}; this code needs "
             "4.x (5.x removed TrainingArguments(warmup_ratio=...)).")
print("transformers", transformers.__version__)
import torch
p = torch.cuda.get_device_properties(0)
print("torch", torch.__version__, "| cuda", torch.version.cuda)
print(f"device {p.name}  {p.total_memory/1e9:.0f} GB  sm_{p.major}{p.minor}")
assert torch.cuda.is_available(), "CUDA not available"
# Prove this wheel can actually emit kernels for this card.
torch.zeros(8, 8, device="cuda") @ torch.zeros(8, 8, device="cuda")
torch.cuda.synchronize()
print("CUDA kernel launch OK")
PY

# ---------------------------------------------------------------- 2. gates
log "2/8  Frozen-config hashes + test suite"
python -c "import sys;sys.path.insert(0,'.');from src.utils import verify_config_hashes;verify_config_hashes();print('config hashes OK')"
python -m pytest tests/ -q -m "not slow" --no-header
python -m src.training.run_grid --config "$CONFIG" --dry-run | head -3

# ---------------------------------------------------------------- 3. prep
log "3/8  Data, transplant artifacts, C1 controls  (~2 h, mostly CPU)"
CONFIG="$CONFIG" DEVICE=cuda bash run_all.sh --prep-only

# ---------------------------------------------------------------- 4. health
log "4/8  Transplant health — read this, it gates the whole M3 arm"
python - <<'PY'
import json, pathlib
p = pathlib.Path("results/cross_base_transplant_quality.json")
if not p.exists():
    print("cross_base_transplant_quality.json missing"); raise SystemExit
q = json.loads(p.read_text(encoding="utf-8"))
dead = []
for base, r in q["per_base"].items():
    t = r.get("top1_transplanted_canonical") or {}
    b = r.get("top1_base") or {}
    print(f"{base:6s} dBPC={r.get('C1c_delta_bpc'):>9}  norm_ratio={r.get('norm_ratio')}  "
          f"healthy={r.get('norm_ratio_healthy')}")
    print(f"       top-1  base {b.get('correct')}/{b.get('masked')}"
          f"  ->  transplanted {t.get('correct')}/{t.get('masked')}")
    if (t.get("accuracy") or 0) < 0.01:
        dead.append(base)
if len(dead) == 2:
    print("\n*** BOTH transplanted models score ~0 top-1.")
    print("*** The swap destroyed masked-LM ability. This is a reportable")
    print("*** finding, and it weakens the xlmr zero-effect control: a null")
    print("*** there cannot separate 'no deficit to fix' from 'we broke it'.")
elif dead:
    print(f"\n*** {dead} destroyed, the other survived. An effect that appears")
    print("*** only in the broken one is an artifact of the surgery.")
else:
    print("\nBoth transplanted models still predict. Surgery was survivable.")
PY

# ---------------------------------------------------------------- 5. benchmark
log "5/8  Micro-benchmark — real s/step and real per-process VRAM"
bash scripts/a100_benchmark.sh

# ---------------------------------------------------------------- 6. streams
log "6/8  Set stream count from the MEASURED numbers"
python - <<'PY'
import hashlib, json, os, re, pathlib
b = json.loads(pathlib.Path("results/hardware_benchmark.json").read_text(encoding="utf-8"))
per_proc = b["per_process_peak_gb"]

import torch
card = torch.cuda.get_device_properties(0).total_memory / 1e9
ceiling = round(card * 0.85, 1)          # 15% headroom; we rented the whole card
by_vram = max(1, int(ceiling // per_proc))
by_cpu = max(1, os.cpu_count() // 2)     # >=2 vCPU per stream: the host issues the launches
streams = max(1, min(8, by_vram, by_cpu))

print(f"per-process peak {per_proc:.2f} GB | card {card:.0f} GB -> ceiling {ceiling} GB")
print(f"streams by VRAM {by_vram} | by vCPU {by_cpu} ({os.cpu_count()} cores) -> USING {streams}")
if by_cpu < by_vram:
    print("  (CPU-bound, as expected: this workload is kernel-launch-bound)")

cfgp = pathlib.Path("configs/experiment.yaml")
t = cfgp.read_text(encoding="utf-8")
t = re.sub(r"(  vram_per_stream_gb: )[\d.]+", rf"\g<1>{per_proc:.2f}", t)
t = re.sub(r"(  vram_ceiling_gb: )[\d.]+",    rf"\g<1>{ceiling}", t)
t = re.sub(r"(  parallel_streams: )\d+",      rf"\g<1>{streams}", t)
t = re.sub(r"(  phase1_streams: )\d+",        rf"\g<1>{min(4, streams)}", t)
cfgp.write_text(t, encoding="utf-8")

lockp = pathlib.Path("configs/CONFIG_HASHES.lock")
lock = json.loads(lockp.read_text(encoding="utf-8"))
lock["configs/experiment.yaml"] = hashlib.sha256(
    cfgp.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
lockp.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("config updated and re-locked (run.* keys are execution machinery, not science)")
PY
python -c "import sys;sys.path.insert(0,'.');from src.utils import verify_config_hashes;verify_config_hashes();print('re-locked OK')"

# ---------------------------------------------------------------- 7. Tranche A
log "7/8  TRANCHE A — 10 runs, single stream (a clean timing baseline)"
# DELIBERATELY validation-only: Tranche A is the pipeline smoke test, not a
# reportable cell. The held-out split is spent once, by the full grid.
python -m src.training.run_grid --config "$CONFIG" --tranche A --phase 2 --streams 1

# ---------------------------------------------------------------- 8. decision
log "8/8  THE DECISION GATE (pre-registered, run plan §8)"
python -m src.analysis.aggregate --config "$CONFIG"
python -m src.analysis.decompose --config "$CONFIG"
python - <<'PY'
import json, pathlib
cells = {(c["base"], c["condition"]): c for c in
         json.loads(pathlib.Path("results/decompose.json").read_text(encoding="utf-8"))["cells"]}
a, b = cells[("xlm15", "baza")], cells[("xlm15", "tokenizator")]
print(f"baza        escape {a['escape_rate_count']}  conditional F1 {a['conditional_macro_f1']}")
print(f"tokenizator escape {b['escape_rate_count']}  conditional F1 {b['conditional_macro_f1']}")
nb, nt = a["n_escaped"], b["n_escaped"]
print("=" * 66)
if nb == 0 and nt == 0:
    print("NEITHER ESCAPES -> STOP. Do NOT run stage 2 and do NOT tune the")
    print("optimizer. n=2000 is below the detectability threshold; re-plan at")
    print("n=10000. Destroy the instance after downloading results/.")
elif nt >= 3 and nb < 3:
    print("tokenizator ESCAPES, baza DOES NOT -> strong M3 evidence, headline")
    print("result. RUN STAGE 2.")
elif nb >= 3 and nt >= 3:
    print("BOTH ESCAPE -> pipeline works and the task is learnable. RUN STAGE 2.")
else:
    print("ERRATIC -> seed variance may dominate at this size. Report escape")
    print("rate as the primary outcome. Discuss before spending the next 7 h.")
PY

python - <<PY
mins = ($(date +%s) - $STAGE_T0) / 60
print(f"\nStage 1 wall-clock: {mins:.0f} min")
PY

log "STAGE 1 COMPLETE — nothing beyond Tranche A was launched."
echo "  Read the decision above. If it says RUN STAGE 2:"
echo "      bash scripts/vastai_stage2.sh"
