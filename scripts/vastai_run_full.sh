#!/usr/bin/env bash
# =====================================================================
#  FULL GRID, ONE COMMAND. No pilot, no Tranche-A gate, no benchmark.
#
#  For when the exploratory work is already done and you just want the
#  164 runs. Runs unattended start to finish.
#
#      bash scripts/vastai_run_full.sh
#
#  Env overrides:
#      STREAMS=6      force the Phase-2 stream count (default: derived)
#      P1_STREAMS=4   force Phase-1 workers          (default: min(4,STREAMS))
#      RUN_TESTS=1    run the test suite first       (default: skipped)
#      SKIP_PREP=1    assert artifacts exist, refuse to build them
#
#  WHAT THIS STILL DOES, AND WHY IT IS NOT A TEST
#  ----------------------------------------------
#  It builds `artifacts/data/*.jsonl` and the six transplant artifacts IF
#  THEY ARE ABSENT. Those are the grid's INPUTS — Phase 1 loads them on its
#  first step — and a freshly rented instance has an empty disk. They are
#  not a check of anything.
#
#  IF YOU ALREADY BUILT THEM, UPLOAD THEM AND THIS SKIPS THE ~2 HOURS:
#      # on your machine, from a box that has them:
#      tar czf artifacts.tgz artifacts/data artifacts/transplanted__*
#      scp -P <PORT> artifacts.tgz root@<HOST>:/workspace/
#      # on the instance, in the project dir:
#      tar xzf /workspace/artifacts.tgz
#  The transplant is deterministic and pinned to the donor revision, so a
#  previously built `omp_k64_rescaled` is byte-for-byte the artifact this
#  grid wants. Reusing it is correct, not a shortcut.
# =====================================================================
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CONFIG="${CONFIG:-configs/experiment.yaml}"
T0=$(date +%s)

log()  { printf '\n\033[1;35m==> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m!!! %s\033[0m\n' "$*"; }

if [ -z "${TMUX:-}" ]; then
  warn "Not inside tmux. An SSH drop kills this run. Ctrl-C, then: tmux new -s grid"
  sleep 8
fi

# ---------------------------------------------------------------- 1. hardware
log "1/6  Hardware + environment"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "vCPU $(nproc) | free disk $(df -BG --output=avail / | tail -1 | tr -d ' ')"

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
import sys, torch, transformers
# transformers 5.x REMOVED TrainingArguments(warmup_ratio=...), which
# src.training.finetune._train_stage passes on every call. On a 5.x image every
# one of the 164 runs would die with TypeError at its first TrainingArguments.
# Fail here, in second one, rather than 164 times.
_major = int(transformers.__version__.split(".")[0])
if _major != 4:
    sys.exit(f"STOP: transformers {transformers.__version__}. This code needs "
             "4.x — 5.x removed TrainingArguments(warmup_ratio=...). Run "
             "`pip install 'transformers>=4.44,<5'` and start again.")
p = torch.cuda.get_device_properties(0)
if any(b in p.name.upper() for b in ("B200", "B300")):
    sys.exit(f"STOP: {p.name} is Blackwell (sm_100). torch 2.4.1+cu121 has no "
             "kernels for it. Destroy this instance and rent an H100 SXM.")
torch.zeros(8, 8, device="cuda") @ torch.zeros(8, 8, device="cuda")
torch.cuda.synchronize()
print(f"{p.name}  {p.total_memory/1e9:.0f} GB  sm_{p.major}{p.minor}  kernels OK")
print(f"torch {torch.__version__} | transformers {transformers.__version__}")
PY

# Cheap and it has caught real problems: a mismatched config here means every
# number downstream is silently from a different experiment. ~1 second.
python -c "import sys;sys.path.insert(0,'.');from src.utils import verify_config_hashes;verify_config_hashes();print('config hashes OK')"

if [ "${RUN_TESTS:-0}" = "1" ]; then
  python -m pytest tests/ -q -m "not slow" --no-header
fi

# ---------------------------------------------------------------- 2. streams
log "2/6  Stream count (derived from this box, no benchmark run)"
python - <<'PY'
import hashlib, json, os, re, pathlib, torch

# 8.6 GB is the MEASURED per-process peak at batch 32 (HANDOFF §12). We are
# not re-measuring it; we are applying it.
PER_PROC = 8.6
card = torch.cuda.get_device_properties(0).total_memory / 1e9
ceiling = round(card * 0.85, 1)                 # 15% headroom; whole card is ours
by_vram = max(1, int(ceiling // PER_PROC))
by_cpu = max(1, os.cpu_count() // 2)            # the host issues every kernel launch
streams = int(os.environ.get("STREAMS") or max(1, min(8, by_vram, by_cpu)))
p1 = int(os.environ.get("P1_STREAMS") or min(4, streams))

print(f"card {card:.0f} GB -> ceiling {ceiling} GB | per-process {PER_PROC} GB (measured)")
print(f"by VRAM {by_vram} | by vCPU {by_cpu} ({os.cpu_count()} cores) -> STREAMS={streams} P1={p1}")
if by_cpu < by_vram:
    print("  CPU-bound, as expected for a kernel-launch-bound workload.")

cfgp = pathlib.Path("configs/experiment.yaml")
t = cfgp.read_text(encoding="utf-8")
t = re.sub(r"(  vram_per_stream_gb: )[\d.]+", rf"\g<1>{PER_PROC}", t)
t = re.sub(r"(  vram_ceiling_gb: )[\d.]+",    rf"\g<1>{ceiling}", t)
t = re.sub(r"(  parallel_streams: )\d+",      rf"\g<1>{streams}", t)
t = re.sub(r"(  phase1_streams: )\d+",        rf"\g<1>{p1}", t)
cfgp.write_text(t, encoding="utf-8")

lockp = pathlib.Path("configs/CONFIG_HASHES.lock")
lock = json.loads(lockp.read_text(encoding="utf-8"))
lock["configs/experiment.yaml"] = hashlib.sha256(
    cfgp.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
lockp.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("config updated + re-locked (run.* keys are execution machinery, not science)")
PY

# ---------------------------------------------------------------- 3. inputs
log "3/6  Grid inputs — reuse if present, build only what is missing"
NEED_PREP=0
python - <<'PY' || NEED_PREP=1
import sys, pathlib
sys.path.insert(0, ".")
from src.utils import load_config, resolve_bases, transplant_dir
cfg = load_config("configs/experiment.yaml")
missing = [str(p) for p in
           [pathlib.Path(cfg.experiment.artifacts_dir) / "data" / f"{n}.jsonl"
            for n in ("az_train", "az_val", "az_test", "tr_train", "tr_train_scrambled")]
           if not p.exists()]
for b in resolve_bases(cfg):
    for tag in ("omp_k64_rescaled", "mean_k64_rescaled", "random_coef_k64_rescaled"):
        d = transplant_dir(cfg, b.short, tag)
        if not (d / "config.json").exists():
            missing.append(str(d))
if missing:
    print(f"{len(missing)} input(s) missing, first: {missing[0]}")
    sys.exit(1)
print("All grid inputs already present — skipping the ~2 h build.")
PY

if [ "$NEED_PREP" = "1" ]; then
  if [ "${SKIP_PREP:-0}" = "1" ]; then
    warn "SKIP_PREP=1 but inputs are missing. Upload artifacts.tgz (see header)."
    exit 1
  fi
  log "     Building inputs (data + 6 transplant artifacts). ~2 h, mostly CPU."
  CONFIG="$CONFIG" DEVICE=cuda bash run_all.sh --prep-only
else
  log "     Inputs reused."
fi

# ---------------------------------------------------------------- 4. phase 1
STREAMS_CFG=$(python -c "import sys;sys.path.insert(0,'.');from src.utils import load_config;print(int(load_config('$CONFIG').run.get('parallel_streams',1)))")
P1_CFG=$(python -c "import sys;sys.path.insert(0,'.');from src.utils import load_config;print(int(load_config('$CONFIG').run.get('phase1_streams',1)))")

log "4/6  PHASE 1 — 30 shared Turkish checkpoints ($P1_CFG workers)"
python -m src.training.run_grid --config "$CONFIG" --phase 1 --phase1-streams "$P1_CFG"
python - <<'PY'
import json, pathlib
m = json.loads(pathlib.Path("artifacts/tr_stage_cache/phase1_manifest.json").read_text(encoding="utf-8"))
failed = [e for e in m["entries"] if e.get("status") == "FAILED"]
built = sum(1 for e in m["entries"] if e.get("status") == "BUILT")
print(f"sealed={m['sealed']} checkpoints={m['n_distinct_checkpoints']} "
      f"built={built} reused={m['n_distinct_checkpoints']-built}")
assert m["sealed"] and not failed, f"Phase 1 incomplete: {failed[:1]}"
PY

# ---------------------------------------------------------------- 5. phase 2
# EVAL_SPLIT (added 2026-09-07): "test" spends the held-out split ONCE per run.
# Training-time evaluation stays on validation either way (see configs/FROZEN.md,
# "2026-09-07"), so this adds a terminal test evaluation, not a different model.
# Export EVAL_SPLIT=val to run validation-only.
EVAL_SPLIT="${EVAL_SPLIT:-test}"
if [ "$EVAL_SPLIT" = "test" ]; then
  TEST_FLAGS="--eval-split test --allow-test-eval"
else
  TEST_FLAGS=""
fi

# ---------------------------------------------------------------- memory trace
# Added 2026-09-08. The compute-compliance claim was previously "8.6 GB per
# process", measured once, against a budget expressed PER TEAM. With N
# concurrent streams those are different quantities, and the aggregate was
# never measured — so the compliance statement could not be substantiated
# either way. This samples DEVICE-TOTAL used memory for the whole of Phase 2
# and writes a trace plus its peak, which is the number the paper needs. It
# also records the count so wall-clock, process-hours and device-hours can be
# reported separately instead of being conflated.
MEMTRACE="results/gpu_memory_trace.csv"
mkdir -p results
echo "utc,gpu_index,mem_used_mib,mem_total_mib,util_pct" > "$MEMTRACE"
(
  while :; do
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null \
      | sed "s|^|$(date -u +%Y-%m-%dT%H:%M:%SZ),|" >> "$MEMTRACE" || true
    sleep 10
  done
) & MEMPID=$!
trap 'kill $MEMPID 2>/dev/null || true' EXIT

log "5/6  PHASE 2 — 164 runs ($STREAMS_CFG streams), cache READ-ONLY, eval_split=$EVAL_SPLIT"
PHASE2_T0=$(date -u +%s)
python -m src.training.run_grid --config "$CONFIG" --phase 2 --streams "$STREAMS_CFG" $TEST_FLAGS
PHASE2_T1=$(date -u +%s)

kill $MEMPID 2>/dev/null || true
python - <<'PY' "$MEMTRACE" "$PHASE2_T0" "$PHASE2_T1" "$STREAMS_CFG"
import csv, json, sys, pathlib
trace, t0, t1, streams = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rows = list(csv.DictReader(open(trace, encoding="utf-8")))
peak_by_gpu, tot = {}, {}
for r in rows:
    try:
        g = int(r["gpu_index"]); used = int(r["mem_used_mib"])
    except (ValueError, KeyError):
        continue
    peak_by_gpu[g] = max(peak_by_gpu.get(g, 0), used)
    tot[g] = int(r["mem_total_mib"])
wall_h = (t1 - t0) / 3600.0
out = {
    "source": trace, "n_samples": len(rows), "sample_interval_sec": 10,
    "phase2_wall_clock_hours": round(wall_h, 4),
    "streams_concurrent": streams,
    "device_peak_mem_used_mib": peak_by_gpu,
    "device_total_mem_mib": tot,
    "device_peak_mem_used_gb": {k: round(v / 1024, 2) for k, v in peak_by_gpu.items()},
    "note": ("DEVICE-TOTAL peak across all concurrent streams, sampled every "
             "10 s for the whole of Phase 2. This is the per-team quantity. "
             "Per-process peak is a different number and must not be "
             "substituted for it. Wall-clock is reported separately from the "
             "sum of per-run runtimes (process-hours), which is not device "
             "occupancy time."),
}
pathlib.Path("results/gpu_memory_summary.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8")
print("MEMORY device peak (GB):", out["device_peak_mem_used_gb"],
      "| phase-2 wall:", out["phase2_wall_clock_hours"], "h")
PY
python - <<'PY'
import json, pathlib
p = pathlib.Path("results/launcher_state_phase2.json")
if p.exists():
    st = json.loads(p.read_text(encoding="utf-8"))
    print(f"completed {len(st['completed'])} | skipped {len(st['skipped_existing'])} "
          f"| failed {len(st['failures'])} | {st['elapsed_sec']/3600:.2f} h "
          f"| streams {st['streams_active']}")
    for f in st["failures"]:
        print("  FAILED:", f)
PY

# ---------------------------------------------------------------- 6. analysis
log "6/6  Analysis"
bash run_all.sh --analysis-only

python - <<'PY'
import json, pathlib
cross = json.loads(pathlib.Path("results/decompose.json").read_text(encoding="utf-8"))["cross_base_comparison"]
if cross.get("status") != "MEASURED":
    print("cross-base comparison: NOT MEASURED —", cross.get("missing"))
else:
    for base, r in cross["bases"].items():
        print(f"{base}: escape {r['baseline_escape_rate']} -> {r['omp_escape_rate']} "
              f"(diff {r['escape_rate_difference']:+.3f}); conditional-F1 diff "
              f"{r['conditional_macro_f1_difference']}")
    c = cross["cross_base"]
    print(f"\nescape-rate effect difference    : {c['escape_rate_effect_difference']}")
    print(f"conditional-F1 effect difference : {c['conditional_macro_f1_effect_difference']}")
PY

python - <<PY
h = ($(date +%s) - $T0) / 3600
print(f"\nTotal wall-clock: {h:.2f} h")
PY

log "DONE"
cat <<'EOF'
  Tables : results/paper_tables.md
  Figures: figures/
  Headline: results/decompose.json -> cross_base_comparison

  BEFORE DESTROYING THE INSTANCE:
      zip -r /workspace/results_full.zip results figures
  then from your laptop:
      scp -P <PORT> root@<HOST>:/workspace/results_full.zip .
  Open it, confirm results/paper_tables.md is inside, THEN destroy.
EOF
