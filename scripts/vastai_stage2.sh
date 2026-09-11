#!/usr/bin/env bash
# =====================================================================
#  VAST.AI STAGE 2 — the full 164-run grid, then the analysis.
#
#  ~6-8 h on an H100 SXM at the stream count stage 1 measured.
#  Run ONLY after stage 1's decision gate said to proceed.
#
#  Usage (inside tmux!):
#      bash scripts/vastai_stage2.sh
#
#  Resumable: every finished run is a durable file and `run.skip_existing`
#  skips it, so re-running this script after any interruption continues
#  rather than restarting. Phase 1 is resumable the same way.
# =====================================================================
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
CONFIG="${CONFIG:-configs/experiment.yaml}"
# EVAL_SPLIT (2026-09-08) — must match scripts/vastai_run_full.sh, or the entry
# point the examiner uses decides whether the held-out test split is ever
# measured. The brief requires the one-command reproduction to reproduce the
# HEADLINE numbers, and those now include test figures, so "test" is the
# default here too. Training-time evaluation stays on validation either way:
# this adds one evaluation of the SELECTED checkpoint on az_test.
#   EVAL_SPLIT=val bash <this script>   -> validation only
EVAL_SPLIT="${EVAL_SPLIT:-test}"
if [ "$EVAL_SPLIT" = "test" ]; then
  TEST_FLAGS="--eval-split test --allow-test-eval"
else
  TEST_FLAGS=""
fi

STAGE_T0=$(date +%s)

log()  { printf '\n\033[1;35m==> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m!!! %s\033[0m\n' "$*"; }

if [ -z "${TMUX:-}" ]; then
  warn "You are NOT inside tmux. An SSH drop will kill this ~7-hour run."
  warn "Ctrl-C now, then:  tmux new -s grid   and re-run this script."
  sleep 10
fi

# Stage 1 wrote the measured stream count into the config; read it back so the
# two stages cannot silently disagree.
STREAMS=$(python -c "import sys;sys.path.insert(0,'.');from src.utils import load_config;print(int(load_config('$CONFIG').run.get('parallel_streams',1)))")
P1_STREAMS=$(python -c "import sys;sys.path.insert(0,'.');from src.utils import load_config;print(int(load_config('$CONFIG').run.get('phase1_streams',1)))")

log "0/4  Pre-flight"
python -c "import sys;sys.path.insert(0,'.');from src.utils import verify_config_hashes;verify_config_hashes();print('config hashes OK')"

# Stage 1 measures per-process VRAM and writes the stream count into the
# config. Without it the config still holds the SAFE defaults (1 stream, the
# brief's 10 GB team ceiling) — correct for the shared workstation, but on a
# rented 80 GB card that silently turns a ~7 h grid into a ~22 h one.
if [ ! -f results/hardware_benchmark.json ]; then
  warn "results/hardware_benchmark.json is missing — stage 1 was not run here."
  warn "Streams are at the safe default ($STREAMS). On a rented H100 that is"
  warn "roughly 3x slower than necessary, and Tranche A's decision gate was"
  warn "never reached. Strongly consider: bash scripts/vastai_stage1.sh"
  warn "Continuing in 20 s — Ctrl-C to stop."
  sleep 20
fi
if [ "$STREAMS" -eq 1 ]; then
  warn "Running SERIAL (parallel_streams=1). Expect ~20-33 h rather than ~7 h."
fi

echo "  Phase 1 workers : $P1_STREAMS"
echo "  Phase 2 streams : $STREAMS"
echo "  free disk       : $(df -BG --output=avail / | tail -1 | tr -d ' ')"
DONE_BEFORE=$(ls results/runs/*.json 2>/dev/null | wc -l | tr -d ' ')
echo "  runs already complete: $DONE_BEFORE / 164"

# ---------------------------------------------------------------- Phase 1
log "1/4  PHASE 1 — build and seal the 30 shared Turkish checkpoints"
echo "  Safe to parallelise: Phase 1's work list is deduplicated (30 specs ->"
echo "  30 distinct cache keys), so no two workers can target the same path."
echo "  The invariant is asserted at runtime, not assumed."
python -m src.training.run_grid --config "$CONFIG" --phase 1 --phase1-streams "$P1_STREAMS"

python - <<'PY'
import json, pathlib
m = json.loads(pathlib.Path("artifacts/tr_stage_cache/phase1_manifest.json").read_text(encoding="utf-8"))
failed = [e for e in m["entries"] if e.get("status") == "FAILED"]
built = sum(1 for e in m["entries"] if e.get("status") == "BUILT")
print(f"sealed={m['sealed']}  checkpoints={m['n_distinct_checkpoints']}  "
      f"built now={built}  reused={m['n_distinct_checkpoints']-built}  workers={m.get('phase1_streams')}")
assert m["sealed"] and not failed, f"Phase 1 incomplete: {failed[:1]}"
PY

# ---------------------------------------------------------------- Phase 2
log "2/4  PHASE 2 — the 164-run grid (cache READ-ONLY, eval_split=$EVAL_SPLIT)"
echo "  Write the paper while this runs. That is the point of the 7-hour"
echo "  option over the 22-hour one."
python -m src.training.run_grid --config "$CONFIG" --phase 2 --streams "$STREAMS" $TEST_FLAGS

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
    print("\n" + st["timing_caveat"])
PY

# ---------------------------------------------------------------- analysis
log "3/4  Analysis — every table and figure regenerated from results/"
bash run_all.sh --analysis-only

log "4/4  Headline"
python - <<'PY'
import json, pathlib
d = json.loads(pathlib.Path("results/decompose.json").read_text(encoding="utf-8"))
cross = d["cross_base_comparison"]
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
    print("\nThe M3 claim needs the effect PRESENT in xlm15 (no Azerbaijani in")
    print("pre-training) and ABSENT in xlmr (has it). Check the control is")
    print("healthy before reading the treatment — a broken control manufactures")
    print("a false confirmation.")
PY

python - <<PY
mins = ($(date +%s) - $STAGE_T0) / 60
print(f"\nStage 2 wall-clock: {mins/60:.1f} h")
PY

log "STAGE 2 COMPLETE"
cat <<'EOF'
  Tables : results/paper_tables.md
  Figures: figures/
  Headline: results/decompose.json -> cross_base_comparison

  NOW, BEFORE DESTROYING THE INSTANCE:
      zip -r /workspace/results_full.zip results figures
  then from your laptop:
      scp -P <PORT> root@<HOST>:/workspace/results_full.zip .
  Open the zip and confirm results/paper_tables.md is in it. THEN destroy.
  A destroyed instance takes its disk with it. There is no undo.
EOF
