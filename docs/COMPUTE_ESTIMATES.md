# Compute estimates — derived, not asserted

Every number below is either **measured** (cited to a file in `results/` or to
`HANDOFF.md`), **derived** by arithmetic from a measured number, or
**projected** with its assumption stated. Nothing here is a guess presented as
a measurement. Projections are replaced by `results/hardware_benchmark.json`
once `scripts/a100_benchmark.sh` has run on the target machine.

---

## 1. The workload, in optimizer steps

| Component | Count | Source |
|---|---:|---|
| Azerbaijani runs | 114 | `run_grid --dry-run`, verified |
| AZ optimizer steps | 114 x 2,000 = **228,000** | `training.max_steps` |
| Distinct Turkish checkpoints | 30 | `run_grid --dry-run`, verified |
| TR optimizer steps | 30 x 1,126 = **33,780** | 18,000 train rows / eff. batch 32 = 563/epoch x 2 epochs |
| Validation forward batches | 114 x 10 x 88 = **100,320** | 2,791 val rows / `eval_batch_size` 32, ~10 passes/run |
| Validation in step-equivalents | ~**33,000** | a forward-only batch is ~1/3 of a fwd+bwd step |
| **Total step-equivalents** | **~295,000** | |

The run plan quotes ~280,000. The difference is the validation term: it was
computed at `eval_batch_size: 64`, which this pass lowered to 32 (see §3).

## 2. Peak VRAM per process — the number that decides everything

This is where the run plan is **too optimistic**, and it matters because
exceeding the brief's allocation is an automatic deduction.

Optimizer state does **not** shrink with batch size. For a model of `P`
parameters, AdamW with fp16 autocast holds fp32 weights, fp32 gradients and two
fp32 moment buffers — `16P` bytes, fixed, regardless of `batch_size`:

| Configuration | Params | Fixed state (16P) | Source for P |
|---|---:|---:|---|
| xlm15 + original tokenizer (vocab 95,000) | 248,978,434 | **3.98 GB** | `HANDOFF.md` §13 |
| xlmr + original tokenizer (vocab 250,002) | 278,045,186 | **4.45 GB** | `HANDOFF.md` §13 |
| xlm15 + OMP transplant (vocab 32,770) | 185,254,914 | 2.96 GB | `HANDOFF.md` §13 |
| xlmr + OMP transplant (vocab 32,770) | 111,211,010 | 1.78 GB | `HANDOFF.md` §13 |

Adding the CUDA context (~0.4–0.6 GB), activations, and allocator overhead:

| Setting | Activations | **Projected peak** | Against a 10 GB team cap |
|---|---:|---:|---|
| **batch 32, accum 1 (CURRENT, pre-registered)** | ~2.5–3.5 GB | **~8.6 GB** *(measured, HANDOFF §12)* | **86% — fits, at one stream** |
| batch 8, accum 4 (authorised fallback) | ~0.5–0.8 GB | ~5.2–5.6 GB *(projected)* | 52–56% |

**Reverted to batch 32 / accum 1 on 2026-09-05** (`configs/FROZEN.md`).
Accumulation was adopted to make 2–3 streams fit the team cap — but the
~4.45 GB batch-size-independent AdamW floor means two streams do not fit
10 GB in *either* configuration, so the premise failed. At one stream 8.6 GB
is already under the cap, and accumulation costs ~2.1x wall-clock (§3).

**Consequence the run plan misses.** §6.1 states per-process memory "drops to
roughly 3–4 GB" and §6.3 plans "2 to 3 concurrent runs". Both understate the
fixed term. At a projected ~5.5 GB per process, against the brief's allocation:

| Streams | Projected peak | 10 GB cap | 12 GB cap |
|---:|---:|---|---|
| 1 | 5.5 GB | fits | fits |
| 2 | 11.0 GB | **exceeds** | fits, no margin |
| 3 | 16.5 GB | **exceeds** | **exceeds** |

**Three streams are not reachable inside the brief's allocation for this
model.** `run.vram_ceiling_gb` (10.0) plus `resolve_stream_count()` already
enforce this — a configured `parallel_streams: 3` is silently capped to what
actually fits and the reduction is logged. Do not raise the ceiling to defeat
the guard; the allocation, not the card, is the constraint.

`scripts/a100_benchmark.sh` measures the real per-process peak and prints the
`run.vram_per_stream_gb` / `run.parallel_streams` values to write back. **Do
that before launching the grid** — the table above is arithmetic, not a
measurement.

## 3. Throughput and wall-clock

Gradient accumulation does not change total FLOPs, but this workload is
**kernel-launch-bound, not compute-bound**: Azerbaijani sequences average 29.6
tokens (p95 = 79), so a batch is roughly 2,000 tokens and the GPU idles between
tiny kernels (run plan §6.3). Splitting one batch of 32 into 4 micro-batches of
8 therefore multiplies the launch overhead that already dominates. The run
plan's §6.1 claim that "run length is unchanged" is true **in optimizer steps**
but not in wall-clock.

Projected A100 per-optimizer-step, from the plan's 0.25 s central estimate at
batch 32:

| Configuration | s/optimizer-step | Serial wall-clock (295k step-equivalents) |
|---|---:|---:|
| **batch 32, accum 1 (CURRENT)** | **0.25 (central)** | **~20 h** |
| batch 32, accum 1 | 0.40 (pessimistic) | ~33 h |
| batch 8, accum 4 | ~0.45–0.60 (projected) | ~37–49 h |

**Accumulation is a ~2.1x wall-clock penalty here.** It buys memory
margin, not speed, and at one stream there is no margin to buy.

### The trade-off, resolved 2026-09-05

For this model and this budget, **the simplest compliant configuration is also
the fastest**: serial, batch 32, one stream, ~8.6 GB, ~20–33 h — inside both
the ~2-day window and the 10–12 GB allocation, with no parallel machinery at
all. Gradient accumulation buys **margin**, not speed:

- **For 8x4 + 2 streams:** 8.6 GB is 72–86% of the allowance with *zero*
  headroom. The rare 256-token batch can spike it, and the brief penalises
  exceeding the allocation on a machine shared with other teams. 5.5 GB per
  process restores real margin, and two streams recover the throughput the
  accumulation costs.
- **Against it:** more moving parts, and results are not bit-identical to any
  non-accumulated run (dropout-mask RNG order — recorded in `configs/FROZEN.md`).

**Resolution: batch 32 / accum 1**, restored to the pre-registered value.
It is compliant on the shared workstation (8.6 GB < 10 GB) at one stream,
and it is also the fastest configuration on a rented GPU. **One config
serves both venues**; only `run.parallel_streams` differs, and that is
execution machinery recorded per run in `streams_active`, not a training
parameter. Whichever venue is used, the setting must be identical across
every reported run; mixing them quarantines the results.

## 4. Per-tranche estimates

Assuming the current 8x4 configuration. `s/step` is the projected A100 range;
T4 is projected at ~2.2x the A100 step time for this launch-bound workload.

At **batch 32 / accum 1** (the current config). A100 1-stream is the shared
workstation; H100 4-stream is a rented card where the team cap does not bind.

| Tranche | Runs | Steps (incl. eval) | A100, 1 stream | H100, 4 streams | Colab T4 |
|---|---:|---:|---:|---:|---:|
| **A** — xlm15, n=2000, baza + tokenizator, 5 seeds | 10 | ~22,900 | **1.6–2.5 h** | **0.4–0.7 h** | 3.5–5.6 h |
| **B** — xlmr, same | 10 | ~22,900 | 1.6–2.5 h | 0.4–0.7 h | 3.5–5.6 h |
| **C** — Turkish cache (Phase 1) | — | 33,780 | 2.3–3.8 h | 0.7–1.1 h at `--phase1-streams 4` | 5.2–8.3 h |
| **D** — Turkish-bearing conditions | 30 | ~68,700 | 4.8–7.6 h | 1.3–2.1 h | 10.5–16.8 h |
| **E** — data curve, remainder | 64 | ~146,700 | 10.2–16.3 h | 2.8–4.5 h | 22–36 h |
| **Full grid** | 114 | ~295,000 | **20–33 h** | **5.7–9.1 h** | 45–72 h |
| **PILOT** (800 steps, 3 seeds, 2x2) | 12 | ~10,700 | 0.7–1.2 h | — | **1.6–2.6 h** (+1–2 h build) |

**Read the A column first.** Tranche A is ~3 h on an A100 and answers the three
things that block every other decision — real s/step, whether the conditions
escape the constant-prediction basin at all, and per-process VRAM. Everything
in this table after row A is a projection that Tranche A replaces with a
measurement.

### The free-T4 pilot — the cheapest decision in the project

`configs/pilot_t4.yaml` + `notebooks/colab_t4_pilot.ipynb` run a deliberately
non-reportable **GO/NO-GO probe**: the full 2x2 cross-base tokenizer contrast
(2 bases x {`baza`, `tokenizator`} x 3 seeds = **12 runs**) at a reduced
**800-step** budget, writing to an isolated `results_pilot/`.

| Component | Cost on a free T4 |
|---|---:|
| Data download + split | ~10 min |
| OMP transplant build, both bases (CPU-bound) | **60–120 min** |
| Transplant health check (top-1 / BPC) | ~10 min |
| 12 runs x 800 steps + validation | **3–4 h** |
| Analysis | ~2 min |
| **Total** | **~5–6 h**, $0 |

**Why 800 steps.** Basin escape was measured between steps 250 and 630
(`configs/FROZEN.md`), so 800 brackets the decisive window on both sides at
40% of the frozen budget — still ~12.8 epochs at n=2000. **Why 3 seeds.**
Three seeds admit only escape rates of 0, ⅓, ⅔, 1 — too coarse to size an
effect, sufficient to see whether one exists.

**Why this particular 12.** The run plan calls Tranche A + B together "a
complete, reportable result before any Turkish run exists". The pilot is that
same 2x2 at a pilot budget: it is the smallest set of cells that can exhibit
the M3 signature (transplant helps the base without Azerbaijani pre-training,
and does *not* help the one with it), and the only one that can also catch the
failure mode where the transplant helps **both** — which would mean the gain
comes from the surgery rather than from fixing a tokenization deficit, and
would invalidate the cross-base identification.

The transplant artifacts it builds are the **canonical** ones
(`omp_k64_rescaled`, shared `artifacts_dir`), so the pilot's most expensive
hour is reused by the full grid rather than thrown away.

**It is quarantined by construction:** `results_pilot/` is a separate
directory, so `load_runs()` cannot pool pilot records with grid records, and
`configs/pilot_t4.yaml` is itself hash-locked — a silently edited pilot config
fails as loudly as a silently edited frozen one.

### Colab free T4 is not a grid machine

The full grid is **81–108 h** on a T4 — far beyond any free session. Free Colab
also disconnects on idle (~90 min) and caps sessions well below that. The T4
notebook is therefore scoped to:

1. environment + artifact verification, the test suite, and the queue dry run;
2. the transplant build and its C1/C1c/top-1 controls (CPU-feasible);
3. **Tranche A, resumable** — Drive-backed `results/` and cache, so
   `run.skip_existing` picks up exactly where a dropped session left off.

Tranche A itself is 6.3–8.4 h on a T4, i.e. **2–3 free sessions**. Use
`--seeds` to take it a few seeds at a time; each completed run is durable.

## 5. Cost

| Route | Rate (approximate, verify before relying on it) | Full grid | Tranche A |
|---|---|---:|---:|
| Shared A100 workstation (the course machine) | booked window, no cash cost | 22–29 h of the ~48 h window | ~3 h |
| Colab free T4 | $0 | not feasible | 2–3 sessions |
| Colab Pro (~$10/mo, ~100 compute units) | T4 ~1.8 units/h; A100 ~12 units/h | exceeds one month's units on A100 | ~36 units on A100 |
| Colab Pro+ (~$50/mo, ~500 units) | as above | ~260–350 units on A100 | ~36 units |
| Cloud A100 40 GB rental | ~$1.20–2.00/h | **~$26–58** | ~$4–8 |
| **vast.ai H100 SXM** | ~$1.33–2.16/h | **~$14–22** | **~$4–6** |

See **section 8** for the full rented-GPU comparison — including two cards on that list that **cannot run this stack at all**.

Compute-unit rates and rental prices change; treat the two right-hand columns
as orders of magnitude. The course's own A100 window is the intended route and
costs nothing — the rental line exists only to price a fallback.

## 6. Disk

| Item | Size | Source |
|---|---:|---|
| Turkish checkpoint cache (30 checkpoints, weights only) | **~27.0 GB** | `HANDOFF.md` §13, per-combo measured |
| Transplant artifacts (6: omp/mean/random_coef x 2 bases, rescaled) | ~4.5 GB | derived from the §13 per-model sizes |
| Result JSON + per-run logs | ~30 MB | `disk_budget_bytes_per_run_overhead` x 114 |
| Datasets (AZ 27,914 + TR 20,000 rows, jsonl) | ~25 MB | `results/splits.json` |
| **Total** | **~32 GB** | |

`execute_queue()` refuses to start when free space is below the projection plus
a 15% margin, and re-checks before every item so the grid stops cleanly between
runs rather than dying mid-write. On the shared workstation put
`run.tr_stage_cache_dir` on the NVMe scratch, not the root SSD — the brief asks
for this explicitly and 27 GB on a 239 GB root is not a safe default.

## 7. What must be measured before the window opens

1. `bash scripts/a100_bootstrap.sh` — environment, pinned donor, config hashes,
   test suite, queue shape.
2. `bash scripts/a100_benchmark.sh` — real s/step for both stages and real
   per-process VRAM; writes `results/hardware_benchmark.json` and prints the
   `parallel_streams` / `vram_per_stream_gb` values to write back.
3. **Tranche A** (`bash run_all.sh --tranche-a`, ~3 h) — then apply the §8
   decision rule in the run plan and recompute this whole table from measured
   numbers before committing the rest of the window.

---

## 8. Rented GPUs (vast.ai) — and why the expensive ones are a trap

### 8.1 Two cards are disqualified before any timing argument

`requirements.lock` pins **`torch==2.4.1+cu121`**. CUDA 12.1 predates Blackwell
and ships no `sm_100` kernels:

| GPU | Arch | Compute capability | Runs the pinned stack? |
|---|---|---|---|
| A100 | Ampere | `sm_80` | yes |
| H100 SXM / PCIe / NVL | Hopper | `sm_90` | yes |
| H200, H200 NVL | Hopper (GH100) | `sm_90` | yes |
| **B200** | Blackwell | `sm_100` | **no** — needs CUDA 12.8+ / torch 2.7+ |
| **B300** | Blackwell Ultra | `sm_100` | **no** — needs CUDA 12.8+ / torch 2.7+ |

On B200/B300 the pinned wheel fails with *"no kernel image is available for
execution on the device."* Upgrading torch to reach them abandons the lock file
— and the lock exists specifically so that results produced on different
machines stay comparable (`requirements.lock` header; `docs/A100_SETUP.md`
Step 2). **Do not rent Blackwell for this project.**

### 8.2 Why a faster GPU barely helps here

This workload is **kernel-launch-bound**. Azerbaijani sequences average 29.6
tokens (p95 = 79); with 8x4 accumulation a micro-batch is 8 x ~80 tokens. The
GPU idles between tiny kernels, so most of a step is fixed overhead that a
bigger card does not touch.

That is measurable, not speculative. The RTX 4060 diagnostic (`HANDOFF.md` §12)
recorded 0.30–0.65 s/step where the A100 is projected at 0.25 — the 4060 has
roughly **a quarter** of the A100's throughput yet ran only ~1.75x slower.
Solving `1 + 3f = 1.75` gives a GPU-compute fraction of **f ≈ 0.25**: three
quarters of each step is overhead.

Step time therefore scales as `0.75 + 0.25 / speedup`. Even an infinitely fast
GPU caps out at a **1.33x** speedup over the A100.

### 8.3 The comparison

Assumptions: 8x4 accumulation, A100 central 0.52 s/optimizer-step, Phase 1 at
4 workers, Phase 2 at 8 streams, ~1.75 h prep (the OMP solve is CPU-bound) and
~0.3 h analysis. Efficiency factors 3.2x at 4 streams and 4.8x at 8 — sub-linear
on purpose; replace them with the measured value from Tranche A.

| GPU | VRAM | s/step | Serial | **Total (P1 x4, P2 x8)** | **Cost @ from-rate** | Cost @ median | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| A100 80 GB *(reference)* | 80 | 0.520 | 44.6 h | 11.4 h | — | — | baseline |
| **H100 SXM** | 80 | 0.462 | 39.9 h | **10.4 h** | **$14** | $22 | **best value** |
| H100 NVL | 80 | 0.471 | 40.6 h | 10.5 h | $16 | $22 | close second |
| H100 PCIe | 80 | 0.477 | 41.1 h | 10.6 h | $27 | $33 | 2x the cost, same speed |
| H200 NVL | 141 | 0.471 | 40.6 h | 10.5 h | $38 | $45 | extra VRAM unused |
| H200 | 141 | 0.462 | 39.9 h | 10.4 h | $40 | $48 | extra VRAM unused |
| ~~B200~~ | 192 | 0.442 | 38.2 h | 10.0 h | $53 | $64 | **won't run** |
| ~~B300~~ | 288 | 0.438 | 37.9 h | 10.0 h | $100 | $100 | **won't run** |

**Read the two right-hand columns together.** Total wall-clock spans
10.0–10.6 h — a **4% spread**. Cost spans **$14 to $100, a 7x spread**. A B300
buys 4% less time for 7x the money, and then does not run.

**Rent the H100 SXM at $1.33/h.** It is simultaneously the cheapest listed and
tied-fastest for this workload.

### 8.4 Tranche A first — validate for ~$4

Run Tranche A before committing to the full grid. Ten runs, no Turkish stage,
so no Phase 1:

| GPU | Tranche A total | Cost @ from | Cost @ median |
|---|---:|---:|---:|
| **H100 SXM** | **~3.0 h** | **$3.90** | $6.40 |
| H100 NVL | ~3.0 h | $4.60 | $6.40 |
| H100 PCIe | ~3.0 h | $7.60 | $9.20 |
| H200 | ~3.0 h | $11.50 | $13.70 |

For **under five dollars** you get real seconds/step, real per-process VRAM,
the escape-behaviour answer, and end-to-end integrity — then re-project this
whole table from measurements instead of assumptions.

### 8.5 Choosing the instance, not just the GPU

- **vCPU is the real stream limit, not VRAM.** At ~5.5 GB/process an 80 GB card
  fits 14 streams and a 288 GB card fits 52 — but the *host* issues every
  kernel launch, and this workload is launch-bound. Budget **>= 2 vCPU per
  stream** and do not exceed 8 streams without measuring. A 8-stream run wants
  >= 16 vCPU; filter vast.ai on that, not on VRAM.
- **Disk: allocate >= 60 GB.** The grid needs ~32 GB (§6) and vast.ai defaults
  are often smaller. The launcher refuses to start below the projection plus a
  15% margin, so an under-sized volume fails fast — but it fails after you have
  started paying.
- **Bandwidth.** ~3 GB of model weights (xlm15 ~1.0 GB, xlmr ~1.1 GB, donor
  ~0.5 GB) plus datasets. Some hosts bill egress/ingress; check before renting.
- **Interruptible instances are a genuine saving here.** Every run writes a
  durable result file and `run.skip_existing` resumes from it, so an
  interrupted grid loses at most one run. Phase 1 is likewise resumable — a
  cached checkpoint is a cache hit on restart. Interruptible pricing on
  vast.ai is frequently well below the on-demand "from" rate.
- **Keep `fp16: true`.** bf16 would be the better numerical choice on Hopper,
  but `training.fp16` is hash-locked and pre-registered, and switching it
  mid-grid makes runs non-comparable. fp16 works correctly on `sm_90`.

### 8.6 Phase 1 can now be parallelised

Phase 1 was a ~4.3 h **serial** block that stream parallelism could not touch —
on an hourly rental that is ~$6–9 of pure latency. It is now optionally
parallel (`--phase1-streams N`, default 1), which cuts it to ~1.4 h at 4
workers.

This does **not** weaken the guarantee two-phase execution exists to provide.
That guarantee removes a *duplicate-request* race: two streams asking for the
same uncached checkpoint and both writing the same path. Phase 1's own work
list is deduplicated by `phase1_required_specs()` — 30 specs, 30 distinct cache
keys — and the cache path is a SHA-256 of the spec, so **no two Phase-1 workers
can ever target the same path**. The invariant is asserted at runtime by
`_assert_specs_are_injective()` and covered by
`tests/test_parallel_execution.py`; if it is ever violated the run stops rather
than silently producing corrupted weights.

### 8.7 Honest reporting if you use rented compute

The brief's ~10–12 GB cap governs the **shared course workstation** ("All teams
share one workstation"), where over-allocating starves other teams. On a
machine you rented alone that specific harm does not apply — but the paper must
still be straight about what was run where:

- Report the **per-process peak** (projected ~5.5 GB) as the figure showing the
  work fits the envelope. That is the honest quantity, and it is what
  `streams_active` in each result file lets a reader reconstruct.
- State the stream count and that concurrent streams ran on rented hardware.
  Do **not** describe an 8-stream, ~44 GB total footprint as fitting a
  10–12 GB envelope.
- `streams_active` is recorded in every result file precisely so a contended
  wall-clock is never mistaken for a run's isolated cost. Quote step time from
  the single-stream benchmark, never from a contended run.
