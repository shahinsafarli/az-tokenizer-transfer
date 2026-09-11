# Runbook — running the real experiments

**Decision date: 2026-09-05.** This file says exactly where to run, in what
configuration, and in what order. It exists because "where and how" was the
open question, and an open question costs more than a wrong-but-revisable
answer.

---

## 0. The constraint that decides everything

It is not the GPU. It is the calendar.

| Item | Status |
|---|---|
| Brief deadline | **7 September 2026, 23:59 Baku** — 2 days |
| `report/report.pdf` (15% of the grade) | **does not exist** |
| `presentation/presentation.pdf` (8%) | **does not exist** |
| `contribution_report.pdf` (automatic deduction if missing) | **does not exist** |
| Defense (12%) | not prepared |
| Result files under the frozen config | **zero** |

**35% of the grade is in artifacts that do not exist**, and the two that go to
Moodle are the *only* things actually submitted — the repository is reached
through a link in the paper's abstract.

So the objective is **not** "use the most GPU". It is: get a complete,
defensible result in the smallest wall-clock, and spend everything else
writing. Every hour of compute you can convert into money, convert.

> If the 9 September date in `project_state_and_run_plan.pdf` is authoritative
> instead, you gain 2 days. **Confirm which is binding before planning around
> it.** The brief is the governing document and says 7 September; plan for
> that and treat 9 September as upside.

---

## 1. The decision

### Rent an H100 SXM on vast.ai. Run the full grid in one sitting.

**Configuration: `batch_size: 32`, `grad_accum: 1`** — already applied, and it
is the *original* pre-registered value (`configs/FROZEN.md`, 2026-09-05
amendment restores it).

| Venue + config | Streams | Peak | **Full grid** | Cost |
|---|---:|---:|---:|---:|
| Institution A100 MIG 20 GB, 8x4 | 1 | 5.5 GB | 44.6 h | free |
| Institution A100 MIG 20 GB, batch 32 | 1 | 8.6 GB | **22.5 h** | free |
| vast.ai H100 SXM, 8x4 | 4 | 22 GB | 13.8 h | $18–30 |
| **vast.ai H100 SXM, batch 32** | **4** | **34 GB** | **7.7 h** | **$10–17** |
| vast.ai H100 SXM, batch 32 | 6 | 52 GB | 6.3 h | $8–14 |

**~$15 buys back roughly 15 hours of wall-clock and removes a scheduling
dependency.** With two days left and no paper, that is the cheapest thing you
will buy all week.

### Why not the institution A100?

Nothing is wrong with it — 22.5 h at batch 32 does fit a ~2-day window. But:

- it is **time-shared and booked**, so you inherit a queue and a fixed slot;
- a failed run inside a booked window may not be re-runnable;
- 22.5 h of a 48 h window leaves the paper competing with the compute.

**Use it as the fallback**, not the primary. If vast.ai is unavailable or the
card misbehaves, the institution A100 at batch 32 / 1 stream still gets you
there — the config is identical, only `--streams 1`.

### Why not H200 / A100 on vast.ai?

This workload is **kernel-launch-bound**, not compute-bound. Measured anchor:
an RTX 4060 has ~¼ an A100's throughput yet ran only ~1.75x slower, which puts
the GPU-compute fraction at ~25%. Step time scales as `0.75 + 0.25/speedup`, so
**even an infinitely fast card caps at 1.33x**. Every workable card lands
within ~4% of the others on wall-clock while price varies 7x.

**H100 SXM is the cheapest listed *and* tied-fastest.** H200 costs ~3x more for
the same hours. And **B200/B300 will not run at all** — `requirements.lock`
pins `torch==2.4.1+cu121`, and CUDA 12.1 has no `sm_100` kernels.

### Why batch 32 rather than the 8x4 you used for the T4

8x4 was adopted to make 2–3 streams fit the brief's ~10–12 GB team cap. That
premise does not hold: the ~4.45 GB of AdamW state is **batch-size
independent**, so two streams exceed 10 GB in *either* configuration. And at
one stream 8.6 GB is already under the cap — there is no margin to buy.

Meanwhile accumulation costs **~2.1x wall-clock** on a launch-bound workload,
because splitting a batch of 32 into four micro-batches multiplies exactly the
overhead that dominates.

Restoring batch 32 also removes the dropout-RNG-ordering caveat: at
`grad_accum: 1` a batch is consumed in one pass, which is what the original
pre-registration assumed.

**This change was free to make** — zero schema-v3 results existed on
2026-09-05. **It stops being free the moment your first real run completes.**

---

## 2. What NOT to change

You asked about tweaking for a bigger GPU. Here is the honest list.

| Tempting change | Verdict |
|---|---|
| `batch_size` beyond 32 | **No.** 32 is the pre-registered *effective* batch. Changing it changes the optimization, not just the speed, and invalidates every design justification in `configs/FROZEN.md`. |
| `max_steps` below 2000 | **No.** The step budget is the whole reason the data curve is interpretable. |
| `lr`, warmup, weight decay | **No.** Pre-registered and explicitly unchanged. |
| `fp16` -> `bf16` | **No**, not now. Numerically better on Hopper, but it changes results and must then be uniform across every run. Not worth it with 2 days left. |
| `torch.compile` / CUDA graphs | **No**, not now. It is genuinely the *right* optimization for a launch-bound workload and could cut step time substantially — but it changes numerics, adds compile time, and can fail inside HF `Trainer`. This is the one to revisit if you ever re-run with time to spare. |
| TF32 matmuls | **No**, not now. Small gain on a launch-bound workload; changes numerics. |
| `eval_batch_size` | Already 64. Forward-only, not pre-registered. Safe to tune if the benchmark shows headroom. |
| `parallel_streams` | **Yes — this is the lever.** Execution machinery, not science, and recorded per run in `streams_active`. |
| `phase1_streams` | **Yes.** Cuts the ~2.3–3.8 h serial Turkish cache to ~1 h. Safe because Phase 1's work list is deduplicated (asserted at runtime). |

**The rule that matters:** whatever you pick must be identical across every
*reported* run. Mixing configurations quarantines results.

---

## 3. The sequence

Use `notebooks/vastai_h100.ipynb`. It already implements this; the steps below
are what it does, so you can also run them over SSH.

### Before you rent — filter the vast.ai listing on

- **GPU:** H100 SXM (accept H100 NVL at a similar price)
- **vCPU: >= 12** — the *host* issues every kernel launch and launches are the
  bottleneck. Budget ~2 vCPU per stream. **This matters more than VRAM.**
- **Disk: >= 60 GB** (the grid needs ~32 GB)
- **Interruptible is safe here** — every run writes a durable result file and
  `run.skip_existing` resumes, so an interruption costs at most one run.

### Step 0 — sanity (free, 1 min)

```bash
nvidia-smi; nproc; df -h /
```
If it says **B200** or **B300**, destroy the instance. `sm_100` will not run
the pinned wheel.

### Step 1 — environment (~10 min)

```bash
pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu121
python -c "from src.utils import verify_config_hashes; verify_config_hashes(); print('OK')"
python -m pytest tests/ -q -m "not slow"
```

> **2026-09-08 — do not run the requirements.lock line above as written.** `requirements.lock` does not resolve (`fsspec==2026.7.0` vs `datasets==2.21.0`, which needs `fsspec[http]<=2024.6.1`). See that file's own header for what the grid actually ran on.

Expect **91 passed**. Anything else: stop.

### Step 2 — data, transplant, controls (~2 h, mostly CPU)

```bash
CONFIG=configs/experiment.yaml DEVICE=cuda bash run_all.sh --prep-only
```

Then **read `results/cross_base_transplant_quality.json` before going on.**
If both transplanted models show ~0 top-1 on ~1,300 masked positions, the
transplant has destroyed masked-LM ability. That is a reportable finding, and
it weakens the `xlmr` zero-effect control — a null there could then mean "no
deficit to fix" *or* "we broke it". Decide consciously whether to proceed.

### Step 3 — benchmark (~10 min)

```bash
bash scripts/a100_benchmark.sh
```
Writes `results/hardware_benchmark.json` and prints the
`run.vram_per_stream_gb` / `run.parallel_streams` values to write back. The
notebook does the write-back and hash re-lock for you. **Replace every
projection in `docs/COMPUTE_ESTIMATES.md` with these measurements.**

### Step 4 — Tranche A (~30 min), then STOP and read

```bash
python -m src.training.run_grid --config configs/experiment.yaml --tranche A --phase 2 --streams 1
python -m src.analysis.aggregate --config configs/experiment.yaml
python -m src.analysis.decompose --config configs/experiment.yaml
```

Apply the pre-registered rule (run plan §8):

| Outcome | Action |
|---|---|
| Both conditions escape in >= 3/5 seeds | Proceed |
| `tokenizator` escapes reliably, `baza` does not | **Headline result.** Proceed |
| Neither escapes in any seed | **STOP.** n=2000 is below the detectability threshold. Move to n=10000. Do **not** tune the optimizer |
| Erratic, uncorrelated with condition | Report escape rate as the primary outcome; raise seeds at the reference size only |

Half an hour here protects the other seven.

### Step 5 — the full grid (~6–8 h, unattended)

```bash
python -m src.training.run_grid --config configs/experiment.yaml --phase 1 --phase1-streams 4
python -m src.training.run_grid --config configs/experiment.yaml --phase 2 --streams 4
```

**Write the paper while this runs.** That is the entire point of choosing the
7.7 h option over the 22.5 h one.

### Step 6 — analysis

```bash
bash run_all.sh --analysis-only
```
Produces `results/paper_tables.md`, `results/decompose.json` and the figures,
every number carrying the result files behind it.

### Step 7 — get the results off the box *before* destroying it

```bash
zip -r results_full.zip results figures
```
Download it, open it, **then** destroy the instance.

---

## 4. The test set

Everything above evaluates on **validation**. The test set is touched exactly
once, at the end, through an explicit opt-in that appends to
`results/test_evaluation_ledger.jsonl`:

```bash
python -m src.training.finetune --config configs/experiment.yaml \
    --base primary --condition tokenizator --train-size 2000 --seed 42 \
    --eval-split test --allow-test-eval
```


---

## 5. If you have to cut

At ~7.7 h for all 114 runs there is no reason to cut. But if you fall behind,
drop in this order — and say so in the paper:

1. **Tranche E** (data curve, 64 runs, ~3–4.5 h at 4 streams). Costs you the
   "where does the effect vanish" curve. Everything at n=2000 still stands:
   both bases, all conditions, both placebos, the Turkish contrast.
2. **Tranche D** (Turkish conditions, 30 runs + Phase 1). Costs you M1 vs M2 —
   the original Kardeş-NLU question. Expensive to lose scientifically.
3. **Never cut** Tranche A + B or the placebo conditions. A+B is the headline
   cross-base result; the placebos are what stop a reviewer reading the
   transplant effect as an artifact of the surgery.

---

## 6. Reporting honestly if you rent

The brief's ~10–12 GB cap governs the **shared workstation**, where
over-allocating starves other teams. On a box you rented alone that harm does
not apply — but the paper must still be straight:

- Report the **per-process peak** (~8.6 GB measured) as the figure showing the
  work fits the envelope. `streams_active` in each result file lets a reader
  reconstruct it.
- State the stream count and that concurrent streams ran on rented hardware.
  Do **not** describe a 4-stream ~34 GB total as fitting a 10–12 GB envelope.
- Quote step time from the **single-stream benchmark**, never from a contended
  run.
- Name the rented hardware in Experimental Setup alongside the A100 window.
