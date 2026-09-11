# vast.ai — complete step-by-step

Everything from "I have no account" to "results are on my laptop, instance
destroyed". Follow it in order.

---

## Answering your question first: where is the vCPU filter?

**There is no vCPU slider in the console's left filter bar.** That is why you
could not find it. There are two ways to get at it:

### Option A — read it off the offer card (web console)

Every offer card shows **CPU type and allocated cores as a portion of the
machine total**, e.g. `AMD EPYC 7763  16.0/128 cores`. The **first** number is
what *you* get. That is the number that matters — ignore the second.

So: search normally, then **read the CPU line on each card** and reject
anything under ~12.

Why it matters here: H100 SXM machines are usually 8-GPU boxes. Renting 1 GPU
gives you roughly ⅛ of the CPU. On a 64-core host that is 8 cores; on a
128-core host, 16. Our workload is **kernel-launch-bound** — the host CPU
issues every GPU kernel — so cores, not VRAM, cap how many parallel streams
actually help.

### Option B — the CLI, where it *is* a real filter (recommended)

The field is **`cpu_cores_effective`** ("virtual cpus you get"). This is the
reliable way, and it takes two minutes to set up.

```bash
curl -fsSL https://vast.ai/install.sh | bash
vastai set api-key YOUR_API_KEY        # Console -> Keys -> API Keys
```

```bash
vastai search offers \
  'gpu_name=H100_SXM num_gpus=1 cpu_cores_effective>=12 cpu_ram>=48 disk_space>=100 reliability>0.98 rentable=true verified=true' \
  -o 'dph' --limit 20
```

`-o 'dph'` sorts by price ascending, so the cheapest qualifying machine is at
the top. Add a `-` to sort descending (`-o 'dph-'`).

If that returns nothing, relax in this order: `verified=true` (drop it),
`reliability>0.98` → `>0.95`, `cpu_cores_effective>=12` → `>=8`. **Relax the
GPU last** — H100 SXM is both the cheapest and the fastest for this workload.

---

## Step 1 — Account and credit

1. Sign up at **https://cloud.vast.ai**
2. **Billing → Add Credit.** Put in **$25**. The full grid is $10–17; the rest
   is margin for a re-run.
3. **Keys → API Keys → Copy** (only needed for the CLI route).

---

## Step 2 — Find a machine

**Console → Search** (or https://cloud.vast.ai/create/).

Set these in the left/top filter bars:

| Filter | Value |
|---|---|
| GPU | **H100 SXM** (accept H100 NVL at similar price) |
| Number of GPUs | **1** |
| **Disk Space slider** | **100 GB** ← see the warning below |
| Reliability | **> 0.98** |
| Sort by | **Price (ascending)** or `$/hr` |

Then **read the CPU line on each card** and take the cheapest with **≥ 12
allocated cores**.

> ### ⚠️ The disk slider is not just a filter
> It is **also the amount of disk your instance is created with, and it cannot
> be changed afterwards.** Set it to **100 GB before you rent.** The grid needs
> ~32 GB (27 GB Turkish checkpoint cache + artifacts), plus the image and pip
> packages. Storage is ~$0.10–0.20/GB/month, so 100 GB for 12 hours costs a few
> cents. Running out of disk mid-grid means renting again.

**Do NOT rent a B200 or B300.** `requirements.lock` pins `torch==2.4.1+cu121`,
and CUDA 12.1 has no Blackwell (`sm_100`) kernels — it fails with *"no kernel
image is available for execution on the device."*

### On-demand vs interruptible

- **On-demand** = the listed price, runs until you stop it. **Use this.**
- **Interruptible** = you bid; a higher bidder evicts you. Cheaper, and *safe
  for us* (every finished run is durable and `run.skip_existing` resumes) —
  but with two days left, a mid-grid eviction costs attention you do not have.
  Take on-demand.

---

## Step 3 — Pick the image, then rent

Click **Change Template** (upper-left of the search page).

Use a PyTorch image matching the pinned stack:

```
pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
```

If that tag is unavailable, vast.ai's **default PyTorch template** is fine —
`requirements.lock` installs the correct torch either way, it just takes a few
minutes longer.

Make sure **SSH** is enabled in the template (it is, by default).

Then click **RENT** on your chosen card.

**CLI equivalent** (`ID` is the offer id from the search output):

```bash
vastai create instance ID \
  --image pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime \
  --disk 100 --ssh --direct
```

---

## Step 4 — Connect

**Console → Instances.** Wait until the card says **RUNNING** (1–3 minutes
while the image pulls).

Click the **key icon** (`>_ Connect`) to get a command like:

```bash
ssh -p 41234 root@ssh5.vast.ai -L 8080:localhost:8080
```

Paste that into your terminal. On Windows use PowerShell or Windows Terminal —
`ssh` is built in.

```bash
vastai show instances       # CLI equivalent
```

---

## Step 5 — Get the code onto the box

**Easiest — from your laptop, in the folder containing the zip:**

```bash
scp -P 41234 az-tokenizer-transfer_FOR_COLAB.zip root@ssh5.vast.ai:/workspace/
```

*(Use the same port and host as your ssh command.)*

Then **on the instance**:

```bash
cd /workspace
apt-get update -qq && apt-get install -y -qq unzip tmux
unzip -q az-tokenizer-transfer_FOR_COLAB.zip
cd az-tokenizer-transfer_snapshot_*
ls          # you should see configs/ src/ scripts/ run_all.sh
```

*(If you have pushed to GitHub instead: `git clone <url> && cd <repo>`.)*

---

## Step 6 — 🔴 Start tmux. Do not skip this.

```bash
tmux new -s grid
```

Everything from here runs **inside tmux**. If your laptop sleeps, your Wi-Fi
drops, or you close the terminal, the run keeps going on the server.

- **Detach** (leave it running): `Ctrl+B`, then `D`
- **Reattach** later: `ssh` back in, then `tmux attach -t grid`

Without tmux, an 8-hour run dies the first time your SSH connection blinks.

---

## Step 7 — Stage 1: prep, benchmark, Tranche A (~2.5–3 h)

```bash
bash scripts/vastai_stage1.sh
```

This does, in order: install → config-hash check → test suite (**expect 91
passed**) → data → transplant build → transplant health check → GPU benchmark →
**Tranche A** → prints the decision.

Then **detach** (`Ctrl+B`, `D`) and go do something else. Reattach with
`tmux attach -t grid`.

### 🚦 When it finishes, READ THE OUTPUT before continuing

Two checks decide whether Stage 2 is worth running:

**A. Transplant health** — if both transplanted models score ~0 top-1 on
~1,300 masked positions, the tokenizer swap destroyed the models' language
ability. That is a reportable finding, and it weakens the `xlmr` control (a
null there could mean "no deficit to fix" *or* "we broke it").

**B. Tranche A decision rule** (pre-registered, run plan §8):

| Outcome | Action |
|---|---|
| Both conditions escape in ≥ 3/5 seeds | **Proceed to Stage 2** |
| `tokenizator` escapes reliably, `baza` does not | **Headline result.** Proceed |
| Neither escapes in any seed | **STOP.** n=2000 is below the detectability threshold. Do **not** run Stage 2, do **not** tune the optimizer. Destroy the instance and re-plan at n=10000 |
| Erratic / uncorrelated with condition | Report escape rate as the primary outcome; discuss before spending more |

---

## Step 8 — Stage 2: the full grid (~6–8 h, unattended)

```bash
bash scripts/vastai_stage2.sh
```

Phase 1 (Turkish cache, 4 workers) → Phase 2 (164 runs, 4 streams) → full
analysis. Detach and leave it.

**Write the paper while this runs.** That is the whole reason for choosing the
7.7-hour option over the 22.5-hour one.

Check progress any time:

```bash
tmux attach -t grid
ls results/runs/*.json | wc -l      # out of 164
```

---

## Step 9 — Get the results off *before* destroying anything

**On the instance:**

```bash
cd /workspace/az-tokenizer-transfer_snapshot_*
zip -r /workspace/results_full.zip results figures
ls -lh /workspace/results_full.zip
```

**From your laptop:**

```bash
scp -P 41234 root@ssh5.vast.ai:/workspace/results_full.zip .
```

**Open the zip and confirm it contains `results/paper_tables.md` and
`results/decompose.json`.** Only then:

**Console → Instances → DESTROY** (or `vastai destroy instance <id>`).

> A destroyed instance takes its disk with it. There is no undo. Download,
> verify, *then* destroy.

---

## Cost check

| Item | Estimate |
|---|---|
| Stage 1 (~3 h) | $4–6 |
| Stage 2 (~7 h) | $9–15 |
| Storage, 100 GB × ~12 h | ~$0.03 |
| **Total** | **~$13–21** |

Watch the meter in **Console → Instances**. Billing stops on **DESTROY**, not
on "stopped" — a stopped instance still bills for storage.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Search returns nothing | Drop `verified=true`, lower reliability to 0.95, lower cores to 8. Keep H100 SXM |
| `CONFIG HASH CHECK FAILED` | A config file was edited. Re-extract from the zip. Do not "fix" it by re-locking |
| Tests fail (not 91 passed) | Stop and send the output. Do not train |
| `no kernel image is available` | You rented a Blackwell (B200/B300). Destroy and rent an H100 |
| `DISK PRE-FLIGHT FAILED` | Disk too small. It cannot be resized — destroy and re-rent with 100 GB |
| SSH drops mid-run | Fine if you used tmux. Reconnect, `tmux attach -t grid` |
| Lost the tmux session | `tmux ls` to list, `tmux attach -t <name>` |
| Out of credit mid-run | Instance is stopped, disk survives briefly. Add credit fast and restart; finished runs resume |
| Phase 2 says `PHASE-2 CACHE MISS` | Phase 1 did not finish. Re-run Stage 2 — Phase 1 is resumable |

---

## The one-page version

```bash
# laptop
scp -P PORT az-tokenizer-transfer_FOR_COLAB.zip root@HOST:/workspace/
ssh -p PORT root@HOST

# instance
cd /workspace && apt-get update -qq && apt-get install -y -qq unzip tmux
unzip -q az-tokenizer-transfer_FOR_COLAB.zip && cd az-tokenizer-transfer_snapshot_*
tmux new -s grid
bash scripts/vastai_stage1.sh        # ~3 h, then READ THE DECISION
bash scripts/vastai_stage2.sh        # ~7 h, write the paper meanwhile
zip -r /workspace/results_full.zip results figures

# laptop
scp -P PORT root@HOST:/workspace/results_full.zip .
# verify the zip opens, THEN destroy the instance
```

Sources: [Finding & Renting Instances](https://docs.vast.ai/guides/instances/choosing/find-and-rent) ·
[CLI search-offers reference](https://docs.vast.ai/cli/reference/search-offers) ·
[Vast.ai CLI](https://vast.ai/developers/cli)
