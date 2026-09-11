# Environment as run — 164-run grid

Captured 2026-09-09, on the same instance, immediately after the grid finished
164/164. Evidence files: `env_freeze_as_run.txt` (untouched `pip freeze`,
117 lines) and `env_hardware_as_run.txt` (`nvidia-smi`, torch probe, `lscpu`,
`free`, `df`).

## Hardware

| Item | Value |
|---|---|
| GPU | 1 × NVIDIA H100 NVL |
| GPU memory (device total) | 95,830 MiB (torch reports 99.46 GB) |
| Driver / host CUDA | 595.71.05 / 13.2 |
| torch build | 2.11.0+cu128 (`torch.version.cuda == 12.8`) |
| CPU | AMD EPYC 9V84, 40 vCPU (1 thread/core) |
| RAM | 313 GiB |
| Disk | 789 GB overlay, 478 GB used at capture |
| Virtualisation | Microsoft Hyper-V |

## Compute cost

| Measure | Value | Source |
|---|---|---|
| Runs | 164 (0 failures) | `results/launcher_state.json` |
| Concurrent streams | 1 throughout | `streams_active` in all 164 run files |
| Process-hours (sum of per-run runtime) | 8.4 h | `runtime_sec`, 164 files |
| — of which Azerbaijani fine-tuning stage | 8.0 h | `az_stage_runtime_sec` |
| Wall-clock | ≈9.5 h | launcher log timestamps |
| Device-hours | ≈9.5 GPU-hours (1 GPU × wall-clock) | derived |
| **Peak GPU memory (process or device)** | **NOT MEASURED** | see below |

The launcher's GPU-memory sampler did not run: there is no
`results/gpu_memory_trace.csv` and no `results/gpu_memory_summary.json` in the
output, and no run record carries `max_memory_allocated`. `nvidia-smi` at
capture time (after the grid ended) shows 97 MiB in use of 95,830 MiB, which
measures an idle GPU and says nothing about the run. Peak memory during the
grid is therefore unmeasured and must be reported as such, not estimated.

Note that the configured `vram_ceiling_gb = 10.0` capped the launcher to a
single stream. On a 94 GB device that is conservative by roughly an order of
magnitude; the ceiling was frozen before the GPU was known and was not changed
after the fact.

## Software stack: pre-run pins vs. what actually ran

| Package | Pre-run `requirements.lock` | Actually run |
|---|---|---|
| torch | 2.4.1+cu121 | **2.11.0+cu128** |
| transformers | 4.44.2 | **4.57.6** |
| tokenizers | 0.19.1 | **0.22.2** |
| datasets | 2.21.0 | **5.0.1** |
| numpy | 1.26.4 | **2.5.3** |
| scipy | 1.14.1 | **1.18.1** |
| scikit-learn | 1.5.2 | **1.9.0** |
| pandas | 2.2.3 | **3.0.5** |
| matplotlib | 3.9.2 | **3.11.1** |
| sentencepiece | 0.2.0 | **0.2.2** |
| pytest | 8.3.3 | **9.1.1** |
| sacremoses | 0.2.0 | 0.2.0 |
| protobuf | *absent* | **7.36.1** |

Two disclosures follow from this table, and both belong in the report:

1. **The shipped pin files did not describe the run environment.** The pre-run
   `requirements.lock` was unresolvable on the host — `fsspec==2026.7.0`
   conflicts with `datasets==2.21.0`, which requires `fsspec[http]<=2024.6.1`
   — so the install step failed and the stack was resolved fresh. The lock's own
   header instructed that the stack must *not* be resolved fresh on the run
   host; that instruction could not be honoured. `requirements.txt` and
   `requirements.lock` have now been regenerated from the post-run freeze and
   describe the environment the reported numbers were computed on.

2. **`sacremoses` and `protobuf` were undeclared dependencies.** Both had to be
   installed mid-session after import failures: `sacremoses` is required by the
   XLM-15 tokenizer, `protobuf` by sentencepiece model conversion. Neither
   appeared in either pin file. They are declared now.

Neither affects the validity of the results — every run in `results/` was
produced under the single stack recorded here, and the per-run best-checkpoint
restore was verified numerically inside that same environment — but the report
must not claim the shipped pins reproduce the grid, because they do not.

## `packaging`

The freeze records `packaging` as a conda-built local path
(`file:///home/conda/...`), so its exact version is not recoverable from the
capture. It is listed unpinned in `requirements.lock` with that caveat.
