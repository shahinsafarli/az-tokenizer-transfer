# Manual transfer manifest — files not carried by `git clone`

`artifacts/` is gitignored (see `.gitignore`) — model weights and derived
data don't belong in git history. Everything below must be copied to the
A100 host by hand (`scp`, `rsync`, etc.) after the clone, before running
`scripts/a100_bootstrap.sh`. `configs/`, `requirements.lock`, and everything
else the bootstrap checks travels via the git clone itself and needs no
manual step — see `docs/A100_SETUP.md`.

Generated 2026-09-04 from the working tree at commit `5ca0aa3` (or later —
if these hashes don't match what's on the source machine now, the source
tree has changed since this manifest was written; regenerate it, don't
trust a stale manifest silently).

## Required for the frozen 114-run grid

Three canonical (`_rescaled`) transplant tags × two bases, plus the AZ/TR
data splits. **Total: ~3.57 GB, 38 files.**

`docs/a100_transfer_checksums.sha256` (committed alongside this file,
`sha256sum -c` format, same 38 entries as the table below) travels via the
git clone itself — no separate transfer needed for the checksum file, only
for the `artifacts/` content it checks. After transferring, verify from the
repo root on the A100 host:
```bash
sha256sum -c docs/a100_transfer_checksums.sha256
```
Every line must print `OK`; anything else means that file did not arrive
intact — re-transfer it, don't proceed.

| File | Size | SHA-256 |
|---|---:|---|
| `artifacts/transplanted__xlm15__omp_k64_rescaled/config.json` | 1,605 B | `6db9db56f3281523f2039c96bfcd2821a7ac66a29b7d2ecc517de7565d65bf19` |
| `artifacts/transplanted__xlm15__omp_k64_rescaled/generation_config.json` | 117 B | `0e8eac89aadba1a73970ecac46dc318af9f45e6042ad450501bc366fcdf36b13` |
| `artifacts/transplanted__xlm15__omp_k64_rescaled/model.safetensors` | 741,163,496 B (707 MiB) | `32fcb5f5693de8295a94a8a790ac3b1582f29a3437cbd25b3b82aeef2d62b746` |
| `artifacts/transplanted__xlm15__omp_k64_rescaled/special_tokens_map.json` | 782 B | `c48720c677638c507233a173d6b7a8fc62d5c4df493ae628caf6908d375eff96` |
| `artifacts/transplanted__xlm15__omp_k64_rescaled/tokenizer.json` | 988,137 B | `eb0168fac8876fcdbc42202f7c7f0cfbd1348dac33639531bc686bbb2a6fbe03` |
| `artifacts/transplanted__xlm15__omp_k64_rescaled/tokenizer_config.json` | 19,581 B | `6ae45e0f2a403e7fec16c54c048cd3cd5666e31f6f7663f7869749341a3d3163` |
| `artifacts/transplanted__xlm15__mean_k64_rescaled/config.json` | 1,606 B | `a0833570dd59e57a5ab7a18d12caf255b3cbaa54c2f63145acb4529229c625fb` |
| `artifacts/transplanted__xlm15__mean_k64_rescaled/generation_config.json` | 117 B | `0e8eac89aadba1a73970ecac46dc318af9f45e6042ad450501bc366fcdf36b13` |
| `artifacts/transplanted__xlm15__mean_k64_rescaled/model.safetensors` | 741,163,496 B (707 MiB) | `6be3f0193cbcd58909fdf85ddd8a54a2f2f5cb3254f19fc3145f2df4ef0c9b61` |
| `artifacts/transplanted__xlm15__mean_k64_rescaled/special_tokens_map.json` | 782 B | `c48720c677638c507233a173d6b7a8fc62d5c4df493ae628caf6908d375eff96` |
| `artifacts/transplanted__xlm15__mean_k64_rescaled/tokenizer.json` | 988,137 B | `eb0168fac8876fcdbc42202f7c7f0cfbd1348dac33639531bc686bbb2a6fbe03` |
| `artifacts/transplanted__xlm15__mean_k64_rescaled/tokenizer_config.json` | 19,581 B | `6ae45e0f2a403e7fec16c54c048cd3cd5666e31f6f7663f7869749341a3d3163` |
| `artifacts/transplanted__xlm15__random_coef_k64_rescaled/config.json` | 1,613 B | `3cbc79bd7e6f0f932bbb2c702ee5c9f4a6ae347a57fd727623e2937d077cc1ac` |
| `artifacts/transplanted__xlm15__random_coef_k64_rescaled/generation_config.json` | 117 B | `0e8eac89aadba1a73970ecac46dc318af9f45e6042ad450501bc366fcdf36b13` |
| `artifacts/transplanted__xlm15__random_coef_k64_rescaled/model.safetensors` | 741,163,496 B (707 MiB) | `749906e3e57c6b4e3ab1803fbd83775eabf25cb30610d17b78ab8be6caa98bbe` |
| `artifacts/transplanted__xlm15__random_coef_k64_rescaled/special_tokens_map.json` | 782 B | `c48720c677638c507233a173d6b7a8fc62d5c4df493ae628caf6908d375eff96` |
| `artifacts/transplanted__xlm15__random_coef_k64_rescaled/tokenizer.json` | 988,137 B | `eb0168fac8876fcdbc42202f7c7f0cfbd1348dac33639531bc686bbb2a6fbe03` |
| `artifacts/transplanted__xlm15__random_coef_k64_rescaled/tokenizer_config.json` | 19,581 B | `6ae45e0f2a403e7fec16c54c048cd3cd5666e31f6f7663f7869749341a3d3163` |
| `artifacts/transplanted__xlmr__omp_k64_rescaled/config.json` | 756 B | `cc124b4ab8769b9d9248fd91a1d36fbf82a2cb4d7e896e6d7180619272c73d68` |
| `artifacts/transplanted__xlmr__omp_k64_rescaled/model.safetensors` | 444,999,328 B (424 MiB) | `724c6b319c762250216b55119966439f1acafd05b099c7f92bf2cf9c40a5e952` |
| `artifacts/transplanted__xlmr__omp_k64_rescaled/special_tokens_map.json` | 782 B | `c48720c677638c507233a173d6b7a8fc62d5c4df493ae628caf6908d375eff96` |
| `artifacts/transplanted__xlmr__omp_k64_rescaled/tokenizer.json` | 988,137 B | `eb0168fac8876fcdbc42202f7c7f0cfbd1348dac33639531bc686bbb2a6fbe03` |
| `artifacts/transplanted__xlmr__omp_k64_rescaled/tokenizer_config.json` | 19,581 B | `6ae45e0f2a403e7fec16c54c048cd3cd5666e31f6f7663f7869749341a3d3163` |
| `artifacts/transplanted__xlmr__mean_k64_rescaled/config.json` | 757 B | `0d3ea624030a1cd74a8dff1630aab49778a87fe1bedcfdc5df7b1c6b25ab78b4` |
| `artifacts/transplanted__xlmr__mean_k64_rescaled/model.safetensors` | 444,999,328 B (424 MiB) | `4dab7d5b9f917c0a4959f9cbaa3e290f14ea8416bbf2725f3208dd29d76babe2` |
| `artifacts/transplanted__xlmr__mean_k64_rescaled/special_tokens_map.json` | 782 B | `c48720c677638c507233a173d6b7a8fc62d5c4df493ae628caf6908d375eff96` |
| `artifacts/transplanted__xlmr__mean_k64_rescaled/tokenizer.json` | 988,137 B | `eb0168fac8876fcdbc42202f7c7f0cfbd1348dac33639531bc686bbb2a6fbe03` |
| `artifacts/transplanted__xlmr__mean_k64_rescaled/tokenizer_config.json` | 19,581 B | `6ae45e0f2a403e7fec16c54c048cd3cd5666e31f6f7663f7869749341a3d3163` |
| `artifacts/transplanted__xlmr__random_coef_k64_rescaled/config.json` | 764 B | `a86d8a158098f0b71a0dbc07a6e174ff09dc2b5bdf64ce6a7722b84334f34ad9` |
| `artifacts/transplanted__xlmr__random_coef_k64_rescaled/model.safetensors` | 444,999,328 B (424 MiB) | `79205ec727350f1d063ee8eb5496b4ca04eda4951fe75da4a458686e884fc8e8` |
| `artifacts/transplanted__xlmr__random_coef_k64_rescaled/special_tokens_map.json` | 782 B | `c48720c677638c507233a173d6b7a8fc62d5c4df493ae628caf6908d375eff96` |
| `artifacts/transplanted__xlmr__random_coef_k64_rescaled/tokenizer.json` | 988,137 B | `eb0168fac8876fcdbc42202f7c7f0cfbd1348dac33639531bc686bbb2a6fbe03` |
| `artifacts/transplanted__xlmr__random_coef_k64_rescaled/tokenizer_config.json` | 19,581 B | `6ae45e0f2a403e7fec16c54c048cd3cd5666e31f6f7663f7869749341a3d3163` |
| `artifacts/data/az_test.jsonl` | 642,347 B | `d532de0444e5e469a8f8b927a1e83c55d637594e7df9df9de2ec5695655c90a6` |
| `artifacts/data/az_train.jsonl` | 3,193,006 B | `9991e8159ae7150401a1fc1f8bedf85e4c9674a58834e3df1765a43cc7cc7c68` |
| `artifacts/data/az_val.jsonl` | 428,221 B | `6265a67a8dca1bdc08375e178e4ce1f735b11493d12bbfc1edaa98c83b51c662` |
| `artifacts/data/tr_train.jsonl` | 2,435,172 B | `5876893f564a0046ad96d9fe475e56a1789b7c0b2ca1c0e0160a93d1e2cb62c1` |
| `artifacts/data/tr_train_scrambled.jsonl` | 2,433,778 B | `330f575541cddf411b0b074fe72b39e0dae243755f19b3675405ea04277ef060` |

Note the identical hashes across sibling files within a base: all three
`xlm15__*_rescaled/generation_config.json` share `0e8eac89...` and all
`tokenizer.json`/`tokenizer_config.json`/`special_tokens_map.json` files are
identical within a base (same donor tokenizer regardless of reconstruction
method) — that's expected, not a copy-paste error in this table. Only
`config.json` (embeds a content hash / differs trivially) and
`model.safetensors` (the actual reconstructed embedding values) differ
between the omp / mean / random_coef siblings.

## Present locally, NOT required by the frozen queue

Six non-`_rescaled` directories (`transplanted__<base>__<method>_k64`,
without the `_rescaled` suffix) exist in `artifacts/` — ~3.4 GB total. These
are the pre-rescale ablation baseline artifacts (`canonical_transplant_tag()`
in `src/utils.py` always resolves to the `_rescaled` variant; the plain
variant is retained only for the no-rescale-vs-rescale ablation comparison
already computed and recorded in `results/`, e.g.
`embedding_norm_report__*.json`). **Skip these** unless you specifically
need to reproduce that already-completed ablation on the A100 too — the
114-run grid does not read them.

## What travels via `git clone` instead (no manual step)

`configs/experiment.yaml`, `configs/FROZEN.md`, `configs/CONFIG_HASHES.lock`,
`requirements.lock`, all of `src/`, `docs/`, `scripts/`, `tests/`, and
`results/` (small — result JSONs, ledgers, existing figures) are all
tracked in git. `scripts/a100_bootstrap.sh` verifies the config hashes
against the transferred/cloned tree; it does not and cannot verify
`artifacts/` content beyond what's in this manifest, since `artifacts/` is
gitignored by design (large binaries don't belong in git history).

## Test-evaluation disclosure

See `docs/TEST_AUDIT.md` — read before transferring anything, not after.
None of the files in the required-transfer table above carry
test predictions or test metrics — the six `_rescaled` transplant artifacts
are model weights, and the five `artifacts/data/*.jsonl` files are the
frozen splits themselves (`az_test.jsonl` is transferred because the split
must be identical across machines, not because it's been evaluated against).
