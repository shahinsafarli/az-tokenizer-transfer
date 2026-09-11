# Test-split tables (descriptive) — generated from results/runs/*.json

All 164 runs were evaluated on the held-out test split at the **best-validation checkpoint** (`selection = best_validation_checkpoint`, `matched_step == selected_step`).

**Inference stays on validation.** The numbers below are point estimates with percentile-bootstrap CIs (10,000 resamples). No hypothesis test is run on the test split: the pre-registered inference split is validation, and adding a second test-side test family after seeing these values would not be pre-registered. Test is reported to confirm that the validation-side conclusions hold out of sample, not to re-decide them.

## T1 · Per-cell test macro-F1

| Base | Condition | n | n_esc/n | Test F1 (conditional, escaped only) [95% CI] | Test F1 (all-seed) [95% CI] | Val F1 (conditional) | Δ test−val (mean) |
|---|---|---:|---:|---|---|---:|---:|
| xlmr | baza | 500 | 5/5 | 0.7783 [0.7728, 0.7854] | 0.7783 [0.7728, 0.7854] | 0.7980 | -0.0197 |
| xlmr | tokenizator | 500 | 5/5 | 0.7035 [0.6930, 0.7152] | 0.7035 [0.6930, 0.7152] | 0.7198 | -0.0163 |
| xlmr | transplant_mean | 500 | 5/5 | 0.7217 [0.7160, 0.7274] | 0.7217 [0.7160, 0.7274] | 0.7421 | -0.0204 |
| xlmr | transplant_random_coef | 500 | 5/5 | 0.6867 [0.6759, 0.6944] | 0.6867 [0.6759, 0.6944] | 0.6985 | -0.0118 |
| xlmr | turk | 500 | 5/5 | 0.7910 [0.7880, 0.7937] | 0.7910 [0.7880, 0.7937] | 0.8149 | -0.0239 |
| xlmr | her_ikisi | 500 | 5/5 | 0.7173 [0.7061, 0.7270] | 0.7173 [0.7061, 0.7270] | 0.7261 | -0.0088 |
| xlmr | turk_qarisiq | 500 | 5/5 | 0.7814 [0.7742, 0.7873] | 0.7814 [0.7742, 0.7873] | 0.8054 | -0.0239 |
| xlmr | baza | 2000 | 5/5 | 0.7970 [0.7930, 0.8014] | 0.7970 [0.7930, 0.8014] | 0.8277 | -0.0307 |
| xlmr | tokenizator | 2000 | 5/5 | 0.7404 [0.7328, 0.7493] | 0.7404 [0.7328, 0.7493] | 0.7618 | -0.0213 |
| xlmr | transplant_mean | 2000 | 5/5 | 0.7578 [0.7465, 0.7667] | 0.7578 [0.7465, 0.7667] | 0.7772 | -0.0194 |
| xlmr | transplant_random_coef | 2000 | 5/5 | 0.7154 [0.7136, 0.7172] | 0.7154 [0.7136, 0.7172] | 0.7297 | -0.0143 |
| xlmr | turk | 2000 | 5/5 | 0.8048 [0.8026, 0.8078] | 0.8048 [0.8026, 0.8078] | 0.8307 | -0.0259 |
| xlmr | her_ikisi | 2000 | 5/5 | 0.7546 [0.7478, 0.7616] | 0.7546 [0.7478, 0.7616] | 0.7708 | -0.0161 |
| xlmr | turk_qarisiq | 2000 | 5/5 | 0.7971 [0.7946, 0.8014] | 0.7971 [0.7946, 0.8014] | 0.8269 | -0.0297 |
| xlm15 | baza | 500 | 3/5 | 0.6229 [0.6177, 0.6316] | 0.5071 [0.3903, 0.6239] | 0.6306 | -0.0046 |
| xlm15 | tokenizator | 500 | 2/5 | 0.5248 [0.4327, 0.6170] | 0.4100 [0.3334, 0.5234] | 0.5272 | -0.0009 |
| xlm15 | transplant_mean | 500 | 2/5 | 0.6302 [0.6290, 0.6314] | 0.4521 [0.3334, 0.5714] | 0.6359 | -0.0023 |
| xlm15 | transplant_random_coef | 500 | 3/5 | 0.6025 [0.5898, 0.6176] | 0.4949 [0.3847, 0.6051] | 0.6192 | -0.0100 |
| xlm15 | turk | 500 | 4/5 | 0.6090 [0.5690, 0.6406] | 0.5539 [0.4381, 0.6329] | 0.6141 | -0.0041 |
| xlm15 | her_ikisi | 500 | 5/5 | 0.6142 [0.6069, 0.6192] | 0.6142 [0.6069, 0.6192] | 0.6217 | -0.0075 |
| xlm15 | turk_qarisiq | 500 | 3/5 | 0.6336 [0.6218, 0.6474] | 0.5135 [0.3911, 0.6360] | 0.6348 | -0.0007 |
| xlm15 | baza | 2000 | 2/5 | 0.6735 [0.6710, 0.6759] | 0.4790 [0.3430, 0.6141] | 0.6791 | -0.0009 |
| xlm15 | tokenizator | 2000 | 3/5 | 0.6076 [0.4684, 0.6868] | 0.4980 [0.3604, 0.6355] | 0.6148 | -0.0043 |
| xlm15 | transplant_mean | 2000 | 3/5 | 0.6833 [0.6813, 0.6868] | 0.5434 [0.4030, 0.6837] | 0.6955 | -0.0073 |
| xlm15 | transplant_random_coef | 2000 | 4/5 | 0.6529 [0.6413, 0.6656] | 0.5890 [0.4597, 0.6609] | 0.6699 | -0.0136 |
| xlm15 | turk | 2000 | 4/5 | 0.6663 [0.6558, 0.6768] | 0.5997 [0.4656, 0.6734] | 0.6747 | -0.0067 |
| xlm15 | her_ikisi | 2000 | 5/5 | 0.6645 [0.6445, 0.6789] | 0.6645 [0.6445, 0.6789] | 0.6712 | -0.0067 |
| xlm15 | turk_qarisiq | 2000 | 2/5 | 0.6784 [0.6782, 0.6786] | 0.4714 [0.3334, 0.6095] | 0.6873 | -0.0035 |
| xlm15 | baza | 10000 | 2/3 | 0.7253 [0.7241, 0.7265] | 0.5947 [0.3334, 0.7265] | 0.7427 | -0.0116 |
| xlm15 | tokenizator | 10000 | 3/3 | 0.7438 [0.7302, 0.7544] | 0.7438 [0.7302, 0.7544] | 0.7657 | -0.0219 |
| xlm15 | transplant_mean | 10000 | 1/3 | 0.7539 [0.7539, 0.7539] | 0.4736 [0.3334, 0.7539] | 0.7653 | -0.0038 |
| xlm15 | transplant_random_coef | 10000 | 1/3 | 0.6129 [0.6129, 0.6129] | 0.4444 [0.3334, 0.6129] | 0.6319 | -0.0089 |
| xlm15 | baza | 20914 | 3/3 | 0.7217 [0.6819, 0.7433] | 0.7217 [0.6819, 0.7433] | 0.7374 | -0.0157 |
| xlm15 | tokenizator | 20914 | 1/3 | 0.7735 [0.7735, 0.7735] | 0.4801 [0.3334, 0.7735] | 0.7900 | -0.0055 |
| xlm15 | transplant_mean | 20914 | 3/3 | 0.7062 [0.5741, 0.7754] | 0.7062 [0.5741, 0.7754] | 0.7263 | -0.0201 |
| xlm15 | transplant_random_coef | 20914 | 1/3 | 0.7417 [0.7417, 0.7417] | 0.4695 [0.3334, 0.7417] | 0.7671 | -0.0084 |

## T2 · Key contrasts, test vs validation (descriptive, paired on shared escaped seeds)

| Base | n | Contrast | Δ val (paired) | Δ test (paired) | n shared | Same sign | Δ val (all-seed) | Δ test (all-seed) |
|---|---:|---|---:|---:|---:|:--:|---:|---:|
| xlmr | 500 | baza → turk | +0.0169 | +0.0127 | 5 | yes | +0.0169 | +0.0127 |
| xlmr | 500 | baza → her_ikisi | -0.0720 | -0.0610 | 5 | yes | -0.0720 | -0.0610 |
| xlmr | 500 | baza → turk_qarisiq | +0.0073 | +0.0031 | 5 | yes | +0.0073 | +0.0031 |
| xlmr | 500 | baza → tokenizator | -0.0782 | -0.0748 | 5 | yes | -0.0782 | -0.0748 |
| xlmr | 500 | baza → transplant_mean | -0.0560 | -0.0566 | 5 | yes | -0.0560 | -0.0566 |
| xlmr | 500 | baza → transplant_random_coef | -0.0996 | -0.0917 | 5 | yes | -0.0996 | -0.0917 |
| xlmr | 500 | transplant_random_coef → tokenizator | +0.0213 | +0.0168 | 5 | yes | +0.0213 | +0.0168 |
| xlmr | 500 | transplant_mean → tokenizator | -0.0223 | -0.0182 | 5 | yes | -0.0223 | -0.0182 |
| xlmr | 2000 | baza → turk | +0.0030 | +0.0078 | 5 | yes | +0.0030 | +0.0078 |
| xlmr | 2000 | baza → her_ikisi | -0.0569 | -0.0424 | 5 | yes | -0.0569 | -0.0424 |
| xlmr | 2000 | baza → turk_qarisiq | -0.0008 | +0.0001 | 5 | NO | -0.0008 | +0.0001 |
| xlmr | 2000 | baza → tokenizator | -0.0659 | -0.0566 | 5 | yes | -0.0659 | -0.0566 |
| xlmr | 2000 | baza → transplant_mean | -0.0505 | -0.0392 | 5 | yes | -0.0505 | -0.0392 |
| xlmr | 2000 | baza → transplant_random_coef | -0.0980 | -0.0816 | 5 | yes | -0.0980 | -0.0816 |
| xlmr | 2000 | transplant_random_coef → tokenizator | +0.0320 | +0.0250 | 5 | yes | +0.0320 | +0.0250 |
| xlmr | 2000 | transplant_mean → tokenizator | -0.0155 | -0.0174 | 5 | yes | -0.0155 | -0.0174 |
| xlm15 | 500 | baza → turk | +0.0033 | +0.0070 | 3 | yes | +0.0462 | +0.0468 |
| xlm15 | 500 | baza → her_ikisi | -0.0124 | -0.0119 | 3 | yes | +0.1100 | +0.1071 |
| xlm15 | 500 | baza → turk_qarisiq | -0.0008 | +0.0021 | 2 | NO | +0.0025 | +0.0064 |
| xlm15 | 500 | baza → tokenizator | -0.0991 | -0.0937 | 2 | yes | -0.1008 | -0.0971 |
| xlm15 | 500 | baza → transplant_mean | +0.0056 | +0.0113 | 1 | yes | -0.0573 | -0.0550 |
| xlm15 | 500 | baza → transplant_random_coef | -0.0297 | -0.0418 | 1 | yes | -0.0068 | -0.0122 |
| xlm15 | 500 | transplant_random_coef → tokenizator | n/a | n/a | 0 | — | -0.0940 | -0.0849 |
| xlm15 | 500 | transplant_mean → tokenizator | -0.0128 | -0.0120 | 1 | yes | -0.0435 | -0.0422 |
| xlm15 | 2000 | baza → turk | -0.0119 | -0.0096 | 2 | yes | +0.1265 | +0.1207 |
| xlm15 | 2000 | baza → her_ikisi | +0.0022 | -0.0024 | 2 | NO | +0.1913 | +0.1855 |
| xlm15 | 2000 | baza → turk_qarisiq | n/a | n/a | 0 | — | -0.0049 | -0.0076 |
| xlm15 | 2000 | baza → tokenizator | +0.0156 | -0.0082 | 1 | NO | +0.0224 | +0.0189 |
| xlm15 | 2000 | baza → transplant_mean | +0.0264 | +0.0106 | 2 | yes | +0.0708 | +0.0643 |
| xlm15 | 2000 | baza → transplant_random_coef | -0.0067 | -0.0305 | 2 | yes | +0.1227 | +0.1099 |
| xlm15 | 2000 | transplant_random_coef → tokenizator | -0.0539 | -0.0465 | 3 | yes | -0.1003 | -0.0910 |
| xlm15 | 2000 | transplant_mean → tokenizator | -0.0001 | -0.0043 | 2 | yes | -0.0484 | -0.0454 |
| xlm15 | 10000 | baza → tokenizator | +0.0206 | +0.0170 | 2 | yes | +0.1594 | +0.1492 |
| xlm15 | 10000 | baza → transplant_mean | n/a | n/a | 0 | — | -0.1289 | -0.1211 |
| xlm15 | 10000 | baza → transplant_random_coef | -0.1137 | -0.1136 | 1 | yes | -0.1530 | -0.1503 |
| xlm15 | 10000 | transplant_random_coef → tokenizator | +0.1166 | +0.1174 | 1 | yes | +0.3124 | +0.2994 |
| xlm15 | 10000 | transplant_mean → tokenizator | +0.0051 | -0.0070 | 1 | NO | +0.2883 | +0.2702 |
| xlm15 | 20914 | baza → tokenizator | +0.0365 | +0.0302 | 1 | yes | -0.2517 | -0.2416 |
| xlm15 | 20914 | baza → transplant_mean | -0.0111 | -0.0155 | 3 | yes | -0.0111 | -0.0155 |
| xlm15 | 20914 | baza → transplant_random_coef | +0.0094 | +0.0019 | 1 | yes | -0.2594 | -0.2522 |
| xlm15 | 20914 | transplant_random_coef → tokenizator | n/a | n/a | 0 | — | +0.0077 | +0.0106 |
| xlm15 | 20914 | transplant_mean → tokenizator | +0.2123 | +0.1994 | 1 | yes | -0.2406 | -0.2261 |

## T3 · Generalisation gap (Δ = test − validation, matched weights)

- n runs: 164
- mean: -0.0126; median: -0.0131; sd: 0.0116
- range: -0.0385 … +0.0095
- negative (test < val): 117/164
- xlmr: mean -0.0202, median -0.0202, n=70
- xlm15: mean -0.0069, median -0.0014, n=94

## T4 · Sign agreement between validation and test contrasts

- **xlmr**: sign agrees 15/16 paired contrasts; 13/13 among contrasts with |Δ val| ≥ 0.01; Pearson r(Δ val, Δ test) = 0.9933; mean |Δ test| / |Δ val| = 0.869
- **xlm15**: sign agrees 18/22 paired contrasts; 13/14 among contrasts with |Δ val| ≥ 0.01; Pearson r(Δ val, Δ test) = 0.9907; mean |Δ test| / |Δ val| = 0.985
- **ALL**: sign agrees 33/38 paired contrasts; 26/27 among contrasts with |Δ val| ≥ 0.01; Pearson r(Δ val, Δ test) = 0.9898; mean |Δ test| / |Δ val| = 0.931

Every sign disagreement occurs where |Δ val| < 0.01 (numerically indistinguishable from zero) or where fewer than 3 seeds escape in both arms. No conclusion that survives Holm on validation reverses direction on test.


## T5 · Test-side confirmation of the 27 validation survivors

The 27 contrasts below are **not** chosen by looking at test. They are exactly the contrasts that survive Holm-Bonferroni on validation across the pre-registered family of 86 tests (`results/stats.csv`, `survives_family_correction = true`). The family is fixed by the validation analysis; the test column is a confirmation pass over that fixed set, Holm-corrected within the 27. This is validation-selection followed by test-confirmation, not test-side hypothesis search.

Every survivor is on **xlmr**, where 5/5 seeds escape in every cell — so conditional and all-seed macro-F1 coincide and none of these results depends on the choice of estimand.

| Base | n | Contrast | Δ val | Δ test | n | t (test) | p (test) | Holm within 27 | Confirmed |
|---|---:|---|---:|---:|---:|---:|---:|---:|:--:|
| xlmr | 500 | baza → tokenizator | -0.0782 | -0.0748 | 5 | -10.715 | 4.30e-04 | 5.20e-03 | YES |
| xlmr | 500 | baza → transplant_mean | -0.0560 | -0.0566 | 5 | -8.840 | 9.04e-04 | 8.54e-03 | YES |
| xlmr | 500 | baza → transplant_random_coef | -0.0996 | -0.0917 | 5 | -15.652 | 9.73e-05 | 2.04e-03 | YES |
| xlmr | 500 | her_ikisi → turk | +0.0889 | +0.0737 | 5 | +10.473 | 4.70e-04 | 5.20e-03 | YES |
| xlmr | 500 | her_ikisi → turk_qarisiq | +0.0793 | +0.0641 | 5 | +8.972 | 8.54e-04 | 8.54e-03 | YES |
| xlmr | 500 | tokenizator → turk | +0.0951 | +0.0875 | 5 | +12.263 | 2.54e-04 | 4.83e-03 | YES |
| xlmr | 500 | tokenizator → turk_qarisiq | +0.0855 | +0.0779 | 5 | +11.418 | 3.36e-04 | 5.07e-03 | YES |
| xlmr | 500 | transplant_mean → transplant_random_coef | -0.0436 | -0.0350 | 5 | -4.810 | 8.59e-03 | 1.72e-02 | YES |
| xlmr | 500 | transplant_mean → turk | +0.0729 | +0.0693 | 5 | +17.204 | 6.70e-05 | 1.61e-03 | YES |
| xlmr | 500 | transplant_mean → turk_qarisiq | +0.0633 | +0.0597 | 5 | +13.556 | 1.71e-04 | 3.43e-03 | YES |
| xlmr | 500 | transplant_random_coef → turk | +0.1165 | +0.1043 | 5 | +16.160 | 8.58e-05 | 1.89e-03 | YES |
| xlmr | 500 | transplant_random_coef → turk_qarisiq | +0.1069 | +0.0947 | 5 | +12.008 | 2.76e-04 | 4.96e-03 | YES |
| xlmr | 2000 | baza → her_ikisi | -0.0569 | -0.0424 | 5 | -7.314 | 1.86e-03 | 1.08e-02 | YES |
| xlmr | 2000 | baza → tokenizator | -0.0659 | -0.0566 | 5 | -10.918 | 4.00e-04 | 5.20e-03 | YES |
| xlmr | 2000 | baza → transplant_mean | -0.0505 | -0.0392 | 5 | -7.200 | 1.97e-03 | 1.08e-02 | YES |
| xlmr | 2000 | baza → transplant_random_coef | -0.0980 | -0.0816 | 5 | -36.967 | 3.20e-06 | 8.31e-05 | YES |
| xlmr | 2000 | her_ikisi → transplant_random_coef | -0.0411 | -0.0392 | 5 | -8.092 | 1.27e-03 | 9.53e-03 | YES |
| xlmr | 2000 | her_ikisi → turk | +0.0599 | +0.0502 | 5 | +11.153 | 3.68e-04 | 5.15e-03 | YES |
| xlmr | 2000 | her_ikisi → turk_qarisiq | +0.0561 | +0.0425 | 5 | +11.947 | 2.81e-04 | 4.96e-03 | YES |
| xlmr | 2000 | tokenizator → transplant_random_coef | -0.0320 | -0.0250 | 5 | -4.653 | 9.64e-03 | 1.72e-02 | YES |
| xlmr | 2000 | tokenizator → turk | +0.0690 | +0.0644 | 5 | +11.590 | 3.17e-04 | 5.07e-03 | YES |
| xlmr | 2000 | tokenizator → turk_qarisiq | +0.0651 | +0.0567 | 5 | +17.020 | 6.99e-05 | 1.61e-03 | YES |
| xlmr | 2000 | transplant_mean → transplant_random_coef | -0.0475 | -0.0424 | 5 | -6.928 | 2.28e-03 | 1.08e-02 | YES |
| xlmr | 2000 | transplant_mean → turk | +0.0535 | +0.0470 | 5 | +8.224 | 1.19e-03 | 9.53e-03 | YES |
| xlmr | 2000 | transplant_mean → turk_qarisiq | +0.0496 | +0.0393 | 5 | +7.376 | 1.80e-03 | 1.08e-02 | YES |
| xlmr | 2000 | transplant_random_coef → turk | +0.1010 | +0.0894 | 5 | +61.226 | 4.26e-07 | 1.15e-05 | YES |
| xlmr | 2000 | transplant_random_coef → turk_qarisiq | +0.0972 | +0.0817 | 5 | +34.301 | 4.31e-06 | 1.08e-04 | YES |

**27 of 27 confirmed on test** (same sign and Holm-adjusted p < 0.05 within the 27).
Largest test-side Holm-adjusted p in the set: 1.717e-02 (xlmr n=500, transplant_mean → transplant_random_coef).
Mean shrinkage |Δ test| / |Δ val| across the 27: 0.874.

No contrast reverses sign. The effects are uniformly slightly smaller on test, which is the expected direction for a checkpoint selected on validation.


## T6 · Exact-permutation floor — the honest limit of 5 seeds

The parametric paired t-tests above report p-values as small as 4e-07. Those numbers come from the normal model, not from the data's exchangeability. With 5 seeds a paired sign-flip permutation test has only 2^5 = 32 arrangements, so the smallest two-sided p it can produce is 2/32 = **0.0625** — above 0.05.

Exact paired permutation p, computed over all 32 sign flips, for the same 27 contrasts:

- validation: every one of the 27 attains the floor, p = 0.0625 (distinct values observed: 0.0625)
- test: same, p = 0.0625 (distinct values: 0.0625)
- seed-level sign unanimity: 27/27 on validation, 27/27 on test

**What this means, stated plainly.** Every one of the 27 contrasts is as extreme as 5 paired seeds can possibly make it: all five seed differences point the same way, in both splits, in all 27 contrasts. That is the maximum evidence this design can produce. But it also means no contrast can reach p < 0.05 under an exact paired test, and the pre-registered t-test p-values below 0.0625 are model-based, not distribution-free.

The pre-registered analysis (Welch and paired t with Holm over 86 tests) stands as specified and is what the paper reports. This section is the sensitivity note that must accompany it: the significance claims rest on the normality assumption, and the design's distribution-free ceiling with 5 seeds is 0.0625. Raising seeds is the only fix; it was not done, and the report should say so rather than let the 4e-07 stand unqualified.

