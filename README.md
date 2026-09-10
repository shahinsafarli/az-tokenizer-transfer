# Türkcədən Azərbaycan Dilinə Transfer: Dil Biliyi, Yoxsa Tokenizasiya Effekti?

Deep Learning Final Project · Track 1 (Pure Research) · DLE-AI-202, Cohort I 2026
AI Academy · National Artificial Intelligence Center

---

## Tədqiqat sualı

Kardeş-NLU (EACL 2024) göstərib ki, modeli əvvəlcə türkcə, sonra azərbaycanca
fine-tune etmək nəticəni yaxşılaşdırır — amma **niyə** işlədiyini izah etməyib.

Biz soruşuruq: **türk mərhələsini tamamilə çıxarıb, sadəcə modelin tokenizatorunu
Azərbaycan dilinə uyğunlaşdırsaq, həmin faydanın nə qədərini geri qaytara bilərik?**

## Eksperiment dizaynı (2×2 + nəzarət)

| № | Şərt | Tokenizator | Pipeline |
|---|------|-------------|----------|
| 1 | `baza` | orijinal XLM-R | birbaşa AZ fine-tune |
| 2 | `turk` | orijinal XLM-R | TR fine-tune → AZ fine-tune |
| 3 | `tokenizator` | OMP-transplant | birbaşa AZ fine-tune |
| 4 | `her_ikisi` | OMP-transplant | TR fine-tune → AZ fine-tune |
| 5 | `turk_qarisiq` | orijinal XLM-R | **qarışdırılmış** TR → AZ fine-tune (nəzarət C4) |

**Əsas nəticə:**

```
bərpa_nisbəti (%) = (Şərt3 − Şərt1) / (Şərt2 − Şərt1) × 100
```

> ⚠️ **Terminologiya.** Bu, *əvəzedicilik* ölçüsüdür, *parçalanma* deyil.
> Şərt 2-də tokenizator dəyişmir — orada türkcə fine-tune AZ mətninin işlətdiyi
> **paylaşılan subword-lərin embedding-lərini isindirir**. Buna görə
> «türk faydasının X%-i tokenizasiyadandır» **DEMƏYİN**. Düzgün ifadə:
> «tokenizator adaptasiyası türk mərhələsinin faydasının X%-ni təkbaşına bərpa edir».
> Leksik və sintaktik effektləri ayırmaq üçün Şərt 5 var.

---

## Sürətli başlanğıc

```bash
git clone <repo-url> && cd az-tokenizer-transfer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# ADDIM 0 — configs/experiment.yaml-da "FILL" sahələrini doldurun (aşağı bax)
nano configs/experiment.yaml

pytest -q                    # vahid testləri (model yükləmir, ~5 saniyə)
bash run_all.sh              # tam pipeline
```

### Headline nəticənin bir əmrlə təkrarı

```bash
bash run_all.sh
# → results/decompose.json  →  cross_base_comparison
# → results/paper_tables.md (hər xanа: escape rate + şərti macro-F1)
```

`cross_base_comparison.cross_base` iki bazanın transplant effektlərinin
FƏRQİDİR — məqalənin baş rəqəmi budur.

---

## ADDIM 0 — konfiqurasiyanın doldurulması (məcburi)

`configs/experiment.yaml` faylında `FILL` yazılmış sahələr:

| Sahə | Nə yazılmalı |
|------|--------------|
| `data.az.hf_name` | Azərbaycan tapşırıq dataseti (HF adı) |
| `data.az.text_column` / `label_column` | həmin datasetin sütun adları |
| `data.tr.hf_name` | Türk tapşırıq dataseti |
| `data.tr.text_column` / `label_column` | həmin datasetin sütun adları |

Doldurulmasa skriptlər aydın xəta ilə dayanır (`src/utils.py::_check_fill`).

Sütun adlarını bilmirsinizsə:

```python
from datasets import load_dataset
ds = load_dataset("<ad>", split="train")
print(ds.column_names, len(ds))
print(ds[0])
```

---

## Addım-addım icra

| # | Əmr | Nə edir | GPU | Vaxt |
|---|-----|---------|-----|------|
| 1 | `python -m src.tokenization.anchors --config configs/experiment.yaml` | **GO/NO-GO qapısı** — ortaq token sayı | yox | ~1 dəq |
| 2 | `python -m src.tokenization.fertility --config ...` | Şəkil 1 üçün ölçmələr | yox | ~1 dəq |
| 3 | `python -m src.data.splits --config ...` | train/val/test fiksasiyası | yox | ~2 dəq |
| 4 | `python -m src.data.scramble --config ...` | qarışdırılmış TR (C4) | yox | ~10 san |
| 5 | `python -m src.transplant.build --config ...` | OMP transplantı | opsional | 5–20 dəq |
| 6 | `python -m src.transplant.controls --config ... --transplanted artifacts/transplanted__omp_k64` | **C1 nəzarətləri** | opsional | ~3 dəq |
| 6b | `python -m src.transplant.top1_accuracy --config ...` | top-1 MLM diaqnostikası [C1c] | opsional | ~2 dəq |
| 6c | `python -m src.transplant.cross_base_quality --config ...` | çarpaz-baza transplant keyfiyyəti | yox | ~10 san |
| 7a | `python -m src.training.run_grid --config ... --phase 1` | türk keşi (30 checkpoint, SERİAL) | **bəli** | 4–6 saat |
| 7b | `python -m src.training.run_grid --config ... --phase 2 --streams N` | 164 fine-tuning run-u | **bəli** | ~3.4 saat (H100, 8 axın) |
| 8 | `python -m src.analysis.{aggregate,stats,decompose} --config ...` | analiz | yox | ~2 dəq |
| 9 | `python -m src.analysis.report --config ...` | cədvəllər + Şəkillər | yox | ~1 dəq |

Tranche A (run plan §7 — 10 run, türk mərhələsi YOX, ~3 saat A100-də):

```bash
bash run_all.sh --tranche-a
```

### 1-ci addım kritikdir — GO/NO-GO

```
verdict = GO             ortaq anchor ≥ 20 000   → davam
verdict = GO_WITH_CARE   5 000 – 20 000          → transplant.k-nı 8–32-yə salın
verdict = NO_GO          < 5 000                 → DAYANIN, donoru dəyişin
```

Başqa donor sınamaq:

```bash
python -m src.tokenization.anchors --config configs/experiment.yaml --donor <hf/model>
```

---

## Nəzarət eksperimentləri — bunlar olmadan nəticələr şərh edilə bilməz

| Kod | Nə yoxlayır | Harada |
|-----|-------------|--------|
| **C1a** | Eyniyyət transplantı bit-bərabərdirmi? (kod düzgünlüyü) | `src/transplant/controls.py` |
| **C1b** | OMP yenidənqurma keyfiyyəti (kosinus) | `src/transplant/omp.py::reconstruction_error` |
| **C1c** | Transplantın öz zərəri — **BPC** (perplexity DEYİL) | `src/transplant/controls.py` |
| **C2** | Anchor örtüşməsi kanonik formada | `src/tokenization/{canon,anchors}.py` |
| **C3** | TR mərhələsindən sonra təsnifat başı atılır | `src/training/finetune.py` |
| **C4** | Qarışdırılmış türkcə — leksik vs sintaktik | `src/data/scramble.py` |

**Niyə BPC, perplexity yox:** fərqli tokenizatorlar mətni fərqli sayda tokenə
bölür, ona görə token başına perplexity müqayisə oluna bilməz. BPC hərfə görə
normallaşdırılıb və müqayisə edilə biləndir.

**Ablasiyalar:**

```bash
python -m src.transplant.build --config ... --method mean        --tag mean_init
python -m src.transplant.build --config ... --method random_coef --tag rand_coef
python -m src.transplant.build --config ... --k 8   --tag omp_k8
python -m src.transplant.build --config ... --k 128 --tag omp_k128
```

---

## Yarıda qalma planı (brief tövsiyəsi)

GPU pəncərəsi bitə bilər. Prioritet sırası koda daxildir
(`run.priority_first: true`): əvvəlcə **bütün şərtlərin 2000 nümunə** variantı
işlədilir, beləliklə tam müqayisə cədvəliniz olur.

```bash
python -m src.training.orchestrate --config ... --only-priority
```

**Minimum zəmanətli nəticə** (≈1 saat): yalnız Şərt 1 və 3 —

```bash
python -m src.training.orchestrate --config ... --only-priority \
    --conditions baza,tokenizator
```

Resume avtomatikdir: `run.skip_existing: true` — mövcud nəticə faylları atlanır,
eyni əmri təkrar işlədin.

---

## Repo strukturu

```
configs/experiment.yaml        bütün parametrlər, seed-lər, yollar
run_all.sh                     tək giriş nöqtəsi (brief tələbi)
src/
  utils.py                     konfiq, seed, IO
  tokenization/canon.py        ▁ / ## normalizasiyası           [C2]
  tokenization/anchors.py      GO/NO-GO qapısı                  [C2]
  tokenization/fertility.py    Şəkil 1 ölçmələri
  transplant/omp.py            OMP nüvəsi (əmsal köçürməsi)
  transplant/build.py          transplant olunmuş model
  transplant/controls.py       eyniyyət + BPC                   [C1]
  data/splits.py               sabit bölgü (dedup BÖLGÜDƏN ƏVVƏL)
  data/scramble.py             qarışdırılmış türkcə             [C4]
  data/audit.py                əl ilə etiket auditi (export/verify/score) [T6]
  data/truncation.py           kəsmə konfoundunun ölçülməsi     [T5]
  data/profile_candidates.py   dataset namizədlərinin profillənməsi [T4]
  transplant/eval_text.py      C1c/top-1 üçün TƏK korpus mənbəyi
  training/finetune.py         tək run (baş atılması daxil)     [C3]
  training/phases.py           iki fazalı icra + paralel axınlar
  training/run_grid.py         qridin giriş nöqtəsi (faza 1/2, tranche)
  training/orchestrate.py      növbə qurulması, serial icra, resume
  analysis/aggregate.py        orta ± std
  analysis/stats.py            t-test, bootstrap CI, Cohen's d
  analysis/decompose.py        BƏRPA NİSBƏTİ (əsas nəticə)
  analysis/errors.py           səhv taksonomiyası
  analysis/figures.py          Şəkil 1–4
tests/                         86 test (vahid + qrid + paralellik + ucdan-uca)
notebooks/colab_t4.ipynb       Tranche A — pulsuz Colab T4
notebooks/a100.ipynb           tam qrid — A100
docs/COMPUTE_ESTIMATES.md      vaxt / VRAM / disk / xərc — çıxarılışı ilə
results/                       JSON/CSV (commit edilir)
figures/                       PNG (commit edilir)
```

## Notebook-lar

**Yeni başlayan üçün:** `notebooks/START_HERE.md` — Colab-ı heç işlətməmiş
adam üçün addım-addım təlimat.

| Notebook | Nə edir | Vaxt | Xərc |
|---|---|---|---|
| **`notebooks/colab_t4_pilot.ipynb`** | **GO/NO-GO probu — 2×2 çarpaz-baza kontrastı (12 run, 800 addım)** | **~5–6 saat** | **$0** |
| `notebooks/colab_t4.ipynb` | Doğrulama + data + transplant + **Tranche A** (resume ilə) | 6.3–8.4 saat (2–3 sessiya) | $0 |
| `notebooks/a100.ipynb` | Benchmark → Tranche A → qərar → Faza 1 → Faza 2 → analiz | 41–55 saat (1 axın) | kurs pəncərəsi |
| `notebooks/vastai_h100.ipynb` | Eyni ardıcıllıq, saatlıq kirayə GPU üçün | **~10.4 saat** | **~$14** |

### Pilot — ən ucuz qərar (pulsuz T4)

`configs/pilot_t4.yaml` DONDURULMUŞ konfiq DEYİL və `results_pilot/`-a yazır —
nəticələri hesabata GETMİR və `load_runs()` onları qrid nəticələri ilə
qarışdıra BİLMİR. Məqsədi tək sual: **tam qridi işlətməyə dəyərmi?**

12 run = 2 baza × {baza, tokenizator} × 3 seed, 800 addım. Bu, M3 imzasını
(transplant AZ görməmiş bazaya kömək edir, görmüşə etmir) göstərə bilən ƏN
KİÇİK hüceyrə dəstidir — və eyni zamanda transplantın HƏR İKİ bazaya kömək
etməsi halını tutur, ki bu, effektin tokenizasiya defisitindən yox,
əməliyyatın özündən gəldiyi demək olardı.

Qurduğu transplant artefaktı KANONİKDİR (`omp_k64_rescaled`) — yəni pilotun
ən bahalı saatı tam qrid tərəfindən TƏKRAR İSTİFADƏ olunur.

### Kirayə GPU (vast.ai) — qısa xülasə

Bu iş yükü **kernel-launch-bound**-dur (cümlələr orta 29.6 token), ona görə
daha güclü GPU demək olar heç nə vermir: işlək kartların HAMISI 10.0–10.6 saat
aralığındadır (**4% fərq**), qiymət isə **7× fərqlənir**.

| GPU | Pinlənmiş stack işləyir? | Cəmi | Xərc (from) |
|---|---|---:|---:|
| **H100 SXM** | bəli (`sm_90`) | **~10.4 s** | **~$14** ← bunu kirayələyin |
| H100 NVL | bəli | ~10.5 s | ~$16 |
| H100 PCIe | bəli | ~10.6 s | ~$27 |
| H200 / H200 NVL | bəli | ~10.4 s | ~$38–40 |
| **B200 / B300** | **XEYR — `sm_100`** | — | $53–100 |

⚠️ `requirements.lock` `torch==2.4.1+cu121` pinləyir; **CUDA 12.1-də Blackwell
(sm_100) kernel-ləri YOXDUR** — B200/B300 "no kernel image is available"
verəcək. Onlara çatmaq üçün torch-u yeniləmək versiya kilidini pozar.

Tam çıxarılış və instance seçimi (vCPU ≥ 2/axın, disk ≥ 60 GB, interruptible
təhlükəsizdir): `docs/COMPUTE_ESTIMATES.md` §8.

Pulsuz T4 qrid maşını DEYİL: tam qrid orada 81–108 saat çəkər. T4 notebook-u
`results/` və keşi Drive-a bağlayır, ona görə kəsilən sessiya itirilmiş iş yox,
sadəcə fasilədir (`run.skip_existing`).

## Təkrarlanabilirlik

- Bütün seed-lər `configs/experiment.yaml`-dan oxunur
  (`experiment.seeds`, `data.split.seed`)
- `torch.use_deterministic_algorithms(True)`, `cudnn.deterministic = True`
- Data bölgüsü seed-i model seed-lərindən **ayrıdır** → fərqli modellər eyni bölgüdə
- Test dəstinə **yalnız bir dəfə**, yekun nəticələr üçün baxılır
- `requirements.txt` pinned

## Kompüter büdcəsi

Tam rəqəmlər və onların NECƏ çıxarıldığı: **`docs/COMPUTE_ESTIMATES.md`**.
Aşağıdakılar pre-registrasiya olunmuş konfiqurasiya üçündür:
`batch_size: 32`, `grad_accum: 1` (2026-09-04 accumulation düzəlişi
2026-09-05-də GERİ ALINDI — `configs/FROZEN.md`).

Faza 2 sətri PROYEKSİYA deyil, ÖLÇMƏDİR: 2026-09-06 tarixli 114-run qridi
H100 SXM-də 8 axınla 8,614 saniyə (2.39 saat) çəkdi; per-run orta 592 s
8-yollu rəqabət altında qeydə alınıb və təcrid olunmuş dəyər DEYİL
(`results/launcher_state_phase2.json`). 164 run bu ötürmə qabiliyyəti ilə
164/114 × 2.39 ≈ 3.4 saatdır.

| Mərhələ | İş | H100 SXM (8 axın) | VRAM/proses |
|---------|----|------|------|
| Data + 6 transplant artefaktı | prep | ~2 saat (əsasən CPU) | — |
| Faza 1 — TR keşi | 30 checkpoint · 37,500 addım | ~0.5–1 saat | 8.6 GB |
| Faza 2 — AZ fine-tuning | 164 run · 328,000 addım | **~3.4 saat** (ölçülmüş sürətlə) | 8.6 GB |
| **Cəmi (soyuq instans)** | — | **~6 saat** | **8.6 GB/proses** |

Serial ekvivalent: 164 × 592 s ≈ **27 GPU-saat**, brifin ~2 günlük
pəncərəsinin içindədir; tək axının zirvəsi 8.6 GB, ~10–12 GB həddinin
altındadır. 8 axın ancaq ona görə işlədilir ki, icarəyə götürülmüş kartın
HAMISI bizimdir — paylaşılan A100 iş stansiyasında `run.vram_ceiling_gb`
axın sayını 1-ə endirir.

⚠️ **Brifin həddi kartın tutumu DEYİL — komandaya ayrılmış ~10–12 GB-dır.**
Proyeksiya olunan ~5.5 GB/proses o deməkdir ki, **2 axın ~11 GB, 3 axın
~16.5 GB** tutar — 3 axın ayrılmış büdcəyə SIĞMIR. `run.vram_ceiling_gb`
bunu məcbur edir: launcher `parallel_streams`-i real sığana qədər AZALDIR.
Həqiqi rəqəmi `bash scripts/a100_benchmark.sh` ÖLÇÜR — qrid başlamazdan
əvvəl onu konfiqə yazın.

Batch 32 / accum 1 (~8.6 GB) accumulation-lı variantdan SÜRƏTLİDİR;
accumulation sürət yox, **ehtiyat marja** alır — ona görə geri alındı.
Müqayisə `docs/COMPUTE_ESTIMATES.md` §3-dədir; qərarı Tranche A ölçməsi verir.

## Lisenziya və data

Hər dataset üçün mənbə və lisenziya `results/splits.json`-a yazılır və məqalədə
göstərilir. Heç bir ToS-pozan scraping istifadə edilmir.

## Referanslar

- Kardeş-NLU (EACL 2024) — https://aclanthology.org/2024.eacl-long.100/
- Training-Free Tokenizer Transplantation via OMP — https://arxiv.org/abs/2506.06607
- mergekit / tokensurgeon — https://github.com/arcee-ai/mergekit
- Open Foundation Models for Azerbaijani (aLLMA) — https://aclanthology.org/2024.sigturk-1.2/
