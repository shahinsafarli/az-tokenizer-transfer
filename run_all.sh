#!/usr/bin/env bash
# =====================================================================
#  BİR ƏMRLƏ TAM TƏKRAR  —  brief məcburi tələbi
#  ("one-command reproduction of your headline result")
#
#  İşlətmə:
#      bash run_all.sh                    # tam pipeline (164 run)
#      bash run_all.sh --smoke            # növbəni göstər, heç nə işlətmə
#      bash run_all.sh --tranche-a        # run plan §7 Tranche A (10 run, ~2 saat)
#      bash run_all.sh --prep-only        # data + transplant + nəzarətlər, sonra DAYAN
#      bash run_all.sh --analysis-only    # mövcud results/ üzərindən cədvəl və qrafiklər
#
#  Ətraf mühit dəyişənləri:
#      CONFIG=configs/experiment.yaml     konfiq yolu
#      STREAMS=3                          Faza 2 paralel axın sayı (VRAM tavanı ilə məhdudlaşır)
#      DEVICE=cuda                        transplant/diaqnostika cihazı
# =====================================================================
set -euo pipefail

# 2026-09-04 observability fix: a run stalled for 41 min without a single
# validation pass and stdout gave no step-level evidence because it was fully
# buffered. Force unbuffered stdio on every launch from this script.
export PYTHONUNBUFFERED=1
# 2026-09-04 memory-pressure fix (Step 3 of the stall diagnostic; not a
# pre-registered scientific parameter — see configs/FROZEN.md).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-configs/experiment.yaml}"
STREAMS="${STREAMS:-1}"
DEVICE="${DEVICE:-cuda}"
MODE="${1:-full}"
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


log() { printf '\n\033[1;35m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- analiz
# Tək yerdə saxlanılır: həm tam pipeline, həm --analysis-only eyni addımları
# işlədir, ona görə ikisi heç vaxt bir-birindən sürüşə bilməz.
run_analysis() {
  log "Analiz — aqreqasiya, statistika, dekompozisiya"
  python -m src.analysis.aggregate  --config "$CONFIG"
  python -m src.analysis.stats      --config "$CONFIG"
  python -m src.analysis.decompose  --config "$CONFIG"

  log "Məqalə cədvəlləri və qrafikləri (hamısı results/-dan)"
  # HEADLINE nəticə BURADA yaranır, ona görə `errors`-dan ƏVVƏLDİR: səhv
  # taksonomiyası əlavə analizdir və onun uğursuzluğu brifin tələb etdiyi
  # "bir əmrlə təkrarlanan headline nəticə"-ni bloklamamalıdır.
  # `report` cədvəlləri VƏ qrafikləri birlikdə yaradır və hər ölçülmüş rəqəmin
  # arxasındakı fayl yollarını yazır; `figures` onun içindən çağırılır.
  python -m src.analysis.report --config "$CONFIG"

  # 2026-09-08: the token-overlap control feeds a paper figure but was never
  # invoked by any launcher, so its raw export did not exist and that figure
  # could not be reproduced from a clean checkout. Non-fatal: it is a
  # diagnostic, and the brief's one-command requirement is about the HEADLINE.
  python -m src.tokenization.overlap_control --config "$CONFIG" \
    || log "overlap_control failed — Figure 3 raw export missing (non-fatal)"

  log "Səhv taksonomiyası (əlavə analiz — yalnız test qiymətləndirməsindən sonra dolur)"
  python -m src.analysis.errors --config "$CONFIG"
}

if [[ "$MODE" == "--analysis-only" ]]; then
  run_analysis
  log "BİTDİ (yalnız analiz)"
  exit 0
fi

log "0/9  Mühit yoxlanışı"
python -c "import torch,transformers,sklearn,scipy; \
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
python -c "import sys; sys.path.insert(0,'.'); \
from src.utils import verify_config_hashes; verify_config_hashes(); \
print('frozen config hashes OK')"

log "1/9  Anchor analizi  [C2 · GO/NO-GO qapısı]"
python -m src.tokenization.anchors --config "$CONFIG"

log "2/9  Fertility və token örtüşməsi  [Şəkil 1]"
python -m src.tokenization.fertility --config "$CONFIG"

log "3/9  Data bölgüsünün fiksasiyası"
python -m src.data.splits --config "$CONFIG"

log "4/9  Qarışdırılmış türkcə  [C4 · Şərt 5]"
python -m src.data.scramble --config "$CONFIG"

log "4b/9 Truncation konfoundunun ölçülməsi  [T5]"
python -m src.data.truncation --config "$CONFIG"

log "5/9  OMP tokenizator transplantı + placebo variantları"
# Şərt 3 (əsas) və Şərt 6/7 (mean / random_coef placeboları) — HƏR ÜÇÜ
# dondurulmuş növbənin şərtləridir, ona görə üçü də burada qurulur.
python -m src.transplant.build --config "$CONFIG"
python -m src.transplant.build_rescale_variant --config "$CONFIG"
python -m src.transplant.build --config "$CONFIG" --method mean --tag mean_k64
python -m src.transplant.build_rescale_variant --config "$CONFIG" --method mean
python -m src.transplant.build --config "$CONFIG" --method random_coef --tag random_coef_k64
python -m src.transplant.build_rescale_variant --config "$CONFIG" --method random_coef

log "6/9  Nəzarət eksperimentləri  [C1]  — hər baza model üçün, KANONİK artefakt"
# Transplant artefaktı baza modelə spesifikdir VƏ kanonik variant rescale
# olunmuşdur (`canonical_transplant_tag`) — nəzarətlər faktiki işlədilən
# artefaktı ölçməlidir, no-rescale ablasiya baseline-ını yox.
mapfile -t TP_SPECS < <(python -c "
import sys; sys.path.insert(0,'.')
from src.utils import load_config, resolve_bases, transplant_dir, canonical_transplant_tag
c = load_config('$CONFIG', require_complete=False)
for b in resolve_bases(c):
    print(b.role, transplant_dir(c, b.short, canonical_transplant_tag(c)))
")
for spec in "${TP_SPECS[@]}"; do
  role="${spec%% *}"; dir="${spec#* }"
  echo "  → $role : $dir"
  python -m src.transplant.controls --config "$CONFIG" \
      --base "$role" --transplanted "$dir" --device "$DEVICE"
done

# 2026-09-07: the same C1c probe on the two PLACEBO artifacts, so each method
# gets its OWN `controls__<base>__<tag>.json`. Previously only the canonical
# artifact was probed here, which meant a per-method delta BPC could only be
# obtained by re-running this step by hand — and every such run overwrote the
# one base-only filename, leaving no un-clobbered per-method record. These are
# forward passes only (no training): the whole loop is minutes.
# The base-only filename is now reserved for the canonical artifact, so these
# calls cannot displace the OMP numbers.
log "6a-bis/9  C1c (BPC + top-1) on the mean / random_coef placebos"
for role in primary contrast; do
  for tag in mean_k64_rescaled random_coef_k64_rescaled; do
    pdir="artifacts/transplanted__$(python -c "
import sys; sys.path.insert(0,'.')
from src.utils import load_config, resolve_bases, transplant_tag
c = load_config('$CONFIG', require_complete=False)
b = resolve_bases(c, ['$role'])[0]
print(transplant_tag(b.short, '$tag'))
")"
    if [[ -d "$pdir" ]]; then
      echo "  → $role : $pdir"
      python -m src.transplant.controls --config "$CONFIG" \
          --base "$role" --transplanted "$pdir" --device "$DEVICE"
    else
      echo "  → $role : $pdir YOXDUR, atlanır"
    fi
  done
done

log "6b/9  BPC ilə birlikdə top-1 MLM diaqnostikası (EYNİ korpus)"
python -m src.transplant.top1_accuracy --config "$CONFIG" --device "$DEVICE"

log "6c/9  Embedding norm hesabatı və çarpaz-baza transplant keyfiyyəti"
python -m src.transplant.check_embedding_norms --config "$CONFIG"
python -m src.transplant.cross_base_quality --config "$CONFIG"

if [[ "$MODE" == "--prep-only" ]]; then
  log "Hazırlıq tamamlandı; fine-tuning QƏSDƏN başladılmadı"
  echo "  Transplant keyfiyyəti: results/cross_base_transplant_quality.json"
  exit 0
fi

log "7/9  Fine-tuning — iki fazalı icra (eval_split=$EVAL_SPLIT)"
case "$MODE" in
  --smoke)
    python -m src.training.run_grid --config "$CONFIG" --dry-run
    echo "SMOKE: yalnız növbə göstərildi. Tam icra üçün arqumentsiz işlədin."
    exit 0
    ;;
  --tranche-a)
    # Run plan §7: türk mərhələsi YOXDUR, ona görə Faza 1 lazım deyil.
    python -m src.training.run_grid --config "$CONFIG" --tranche A --phase 2 --streams "$STREAMS" $TEST_FLAGS
    ;;
  *)
    python -m src.training.run_grid --config "$CONFIG" --phase 1
    python -m src.training.run_grid --config "$CONFIG" --phase 2 --streams "$STREAMS" $TEST_FLAGS
    ;;
esac

run_analysis

log "BİTDİ"
echo "  Nəticələr : results/"
echo "  Qrafiklər : figures/"
echo "  Cədvəllər : results/paper_tables.md"
echo "  Əsas rəqəm: results/decompose.json"
echo "               → cross_base_comparison.cross_base.escape_rate_effect_difference"
echo "               → cross_base_comparison.cross_base.conditional_macro_f1_effect_difference"
