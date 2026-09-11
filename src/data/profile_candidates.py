"""
DATASET NAMİZƏDLƏRİNİN PROFİLLƏNMƏSİ (T4)  —  brifin meyarlarına qarşı

Bu, generic Hub-axtarışı DEYİL (`src/data/discover.py` heç vaxt yazılmadı və
lazım da olmadı — AZ namizədləri artıq adlandırılmışdı, TR namizədləri isə
birbaşa Hub axtarışı ilə tapıldı). Bu skript SEÇİLMİŞ namizədləri brifin
§3 meyarlarına qarşı profilləyir və qərarı SƏNƏDLƏŞDİRİR.

Ölçülənlər:
  * dedup-dan SONRAKI sətir sayı (brifin ≥10k həddi buna görə oxunmalıdır)
  * sinif sayı və ƏN BÖYÜK sinfin payı (balanssızlıq)
  * lisenziya və gated statusu (brifin etika/lisenziya tələbi)
  * "headroom proxy" — TF-IDF + LogReg macro-F1. BU, YEKUN RƏQƏM DEYİL,
    məlum şəkildə PESSİMİST alt-hədddir; məqsədi tapşırığın nə tamamilə
    həll olunmuş (proxy > 0.85 → transformer üçün yer qalmır), nə də
    öyrənilməz (proxy < 0.60 → etiket küyü şübhəsi) olmadığını göstərməkdir
  * `readme_provenance` — README-də etiketlərin NECƏ yarandığına dair
    açar-söz axtarışı
  * `source_composition_by_label` — namizəd mənbə/provenans sütunu elan
    edirsə, hər etiket × mənbə çarpaz cədvəli

Sonuncu ikisi kosmetik deyil: onlar `hajili/...` (etiketlər emoji-qaydası ilə
yaradılıb, sonra emojilər mətndən SİLİNİB) və `winvoker/...` (Notr sinfi
99.7% Vikipediya mətnindən) namizədlərini rədd etməyə əsas verən şeydir —
hər ikisi meyar-sayına görə "keçirdi".

İşlətmə:
    python -m src.data.profile_candidates --config configs/experiment.yaml --lang all

Çıxış:
    results/dataset_candidates.json      (AZ)
    results/dataset_candidates_tr.json   (TR)
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.utils import (base_argparser, ensure_dir, get_logger, load_config,
                       setup_logging, write_json)

log = get_logger(__name__)

# Brifin §3 mətn/NLP həddi.
MIN_ROWS = 10_000
MAX_MAJORITY_SHARE = 0.90
HEADROOM_BAND = (0.60, 0.85)
MIN_CLASSES_AZ = 3      # yekun qiymətləndirmə dili — qütb + neutral lazımdır
MIN_CLASSES_TR = 2      # ara-mərhələ; təsnifat başı ATILIR [C3], ona görə parite LAZIM DEYİL

# README-də etiket provenansına işarə edən ifadələr.
PROVENANCE_MARKERS = ("emoji", "rule-based", "rule based", "automatically labeled",
                      "automatically labelled", "heuristic", "keyword", "weak supervision",
                      "distant supervision", "wikipedia", "translated", "synthetic")

# Namizədin mənbə/provenans sütunu ola bilən adlar.
SOURCE_COLUMN_CANDIDATES = ("dataset", "source", "origin", "provenance", "corpus", "domain")


def _headroom_proxy(texts: list[str], labels: list, seed: int = 0) -> float | None:
    """
    Simvol n-qram TF-IDF + LogReg macro-F1 (holdout). Məlum PESSİMİSTDİR —
    transformer bundan yuxarı çıxacaq — ona görə YALNIZ zolaq yoxlaması üçün.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline

    if len(set(labels)) < 2:
        return None
    try:
        x_tr, x_te, y_tr, y_te = train_test_split(
            texts, labels, test_size=0.2, random_state=seed, stratify=labels)
    except ValueError:
        return None
    pipe = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                        max_features=200_000, sublinear_tf=True),
        LogisticRegression(max_iter=1000, n_jobs=-1),
    )
    pipe.fit(x_tr, y_tr)
    return round(float(f1_score(y_te, pipe.predict(x_te), average="macro")), 4)


def _readme_provenance(hf_name: str) -> dict:
    """README-də etiket provenansına dair açar sözləri axtarır (yalnız bildirir)."""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(hf_name, "README.md", repo_type="dataset")
        text = Path(path).read_text(encoding="utf-8", errors="replace").casefold()
    except Exception as e:  # noqa: BLE001 — README olmaya bilər, bu, xəta deyil
        return {"available": False, "reason": type(e).__name__, "markers_found": []}
    found = sorted({m for m in PROVENANCE_MARKERS if m in text})
    return {
        "available": True,
        "markers_found": found,
        "note": ("Etiketlərin NECƏ yarandığını ƏL İLƏ oxuyun — bu, yalnız "
                 "diqqət çəkən açar sözlərdir, avtomatik qərar DEYİL."),
    }


def profile_candidate(hf_name: str, text_column: str, label_column: str,
                      *, hf_config: str | None = None, split: str = "train",
                      min_classes: int = 2, max_rows_for_proxy: int = 20_000,
                      seed: int = 0) -> dict:
    """Bir namizədi profilləyir. Şəbəkə tələb edir."""
    from datasets import load_dataset
    from huggingface_hub import HfApi

    from src.data.splits import _dedup

    out: dict = {"hf_name": hf_name, "text_column": text_column,
                 "label_column": label_column, "hf_config": hf_config, "split": split}

    try:
        info = HfApi().dataset_info(hf_name)
        card = info.card_data or {}
        out["licence"] = card.get("license") or card.get("licence") or "none declared"
        out["gated"] = bool(getattr(info, "gated", False))
    except Exception as e:  # noqa: BLE001
        out["licence"] = f"UNKNOWN ({type(e).__name__})"
        out["gated"] = None

    try:
        ds = load_dataset(hf_name, hf_config, split=split)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        out["criteria_passed"] = False
        return out

    out["columns"] = ds.column_names
    missing = [c for c in (text_column, label_column) if c not in ds.column_names]
    if missing:
        out["error"] = f"sütun tapılmadı: {missing}"
        out["criteria_passed"] = False
        return out

    feature = ds.features[label_column]
    to_name = getattr(feature, "int2str", None)
    records = [{"text": str(t).strip(), "label": (to_name(l) if to_name else l)}
               for t, l in zip(ds[text_column], ds[label_column])
               if t is not None and l is not None and str(t).strip()]
    out["rows_raw"] = len(records)

    deduped, dedup_stats = _dedup(records)
    out["dedup"] = dedup_stats
    out["rows_after_dedup"] = len(deduped)

    counts = Counter(str(r["label"]) for r in deduped)
    out["n_classes"] = len(counts)
    out["class_counts"] = dict(counts)
    majority = max(counts.values()) / max(len(deduped), 1) if counts else 1.0
    out["majority_class_share"] = round(majority, 4)

    # Provenans çarpaz cədvəli — namizəd mənbə sütunu elan edirsə.
    source_col = next((c for c in SOURCE_COLUMN_CANDIDATES if c in ds.column_names), None)
    if source_col:
        cross: dict[str, Counter] = {}
        for label, source in zip(ds[label_column], ds[source_col]):
            name = str(to_name(label) if to_name else label)
            cross.setdefault(name, Counter())[str(source)] += 1
        out["source_composition_by_label"] = {
            label: {"top_source": c.most_common(1)[0][0],
                    "top_source_share": round(c.most_common(1)[0][1] / sum(c.values()), 4),
                    "n_sources": len(c)}
            for label, c in cross.items()}
        out["source_composition_note"] = (
            "Bir sinfin ~hamısı TƏK mənbədən gəlirsə, model həmin mənbəni "
            "tanıya bilər — sinfi yox. Bu, konstruksiya-etibarlılığı defektidir.")

    subset = deduped[:max_rows_for_proxy]
    out["headroom_proxy_macro_f1"] = _headroom_proxy(
        [r["text"] for r in subset], [str(r["label"]) for r in subset], seed)
    out["headroom_proxy_note"] = (
        f"Simvol n-qram TF-IDF + LogReg; PESSİMİST alt-hədd. Hədəf zolaq "
        f"{HEADROOM_BAND[0]}–{HEADROOM_BAND[1]}.")
    out["readme_provenance"] = _readme_provenance(hf_name)

    criteria = {
        "rows_after_dedup_ge_min": out["rows_after_dedup"] >= MIN_ROWS,
        "n_classes_ge_min": out["n_classes"] >= min_classes,
        "majority_share_ok": majority <= MAX_MAJORITY_SHARE,
        "not_gated": out["gated"] is not True,
        "headroom_in_band": (
            out["headroom_proxy_macro_f1"] is not None
            and HEADROOM_BAND[0] <= out["headroom_proxy_macro_f1"] <= HEADROOM_BAND[1]),
    }
    out["criteria"] = criteria
    out["criteria_passed"] = all(criteria.values())
    out["criteria_note"] = (
        "Meyar SAYI qərar vermir — `readme_provenance` və "
        "`source_composition_by_label` ƏL İLƏ oxunmalıdır; hər ikisi "
        "meyarları keçən namizədləri rədd etməyə əsas ola bilər.")
    return out


CANDIDATES = {
    "az": [
        ("LocalDoc/sentiments_dataset_azerbaijani", "text", "labels"),
        ("hajili/azerbaijani_tweet_emotion_classification", "text", "label"),
    ],
    "tr": [
        ("maydogan/Turkish_SentimentAnalysis_TRSAv1", "review", "score"),
        ("winvoker/turkish-sentiment-analysis-dataset", "text", "label"),
        ("fthbrmnby/turkish_product_reviews", "sentence", "sentiment"),
    ],
}


def main() -> None:
    ap = base_argparser(__doc__)
    ap.add_argument("--lang", default="all", choices=["az", "tr", "all"])
    args = ap.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config, require_complete=False)
    results_dir = ensure_dir(cfg.experiment.results_dir)

    langs = ["az", "tr"] if args.lang == "all" else [args.lang]
    for lang in langs:
        min_classes = MIN_CLASSES_AZ if lang == "az" else MIN_CLASSES_TR
        profiles = []
        for hf_name, text_col, label_col in CANDIDATES[lang]:
            log.info("=" * 70)
            log.info("[%s] profillənir: %s", lang.upper(), hf_name)
            profile = profile_candidate(hf_name, text_col, label_col,
                                        min_classes=min_classes,
                                        seed=int(cfg.data.split.seed))
            profiles.append(profile)
            log.info("  sətir(dedup)=%s  sinif=%s  ən böyük sinif=%s  proxy=%s  keçdi=%s",
                     profile.get("rows_after_dedup"), profile.get("n_classes"),
                     profile.get("majority_class_share"),
                     profile.get("headroom_proxy_macro_f1"),
                     profile.get("criteria_passed"))

        filename = "dataset_candidates.json" if lang == "az" else f"dataset_candidates_{lang}.json"
        write_json({"lang": lang, "min_classes_required": min_classes,
                    "thresholds": {"min_rows": MIN_ROWS,
                                   "max_majority_share": MAX_MAJORITY_SHARE,
                                   "headroom_band": list(HEADROOM_BAND)},
                    "candidates": profiles},
                   results_dir / filename)
        log.info("Yazıldı: %s", results_dir / filename)


if __name__ == "__main__":
    main()
