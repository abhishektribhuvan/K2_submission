# Business Entity Resolution Pipeline

## Overview

End-to-end ML pipeline for resolving business entities across three independent data sources. Given noisy, inconsistent business records from Source 1 (reference), Source 2, and Source 3, the pipeline determines which records refer to the same real-world business entity.

**Architecture:** Sparse TF-IDF Blocking → RapidFuzz Feature Engineering → LightGBM Classification

**Optimized for:** 16 GB RAM, 10-core Intel i5-13450HX

## Pipeline Stages

### 1. Text Preprocessing (`src/preprocess.py`)
- Unicode normalization and non-ASCII cleanup
- Business suffix removal (Inc, Corp, Ltd, Pvt, SARL, etc.)
- Address abbreviation expansion (Rd→Road, St→Street, etc.)
- Numeric token extraction (street numbers, postal codes)
- Combined name+address field creation for blocking

### 2. Candidate Generation / Blocking (`src/blocking.py`)
- **Method:** Sparse TF-IDF with character 3-gram/4-gram analysis
- **Hard filter:** Per-country blocking (S1 only matches S2/S3 in same country)
- **Sublinear TF scaling** (BM25-like behavior)
- **Top-15 candidates** per Source 1 entity via cosine similarity
- Output: `output/candidate_pairs.tsv`

### 3. Feature Engineering (`src/features.py`)
14 language-agnostic pairwise features using `rapidfuzz`:
- **Name features (5):** Token Sort Ratio, Levenshtein Ratio, Jaccard Token Overlap, Partial Ratio, Token Set Ratio
- **Address features (5):** Token Sort Ratio, Levenshtein Ratio, Jaccard Overlap, Character N-gram Cosine, Partial Ratio
- **Numeric features (2):** Address Numeric Jaccard, Exact Numeric Match flag
- **Combined features (2):** Combined Token Sort Ratio, Country Exact Match flag

### 4. Model Training (`src/train.py`)
- **Split:** 80/20 stratified train/validation (preserving singleton ratio)
- **Negative sampling:** Controlled ratio from blocking negatives
- **Classifier:** LightGBM with `scale_pos_weight` for class imbalance
- **Threshold sweep:** Macro-averaged F₀.₅ optimization across [0.50, 0.95]
- **Early stopping:** On validation binary logloss

### 5. Memory-Safe Inference (`src/infer.py`)
- **Streaming batches:** 50,000 Source 1 rows per batch
- **Per-batch pipeline:** Block → Features → Predict → Threshold → Write
- **Validation:** Completeness check (every S1 entity in output) + duplicate detection
- Output: `output/matching_results.tsv`

## Quick Start

### Prerequisites

```bash
# Python 3.9+ required
pip install -r requirements.txt
```

### Directory Structure

```
ML_challange/
├── data/student_resource/dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── code/business_entity_resolution/   ← Run from here
│   ├── src/
│   ├── requirements.txt
│   └── README.md
└── output/                            ← Created at runtime
    ├── matching_results.tsv
    └── candidate_pairs.tsv
```

### Running the Full Pipeline

```bash
cd code/business_entity_resolution

# Full pipeline (train + inference):
python -m src.main

# Quick test with 1% data sample:
python -m src.main --sample 0.01

# Training only:
python -m src.main --train-only

# Inference only (after training):
python -m src.main --infer-only

# Custom batch size:
python -m src.main --batch-size 25000
```

### Expected Runtime

| Stage | Time (approx.) |
|-------|----------------|
| Preprocessing | 15-30 min |
| Blocking (TF-IDF build) | 20-40 min |
| Feature Engineering | 30-60 min |
| LightGBM Training | 5-10 min |
| Threshold Sweep | 2-5 min |
| Inference (all batches) | 60-120 min |
| **Total** | **~2-4 hours** |

## Evaluation Metric

**Macro-averaged F₀.₅** (precision-heavy):

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

- Computed per Source 1 entity, then averaged
- Singletons score 1.0 when correctly predicted as empty
- False merges are penalized 2× more than missed matches

## Key Design Decisions

1. **Character n-grams over word tokens:** Robust to typos, abbreviations, and transliteration noise across US, Indian, and French business names.

2. **Per-country blocking:** Eliminates cross-country false positives while being label-agnostic (no hardcoded country sets).

3. **Geographically agnostic features:** All 14 features are pure string-geometric comparisons — no language models, no geocoding, no regex for specific postal formats.

4. **Sparse operations throughout:** TF-IDF uses `scipy.sparse` matrices; never materializes dense 10M×10M similarity matrices.

5. **Streaming inference:** 50K-row batches keep peak RAM well under 16GB even with 1.7M Source 1 test entities.

## Output Format

Both output files are tab-separated with these exact columns:

**`matching_results.tsv`:**
```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S3-00812
S1-00002	
S1-00003	S2-00193
```

**`candidate_pairs.tsv`:**
```
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	
S1-00003	S2-00193,S3-00555
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| polars | 1.30.0 | Memory-efficient DataFrame operations |
| rapidfuzz | 3.12.2 | Fast fuzzy string matching (C extension) |
| lightgbm | 4.6.0 | Gradient-boosted decision tree classifier |
| scikit-learn | 1.6.1 | TF-IDF vectorizer, cosine similarity |
| scipy | 1.15.2 | Sparse matrix operations |
| numpy | 2.2.5 | Numerical computations |
| tqdm | 4.67.1 | Progress bars |
| joblib | 1.4.2 | Parallel processing utilities |

All dependencies are MIT or Apache 2.0 licensed. LightGBM is MIT licensed with <1M parameters.

## License

Model: LightGBM (MIT License, <8B parameters ✓)
