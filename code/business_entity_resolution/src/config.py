"""
config.py - Central Configuration
===================================
All paths, thresholds, batch sizes, and hyperparameters for the
entity resolution pipeline. Designed for a 16GB RAM system.
"""

import os
from pathlib import Path

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# Base directory: assumes the pipeline is run from code/business_entity_resolution/
# and data lives at ../../data/student_resource/dataset/
BASE_DIR = Path(__file__).resolve().parent.parent  # code/business_entity_resolution/
PROJECT_ROOT = BASE_DIR.parent.parent  # ML_challange/

# Data paths - relative to project root
DATA_DIR = PROJECT_ROOT / "data" / "student_resource" / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

# Training files
TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

# Test files
TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# Output directory - created at runtime
OUTPUT_DIR = PROJECT_ROOT / "output"
MATCHING_RESULTS_FILE = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_FILE = OUTPUT_DIR / "candidate_pairs.tsv"

# Model artifacts directory
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "lgbm_entity_resolution.txt"
THRESHOLD_PATH = MODEL_DIR / "optimal_threshold.txt"

# Cache directory for intermediate Parquet files
CACHE_DIR = BASE_DIR / "cache"

# ============================================================================
# MEMORY & BATCH CONFIGURATION  (tuned for 16GB RAM)
# ============================================================================
# Streaming batch size for reading large TSV files with Polars
POLARS_BATCH_SIZE = 200_000

# Inference batch size for test_source1 processing
INFERENCE_BATCH_SIZE = 100_000

# Number of top candidates to retrieve per Source 1 entity during blocking
TOP_K_CANDIDATES = 15

# TF-IDF vectorizer batch size for fitting (to avoid OOM on 10M+ rows)
TFIDF_FIT_BATCH_SIZE = 500_000

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================
# Common business suffixes to normalize (language-agnostic)
BUSINESS_SUFFIXES = [
    # English
    r'\bincorporated\b', r'\bcorporation\b', r'\bcompany\b',
    r'\blimited\b', r'\bllc\b', r'\bllp\b', r'\bplc\b',
    r'\binc\b', r'\bcorp\b', r'\bco\b', r'\bltd\b',
    # Indian
    r'\bprivate\b', r'\bpvt\b', r'\bngo\b',
    # French (for unseen test data generalization)
    r'\bsarl\b', r'\bsas\b', r'\bsa\b', r'\beurl\b',
    r'\bsci\b', r'\bsnc\b', r'\bsociete\b',
    # Generic
    r'\bgroup\b', r'\bholdings?\b', r'\benterprise[s]?\b',
    r'\bservices?\b', r'\bsolutions?\b', r'\btechnolog(?:y|ies)\b',
    r'\binternational\b', r'\bglobal\b',
]

# Common address abbreviations to normalize (language-agnostic)
ADDRESS_ABBREVIATIONS = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'blvd': 'boulevard',
    'dr': 'drive', 'ln': 'lane', 'ct': 'court', 'pl': 'place',
    'cir': 'circle', 'hwy': 'highway', 'pkwy': 'parkway',
    'apt': 'apartment', 'ste': 'suite', 'fl': 'floor',
    'bldg': 'building', 'dept': 'department', 'rm': 'room',
    'n': 'north', 's': 'south', 'e': 'east', 'w': 'west',
    'ne': 'northeast', 'nw': 'northwest', 'se': 'southeast', 'sw': 'southwest',
    # French address terms
    'bd': 'boulevard', 'av': 'avenue', 'r': 'rue', 'pl': 'place',
    'imp': 'impasse', 'chem': 'chemin', 'rte': 'route',
}

# ============================================================================
# TF-IDF / BLOCKING CONFIGURATION
# ============================================================================
# Analyzer type for TF-IDF vectorizer ('word' for fast word-level tokens)
ANALYZER = 'word'

# Word n-gram range for TF-IDF vectorizer
NGRAM_RANGE = (1, 2)

# Maximum number of features for TF-IDF (controls memory)
MAX_FEATURES = 500_000

# Minimum document frequency for TF-IDF terms
MIN_DF = 2

# Maximum document frequency ratio for TF-IDF terms
MAX_DF = 0.25

# Sublinear TF scaling (log-normalized term frequencies, similar to BM25)
SUBLINEAR_TF = True

# ============================================================================
# LIGHTGBM HYPERPARAMETERS
# ============================================================================
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,        # Use all CPU cores
    'verbose': -1,       # Suppress LightGBM logs during training
    'is_unbalance': False,  # We'll use scale_pos_weight instead
}

# ============================================================================
# THRESHOLD SWEEP CONFIGURATION
# ============================================================================
# Range of thresholds to evaluate for F0.5 optimization
THRESHOLD_MIN = 0.50
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01

# ============================================================================
# TRAIN/VALIDATION SPLIT
# ============================================================================
VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42

# ============================================================================
# PARALLEL PROCESSING
# ============================================================================
# Number of CPU cores to use (-1 = all available)
N_JOBS = -1
# Actual core count for manual parallelism
N_CORES = min(os.cpu_count() or 4, 10)

# ============================================================================
# NEGATIVE SAMPLING
# ============================================================================
# Ratio of negative to positive samples during training
# Higher ratio = harder negatives, better precision
NEGATIVE_RATIO = 5


def ensure_directories():
    """Create all required output directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[CONFIG] Output directory: {OUTPUT_DIR}")
    print(f"[CONFIG] Model directory:  {MODEL_DIR}")
    print(f"[CONFIG] Cache directory:  {CACHE_DIR}")


def validate_data_paths():
    """Check that all required data files exist before pipeline runs."""
    required_train = [TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH]
    required_test = [TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3]

    missing = []
    for p in required_train + required_test:
        if not p.exists():
            missing.append(str(p))

    if missing:
        raise FileNotFoundError(
            f"Missing data files:\n" + "\n".join(f"  - {m}" for m in missing)
        )
    print("[CONFIG] All data files validated successfully.")


if __name__ == "__main__":
    validate_data_paths()
    ensure_directories()
    print(f"[CONFIG] Train Source 1: {TRAIN_SOURCE1}")
    print(f"[CONFIG] Test Source 1:  {TEST_SOURCE1}")
    print(f"[CONFIG] Top-K Candidates: {TOP_K_CANDIDATES}")
    print(f"[CONFIG] Inference Batch Size: {INFERENCE_BATCH_SIZE}")
