"""
train.py - LightGBM Training, Negative Sampling, and Threshold Sweep
======================================================================
Handles the full training pipeline:
1. Split train_source1 into 80/20 train/validation
2. Run blocking on both splits
3. Generate positive/negative labels from ground truth
4. Engineer features for all labeled pairs
5. Train LightGBM binary classifier with class imbalance handling
6. Sweep decision thresholds to maximize macro-averaged F0.5
7. Save model and optimal threshold for inference

The F0.5 metric heavily penalizes false merges (precision-weighted),
which aligns with the challenge scoring.
"""

import gc
import json
import numpy as np
import polars as pl
import lightgbm as lgb
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from tqdm import tqdm

from src.config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    MODEL_PATH, THRESHOLD_PATH, MODEL_DIR,
    LGBM_PARAMS, VALIDATION_FRACTION, RANDOM_SEED,
    NEGATIVE_RATIO, TOP_K_CANDIDATES,
    THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP,
    OUTPUT_DIR,
)
from src.preprocess import load_and_preprocess, load_ground_truth
from src.blocking import build_blocking_index, generate_candidates
from src.features import (
    compute_features_batch, FEATURE_NAMES, NUM_FEATURES,
)


def split_train_validation(
    s1_df: pl.DataFrame,
    ground_truth: Dict[str, Set[str]],
    val_fraction: float = VALIDATION_FRACTION,
    seed: int = RANDOM_SEED,
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Split Source 1 data and ground truth into train/validation sets.

    Stratified split: ensures both splits have proportional singletons
    (entities with no matches) and entities with matches.

    Args:
        s1_df: Full Source 1 preprocessed DataFrame
        ground_truth: Full ground truth dict
        val_fraction: Fraction for validation (default 0.20)
        seed: Random seed for reproducibility

    Returns:
        (train_s1_df, val_s1_df, train_gt, val_gt)
    """
    print(f"\n[TRAIN] Splitting Source 1 into {1-val_fraction:.0%} train / "
          f"{val_fraction:.0%} validation...")

    # Get entity IDs that exist in both S1 and ground truth
    all_s1_ids = s1_df["entity_id"].to_list()
    s1_ids_with_gt = [eid for eid in all_s1_ids if eid in ground_truth]
    s1_ids_without_gt = [eid for eid in all_s1_ids if eid not in ground_truth]

    print(f"[TRAIN]   Source 1 entities with ground truth: {len(s1_ids_with_gt):,}")
    print(f"[TRAIN]   Source 1 entities without ground truth: {len(s1_ids_without_gt):,}")

    # Stratify by singleton vs. non-singleton
    singletons = [eid for eid in s1_ids_with_gt if len(ground_truth[eid]) == 0]
    non_singletons = [eid for eid in s1_ids_with_gt if len(ground_truth[eid]) > 0]

    rng = np.random.RandomState(seed)
    rng.shuffle(singletons)
    rng.shuffle(non_singletons)

    n_val_sing = int(len(singletons) * val_fraction)
    n_val_non = int(len(non_singletons) * val_fraction)

    val_ids = set(singletons[:n_val_sing] + non_singletons[:n_val_non])
    train_ids = set(singletons[n_val_sing:] + non_singletons[n_val_non:])

    # Also include entities without ground truth in training
    train_ids.update(s1_ids_without_gt)

    # Split DataFrames
    train_s1_df = s1_df.filter(pl.col("entity_id").is_in(list(train_ids)))
    val_s1_df = s1_df.filter(pl.col("entity_id").is_in(list(val_ids)))

    # Split ground truth
    train_gt = {eid: ground_truth[eid] for eid in train_ids if eid in ground_truth}
    val_gt = {eid: ground_truth[eid] for eid in val_ids if eid in ground_truth}

    print(f"[TRAIN]   Train split: {len(train_s1_df):,} entities "
          f"({len(train_gt):,} with GT)")
    print(f"[TRAIN]   Val split:   {len(val_s1_df):,} entities "
          f"({len(val_gt):,} with GT)")

    return train_s1_df, val_s1_df, train_gt, val_gt


def create_labeled_pairs(
    candidate_pairs: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    negative_ratio: int = NEGATIVE_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[Dict[str, List[str]], List[int]]:
    """
    Create labeled (positive/negative) training pairs from blocking output.

    For each Source 1 entity:
    - Candidates that appear in ground truth are POSITIVE (label=1)
    - Candidates that don't appear in ground truth are NEGATIVE (label=0)
    - If ground truth matches were missed by blocking, they are added as
      positive examples (to train the model on hard positives)

    Negative sampling is controlled to prevent extreme class imbalance.

    Args:
        candidate_pairs: Dict from blocking: s1_id -> [candidate_ids]
        ground_truth: Dict: s1_id -> set of true match IDs
        negative_ratio: Max negatives per positive to keep
        seed: Random seed for negative sampling

    Returns:
        Tuple of:
        - labeled_pairs: Dict mapping s1_id -> [candidate_ids] (reordered)
        - labels: List of 0/1 labels corresponding to pairs
    """
    print(f"\n[TRAIN] Creating labeled pairs (negative_ratio={negative_ratio})...")

    rng = np.random.RandomState(seed)
    labeled_pairs = {}
    all_labels = []

    n_pos_total = 0
    n_neg_total = 0
    n_gt_added = 0  # Ground truth matches not found by blocking

    for s1_id, cands in candidate_pairs.items():
        if s1_id not in ground_truth:
            continue

        true_matches = ground_truth[s1_id]
        positives = []
        negatives = []

        # Classify blocking candidates
        for cand_id in cands:
            if cand_id in true_matches:
                positives.append(cand_id)
            else:
                negatives.append(cand_id)

        # Add ground truth matches missed by blocking
        missed_gt = true_matches - set(cands)
        for missed_id in missed_gt:
            positives.append(missed_id)
            n_gt_added += 1

        # Downsample negatives if too many
        max_neg = max(negative_ratio * max(len(positives), 1), 3)
        if len(negatives) > max_neg:
            rng.shuffle(negatives)
            negatives = negatives[:max_neg]

        # Combine and create labels
        pair_ids = positives + negatives
        pair_labels = [1] * len(positives) + [0] * len(negatives)

        labeled_pairs[s1_id] = pair_ids
        all_labels.extend(pair_labels)

        n_pos_total += len(positives)
        n_neg_total += len(negatives)

    print(f"[TRAIN]   Positive pairs: {n_pos_total:,}")
    print(f"[TRAIN]   Negative pairs: {n_neg_total:,}")
    print(f"[TRAIN]   GT matches added (missed by blocking): {n_gt_added:,}")
    print(f"[TRAIN]   Pos/Neg ratio: 1:{n_neg_total/max(n_pos_total,1):.1f}")

    return labeled_pairs, all_labels


def compute_f05_per_entity(
    true_matches: Set[str],
    predicted_matches: Set[str],
) -> float:
    """
    Compute F0.5 score for a single Source 1 entity.

    F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)

    Special cases:
    - Singleton correctly predicted empty: F0.5 = 1.0
    - Singleton with false predictions:   F0.5 = 0.0
    - Missed entity (no predictions for non-singleton): F0.5 = 0.0

    Args:
        true_matches: Set of ground truth matched IDs
        predicted_matches: Set of predicted matched IDs

    Returns:
        F0.5 score in [0.0, 1.0]
    """
    # Singleton: no true matches
    if len(true_matches) == 0:
        return 1.0 if len(predicted_matches) == 0 else 0.0

    # Non-singleton: compute precision and recall
    if len(predicted_matches) == 0:
        return 0.0  # Missed all matches

    tp = len(true_matches & predicted_matches)
    precision = tp / len(predicted_matches) if predicted_matches else 0.0
    recall = tp / len(true_matches) if true_matches else 0.0

    if precision + recall == 0:
        return 0.0

    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f05


def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
) -> float:
    """
    Compute macro-averaged F0.5 across all Source 1 entities.

    Macro-average: compute F0.5 per entity, then average.
    Every Source 1 entity contributes equally regardless of match count.

    Args:
        ground_truth: Dict s1_id -> set of true match IDs
        predictions: Dict s1_id -> set of predicted match IDs

    Returns:
        Macro-averaged F0.5 score
    """
    scores = []
    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, set())
        score = compute_f05_per_entity(true_matches, pred_matches)
        scores.append(score)

    return np.mean(scores) if scores else 0.0


def threshold_sweep(
    val_gt: Dict[str, Set[str]],
    val_s1_ids: List[str],
    val_cand_ids: List[str],
    val_probabilities: np.ndarray,
    threshold_min: float = THRESHOLD_MIN,
    threshold_max: float = THRESHOLD_MAX,
    threshold_step: float = THRESHOLD_STEP,
) -> Tuple[float, float]:
    """
    Sweep decision thresholds to find the one maximizing macro F0.5.

    For each threshold in [threshold_min, threshold_max]:
    1. Apply threshold to predicted probabilities
    2. Group predictions by Source 1 entity
    3. Compute macro F0.5 against validation ground truth

    Args:
        val_gt: Validation ground truth
        val_s1_ids: Source 1 entity IDs for each pair
        val_cand_ids: Candidate entity IDs for each pair
        val_probabilities: Predicted match probabilities
        threshold_min: Start of sweep range
        threshold_max: End of sweep range
        threshold_step: Step size for sweep

    Returns:
        (best_threshold, best_f05_score)
    """
    print(f"\n[TRAIN] Sweeping thresholds from {threshold_min} to {threshold_max} "
          f"(step={threshold_step})...")

    thresholds = np.arange(threshold_min, threshold_max + threshold_step/2, threshold_step)
    best_threshold = 0.5
    best_f05 = 0.0

    for threshold in tqdm(thresholds, desc="  Threshold sweep"):
        # Apply threshold and group predictions
        predictions = {}
        for s1_id, cand_id, prob in zip(val_s1_ids, val_cand_ids, val_probabilities):
            if s1_id not in predictions:
                predictions[s1_id] = set()
            if prob >= threshold:
                predictions[s1_id].add(cand_id)

        # Ensure all validation entities are in predictions
        for s1_id in val_gt:
            if s1_id not in predictions:
                predictions[s1_id] = set()

        # Compute macro F0.5
        f05 = compute_macro_f05(val_gt, predictions)

        if f05 > best_f05:
            best_f05 = f05
            best_threshold = threshold

    print(f"[TRAIN]   Best threshold: {best_threshold:.3f}")
    print(f"[TRAIN]   Best macro F0.5: {best_f05:.4f}")

    return best_threshold, best_f05


def train_pipeline(
    sample_fraction: Optional[float] = None,
) -> Tuple[lgb.LGBMClassifier, float]:
    """
    Execute the full training pipeline.

    Steps:
    1. Load and preprocess all training data
    2. Split Source 1 into train/validation (80/20)
    3. Build TF-IDF blocking index from S2+S3
    4. Generate candidates for train and validation S1 entities
    5. Create labeled pairs from ground truth
    6. Compute pairwise features
    7. Train LightGBM classifier
    8. Sweep thresholds on validation set
    9. Save model and optimal threshold

    Args:
        sample_fraction: Optional fraction to sample data (for debugging)

    Returns:
        (trained_model, optimal_threshold)
    """
    print("=" * 70)
    print("TRAINING PIPELINE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load and preprocess all data
    # ------------------------------------------------------------------
    print("\n[TRAIN] Step 1: Loading and preprocessing training data...")

    s1_df = load_and_preprocess(str(TRAIN_SOURCE1))
    s2_df = load_and_preprocess(str(TRAIN_SOURCE2))
    s3_df = load_and_preprocess(str(TRAIN_SOURCE3))
    ground_truth = load_ground_truth(str(TRAIN_GROUND_TRUTH))

    if sample_fraction and sample_fraction < 1.0:
        n_sample_s1 = int(len(s1_df) * sample_fraction)
        s1_df = s1_df.head(n_sample_s1)
        s1_sampled_ids = set(s1_df["entity_id"].to_list())
        ground_truth = {k: v for k, v in ground_truth.items() if k in s1_sampled_ids}

        # Collect all ground-truth candidate IDs for sampled S1
        gt_cand_ids = set()
        for matches in ground_truth.values():
            gt_cand_ids.update(matches)

        # Include GT candidate IDs + a sample of other candidates
        n_sample_s2 = int(len(s2_df) * sample_fraction)
        n_sample_s3 = int(len(s3_df) * sample_fraction)
        
        s2_gt = s2_df.filter(pl.col("entity_id").is_in(list(gt_cand_ids)))
        s2_other = s2_df.filter(~pl.col("entity_id").is_in(list(gt_cand_ids))).head(n_sample_s2)
        s2_df = pl.concat([s2_gt, s2_other]).unique(subset=["entity_id"])

        s3_gt = s3_df.filter(pl.col("entity_id").is_in(list(gt_cand_ids)))
        s3_other = s3_df.filter(~pl.col("entity_id").is_in(list(gt_cand_ids))).head(n_sample_s3)
        s3_df = pl.concat([s3_gt, s3_other]).unique(subset=["entity_id"])

        print(f"[TRAIN]   SAMPLED: S1={len(s1_df):,}, S2={len(s2_df):,}, "
              f"S3={len(s3_df):,}, GT={len(ground_truth):,}")

    # ------------------------------------------------------------------
    # Step 2: Split train/validation
    # ------------------------------------------------------------------
    train_s1, val_s1, train_gt, val_gt = split_train_validation(
        s1_df, ground_truth
    )
    del s1_df
    gc.collect()

    # ------------------------------------------------------------------
    # Step 3: Build blocking index from S2+S3
    # ------------------------------------------------------------------
    print("\n[TRAIN] Step 3: Building blocking index...")
    blocker = build_blocking_index(s2_df, s3_df)

    # ------------------------------------------------------------------
    # Step 4: Generate candidates for train and validation
    # ------------------------------------------------------------------
    print("\n[TRAIN] Step 4: Generating candidates...")
    train_candidates = generate_candidates(blocker, train_s1, batch_size=5000)
    val_candidates = generate_candidates(blocker, val_s1, batch_size=5000)

    # Compute blocking recall on validation set
    val_recall_hits = 0
    val_recall_total = 0
    for s1_id, true_matches in val_gt.items():
        cands = set(val_candidates.get(s1_id, []))
        for match in true_matches:
            val_recall_total += 1
            if match in cands:
                val_recall_hits += 1
    blocking_recall = val_recall_hits / val_recall_total if val_recall_total > 0 else 0.0
    print(f"[TRAIN]   Blocking recall on validation: {blocking_recall:.4f} "
          f"({val_recall_hits:,}/{val_recall_total:,})")

    # ------------------------------------------------------------------
    # Step 5: Create labeled pairs
    # ------------------------------------------------------------------
    train_labeled_pairs, train_labels = create_labeled_pairs(
        train_candidates, train_gt
    )

    # ------------------------------------------------------------------
    # Step 6: Compute features
    # ------------------------------------------------------------------
    print("\n[TRAIN] Step 6: Computing pairwise features for training...")

    # Combine S2+S3 for feature lookup
    cols_needed = ["entity_id", "name_clean", "addr_clean", "name_numerics",
                   "addr_numerics", "name_addr_combined", "country_clean"]
    candidates_combined = pl.concat([
        s2_df.select(cols_needed),
        s3_df.select(cols_needed),
    ])

    # Free memory before feature computation
    del s2_df, s3_df
    gc.collect()

    train_features, train_s1_ids, train_cand_ids = compute_features_batch(
        train_s1, candidates_combined, train_labeled_pairs
    )

    # Reconstruct labels aligned with actual computed features.
    # create_labeled_pairs() may include GT matches whose candidate IDs
    # don't exist in the candidate pool (e.g., when sampling), causing
    # compute_features_batch() to skip those pairs. Rebuilding labels
    # from the returned pair IDs guarantees perfect alignment.
    train_labels = []
    for s1_id, cand_id in zip(train_s1_ids, train_cand_ids):
        true_matches = train_gt.get(s1_id, set())
        train_labels.append(1 if cand_id in true_matches else 0)

    print(f"[TRAIN]   Training feature matrix: {train_features.shape}")
    print(f"[TRAIN]   Training labels: {len(train_labels):,} "
          f"(pos={sum(train_labels):,}, neg={len(train_labels)-sum(train_labels):,})")

    # Validation features
    print("\n[TRAIN] Step 6b: Computing pairwise features for validation...")
    val_features, val_s1_ids, val_cand_ids = compute_features_batch(
        val_s1, candidates_combined, val_candidates
    )
    print(f"[TRAIN]   Validation feature matrix: {val_features.shape}")

    # ------------------------------------------------------------------
    # Step 7: Train LightGBM
    # ------------------------------------------------------------------
    print("\n[TRAIN] Step 7: Training LightGBM classifier...")

    # Compute scale_pos_weight for class imbalance (capped at 2.5 to prevent loss explosion)
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    raw_ratio = n_neg / max(n_pos, 1)
    scale_pos_weight = min(2.5, max(1.0, float(raw_ratio)))
    print(f"[TRAIN]   raw pos/neg ratio: {raw_ratio:.2f} -> capped scale_pos_weight: {scale_pos_weight:.2f}")

    params = LGBM_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight

    model = lgb.LGBMClassifier(**params)

    train_labels_arr = np.array(train_labels, dtype=np.int32)

    # Use a subset of validation data for early stopping
    # Create validation labels from ground truth
    val_labels = []
    for s1_id, cand_id in zip(val_s1_ids, val_cand_ids):
        true_matches = val_gt.get(s1_id, set())
        val_labels.append(1 if cand_id in true_matches else 0)
    val_labels_arr = np.array(val_labels, dtype=np.int32)

    model.fit(
        train_features, train_labels_arr,
        eval_set=[(val_features, val_labels_arr)],
        eval_metric='binary_logloss',
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=True),
            lgb.log_evaluation(period=50),
        ],
        feature_name=FEATURE_NAMES,
    )

    print(f"[TRAIN]   Best iteration: {model.best_iteration_}")

    # Feature importance
    importance = model.feature_importances_
    sorted_idx = np.argsort(-importance)
    print("\n[TRAIN]   Feature importance (top 10):")
    for idx in sorted_idx[:10]:
        print(f"[TRAIN]     {FEATURE_NAMES[idx]}: {importance[idx]}")

    # ------------------------------------------------------------------
    # Step 8: Threshold sweep on validation
    # ------------------------------------------------------------------
    print("\n[TRAIN] Step 8: Threshold sweep on validation set...")

    val_probabilities = model.predict_proba(val_features)[:, 1]

    best_threshold, best_f05 = threshold_sweep(
        val_gt, val_s1_ids, val_cand_ids, val_probabilities
    )

    # ------------------------------------------------------------------
    # Step 9: Save model and threshold
    # ------------------------------------------------------------------
    print("\n[TRAIN] Step 9: Saving model and threshold...")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save LightGBM model
    model.booster_.save_model(str(MODEL_PATH))
    print(f"[TRAIN]   Model saved to: {MODEL_PATH}")

    # Save optimal threshold
    with open(str(THRESHOLD_PATH), 'w') as f:
        f.write(f"{best_threshold:.4f}\n")
    print(f"[TRAIN]   Threshold saved to: {THRESHOLD_PATH}")

    # Save training metadata
    metadata = {
        "best_threshold": best_threshold,
        "best_f05": best_f05,
        "blocking_recall": blocking_recall,
        "n_train_pairs": len(train_labels),
        "n_val_pairs": len(val_labels),
        "n_pos_train": int(n_pos),
        "n_neg_train": int(n_neg),
        "scale_pos_weight": scale_pos_weight,
        "best_iteration": model.best_iteration_,
        "feature_names": FEATURE_NAMES,
        "feature_importance": {
            FEATURE_NAMES[i]: int(importance[i]) for i in range(len(FEATURE_NAMES))
        },
    }
    metadata_path = MODEL_DIR / "training_metadata.json"
    with open(str(metadata_path), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[TRAIN]   Metadata saved to: {metadata_path}")

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE")
    print(f"  Best threshold: {best_threshold:.3f}")
    print(f"  Validation macro F0.5: {best_f05:.4f}")
    print(f"  Blocking recall: {blocking_recall:.4f}")
    print("=" * 70)

    return model, best_threshold


if __name__ == "__main__":
    # Run training pipeline
    # Use sample_fraction for quick testing, None for full training
    import sys

    sample = float(sys.argv[1]) if len(sys.argv) > 1 else None
    model, threshold = train_pipeline(sample_fraction=sample)
