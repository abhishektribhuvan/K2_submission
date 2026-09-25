"""
infer.py - Memory-Safe Batched Test Inference
================================================
Processes the test set in streaming batches of 50,000 Source 1 rows
to stay within the 16GB RAM constraint. For each batch:
1. Load and preprocess the S1 batch
2. Query the blocking index for candidates
3. Compute RapidFuzz pairwise features
4. Predict match probabilities with the trained LightGBM model
5. Apply the optimized F0.5 threshold
6. Append results to output/matching_results.tsv

Also builds and writes output/candidate_pairs.tsv for the full test set.
"""

import gc
import numpy as np
import polars as pl
import lightgbm as lgb
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from tqdm import tqdm

from src.config import (
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    MODEL_PATH, THRESHOLD_PATH,
    MATCHING_RESULTS_FILE, CANDIDATE_PAIRS_FILE,
    OUTPUT_DIR, INFERENCE_BATCH_SIZE,
)
from src.preprocess import load_and_preprocess, preprocess_dataframe
from src.blocking import build_blocking_index, TFIDFBlocker
from src.features import (
    compute_features_for_pairs_streaming, FEATURE_NAMES, NUM_FEATURES,
)


def load_model_and_threshold() -> Tuple[lgb.Booster, float]:
    """
    Load the trained LightGBM model and optimal threshold from disk.

    Returns:
        Tuple of (lgb.Booster, threshold_float)

    Raises:
        FileNotFoundError: If model or threshold files don't exist
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Threshold not found at {THRESHOLD_PATH}. Run train.py first.")

    print(f"[INFER] Loading model from: {MODEL_PATH}")
    model = lgb.Booster(model_file=str(MODEL_PATH))

    with open(str(THRESHOLD_PATH), 'r') as f:
        threshold = float(f.read().strip())
    print(f"[INFER] Loaded threshold: {threshold:.4f}")

    return model, threshold


def build_candidate_lookup(
    candidates_df: pl.DataFrame,
) -> Dict[str, dict]:
    """
    Build a dictionary lookup from candidate entity_id to preprocessed fields.

    This allows O(1) access to candidate features during pair computation.

    Args:
        candidates_df: Preprocessed Source 2+3 DataFrame

    Returns:
        Dict mapping entity_id -> dict of preprocessed fields
    """
    print(f"[INFER] Building candidate lookup ({len(candidates_df):,} records)...")

    lookup = {}
    for row in candidates_df.iter_rows(named=True):
        lookup[row["entity_id"]] = {
            "entity_id": row["entity_id"],
            "name_clean": row.get("name_clean", ""),
            "addr_clean": row.get("addr_clean", ""),
            "name_numerics": row.get("name_numerics", ""),
            "addr_numerics": row.get("addr_numerics", ""),
            "name_addr_combined": row.get("name_addr_combined", ""),
            "country_clean": row.get("country_clean", ""),
        }

    print(f"[INFER]   Lookup built: {len(lookup):,} entries")
    return lookup


def inference_pipeline(
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> None:
    """
    Execute the full inference pipeline on the test set.

    Processes Source 1 test data in batches to respect memory limits:
    1. Load and preprocess S2+S3 (kept in memory as the blocking index)
    2. Build per-country TF-IDF blocking index
    3. Stream S1 in batches of `batch_size`
    4. For each batch: block -> features -> predict -> threshold -> write

    Outputs:
    - output/candidate_pairs.tsv: All blocking candidates
    - output/matching_results.tsv: Final matched entity IDs

    Args:
        batch_size: Number of S1 rows per batch (default 50,000)
    """
    print("=" * 70)
    print("INFERENCE PIPELINE")
    print("=" * 70)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load model and threshold
    # ------------------------------------------------------------------
    model, threshold = load_model_and_threshold()

    # ------------------------------------------------------------------
    # Step 2: Load and preprocess S2+S3 candidates
    # ------------------------------------------------------------------
    print("\n[INFER] Step 2: Loading and preprocessing test candidates...")

    s2_df = load_and_preprocess(str(TEST_SOURCE2))
    s3_df = load_and_preprocess(str(TEST_SOURCE3))

    # ------------------------------------------------------------------
    # Step 3: Build blocking index
    # ------------------------------------------------------------------
    print("\n[INFER] Step 3: Building blocking index on test S2+S3...")
    blocker = build_blocking_index(s2_df, s3_df)

    # Build candidate lookup for feature computation
    cols_needed = ["entity_id", "name_clean", "addr_clean", "name_numerics",
                   "addr_numerics", "name_addr_combined", "country_clean"]
    candidates_combined = pl.concat([
        s2_df.select(cols_needed),
        s3_df.select(cols_needed),
    ])
    cand_lookup = build_candidate_lookup(candidates_combined)

    # Free the full DataFrames (index + lookup retain what we need)
    del s2_df, s3_df, candidates_combined
    gc.collect()

    # ------------------------------------------------------------------
    # Step 4: Load and preprocess test Source 1
    # ------------------------------------------------------------------
    print("\n[INFER] Step 4: Loading and preprocessing test Source 1...")
    s1_df = load_and_preprocess(str(TEST_SOURCE1))
    total_s1 = len(s1_df)
    print(f"[INFER]   Total Source 1 entities: {total_s1:,}")

    # ------------------------------------------------------------------
    # Step 5: Global candidate blocking
    # ------------------------------------------------------------------
    print("\n[INFER] Step 5: Querying global candidate blocking index...")
    all_candidates = blocker.query_batch(s1_df, batch_size=1000)
    n_with_cands = sum(1 for v in all_candidates.values() if len(v) > 0)
    print(f"[INFER]   Blocking complete: {n_with_cands:,}/{total_s1:,} entities have candidates")

    # ------------------------------------------------------------------
    # Step 6: Stream features, predict, and write output files
    # ------------------------------------------------------------------
    print("\n[INFER] Step 6: Initializing output files and streaming predictions...")

    # Write headers
    with open(str(MATCHING_RESULTS_FILE), 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")

    with open(str(CANDIDATE_PAIRS_FILE), 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")

    n_batches = (total_s1 + batch_size - 1) // batch_size
    print(f"[INFER] Stream processing {total_s1:,} entities in {n_batches} batches (batch_size={batch_size:,})...")

    processed_ids = set()

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, total_s1)

        print(f"\n[INFER] --- Batch {batch_idx+1}/{n_batches} (rows {start:,}-{end:,}) ---")

        # Extract batch
        s1_batch = s1_df.slice(start, end - start)
        batch_ids = s1_batch["entity_id"].to_list()

        # Subset candidate dict for batch
        batch_cands = {qid: all_candidates.get(qid, []) for qid in batch_ids}

        # Feature computation for batch
        s1_rows = [row for row in s1_batch.iter_rows(named=True)]
        features, feat_s1_ids, feat_cand_ids = compute_features_for_pairs_streaming(
            s1_rows, cand_lookup, batch_cands
        )
        print(f"[INFER]   Computed {features.shape[0]:,} pair features")

        # Prediction
        batch_matches = {}  # s1_id -> set of matched cand_ids
        if features.shape[0] > 0:
            probabilities = model.predict(features)
            print(f"[INFER]   Predictions: min={probabilities.min():.4f}, "
                  f"max={probabilities.max():.4f}, "
                  f"mean={probabilities.mean():.4f}")

            # Apply threshold
            for s1_id, cand_id, prob in zip(feat_s1_ids, feat_cand_ids, probabilities):
                if prob >= threshold:
                    if s1_id not in batch_matches:
                        batch_matches[s1_id] = set()
                    batch_matches[s1_id].add(cand_id)

        n_matched = sum(1 for v in batch_matches.values() if len(v) > 0)
        print(f"[INFER]   Matched {n_matched:,}/{len(batch_ids):,} entities "
              f"above threshold {threshold:.3f}")

        # Write candidate pairs
        with open(str(CANDIDATE_PAIRS_FILE), 'a', encoding='utf-8') as f:
            for s1_id in batch_ids:
                cands = batch_cands.get(s1_id, [])
                seen = set()
                deduped = []
                for cid in cands:
                    if cid not in seen and not cid.startswith("S1-"):
                        seen.add(cid)
                        deduped.append(cid)

                cand_str = ",".join(deduped)
                f.write(f"{s1_id}\t{cand_str}\n")

        # Write matching results
        with open(str(MATCHING_RESULTS_FILE), 'a', encoding='utf-8') as f:
            for s1_id in batch_ids:
                matches = batch_matches.get(s1_id, set())
                deduped = []
                seen = set()
                for mid in matches:
                    if mid not in seen and not mid.startswith("S1-"):
                        seen.add(mid)
                        deduped.append(mid)

                match_str = ",".join(sorted(deduped))
                f.write(f"{s1_id}\t{match_str}\n")

                processed_ids.add(s1_id)

        # Free batch memory
        del s1_batch, batch_cands, features, feat_s1_ids, feat_cand_ids, s1_rows, batch_matches
        gc.collect()

    # ------------------------------------------------------------------
    # Step 7: Validate completeness
    # ------------------------------------------------------------------
    print(f"\n[INFER] Step 7: Validation...")

    all_s1_ids = set(s1_df["entity_id"].to_list())
    missing = all_s1_ids - processed_ids
    if missing:
        print(f"[INFER] WARNING: {len(missing):,} Source 1 entities missing from output!")
        # Append missing entities with empty match lists
        with open(str(MATCHING_RESULTS_FILE), 'a', encoding='utf-8') as f:
            for s1_id in sorted(missing):
                f.write(f"{s1_id}\t\n")
        with open(str(CANDIDATE_PAIRS_FILE), 'a', encoding='utf-8') as f:
            for s1_id in sorted(missing):
                f.write(f"{s1_id}\t\n")
        print(f"[INFER]   Added {len(missing):,} missing entities with empty matches")
    else:
        print(f"[INFER]   All {len(all_s1_ids):,} Source 1 entities present in output [OK]")

    # Check for duplicate S1 IDs in output
    _validate_output_no_duplicates(str(MATCHING_RESULTS_FILE))
    _validate_output_no_duplicates(str(CANDIDATE_PAIRS_FILE))

    print(f"\n[INFER] Output files:")
    print(f"[INFER]   {MATCHING_RESULTS_FILE}")
    print(f"[INFER]   {CANDIDATE_PAIRS_FILE}")

    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)


def _validate_output_no_duplicates(filepath: str) -> None:
    """
    Validate that an output TSV file has no duplicate source1_entity_id rows
    and no duplicate IDs within any comma-separated ID list.

    Args:
        filepath: Path to the output TSV file
    """
    print(f"[INFER] Validating: {filepath}")

    seen_s1 = set()
    n_dup_rows = 0
    n_dup_ids = 0
    line_count = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline()  # Skip header
        for line in f:
            line_count += 1
            parts = line.strip().split('\t')
            if not parts:
                continue

            s1_id = parts[0]

            # Check for duplicate S1 rows
            if s1_id in seen_s1:
                n_dup_rows += 1
            seen_s1.add(s1_id)

            # Check for duplicate IDs in match list
            if len(parts) > 1 and parts[1].strip():
                ids = parts[1].split(',')
                id_set = set()
                for eid in ids:
                    eid = eid.strip()
                    if eid in id_set:
                        n_dup_ids += 1
                    id_set.add(eid)

    if n_dup_rows > 0:
        print(f"[INFER]   WARNING: {n_dup_rows} duplicate source1_entity_id rows!")
    if n_dup_ids > 0:
        print(f"[INFER]   WARNING: {n_dup_ids} duplicate IDs within match lists!")
    if n_dup_rows == 0 and n_dup_ids == 0:
        print(f"[INFER]   No duplicates found [OK] ({line_count:,} rows)")


if __name__ == "__main__":
    inference_pipeline()
