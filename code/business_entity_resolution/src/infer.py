"""
infer.py - Memory-Safe Batched Test Inference with Zero-Copy Candidate Index
=============================================================================
Processes the test set in streaming batches of Source 1 rows
to stay strictly within the 16GB RAM constraint.

Key architectural optimizations:
1. Compact flat index for candidates (400MB vs 5GB dict-of-dicts)
2. Streaming batch-wise blocking query (prevents holding 43M candidate strings)
3. Precision veto enforcement (kills franchise & multi-tenant false merges)
4. Exact challenge formatting validation (tab-separated, deduplicated, unique)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc
import os
import numpy as np
import polars as pl
import lightgbm as lgb
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from src.config import (
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    MODEL_PATH, THRESHOLD_PATH,
    MATCHING_RESULTS_FILE, CANDIDATE_PAIRS_FILE,
    OUTPUT_DIR, INFERENCE_BATCH_SIZE, TOP_K_CANDIDATES,
)
from src.preprocess import load_and_preprocess
from src.blocking import build_blocking_index, CountryBlockingIndex
from src.features import compute_pair_features, evaluate_precision_veto, _compute_features_for_chunk


def load_model_and_threshold() -> Tuple[lgb.Booster, float]:
    """
    Load the trained LightGBM model and optimal threshold from disk.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Threshold not found at {THRESHOLD_PATH}. Run train.py first.")

    print(f"[INFER] Loading model from: {MODEL_PATH}")
    model = lgb.Booster(model_file=str(MODEL_PATH))

    with open(str(THRESHOLD_PATH), 'r') as f:
        threshold = float(f.read().strip())
    print(f"[INFER] Loaded optimal threshold: {threshold:.4f}")

    return model, threshold


def inference_pipeline(batch_size: int = INFERENCE_BATCH_SIZE) -> None:
    """
    Execute zero-copy streaming inference on the test set.
    """
    print("=" * 80)
    print("STARTING ZERO-COPY STREAMING INFERENCE PIPELINE")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load model and threshold
    model, threshold = load_model_and_threshold()

    # 2. Load & preprocess candidate pools (S2 + S3)
    print("\n[INFER] Step 1: Loading and preprocessing test candidates (S2 + S3)...")
    s2_df = load_and_preprocess(str(TEST_SOURCE2))
    s3_df = load_and_preprocess(str(TEST_SOURCE3))

    # Build compact columnar candidate index and free raw data immediately
    cols_needed = ["entity_id", "name_clean", "addr_clean", "name_numerics",
                   "addr_numerics", "name_addr_combined", "country_clean"]
    candidates_combined = pl.concat([
        s2_df.select(cols_needed),
        s3_df.select(cols_needed),
    ]).unique(subset=["entity_id"])

    del s2_df, s3_df
    gc.collect()

    # Build TF-IDF blocking index with pre-freed RAM
    print("\n[INFER] Step 2: Building country blocking index...")
    blocker = CountryBlockingIndex(top_k=TOP_K_CANDIDATES)
    blocker.build_index(candidates_combined)

    print(f"[INFER]   Total unique candidates indexed: {len(candidates_combined):,}")

    # Build compact entity_id -> row_idx map (~400MB RAM vs 5GB dict-of-dicts)
    print("[INFER]   Building compact entity_id -> row_idx index...")
    cand_id_to_idx = {
        eid: idx for idx, eid in enumerate(candidates_combined["entity_id"].to_list())
    }

    # Extract flat Python lists for fastest C-level tuple packing
    c_names = candidates_combined["name_clean"].to_list()
    c_addrs = candidates_combined["addr_clean"].to_list()
    c_n_nums = candidates_combined["name_numerics"].to_list()
    c_a_nums = candidates_combined["addr_numerics"].to_list()
    c_combs = candidates_combined["name_addr_combined"].to_list()
    c_cntrys = candidates_combined["country_clean"].to_list()

    del candidates_combined
    gc.collect()

    # 3. Stream Source 1 in batches
    print(f"\n[INFER] Step 3: Loading test Source 1 queries...")
    s1_df = load_and_preprocess(str(TEST_SOURCE1))
    total_s1 = len(s1_df)
    print(f"[INFER]   Total Source 1 entities: {total_s1:,}")

    processed_ids = set()
    if MATCHING_RESULTS_FILE.exists() and CANDIDATE_PAIRS_FILE.exists() and MATCHING_RESULTS_FILE.stat().st_size > 50:
        with open(str(MATCHING_RESULTS_FILE), 'r', encoding='utf-8') as f:
            f.readline()  # skip header
            for line in f:
                parts = line.split('\t')
                if parts and parts[0].strip():
                    processed_ids.add(parts[0].strip())
        print(f"[INFER] Resuming inference: found {len(processed_ids):,} already processed entities on disk")
    else:
        # Initialize output files with exact challenge headers
        with open(str(MATCHING_RESULTS_FILE), 'w', encoding='utf-8') as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
        with open(str(CANDIDATE_PAIRS_FILE), 'w', encoding='utf-8') as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")

    n_batches = (total_s1 + batch_size - 1) // batch_size
    print(f"[INFER] Stream processing {total_s1:,} entities in {n_batches} batches (batch_size={batch_size:,})...")

    for b_idx in range(n_batches):
        start = b_idx * batch_size
        end = min(start + batch_size, total_s1)
        print(f"\n[INFER] --- Batch {b_idx + 1}/{n_batches} (rows {start:,} to {end:,}) ---")

        s1_batch = s1_df.slice(start, end - start)
        batch_ids = s1_batch["entity_id"].to_list()

        # Check if entire batch was already written to disk
        if all(qid in processed_ids for qid in batch_ids):
            print(f"[INFER]   Batch {b_idx + 1}/{n_batches} already processed ({len(batch_ids):,} entities). Skipping...")
            continue

        # Query candidates with safe high-throughput batch size (8,000 prevents 16GB RAM spikes)
        batch_candidates = blocker.query_batch(s1_batch, batch_size=8000)

        # Enumerate pairs for this batch
        pair_data = []
        pair_s1_ids = []
        pair_cand_ids = []

        s1_names = s1_batch["name_clean"].to_list()
        s1_addrs = s1_batch["addr_clean"].to_list()
        s1_n_nums = s1_batch["name_numerics"].to_list()
        s1_a_nums = s1_batch["addr_numerics"].to_list()
        s1_combs = s1_batch["name_addr_combined"].to_list()
        s1_cntrys = s1_batch["country_clean"].to_list()

        for i, s1_id in enumerate(batch_ids):
            c_list = batch_candidates.get(s1_id, [])
            s1_tup = (
                s1_names[i], s1_addrs[i], s1_n_nums[i],
                s1_a_nums[i], s1_combs[i], s1_cntrys[i]
            )

            for cid in c_list:
                c_idx = cand_id_to_idx.get(cid)
                if c_idx is None:
                    continue

                c_tup = (
                    c_names[c_idx], c_addrs[c_idx], c_n_nums[c_idx],
                    c_a_nums[c_idx], c_combs[c_idx], c_cntrys[c_idx]
                )

                pair_data.append((s1_tup, c_tup))
                pair_s1_ids.append(s1_id)
                pair_cand_ids.append(cid)

        # Compute pair features in parallel across CPU cores
        n_workers = min(10, os.cpu_count() or 4)
        pair_features = []
        if pair_data:
            n_pairs = len(pair_data)
            chunk_size = max(2000, n_pairs // (n_workers * 4))
            chunks = [pair_data[k:k + chunk_size] for k in range(0, n_pairs, chunk_size)]
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                for chunk_res in executor.map(_compute_features_for_chunk, chunks):
                    pair_features.extend(chunk_res)

        print(f"[INFER]   Computed {len(pair_features):,} pair features across {n_workers} CPU cores")

        # Predict with LightGBM
        batch_matches: Dict[str, List[str]] = {qid: [] for qid in batch_ids}

        if pair_features:
            feat_arr = np.array(pair_features, dtype=np.float32)
            probs = model.predict(feat_arr)

            # Precision veto is feature index 24 (1.0 = vetoed, 0.0 = safe)
            vetoes = (feat_arr[:, 24] == 1.0)
            n_vetoed = int(vetoes.sum())
            print(f"[INFER]   Predictions: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
            print(f"[INFER]   Precision vetoes applied: {n_vetoed:,} pairs rejected")

            for s1_id, cid, prob, veto in zip(pair_s1_ids, pair_cand_ids, probs, vetoes):
                # Enforce threshold AND precision veto
                if not veto and prob >= threshold:
                    batch_matches[s1_id].append(cid)

        n_matched = sum(1 for v in batch_matches.values() if len(v) > 0)
        print(f"[INFER]   Matched {n_matched:,}/{len(batch_ids):,} entities above threshold {threshold:.3f}")

        # Append candidate pairs to TSV
        with open(str(CANDIDATE_PAIRS_FILE), 'a', encoding='utf-8') as f_cand:
            for s1_id in batch_ids:
                cands = batch_candidates.get(s1_id, [])
                seen = set()
                deduped = []
                for cid in cands:
                    if cid not in seen and not cid.startswith("S1-"):
                        seen.add(cid)
                        deduped.append(cid)
                c_str = ",".join(deduped)
                f_cand.write(f"{s1_id}\t{c_str}\n")

        # Append matching results to TSV
        with open(str(MATCHING_RESULTS_FILE), 'a', encoding='utf-8') as f_match:
            for s1_id in batch_ids:
                m_list = sorted(set(batch_matches.get(s1_id, [])))
                m_deduped = [m for m in m_list if not m.startswith("S1-")]
                m_str = ",".join(m_deduped)
                f_match.write(f"{s1_id}\t{m_str}\n")
                processed_ids.add(s1_id)

        del pair_features, pair_s1_ids, pair_cand_ids, batch_candidates, batch_matches
        if 'feat_arr' in locals():
            del feat_arr
        if 'probs' in locals():
            del probs
        if 'vetoes' in locals():
            del vetoes
        gc.collect()

    # 4. Validate output files
    print(f"\n[INFER] Step 4: Validation...")
    all_s1_ids = set(s1_df["entity_id"].to_list())
    missing = all_s1_ids - processed_ids
    if missing:
        print(f"[INFER] WARNING: {len(missing):,} Source 1 entities missing from output! Appending...")
        with open(str(MATCHING_RESULTS_FILE), 'a', encoding='utf-8') as f:
            for s1_id in sorted(missing):
                f.write(f"{s1_id}\t\n")
        with open(str(CANDIDATE_PAIRS_FILE), 'a', encoding='utf-8') as f:
            for s1_id in sorted(missing):
                f.write(f"{s1_id}\t\n")

    _validate_output_no_duplicates(str(MATCHING_RESULTS_FILE))
    _validate_output_no_duplicates(str(CANDIDATE_PAIRS_FILE))

    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE: Outputs written and verified.")
    print("=" * 80)


def _validate_output_no_duplicates(filepath: str) -> None:
    """
    Validate that an output TSV file has no duplicate source1_entity_id rows
    and no duplicate IDs within any comma-separated ID list.
    """
    print(f"[INFER] Validating: {filepath}")

    seen_s1 = set()
    n_dup_rows = 0
    n_dup_ids = 0
    line_count = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            line_count += 1
            parts = line.strip().split('\t')
            if not parts:
                continue

            s1_id = parts[0]
            if s1_id in seen_s1:
                n_dup_rows += 1
            seen_s1.add(s1_id)

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
        print(f"[INFER]   Verified: {line_count:,} rows, strictly unique [OK]")


if __name__ == "__main__":
    inference_pipeline()
