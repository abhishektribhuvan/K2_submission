"""
features.py - RapidFuzz Pairwise String Similarity Metrics
============================================================
Computes pairwise lexical features between Source 1 entities and their
candidate matches from blocking. Uses the `rapidfuzz` library for
high-performance string comparison with all CPU cores.

All features are purely string-geometric (character/token-level) and
do not rely on language-specific models, ensuring generalization to
unseen French data.

Feature set:
1. Name Token Sort Ratio (handles word reordering)
2. Name Levenshtein Ratio (edit distance normalized)
3. Name Jaccard Token Overlap (set intersection / union)
4. Name Partial Ratio (best substring match)
5. Address Token Sort Ratio
6. Address Levenshtein Ratio
7. Address Jaccard Token Overlap
8. Address Character N-gram Cosine similarity
9. Exact numeric match flag (street numbers, postal codes)
10. Numeric Jaccard overlap
11. Combined name+address Levenshtein ratio
12. Country exact match flag
"""

import numpy as np
import polars as pl
from rapidfuzz import fuzz, distance
from rapidfuzz.distance import Levenshtein
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

from src.config import N_CORES


def jaccard_token_similarity(s1: str, s2: str) -> float:
    """
    Compute Jaccard similarity between token sets of two strings.

    Jaccard = |intersection| / |union|

    This is robust to word reordering and handles multi-word names
    where tokens may appear in different orders across sources.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Jaccard similarity in [0.0, 1.0]
    """
    if not s1 or not s2:
        return 0.0
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union) if union else 0.0


def char_ngram_cosine(s1: str, s2: str, n: int = 3) -> float:
    """
    Compute cosine similarity between character n-gram frequency vectors.

    Creates character n-gram bags from both strings and computes the
    cosine of the angle between their frequency vectors. This is more
    robust to typos than exact token matching.

    Args:
        s1: First string
        s2: Second string
        n: Size of character n-grams (default 3)

    Returns:
        Cosine similarity in [0.0, 1.0]
    """
    if not s1 or not s2:
        return 0.0

    def get_ngrams(s):
        """Extract character n-grams as a frequency dict."""
        ngrams = {}
        for i in range(len(s) - n + 1):
            gram = s[i:i+n]
            ngrams[gram] = ngrams.get(gram, 0) + 1
        return ngrams

    ng1 = get_ngrams(s1)
    ng2 = get_ngrams(s2)

    if not ng1 or not ng2:
        return 0.0

    # Compute dot product and magnitudes
    all_grams = set(ng1.keys()) | set(ng2.keys())
    dot = sum(ng1.get(g, 0) * ng2.get(g, 0) for g in all_grams)
    mag1 = sum(v * v for v in ng1.values()) ** 0.5
    mag2 = sum(v * v for v in ng2.values()) ** 0.5

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot / (mag1 * mag2)


def numeric_overlap(nums1: str, nums2: str) -> Tuple[float, int]:
    """
    Compute overlap between extracted numeric tokens.

    Returns both the Jaccard similarity and an exact-match flag.
    Numeric tokens include street numbers, unit numbers, postal codes.
    This is language-agnostic: works for US ZIP, Indian PIN, French postal.

    Args:
        nums1: Space-separated numeric tokens from entity 1
        nums2: Space-separated numeric tokens from entity 2

    Returns:
        Tuple of (jaccard_similarity, exact_match_flag)
        exact_match_flag is 1 if both non-empty and identical, else 0
    """
    if not nums1 or not nums2:
        return (0.0, 0)

    set1 = set(nums1.split())
    set2 = set(nums2.split())

    if not set1 or not set2:
        return (0.0, 0)

    intersection = set1 & set2
    union = set1 | set2

    jaccard = len(intersection) / len(union) if union else 0.0
    exact = 1 if set1 == set2 else 0

    return (jaccard, exact)


from rapidfuzz.distance import JaroWinkler


def evaluate_precision_veto(
    s1_name: str,
    s1_addr: str,
    s1_addr_nums: str,
    c_name: str,
    c_addr: str,
    c_addr_nums: str,
    name_sort_sim: float,
    name_jw_sim: float,
) -> bool:
    """
    Deterministic precision vetoes to eliminate catastrophic false merges.
    In macro F0.5, a false positive carries a 4x penalty over a false negative.
    
    Returns True if the candidate pair MUST BE VETOED (forced to reject).
    """
    # Veto 1: Multi-tenant building trap
    # If clean names share virtually no similarity (< 0.35),
    # they are different tenants in the same building or street.
    if max(name_sort_sim, name_jw_sim) < 0.35:
        return True

    # Veto 2: Franchise trap (same brand/street, but conflicting street numbers)
    # Only enforce when BOTH entities have non-empty address numerics
    if s1_addr_nums and c_addr_nums:
        s1_num_set = set(s1_addr_nums.split())
        c_num_set = set(c_addr_nums.split())
        # If both entities have numbers, but share ZERO numbers in common
        if s1_num_set and c_num_set and not (s1_num_set & c_num_set):
            return True

    return False


def compute_pair_features(
    s1_name: str, s1_addr: str, s1_name_nums: str, s1_addr_nums: str,
    s1_combined: str, s1_country: str,
    c_name: str, c_addr: str, c_name_nums: str, c_addr_nums: str,
    c_combined: str, c_country: str,
) -> List[float]:
    features = []

    # --- NAME FEATURES (12) ---
    # 1. Token Sort Ratio
    features.append(fuzz.token_sort_ratio(s1_name, c_name) / 100.0)

    # 2. Levenshtein Ratio
    features.append(fuzz.ratio(s1_name, c_name) / 100.0)

    # 3. Jaccard Token Overlap
    features.append(jaccard_token_similarity(s1_name, c_name))

    # 4. Partial Ratio
    features.append(fuzz.partial_ratio(s1_name, c_name) / 100.0)

    # 5. Token Set Ratio
    features.append(fuzz.token_set_ratio(s1_name, c_name) / 100.0)

    # 6. Jaro-Winkler Similarity (prefix weighted)
    features.append(JaroWinkler.similarity(s1_name, c_name))

    # 7. Exact Clean Name Match flag
    features.append(1.0 if (s1_name and s1_name == c_name) else 0.0)

    # 8. Name Character 3-gram Cosine Similarity
    features.append(char_ngram_cosine(s1_name, c_name, n=3))

    # 9. Name Length Ratio
    l1, l2 = len(s1_name), len(c_name)
    features.append(min(l1, l2) / max(l1, l2) if max(l1, l2) > 0 else 0.0)

    # 10. Name Prefix Match (first 4 characters)
    features.append(1.0 if (s1_name and c_name and len(s1_name) >= 4 and len(c_name) >= 4 and s1_name[:4] == c_name[:4]) else 0.0)

    # 11. Name Containment (shorter inside longer)
    features.append(1.0 if (s1_name and c_name and (s1_name in c_name or c_name in s1_name)) else 0.0)

    # 12. Name Token Count Ratio
    tc1, tc2 = len(s1_name.split()), len(c_name.split())
    features.append(min(tc1, tc2) / max(tc1, tc2) if max(tc1, tc2) > 0 else 0.0)

    # --- ADDRESS FEATURES (8) ---
    # 13. Address Token Sort Ratio
    features.append(fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0)

    # 14. Address Levenshtein Ratio
    features.append(fuzz.ratio(s1_addr, c_addr) / 100.0)

    # 15. Address Jaccard Token Overlap
    features.append(jaccard_token_similarity(s1_addr, c_addr))

    # 16. Address Character 3-gram Cosine Similarity
    features.append(char_ngram_cosine(s1_addr, c_addr, n=3))

    # 17. Address Partial Ratio
    features.append(fuzz.partial_ratio(s1_addr, c_addr) / 100.0)

    # 18. Address Exact Match flag
    features.append(1.0 if (s1_addr and s1_addr == c_addr) else 0.0)

    # 19. Address Length Ratio
    al1, al2 = len(s1_addr), len(c_addr)
    features.append(min(al1, al2) / max(al1, al2) if max(al1, al2) > 0 else 0.0)

    # 20. Address Containment (shorter inside longer)
    features.append(1.0 if (s1_addr and c_addr and (s1_addr in c_addr or c_addr in s1_addr)) else 0.0)

    # --- NUMERIC FEATURES (2) ---
    # 21-22. Address numeric overlap (Jaccard + exact match flag)
    addr_num_jaccard, addr_num_exact = numeric_overlap(s1_addr_nums, c_addr_nums)
    features.append(addr_num_jaccard)
    features.append(float(addr_num_exact))

    # --- COMBINED FEATURES (2) ---
    # 23. Combined name+address Levenshtein Ratio
    features.append(fuzz.token_sort_ratio(s1_combined, c_combined) / 100.0)

    # 24. Country exact match flag
    features.append(1.0 if s1_country == c_country else 0.0)

    # 25. Precision Veto Flag (1.0 if pair violates street number or name disjointness, else 0.0)
    veto = evaluate_precision_veto(
        s1_name, s1_addr, s1_addr_nums,
        c_name, c_addr, c_addr_nums,
        features[0], features[5]
    )
    features.append(1.0 if veto else 0.0)

    return features


# Feature names for LightGBM and analysis
FEATURE_NAMES = [
    "name_token_sort_ratio",
    "name_levenshtein_ratio",
    "name_jaccard_overlap",
    "name_partial_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "name_exact_match",
    "name_char_ngram_cosine",
    "name_length_ratio",
    "name_prefix_match_4",
    "name_containment",
    "name_token_count_ratio",
    "addr_token_sort_ratio",
    "addr_levenshtein_ratio",
    "addr_jaccard_overlap",
    "addr_char_ngram_cosine",
    "addr_partial_ratio",
    "addr_exact_match",
    "addr_length_ratio",
    "addr_containment",
    "addr_numeric_jaccard",
    "addr_numeric_exact_match",
    "combined_token_sort_ratio",
    "country_exact_match",
    "precision_veto_flag",
]

NUM_FEATURES = len(FEATURE_NAMES)


def _compute_features_for_chunk(args):
    """
    Worker function for parallel feature computation.

    Processes a chunk of (source1_row, candidate_row) pairs and
    returns the feature matrix for that chunk.

    Args:
        args: Tuple of (chunk_data,) where chunk_data is a list of
              tuples containing (s1_fields, cand_fields)

    Returns:
        List of feature vectors for the chunk
    """
    chunk_data = args
    results = []
    for s1_fields, c_fields in chunk_data:
        features = compute_pair_features(*s1_fields, *c_fields)
        results.append(features)
    return results


def compute_features_batch(
    s1_df: pl.DataFrame,
    candidates_df: pl.DataFrame,
    candidate_pairs: Dict[str, List[str]],
    max_pairs: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute features for all (Source 1, candidate) pairs in batch.

    This is the main entry point for feature engineering. It takes
    the blocking output (candidate_pairs dict) and computes pairwise
    features for each (S1, candidate) pair.

    Args:
        s1_df: Preprocessed Source 1 DataFrame
        candidates_df: Preprocessed Source 2+3 DataFrame (combined)
        candidate_pairs: Dict from blocking: s1_id -> [candidate_ids]
        max_pairs: Optional limit on total pairs (for debugging)

    Returns:
        Tuple of:
        - feature_matrix: np.ndarray of shape (n_pairs, NUM_FEATURES)
        - s1_ids: List of Source 1 entity IDs for each pair
        - cand_ids: List of candidate entity IDs for each pair
    """
    print(f"[FEATURES] Computing pairwise features...")

    # Build lookup dictionaries for fast access
    # Map entity_id -> row data for Source 1
    s1_lookup = {}
    for row in s1_df.iter_rows(named=True):
        s1_lookup[row["entity_id"]] = (
            row.get("name_clean", ""),
            row.get("addr_clean", ""),
            row.get("name_numerics", ""),
            row.get("addr_numerics", ""),
            row.get("name_addr_combined", ""),
            row.get("country_clean", ""),
        )

    # Map entity_id -> row data for candidates (S2+S3)
    cand_lookup = {}
    for row in candidates_df.iter_rows(named=True):
        cand_lookup[row["entity_id"]] = (
            row.get("name_clean", ""),
            row.get("addr_clean", ""),
            row.get("name_numerics", ""),
            row.get("addr_numerics", ""),
            row.get("name_addr_combined", ""),
            row.get("country_clean", ""),
        )

    # Enumerate all pairs
    pair_data = []  # List of (s1_fields, cand_fields)
    s1_ids = []
    cand_ids = []

    for s1_id, cand_list in candidate_pairs.items():
        if s1_id not in s1_lookup:
            continue
        s1_fields = s1_lookup[s1_id]

        for cand_id in cand_list:
            if cand_id not in cand_lookup:
                continue
            c_fields = cand_lookup[cand_id]
            pair_data.append((s1_fields, c_fields))
            s1_ids.append(s1_id)
            cand_ids.append(cand_id)

            if max_pairs and len(pair_data) >= max_pairs:
                break
        if max_pairs and len(pair_data) >= max_pairs:
            break

    n_pairs = len(pair_data)
    print(f"[FEATURES]   Total pairs to compute: {n_pairs:,}")

    if n_pairs == 0:
        return np.array([]).reshape(0, NUM_FEATURES), [], []

    # Compute features in parallel using thread pool
    # (RapidFuzz releases GIL, so threads are effective)
    chunk_size = max(1000, n_pairs // (N_CORES * 4))
    chunks = [pair_data[i:i+chunk_size] for i in range(0, n_pairs, chunk_size)]

    print(f"[FEATURES]   Processing {len(chunks)} chunks across {N_CORES} cores...")

    all_features = []
    with ThreadPoolExecutor(max_workers=N_CORES) as executor:
        futures = [executor.submit(_compute_features_for_chunk, chunk) for chunk in chunks]

        for i, future in enumerate(tqdm(futures, desc="  Computing features")):
            chunk_features = future.result()
            all_features.extend(chunk_features)

    feature_matrix = np.array(all_features, dtype=np.float32)
    print(f"[FEATURES]   Feature matrix shape: {feature_matrix.shape}")

    # Clean up
    del pair_data, all_features, s1_lookup, cand_lookup
    import gc
    gc.collect()

    return feature_matrix, s1_ids, cand_ids


def compute_features_for_pairs_streaming(
    s1_rows: List[dict],
    cand_rows_lookup: Dict[str, dict],
    candidate_pairs: Dict[str, List[str]],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Streaming feature computation for inference.

    Optimized for memory-constrained inference where we process
    one batch of Source 1 entities at a time.

    Args:
        s1_rows: List of Source 1 row dicts (preprocessed)
        cand_rows_lookup: Dict mapping entity_id -> preprocessed row dict
        candidate_pairs: Dict mapping s1_id -> [candidate_ids]

    Returns:
        Tuple of (feature_matrix, s1_ids, cand_ids)
    """
    pair_data = []
    s1_ids = []
    cand_ids = []

    for s1_row in s1_rows:
        s1_id = s1_row["entity_id"]
        s1_fields = (
            s1_row.get("name_clean", ""),
            s1_row.get("addr_clean", ""),
            s1_row.get("name_numerics", ""),
            s1_row.get("addr_numerics", ""),
            s1_row.get("name_addr_combined", ""),
            s1_row.get("country_clean", ""),
        )

        for cand_id in candidate_pairs.get(s1_id, []):
            if cand_id not in cand_rows_lookup:
                continue
            c_row = cand_rows_lookup[cand_id]
            c_fields = (
                c_row.get("name_clean", ""),
                c_row.get("addr_clean", ""),
                c_row.get("name_numerics", ""),
                c_row.get("addr_numerics", ""),
                c_row.get("name_addr_combined", ""),
                c_row.get("country_clean", ""),
            )
            pair_data.append((s1_fields, c_fields))
            s1_ids.append(s1_id)
            cand_ids.append(cand_id)

    if not pair_data:
        return np.array([]).reshape(0, NUM_FEATURES), [], []

    # Compute features (single-threaded for small batches, parallel for large)
    n_pairs = len(pair_data)
    if n_pairs > 10000:
        chunk_size = max(1000, n_pairs // (N_CORES * 4))
        chunks = [pair_data[i:i+chunk_size] for i in range(0, n_pairs, chunk_size)]

        all_features = []
        with ThreadPoolExecutor(max_workers=N_CORES) as executor:
            futures = [executor.submit(_compute_features_for_chunk, chunk) for chunk in chunks]
            for future in futures:
                all_features.extend(future.result())
    else:
        all_features = _compute_features_for_chunk(pair_data)

    return np.array(all_features, dtype=np.float32), s1_ids, cand_ids


if __name__ == "__main__":
    # Quick test with synthetic data
    print(f"Feature names ({NUM_FEATURES}): {FEATURE_NAMES}")

    # Test feature computation
    features = compute_pair_features(
        "acme corporation", "123 main street new york ny", "123", "123 10001",
        "acme corporation 123 main street new york ny", "us",
        "acme corp", "123 main st new york", "123", "123",
        "acme corp 123 main st new york", "us",
    )
    print(f"\nTest features ({len(features)}):")
    for name, val in zip(FEATURE_NAMES, features):
        print(f"  {name}: {val:.4f}")
