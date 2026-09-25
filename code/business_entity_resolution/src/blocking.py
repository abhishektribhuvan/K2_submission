"""
blocking.py - Sparse TF-IDF Candidate Generation
===================================================
Implements a memory-efficient blocking strategy using sparse TF-IDF
character n-gram vectors and cosine similarity for candidate retrieval.

Key design decisions:
1. Country-level hard filtering: Source 1 records only match candidates
   from the same country (prevents cross-country false positives).
2. Character 3-gram/4-gram TF-IDF with sublinear TF scaling (mimics BM25
   behavior) for robustness to typos and transliteration.
3. Sparse matrix operations throughout to stay within 16GB RAM for
   10M+ total records.
4. Batched query processing to control peak memory usage.
"""

import gc
import numpy as np
import polars as pl
from scipy.sparse import vstack as sparse_vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from src.config import (
    ANALYZER,
    NGRAM_RANGE,
    MAX_FEATURES,
    MIN_DF,
    MAX_DF,
    SUBLINEAR_TF,
    TOP_K_CANDIDATES,
    CANDIDATE_PAIRS_FILE,
    TFIDF_FIT_BATCH_SIZE,
)


class TFIDFBlocker:
    """
    TF-IDF-based blocking index for candidate generation.

    Builds per-country TF-IDF indexes on combined name+address text.
    For each Source 1 query, retrieves the top-K most similar candidates
    from Source 2/Source 3 within the same country using cosine similarity.

    Attributes:
        top_k: Number of candidates to retrieve per query
        vectorizers: Dict mapping country -> fitted TfidfVectorizer
        candidate_matrices: Dict mapping country -> sparse TF-IDF matrix
        candidate_ids: Dict mapping country -> list of entity_id strings
    """

    def __init__(self, top_k: int = TOP_K_CANDIDATES):
        self.top_k = top_k
        self.vectorizers: Dict[str, TfidfVectorizer] = {}
        self.candidate_matrices = {}
        self.candidate_ids: Dict[str, List[str]] = {}

    def build_index(
        self,
        candidates_df: pl.DataFrame,
        text_col: str = "name_addr_combined",
        country_col: str = "country_clean",
        id_col: str = "entity_id",
    ) -> None:
        """
        Build per-country TF-IDF indexes from candidate (Source 2+3) data.

        For each unique country in the candidate data:
        1. Extract the combined text column
        2. Fit a TF-IDF vectorizer on those texts
        3. Transform texts into a sparse TF-IDF matrix
        4. Store the vectorizer, matrix, and entity IDs for querying

        Args:
            candidates_df: Preprocessed DataFrame of S2+S3 records
            text_col: Column containing combined name+address text
            country_col: Column containing cleaned country label
            id_col: Column containing entity_id
        """
        # Get unique countries
        countries = candidates_df[country_col].unique().to_list()
        print(f"[BLOCKING] Building TF-IDF indexes for {len(countries)} countries: {countries}")

        for country in countries:
            print(f"\n[BLOCKING] Processing country: '{country}'")

            # Filter candidates for this country
            country_df = candidates_df.filter(pl.col(country_col) == country)
            n_candidates = len(country_df)
            print(f"[BLOCKING]   {n_candidates:,} candidate records")

            # Extract texts and IDs
            texts = country_df[text_col].to_list()
            ids = country_df[id_col].to_list()

            # Replace None/empty with a placeholder to avoid vectorizer issues
            texts = [t if t and str(t).strip() else "unknown" for t in texts]

            # Create and fit TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                analyzer=ANALYZER,         # Fast word-level token analysis
                ngram_range=NGRAM_RANGE,    # Word 1-2 n-grams
                max_features=MAX_FEATURES,  # Cap vocabulary to control memory
                min_df=MIN_DF,              # Ignore very rare terms
                max_df=MAX_DF,              # Ignore ubiquitous background terms (>25%)
                sublinear_tf=SUBLINEAR_TF,  # Log-normalize TF (BM25-like)
                dtype=np.float32,           # Use float32 to halve memory
                norm='l2',                  # L2-normalize for cosine sim via dot product
            )

            # Fit and transform in one pass
            print(f"[BLOCKING]   Fitting TF-IDF vectorizer (ngrams={NGRAM_RANGE}, "
                  f"max_features={MAX_FEATURES:,})...")

            # For very large candidate sets, fit in batches
            if n_candidates > TFIDF_FIT_BATCH_SIZE:
                # Fit on a subsample, then transform in batches
                print(f"[BLOCKING]   Large dataset - fitting on subsample of "
                      f"{TFIDF_FIT_BATCH_SIZE:,} records")

                # Fit on a representative sample
                sample_indices = np.random.RandomState(42).choice(
                    n_candidates,
                    size=min(TFIDF_FIT_BATCH_SIZE, n_candidates),
                    replace=False
                )
                sample_texts = [texts[i] for i in sample_indices]
                vectorizer.fit(sample_texts)
                del sample_texts
                gc.collect()

                # Transform in batches
                print(f"[BLOCKING]   Transforming {n_candidates:,} records in batches...")
                batch_matrices = []
                for batch_start in range(0, n_candidates, TFIDF_FIT_BATCH_SIZE):
                    batch_end = min(batch_start + TFIDF_FIT_BATCH_SIZE, n_candidates)
                    batch_texts = texts[batch_start:batch_end]
                    batch_matrix = vectorizer.transform(batch_texts)
                    batch_matrices.append(batch_matrix)
                    if batch_start % (TFIDF_FIT_BATCH_SIZE * 2) == 0:
                        print(f"[BLOCKING]     Transformed {batch_end:,}/{n_candidates:,}")

                tfidf_matrix = sparse_vstack(batch_matrices, format='csr')
                del batch_matrices
                gc.collect()
            else:
                tfidf_matrix = vectorizer.fit_transform(texts)

            print(f"[BLOCKING]   TF-IDF matrix shape: {tfidf_matrix.shape}, "
                  f"nnz: {tfidf_matrix.nnz:,}")

            # Prune hyper-frequent feature columns (>5,000 docs) to eliminate dot product bottleneck
            col_counts = np.diff(tfidf_matrix.tocsc().indptr)
            frequent_cols = np.where(col_counts > 5000)[0]
            if len(frequent_cols) > 0:
                print(f"[BLOCKING]   Pruning {len(frequent_cols):,} hyper-frequent features (>5k docs)...")
                tfidf_csc = tfidf_matrix.tocsc()
                for col in frequent_cols:
                    start_i = tfidf_csc.indptr[col]
                    end_i = tfidf_csc.indptr[col + 1]
                    tfidf_csc.data[start_i:end_i] = 0.0
                tfidf_csc.eliminate_zeros()
                tfidf_matrix = tfidf_csc.tocsr()
                print(f"[BLOCKING]   Pruned TF-IDF matrix nnz: {tfidf_matrix.nnz:,}")

            # Store the index components
            self.vectorizers[country] = vectorizer
            self.candidate_matrices[country] = tfidf_matrix
            self.candidate_ids[country] = ids

            del texts
            gc.collect()

        print(f"\n[BLOCKING] Index build complete for {len(countries)} countries")

    def query_batch(
        self,
        query_df: pl.DataFrame,
        text_col: str = "name_addr_combined",
        country_col: str = "country_clean",
        id_col: str = "entity_id",
        batch_size: int = 5000,
    ) -> Dict[str, List[str]]:
        """
        Retrieve top-K candidates for a batch of Source 1 queries.

        For each query:
        1. Look up the country-specific index
        2. Transform query text with the same TF-IDF vectorizer
        3. Compute cosine similarity against all candidates in that country
        4. Return top-K candidate IDs sorted by similarity (descending)

        Args:
            query_df: Preprocessed DataFrame of Source 1 queries
            text_col: Combined text column for queries
            country_col: Country column for routing to correct index
            id_col: Entity ID column
            batch_size: Sub-batch size for cosine similarity computation

        Returns:
            Dict mapping query entity_id -> list of candidate entity_ids
        """
        results = {}
        total = len(query_df)

        # Group queries by country for efficient processing
        for country in query_df[country_col].unique().to_list():
            country_queries = query_df.filter(pl.col(country_col) == country)
            n_queries = len(country_queries)

            if country not in self.vectorizers:
                # No candidates exist for this country - return empty lists
                print(f"[BLOCKING] WARNING: No index for country '{country}' "
                      f"({n_queries:,} queries will have no candidates)")
                for qid in country_queries[id_col].to_list():
                    results[qid] = []
                continue

            vectorizer = self.vectorizers[country]
            candidate_matrix = self.candidate_matrices[country]
            candidate_ids = self.candidate_ids[country]
            n_candidates = len(candidate_ids)

            # Use batch_size = 5,000 for high-throughput word token sparse dot product
            effective_batch = min(batch_size, 5000)

            print(f"[BLOCKING] Querying {n_queries:,} records for country '{country}' "
                  f"against {n_candidates:,} candidates "
                  f"(batch_size={effective_batch:,})")

            # Extract query data
            query_texts = country_queries[text_col].to_list()
            query_ids = country_queries[id_col].to_list()
            query_texts = [t if t and str(t).strip() else "unknown" for t in query_texts]

            # Transpose candidate matrix once for efficient sparse dot product
            candidate_matrix_T = candidate_matrix.T.tocsc()

            # Process in sub-batches
            for i in tqdm(range(0, n_queries, effective_batch),
                         desc=f"  Blocking ({country})",
                         total=(n_queries + effective_batch - 1) // effective_batch):
                batch_end = min(i + effective_batch, n_queries)
                batch_texts = query_texts[i:batch_end]
                batch_ids = query_ids[i:batch_end]

                # Transform queries (already L2-normalized by vectorizer)
                query_matrix = vectorizer.transform(batch_texts)

                # Sparse dot product: since both matrices are L2-normalized,
                # dot product = cosine similarity.
                sim_sparse = (query_matrix @ candidate_matrix_T).tocsr()

                # Fast SciPy CSR memory slicing using indptr / indices / data
                indptr = sim_sparse.indptr
                indices = sim_sparse.indices
                data = sim_sparse.data

                for j, qid in enumerate(batch_ids):
                    start_idx = indptr[j]
                    end_idx = indptr[j + 1]

                    if start_idx == end_idx:
                        results[qid] = []
                        continue

                    row_vals = data[start_idx:end_idx]
                    row_cols = indices[start_idx:end_idx]

                    n_elem = len(row_vals)
                    if n_elem <= self.top_k:
                        top_local = np.argsort(-row_vals)
                    else:
                        top_local = np.argpartition(-row_vals, self.top_k)[:self.top_k]
                        top_local = top_local[np.argsort(-row_vals[top_local])]

                    top_candidates = []
                    for li in top_local:
                        if row_vals[li] > 0.0:
                            cid = candidate_ids[row_cols[li]]
                            if cid != qid and cid not in top_candidates:
                                top_candidates.append(cid)

                    results[qid] = top_candidates

                del sim_sparse, query_matrix
                gc.collect()

            del candidate_matrix_T
            gc.collect()

        # Ensure every query has an entry (even if empty)
        for qid in query_df[id_col].to_list():
            if qid not in results:
                results[qid] = []

        return results

    def save_candidate_pairs(
        self,
        candidates: Dict[str, List[str]],
        output_path: str,
    ) -> None:
        """
        Write candidate pairs to TSV file.

        Format:
            source1_entity_id<TAB>candidate_entity_ids
            S1-001<TAB>S2-047,S3-812
            S1-002<TAB>
            ...

        Validates no duplicates exist within any candidate list.

        Args:
            candidates: Dict mapping source1_id -> list of candidate IDs
            output_path: Path to write the TSV file
        """
        print(f"[BLOCKING] Writing candidate pairs to: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")

            for s1_id in sorted(candidates.keys()):
                cand_list = candidates[s1_id]

                # Deduplicate while preserving order
                seen = set()
                deduped = []
                for cid in cand_list:
                    if cid not in seen:
                        seen.add(cid)
                        deduped.append(cid)

                # Validate: no Source 1 IDs in candidate list
                deduped = [c for c in deduped if not c.startswith("S1-")]

                cand_str = ",".join(deduped)
                f.write(f"{s1_id}\t{cand_str}\n")

        n_with_cands = sum(1 for v in candidates.values() if len(v) > 0)
        n_empty = sum(1 for v in candidates.values() if len(v) == 0)
        print(f"[BLOCKING] Written {len(candidates):,} entries "
              f"({n_with_cands:,} with candidates, {n_empty:,} empty)")


def build_blocking_index(s2_df: pl.DataFrame, s3_df: pl.DataFrame) -> TFIDFBlocker:
    """
    Build the blocking index from Source 2 and Source 3 data.

    Concatenates S2 and S3 into a single candidate pool, then
    builds per-country TF-IDF indexes.

    Args:
        s2_df: Preprocessed Source 2 DataFrame
        s3_df: Preprocessed Source 3 DataFrame

    Returns:
        Fitted TFIDFBlocker instance
    """
    print("[BLOCKING] Combining Source 2 and Source 3 candidates...")

    # Select only needed columns to minimize memory
    cols = ["entity_id", "name_addr_combined", "country_clean"]
    candidates_df = pl.concat([
        s2_df.select(cols),
        s3_df.select(cols),
    ])
    print(f"[BLOCKING] Total candidates: {len(candidates_df):,}")

    # Build the TF-IDF index
    blocker = TFIDFBlocker(top_k=TOP_K_CANDIDATES)
    blocker.build_index(candidates_df)

    del candidates_df
    gc.collect()

    return blocker


def generate_candidates(
    blocker: TFIDFBlocker,
    s1_df: pl.DataFrame,
    output_path: Optional[str] = None,
    batch_size: int = 5000,
) -> Dict[str, List[str]]:
    """
    Generate candidate pairs for all Source 1 entities.

    Args:
        blocker: Fitted TFIDFBlocker instance
        s1_df: Preprocessed Source 1 DataFrame
        output_path: Optional path to save candidate_pairs.tsv
        batch_size: Sub-batch size for cosine similarity

    Returns:
        Dict mapping source1_id -> list of candidate IDs
    """
    print(f"\n[BLOCKING] Generating candidates for {len(s1_df):,} Source 1 entities...")

    candidates = blocker.query_batch(s1_df, batch_size=batch_size)

    # Save to file if path provided
    if output_path:
        blocker.save_candidate_pairs(candidates, output_path)

    # Statistics
    n_total = len(candidates)
    n_with = sum(1 for v in candidates.values() if len(v) > 0)
    avg_cands = np.mean([len(v) for v in candidates.values()]) if candidates else 0
    print(f"[BLOCKING] Candidate generation complete:")
    print(f"[BLOCKING]   Total Source 1 entities: {n_total:,}")
    print(f"[BLOCKING]   With candidates: {n_with:,}")
    print(f"[BLOCKING]   Average candidates per entity: {avg_cands:.1f}")

    return candidates


if __name__ == "__main__":
    # Quick test with a small sample
    from src.config import TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3
    from src.preprocess import load_and_preprocess

    print("Loading small sample for blocking test...")
    s1 = load_and_preprocess(str(TRAIN_SOURCE1))
    s1 = s1.head(100)  # Just 100 queries for testing

    s2 = load_and_preprocess(str(TRAIN_SOURCE2))
    s2 = s2.head(10000)  # 10K candidates

    s3 = load_and_preprocess(str(TRAIN_SOURCE3))
    s3 = s3.head(10000)

    blocker = build_blocking_index(s2, s3)
    candidates = generate_candidates(blocker, s1)

    for s1_id, cands in list(candidates.items())[:5]:
        print(f"  {s1_id}: {len(cands)} candidates -> {cands[:3]}...")
