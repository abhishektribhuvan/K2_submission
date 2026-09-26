import polars as pl
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor
import time
import sys

print("Testing Word-level (1-2 gram) Blocking Speed & Recall...", flush=True)

# 1. Load Ground Truth to test recall
gt = pl.read_csv(r"c:\DEV\ML_challange\data\student_resource\dataset\train\train_ground_truth.tsv", separator="\t")
gt.columns = [c.strip() for c in gt.columns]
gt_matches = gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != "")).head(1500)

gt_map = {}
all_matched_cands = set()
for row in gt_matches.iter_rows(named=True):
    s1_id = str(row["source1_entity_id"]).strip()
    m_ids = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()]
    gt_map[s1_id] = set(m_ids)
    all_matched_cands.update(m_ids)

cols = ["entity_id", "country_clean", "name_addr_combined"]
s1 = pl.scan_parquet(r"c:\DEV\ML_challange\code\business_entity_resolution\cache\train_source1_preprocessed.parquet").select(cols)
s2 = pl.scan_parquet(r"c:\DEV\ML_challange\code\business_entity_resolution\cache\train_source2_preprocessed.parquet").select(cols)

s1_sub = s1.filter(pl.col("entity_id").is_in(list(gt_map.keys()))).collect()
s2_sub = s2.filter(pl.col("country_clean") == "us").head(100000).collect()
s2_extra = s2.filter(pl.col("entity_id").is_in(list(all_matched_cands))).collect()
s2_sub = pl.concat([s2_sub, s2_extra]).unique(subset=["entity_id"])

s1_texts = s1_sub["name_addr_combined"].to_list()
s2_texts = s2_sub["name_addr_combined"].to_list()
s1_ids = s1_sub["entity_id"].to_list()
s2_ids = s2_sub["entity_id"].to_list()

print(f"Sampled {len(s1_ids)} query entities vs {len(s2_ids):,} candidate entities", flush=True)

# Test configurations
configs = [
    ("Char 3-4 (Current)", "char_wb", (3, 4), 3, 0.015, 120000),
    ("Word 1-2 (min_df=5)", "word", (1, 2), 5, 0.01, 80000),
    ("Word 1-2 (min_df=3)", "word", (1, 2), 3, 0.015, 100000),
]

for name, analyzer, ngrams, min_df, max_df, max_feat in configs:
    t0 = time.time()
    vec = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngrams,
        min_df=min_df,
        max_df=max_df,
        max_features=max_feat,
        sublinear_tf=True,
        dtype=np.float32,
        norm="l2"
    )
    cand_mat = vec.fit_transform(s2_texts)
    q_mat = vec.transform(s1_texts)
    build_time = time.time() - t0
    
    t0 = time.time()
    sim = q_mat @ cand_mat.T
    dot_time = time.time() - t0
    
    # Check Recall
    hits = 0
    total = 0
    top_k = 25
    indptr = sim.indptr
    indices = sim.indices
    data = sim.data
    
    for i, qid in enumerate(s1_ids):
        true_s2 = gt_map.get(qid, set())
        if not true_s2: continue
        true_present = true_s2.intersection(set(s2_ids))
        if not true_present: continue
        total += len(true_present)
        
        s = indptr[i]
        e = indptr[i+1]
        if s == e: continue
        rv = data[s:e]
        rc = indices[s:e]
        if len(rv) <= top_k:
            top_cand_indices = rc
        else:
            p = np.argpartition(-rv, top_k)[:top_k]
            top_cand_indices = rc[p]
        top_cand_ids = {s2_ids[idx] for idx in top_cand_indices}
        hits += len(true_present.intersection(top_cand_ids))
        
    recall = hits / total if total > 0 else 0
    print(f"[{name}]")
    print(f"  Matrix NNZ: {cand_mat.nnz:,} ({cand_mat.nnz/len(s2_texts):.1f} nnz/doc)")
    print(f"  Dot product time for {len(s1_ids)} queries: {dot_time:.4f}s")
    print(f"  Top-25 Recall: {recall*100:.2f}% ({hits}/{total})\n", flush=True)
