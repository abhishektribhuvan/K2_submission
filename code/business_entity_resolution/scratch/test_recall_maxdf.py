import polars as pl
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import sys

print("Testing exact recall across max_df...", flush=True)
gt = pl.read_csv(r"c:\DEV\ML_challange\data\student_resource\dataset\train\train_ground_truth.tsv", separator="\t")
gt.columns = [c.strip() for c in gt.columns]

gt_matches = gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != "")).head(1000)

gt_map = {}
all_matched_cands = set()
for row in gt_matches.iter_rows(named=True):
    s1_id = str(row["source1_entity_id"]).strip()
    m_ids = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()]
    gt_map[s1_id] = set(m_ids)
    all_matched_cands.update(m_ids)

print(f"Sampled {len(gt_map)} query ground truth entities", flush=True)

cols = ["entity_id", "country_clean", "name_addr_combined"]
s1 = pl.scan_parquet(r"c:\DEV\ML_challange\code\business_entity_resolution\cache\train_source1_preprocessed.parquet").select(cols)
s2 = pl.scan_parquet(r"c:\DEV\ML_challange\code\business_entity_resolution\cache\train_source2_preprocessed.parquet").select(cols)

s1_sub = s1.filter(pl.col("entity_id").is_in(list(gt_map.keys()))).collect()
s2_sub = s2.filter(pl.col("country_clean") == "us").head(50000).collect()
s2_extra = s2.filter(pl.col("entity_id").is_in(list(all_matched_cands))).collect()
s2_sub = pl.concat([s2_sub, s2_extra]).unique(subset=["entity_id"])

s1_texts = s1_sub["name_addr_combined"].to_list()
s2_texts = s2_sub["name_addr_combined"].to_list()
s1_ids = s1_sub["entity_id"].to_list()
s2_ids = s2_sub["entity_id"].to_list()

print(f"Test query set: {len(s1_ids)}, Candidate pool: {len(s2_ids):,}", flush=True)

for max_df in [1.0, 0.05, 0.02, 0.01]:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, max_df=max_df if max_df < 1.0 else 1.0, max_features=100000)
    cand_mat = vec.fit_transform(s2_texts)
    q_mat = vec.transform(s1_texts)
    
    t0 = time.time()
    sim = q_mat @ cand_mat.T
    dt = time.time() - t0
    
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
    print(f"max_df={max_df:5}: Recall={recall*100:6.2f}% ({hits}/{total}) | dot_time={dt:.3f}s | cand_nnz={cand_mat.nnz:,}", flush=True)
