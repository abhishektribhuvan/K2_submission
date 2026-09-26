import polars as pl
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor
import time
import gc

print("Testing end-to-end accelerated blocking...", flush=True)

# Load test data
s1 = pl.read_parquet(r"c:\DEV\ML_challange\code\business_entity_resolution\cache\test_source1_preprocessed.parquet").head(10000)
s2 = pl.read_parquet(r"c:\DEV\ML_challange\code\business_entity_resolution\cache\test_source2_preprocessed.parquet").filter(pl.col("country_clean") == "india").head(500000)

texts = s2["name_addr_combined"].to_list()
texts = [t if t and str(t).strip() else "unknown" for t in texts]
cand_ids = s2["entity_id"].to_list()

print(f"Building candidate index for 500,000 India candidates...", flush=True)
t0 = time.time()
vec = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 4),
    min_df=3,
    max_df=0.015,
    max_features=120000,
    sublinear_tf=True,
    dtype=np.float32,
    norm="l2"
)
cand_mat = vec.fit_transform(texts)
cand_mat_T = cand_mat.T
print(f"Index built in {time.time()-t0:.2f}s | shape: {cand_mat.shape} | nnz: {cand_mat.nnz:,}", flush=True)

q_df = s1.filter(pl.col("country_clean") == "india")
q_texts = q_df["name_addr_combined"].to_list()
q_texts = [t if t and str(t).strip() else "unknown" for t in q_texts]
q_ids = q_df["entity_id"].to_list()
n_queries = len(q_ids)

print(f"Transforming {n_queries} queries...", flush=True)
t0 = time.time()
q_mat_all = vec.transform(q_texts)
print(f"Transformed queries in {time.time()-t0:.2f}s", flush=True)

# Parallel query
top_k = 25
sub_batch_size = 500
starts = list(range(0, n_queries, sub_batch_size))

def worker(start_i):
    end_i = min(start_i + sub_batch_size, n_queries)
    sub_ids = q_ids[start_i:end_i]
    sub_q = q_mat_all[start_i:end_i]
    sim = sub_q @ cand_mat_T
    
    indptr = sim.indptr
    indices = sim.indices
    data = sim.data
    
    sub_res = {}
    for j, qid in enumerate(sub_ids):
        s = indptr[j]
        e = indptr[j+1]
        if s == e:
            sub_res[qid] = []
            continue
        rv = data[s:e]
        rc = indices[s:e]
        
        mask = rv > 0.08
        if np.count_nonzero(mask) >= top_k:
            lv = rv[mask]
            lc = rc[mask]
            if len(lv) <= top_k:
                top_l = np.argsort(-lv)
            else:
                top_l = np.argpartition(-lv, top_k)[:top_k]
                top_l = top_l[np.argsort(-lv[top_l])]
            chosen = [cand_ids[lc[idx]] for idx in top_l[:top_k] if lc[idx] < len(cand_ids)]
        else:
            if len(rv) <= top_k:
                top_l = np.argsort(-rv)
            else:
                top_l = np.argpartition(-rv, top_k)[:top_k]
                top_l = top_l[np.argsort(-rv[top_l])]
            chosen = [cand_ids[rc[idx]] for idx in top_l[:top_k] if rc[idx] < len(cand_ids)]
            
        sub_res[qid] = chosen
    return sub_res

print(f"Querying {n_queries} records with 4 threads...", flush=True)
t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    chunk_results = list(ex.map(worker, starts))
dt = time.time() - t0

all_results = {}
for cr in chunk_results:
    all_results.update(cr)

print(f"Done! {n_queries} queries processed in {dt:.2f}s ({n_queries/dt:.1f} queries/s)", flush=True)
non_empty = sum(1 for v in all_results.values() if len(v) > 0)
print(f"Entities with candidates: {non_empty}/{n_queries} ({non_empty/n_queries*100:.1f}%)", flush=True)
