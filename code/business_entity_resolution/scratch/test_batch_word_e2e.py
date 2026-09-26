import polars as pl
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor
import time

print("Testing End-to-End Word Blocking on 10,000 test queries...", flush=True)

s1 = pl.read_parquet(r"c:\DEV\ML_challange\code\business_entity_resolution\cache\test_source1_preprocessed.parquet").head(10000)
s2 = pl.read_parquet(r"c:\DEV\ML_challange\code\business_entity_resolution\cache\test_source2_preprocessed.parquet").filter(pl.col("country_clean") == "india").head(1000000)

cand_texts = s2["name_addr_combined"].to_list()
cand_texts = [t if t and str(t).strip() else "unknown" for t in cand_texts]
cand_ids = s2["entity_id"].to_list()

t0 = time.time()
vec = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.015,
    max_features=80000,
    sublinear_tf=True,
    dtype=np.float32,
    norm="l2"
)
cand_mat = vec.fit_transform(cand_texts)
cand_mat_T = cand_mat.T
print(f"Candidate index for 1,000,000 Indian businesses built in {time.time()-t0:.2f}s | NNZ: {cand_mat.nnz:,}", flush=True)

# India queries
q_df = s1.filter(pl.col("country_clean") == "india")
q_texts = q_df["name_addr_combined"].to_list()
q_texts = [t if t and str(t).strip() else "unknown" for t in q_texts]
q_ids = q_df["entity_id"].to_list()
n_queries = len(q_ids)

print(f"Transforming {n_queries} queries...", flush=True)
t0 = time.time()
q_mat_all = vec.transform(q_texts)

# Parallel retrieval with effective_batch = 2000
sub_batch_size = 2000
starts = list(range(0, n_queries, sub_batch_size))
top_k = 25

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
        
        n_elem = len(rv)
        if n_elem <= top_k:
            top_l = np.argsort(-rv)
        else:
            p = np.argpartition(-rv, top_k)[:top_k]
            top_l = p[np.argsort(-rv[p])]
        sub_res[qid] = [cand_ids[rc[idx]] for idx in top_l[:top_k]]
    return sub_res

print(f"Executing parallel search for {n_queries} queries...", flush=True)
t0 = time.time()
with ThreadPoolExecutor(max_workers=6) as ex:
    chunk_results = list(ex.map(worker, starts))
search_time = time.time() - t0

total_retrieved = sum(len(res) for res in chunk_results)
print(f"Retrieved candidates for {total_retrieved} queries in {search_time:.2f}s ({n_queries/search_time:.1f} queries/s)", flush=True)
