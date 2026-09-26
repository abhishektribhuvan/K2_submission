import polars as pl
import time
import numpy as np
import scipy.sparse as sp

print("Checking actual time on US candidate sample...")
# Load 50k US candidates from parquet
df = pl.read_parquet(r"c:\DEV\ML_challange\code\business_entity_resolution\cache\test_source2_preprocessed.parquet")
us_cand = df.filter(pl.col("country_clean") == "us").head(100000)
print(f"Loaded {len(us_cand)} US candidates")

from sklearn.feature_extraction.text import TfidfVectorizer

t0 = time.time()
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, max_features=100000)
texts = us_cand["name_addr_combined"].to_list()
texts = [t if t and str(t).strip() else "unknown" for t in texts]
X = vec.fit_transform(texts)
print(f"Fit transform: {time.time()-t0:.2f}s, shape: {X.shape}, nnz: {X.nnz}")

XT = X.T

# Query 1000 records
q_texts = texts[:1000]
Q = vec.transform(q_texts)

t0 = time.time()
sim = Q @ XT
t_dot = time.time() - t0
print(f"Dot product 1000 queries vs 100k cands: {t_dot:.3f}s, nnz: {sim.nnz}")

t0 = time.time()
indptr = sim.indptr
indices = sim.indices
data = sim.data
top_k = 25
for j in range(1000):
    s = indptr[j]
    e = indptr[j+1]
    if s == e: continue
    rv = data[s:e]
    rc = indices[s:e]
    mask = rv > 0.08
    if np.count_nonzero(mask) >= top_k:
        lv = rv[mask]
        lc = rc[mask]
        if len(lv) <= top_k:
            idx = np.argsort(-lv)
        else:
            p = np.argpartition(-lv, top_k)[:top_k]
            idx = p[np.argsort(-lv[p])]
    else:
        if len(rv) <= top_k:
            idx = np.argsort(-rv)
        else:
            p = np.argpartition(-rv, top_k)[:top_k]
            idx = p[np.argsort(-rv[p])]
t_extract = time.time() - t0
print(f"Row extraction: {t_extract:.3f}s")
