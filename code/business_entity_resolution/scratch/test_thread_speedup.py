import time
import numpy as np
import scipy.sparse as sp
from concurrent.futures import ThreadPoolExecutor

print("Testing SciPy sparse dot concurrency...")
# Create a mock candidate matrix resembling India (115,000 features x 4.7M candidates, 50M nnz)
# For quick test, create (50,000 x 500,000, 5M nnz)
n_features = 30000
n_candidates = 200000
nnz = 2000000

# Random sparse matrix
data = np.ones(nnz, dtype=np.float32)
row = np.random.randint(0, n_features, size=nnz)
col = np.random.randint(0, n_candidates, size=nnz)
candidate_T = sp.coo_matrix((data, (row, col)), shape=(n_features, n_candidates), dtype=np.float32).tocsc()

print(f"Candidate matrix shape: {candidate_T.shape}, nnz: {candidate_T.nnz}")

# Create queries: 4000 queries
n_queries = 4000
q_nnz = 200000
q_data = np.ones(q_nnz, dtype=np.float32)
q_row = np.random.randint(0, n_queries, size=q_nnz)
q_col = np.random.randint(0, n_features, size=q_nnz)
queries = sp.coo_matrix((q_data, (q_row, q_col)), shape=(n_queries, n_features), dtype=np.float32).tocsr()

# 1. Single-threaded 4 chunks of 1000
sub_queries = [queries[i:i+1000] for i in range(0, 4000, 1000)]

t0 = time.time()
for sq in sub_queries:
    res = sq @ candidate_T
t_seq = time.time() - t0
print(f"Sequential (1 thread) 4x1000: {t_seq:.3f} s")

# 2. Multi-threaded 4 chunks with ThreadPoolExecutor
def worker(sq):
    return sq @ candidate_T

t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(worker, sub_queries))
t_par4 = time.time() - t0
print(f"ThreadPool (4 threads) 4x1000: {t_par4:.3f} s (Speedup: {t_seq/t_par4:.2f}x)")

# 3. Multi-threaded 8 chunks
sub_queries_8 = [queries[i:i+500] for i in range(0, 4000, 500)]
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(worker, sub_queries_8))
t_par8 = time.time() - t0
print(f"ThreadPool (8 threads) 8x500: {t_par8:.3f} s (Speedup: {t_seq/t_par8:.2f}x)")
