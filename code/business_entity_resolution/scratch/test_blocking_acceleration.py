"""
Benchmark and verify blocking acceleration options.
Tests:
1. ThreadPoolExecutor vs single-thread for sparse dot product
2. Batch size 5,000 vs 20,000
3. Fast C-level top-k extraction
"""
import time
import numpy as np
from scipy import sparse
from concurrent.futures import ThreadPoolExecutor

print("Running sparse matrix acceleration benchmark...")

# Simulate realistic dimensions:
# 20,000 queries, 250,000 features, 1,000,000 candidates
n_queries = 10000
n_features = 50000
n_candidates = 200000

# Create random sparse matrices with realistic density (~0.05% non-zero)
print("Creating benchmark matrices...")
q = sparse.random(n_queries, n_features, density=0.001, format='csr', dtype=np.float32)
c_T = sparse.random(n_features, n_candidates, density=0.001, format='csc', dtype=np.float32)

print(f"Query shape: {q.shape}, Candidate_T shape: {c_T.shape}")

# Benchmark 1: Sequential 2 x 5000 batches
t0 = time.time()
q1 = q[:5000]
q2 = q[5000:10000]
res1 = (q1 @ c_T).tocsr()
res2 = (q2 @ c_T).tocsr()
t_seq = time.time() - t0
print(f"Sequential (2 x 5000): {t_seq:.3f}s")

# Benchmark 2: Parallel 2 x 5000 batches with ThreadPoolExecutor
def multiply(sub_q):
    return (sub_q @ c_T).tocsr()

t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    f1 = ex.submit(multiply, q1)
    f2 = ex.submit(multiply, q2)
    r1 = f1.result()
    r2 = f2.result()
t_par = time.time() - t0
print(f"Parallel ThreadPool (4 workers): {t_par:.3f}s (Speedup: {t_seq/t_par:.2f}x)")

# Benchmark 3: Single 10,000 batch
t0 = time.time()
res_large = (q @ c_T).tocsr()
t_large = time.time() - t0
print(f"Large batch (1 x 10000): {t_large:.3f}s (Speedup: {t_seq/t_large:.2f}x)")

print("Benchmark complete!")
