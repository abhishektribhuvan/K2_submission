"""
Benchmark parallel sub-batch execution with ThreadPoolExecutor vs sequential.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from scipy import sparse
import numpy as np

# Load or create realistic test matrices
n_features = 50000
n_candidates = 500000
q_batch = 500

print("Generating benchmark matrices...")
np.random.seed(42)
q = sparse.random(2000, n_features, density=0.002, format='csr', dtype=np.float32)
c_T = sparse.random(n_features, n_candidates, density=0.002, format='csc', dtype=np.float32)

def run_subbatch(start, end):
    sub_q = q[start:end]
    sim = (sub_q @ c_T).tocsr()
    return sim.nnz

# 1. Sequential 4 sub-batches
t0 = time.time()
seq_nnz = [run_subbatch(i, i + q_batch) for i in range(0, 2000, q_batch)]
t_seq = time.time() - t0
print(f"Sequential (4 x 500 queries): {t_seq:.3f}s")

# 2. Parallel 4 sub-batches with ThreadPoolExecutor (max_workers=4)
t0 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    futures = [ex.submit(run_subbatch, i, i + q_batch) for i in range(0, 2000, q_batch)]
    par_nnz = [f.result() for f in futures]
t_par = time.time() - t0
print(f"Parallel ThreadPool (4 workers): {t_par:.3f}s")
print(f"Speedup: {t_seq / t_par:.2f}x")
print(f"Equal: {seq_nnz == par_nnz}")
