"""
Verify pre-filtering speedup on top-k extraction.
"""
import time
import numpy as np

print("Testing pre-filtering vs full argpartition...")

# Simulate 1 query with 100,000 non-zero similarities (mostly small < 0.05)
np.random.seed(42)
vals = np.random.exponential(scale=0.03, size=100_000).astype(np.float32)
# Add some true high matches
vals[:25] = np.random.uniform(0.3, 0.9, size=25).astype(np.float32)
cols = np.arange(100_000, dtype=np.int32)
top_k = 25

# Method 1: Current approach (argpartition on full 100,000 array)
t0 = time.time()
for _ in range(500):
    top_local = np.argpartition(-vals, top_k)[:top_k]
    top_local = top_local[np.argsort(-vals[top_local])]
    res1 = cols[top_local]
t_old = time.time() - t0
print(f"Current method (500 queries): {t_old:.3f}s")

# Method 2: Fast pre-filter (drop < 0.10 threshold, fallback if < 25)
t0 = time.time()
for _ in range(500):
    mask = vals > 0.10
    if np.count_nonzero(mask) >= top_k:
        f_vals = vals[mask]
        f_cols = cols[mask]
        if len(f_vals) <= top_k:
            top_local = np.argsort(-f_vals)
        else:
            top_local = np.argpartition(-f_vals, top_k)[:top_k]
            top_local = top_local[np.argsort(-f_vals[top_local])]
        res2 = f_cols[top_local]
    else:
        top_local = np.argpartition(-vals, top_k)[:top_k]
        top_local = top_local[np.argsort(-vals[top_local])]
        res2 = cols[top_local]
t_new = time.time() - t0
print(f"Pre-filtered method (500 queries): {t_new:.3f}s")
print(f"Speedup: {t_old / t_new:.2f}x")
print(f"Top 5 matches equal: {np.array_equal(res1[:5], res2[:5])}")
