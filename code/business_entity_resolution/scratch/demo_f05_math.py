import polars as pl
import numpy as np

# Let's test the mathematical effect on F0.5
# Suppose an entity has k true matches.
# If we capture m out of k matches with 100% precision:
print("F0.5 score when Precision is 100% (0 false positives) but Recall is incomplete:")
for k in [2, 3, 4, 5]:
    for m in range(1, k + 1):
        p = 1.0
        r = m / k
        f05 = (1.25 * p * r) / (0.25 * p + r)
        print(f"  True matches={k}, Captured={m}/{k} (Recall={r:.2f}) -> F0.5 = {f05:.4f}")
