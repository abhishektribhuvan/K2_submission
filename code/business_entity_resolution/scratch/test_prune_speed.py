import polars as pl
import time
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

df = pl.read_parquet(r"c:\DEV\ML_challange\code\business_entity_resolution\cache\test_source2_preprocessed.parquet")
us_cand = df.filter(pl.col("country_clean") == "us").head(100000)
texts = us_cand["name_addr_combined"].to_list()
texts = [t if t and str(t).strip() else "unknown" for t in texts]

for max_df in [1.0, 0.05, 0.02, 0.01, 0.005]:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=3, max_df=max_df if max_df < 1.0 else 1.0, max_features=100000)
    X = vec.fit_transform(texts)
    XT = X.T
    Q = vec.transform(texts[:1000])
    t0 = time.time()
    sim = Q @ XT
    t_dot = time.time() - t0
    print(f"max_df={max_df:5}: X nnz={X.nnz:,} | dot nnz={sim.nnz:,} | dot time={t_dot:.3f}s")
