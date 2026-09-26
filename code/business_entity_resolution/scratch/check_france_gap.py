import polars as pl
import numpy as np
import lightgbm as lgb
import sys

sys.path.insert(0, r"c:\DEV\ML_challange\code\business_entity_resolution")
from src.features import compute_pair_features

print("Diagnosing French entities specifically...")
s1 = pl.read_parquet(r"c:\DEV\ML_challange\code\business_entity_resolution\cache\test_source1_preprocessed.parquet")
s1_fr = s1.filter(pl.col("country_clean") == "france")
print(f"Total French S1 entities in test: {len(s1_fr):,} ({len(s1_fr)/len(s1)*100:.1f}%)")

s1_us = s1.filter(pl.col("country_clean") == "us")
s1_in = s1.filter(pl.col("country_clean") == "india")
print(f"US S1 entities: {len(s1_us):,} ({len(s1_us)/len(s1)*100:.1f}%)")
print(f"India S1 entities: {len(s1_in):,} ({len(s1_in)/len(s1)*100:.1f}%)")

out = pl.read_csv(r"c:\DEV\ML_challange\output\matching_results.tsv", separator="\t")
out.columns = [c.strip() for c in out.columns]

# Join with test country
s1_country = dict(zip(s1["entity_id"].to_list(), s1["country_clean"].to_list()))
out_with_country = out.with_columns(pl.col("source1_entity_id").map_elements(lambda x: s1_country.get(x, "unknown"), return_dtype=pl.Utf8).alias("country"))

for c in ["us", "india", "france"]:
    sub = out_with_country.filter(pl.col("country") == c)
    n_total = len(sub)
    n_empty = len(sub.filter(pl.col("matched_entity_ids").is_null() | (pl.col("matched_entity_ids") == "")))
    non_empty = sub.filter(~(pl.col("matched_entity_ids").is_null() | (pl.col("matched_entity_ids") == "")))
    m_counts = non_empty["matched_entity_ids"].map_elements(lambda x: len(str(x).split(",")), return_dtype=pl.Int32) if len(non_empty) > 0 else []
    avg_m = m_counts.mean() if len(m_counts) > 0 else 0
    print(f"[{c.upper()}] Total: {n_total:,} | Empty (singletons): {n_empty:,} ({n_empty/n_total*100:.2f}%) | Avg matches: {avg_m:.2f}")
