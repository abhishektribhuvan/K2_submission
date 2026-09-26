import polars as pl
import numpy as np
import lightgbm as lgb
from pathlib import Path
import sys

print("Diagnosing exact Macro F0.5 breakdown...")

# Load model
model = lgb.Booster(model_file=r"c:\DEV\ML_challange\code\business_entity_resolution\models\lgbm_entity_resolution.txt")

# Load a chunk of train queries and candidates
# We have cache/train_source1_preprocessed.parquet, etc.
s1 = pl.read_parquet(r"c:\DEV\ML_challange\code\business_entity_resolution\cache\train_source1_preprocessed.parquet").head(10000)
gt = pl.read_csv(r"c:\DEV\ML_challange\data\student_resource\dataset\train\train_ground_truth.tsv", separator="\t")
gt.columns = [c.strip() for c in gt.columns]

gt_map = {}
for row in gt.iter_rows(named=True):
    s1_id = str(row["source1_entity_id"]).strip()
    m_ids = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()] if row["matched_entity_ids"] else []
    gt_map[s1_id] = set(m_ids)

print(f"Total ground truth mapped: {len(gt_map):,}")

# Check how many entities in train ground truth have multiple matches
match_lens = [len(v) for v in gt_map.values()]
print(f"Average true matches per entity: {np.mean(match_lens):.2f}")
print(f"Entities with 0 matches (singletons): {sum(1 for l in match_lens if l == 0)} ({sum(1 for l in match_lens if l == 0)/len(match_lens)*100:.1f}%)")
print(f"Entities with 1 match: {sum(1 for l in match_lens if l == 1)} ({sum(1 for l in match_lens if l == 1)/len(match_lens)*100:.1f}%)")
print(f"Entities with 2 matches: {sum(1 for l in match_lens if l == 2)} ({sum(1 for l in match_lens if l == 2)/len(match_lens)*100:.1f}%)")
print(f"Entities with 3+ matches: {sum(1 for l in match_lens if l >= 3)} ({sum(1 for l in match_lens if l >= 3)/len(match_lens)*100:.1f}%)")
