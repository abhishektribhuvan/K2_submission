import polars as pl
import numpy as np
import lightgbm as lgb
import sys

sys.path.insert(0, r"c:\DEV\ML_challange\code\business_entity_resolution")
from src.features import compute_pair_features
from src.train import compute_f05_per_entity

print("Comparing Option A (Inference Tuning) vs Current Baseline on Validation Set...")

# Load validation ground truth sample
gt = pl.read_csv(r"c:\DEV\ML_challange\data\student_resource\dataset\train\train_ground_truth.tsv", separator="\t")
gt.columns = [c.strip() for c in gt.columns]

s1 = pl.read_parquet(r"c:\DEV\ML_challange\code\business_entity_resolution\cache\train_source1_preprocessed.parquet").head(3000)
s1_ids = s1["entity_id"].to_list()
s1_set = set(s1_ids)

gt_sample = {}
for row in gt.iter_rows(named=True):
    s1_id = str(row["source1_entity_id"]).strip()
    if s1_id in s1_set:
        m_ids = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()] if row["matched_entity_ids"] else []
        gt_sample[s1_id] = set(m_ids)

all_true_ids = set()
for v in gt_sample.values():
    all_true_ids.update(v)

cols = ["entity_id", "name_clean", "addr_clean", "name_numerics", "addr_numerics", "name_addr_combined", "country_clean"]
s2 = pl.read_parquet(r"c:\DEV\ML_challange\code\business_entity_resolution\cache\train_source2_preprocessed.parquet")
s3 = pl.read_parquet(r"c:\DEV\ML_challange\code\business_entity_resolution\cache\train_source3_preprocessed.parquet")

s2_sub = s2.head(50000).select(cols)
s3_sub = s3.head(50000).select(cols)
s2_true = s2.filter(pl.col("entity_id").is_in(list(all_true_ids))).select(cols)
s3_true = s3.filter(pl.col("entity_id").is_in(list(all_true_ids))).select(cols)
cands = pl.concat([s2_sub, s3_sub, s2_true, s3_true]).unique(subset=["entity_id"])

from src.blocking import CountryBlockingIndex
blocker = CountryBlockingIndex(top_k=25)
blocker.build_index(cands)
candidates = blocker.query_batch(s1, batch_size=2000)

cand_map = {}
for row in cands.iter_rows(named=True):
    cand_map[row["entity_id"]] = (
        row["name_clean"], row["addr_clean"], row["name_numerics"],
        row["addr_numerics"], row["name_addr_combined"], row["country_clean"]
    )

s1_names = s1["name_clean"].to_list()
s1_addrs = s1["addr_clean"].to_list()
s1_n_nums = s1["name_numerics"].to_list()
s1_a_nums = s1["addr_numerics"].to_list()
s1_combs = s1["name_addr_combined"].to_list()
s1_cntrys = s1["country_clean"].to_list()

pair_data = []
pair_s1 = []
pair_cand = []

for i, qid in enumerate(s1_ids):
    c_list = candidates.get(qid, [])
    s1_tup = (s1_names[i], s1_addrs[i], s1_n_nums[i], s1_a_nums[i], s1_combs[i], s1_cntrys[i])
    for cid in c_list:
        if cid in cand_map:
            pair_data.append((s1_tup, cand_map[cid]))
            pair_s1.append(qid)
            pair_cand.append(cid)

features = [compute_pair_features(*s1_t, *c_t) for s1_t, c_t in pair_data]
feat_arr = np.array(features, dtype=np.float32)
vetoes = (feat_arr[:, 24] == 1.0)

model = lgb.Booster(model_file=r"c:\DEV\ML_challange\code\business_entity_resolution\models\lgbm_entity_resolution.txt")
preds = model.predict(feat_arr)

# 1. Baseline (Hard veto + 0.865)
base_preds = {qid: set() for qid in s1_ids}
for qid, cid, prob, is_v in zip(pair_s1, pair_cand, preds, vetoes):
    if not is_v and prob >= 0.865:
        base_preds[qid].add(cid)
base_score = np.mean([compute_f05_per_entity(gt_sample[qid], base_preds[qid]) for qid in s1_ids])
print(f"Current Baseline (Hard Veto + 0.865 Cutoff): Macro F0.5 = {base_score:.4f}")

# 2. Option A1: Softened Veto (Veto only applies if prob < 0.90) + Calibrated Threshold
for th in [0.70, 0.75, 0.80, 0.85]:
    optA_preds = {qid: set() for qid in s1_ids}
    for qid, cid, prob, is_v in zip(pair_s1, pair_cand, preds, vetoes):
        # Soft veto: if prob is very high (> 0.92), trust the model over the address typo
        if is_v and prob < 0.92:
            continue
        if prob >= th:
            optA_preds[qid].add(cid)
    scoreA = np.mean([compute_f05_per_entity(gt_sample[qid], optA_preds[qid]) for qid in s1_ids])
    print(f"Option A (Soft Veto + Threshold {th}): Macro F0.5 = {scoreA:.4f}")

# 3. Option A2: Soft Veto + Adaptive Companion Clustering
# If an entity has a top match >= 0.90, admit companion matches from same cluster down to 0.70
optA2_preds = {qid: set() for qid in s1_ids}
# Group pairs by query
q_pairs = {}
for qid, cid, prob, is_v in zip(pair_s1, pair_cand, preds, vetoes):
    q_pairs.setdefault(qid, []).append((cid, prob, is_v))

for qid in s1_ids:
    p_list = q_pairs.get(qid, [])
    if not p_list: continue
    max_prob = max(p for _, p, _ in p_list)
    effective_th = 0.70 if max_prob >= 0.88 else 0.85
    for cid, prob, is_v in p_list:
        if is_v and prob < 0.92:
            continue
        if prob >= effective_th:
            optA2_preds[qid].add(cid)

scoreA2 = np.mean([compute_f05_per_entity(gt_sample[qid], optA2_preds[qid]) for qid in s1_ids])
print(f"Option A2 (Soft Veto + Adaptive Companion Clustering): Macro F0.5 = {scoreA2:.4f}")
