import polars as pl
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, r"c:\DEV\ML_challange\code\business_entity_resolution")
from src.features import compute_pair_features, evaluate_precision_veto

print("Diagnosing exact validation failure modes on 5,000 entities...", flush=True)

gt = pl.read_csv(r"c:\DEV\ML_challange\data\student_resource\dataset\train\train_ground_truth.tsv", separator="\t")
gt.columns = [c.strip() for c in gt.columns]

s1 = pl.read_parquet(r"c:\DEV\ML_challange\code\business_entity_resolution\cache\train_source1_preprocessed.parquet").head(5000)
s1_ids = s1["entity_id"].to_list()
s1_ids_set = set(s1_ids)

gt_sample = {}
for row in gt.iter_rows(named=True):
    s1_id = str(row["source1_entity_id"]).strip()
    if s1_id in s1_ids_set:
        m_ids = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()] if row["matched_entity_ids"] else []
        gt_sample[s1_id] = set(m_ids)

all_true_ids = set()
for v in gt_sample.values():
    all_true_ids.update(v)

cols = ["entity_id", "name_clean", "addr_clean", "name_numerics", "addr_numerics", "name_addr_combined", "country_clean"]
s2 = pl.read_parquet(r"c:\DEV\ML_challange\code\business_entity_resolution\cache\train_source2_preprocessed.parquet")
s3 = pl.read_parquet(r"c:\DEV\ML_challange\code\business_entity_resolution\cache\train_source3_preprocessed.parquet")

s2_sub = s2.head(80000).select(cols)
s3_sub = s3.head(80000).select(cols)
s2_true = s2.filter(pl.col("entity_id").is_in(list(all_true_ids))).select(cols)
s3_true = s3.filter(pl.col("entity_id").is_in(list(all_true_ids))).select(cols)

cands = pl.concat([s2_sub, s3_sub, s2_true, s3_true]).unique(subset=["entity_id"])
print(f"Sampled {len(s1)} S1 queries against {len(cands):,} candidates", flush=True)

# Build Word TF-IDF Blocker
from src.blocking import CountryBlockingIndex
blocker = CountryBlockingIndex(top_k=25)
blocker.build_index(cands)

# Query candidates
candidates = blocker.query_batch(s1, batch_size=2000)

# Check blocking recall
recall_hits = 0
recall_total = 0
for qid in s1_ids:
    true_set = gt_sample.get(qid, set())
    if not true_set: continue
    c_set = set(candidates.get(qid, []))
    for t in true_set:
        recall_total += 1
        if t in c_set:
            recall_hits += 1
print(f"Blocking Recall on Sample: {recall_hits/recall_total*100:.2f}% ({recall_hits}/{recall_total})", flush=True)

# Prepare candidate fast lookup
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

print(f"Total candidate pairs to score: {len(pair_data):,}", flush=True)

# Compute features with ThreadPool
def feat_worker(chunk):
    return [compute_pair_features(*s1_t, *c_t) for s1_t, c_t in chunk]

chunk_size = 2000
chunks = [pair_data[i:i+chunk_size] for i in range(0, len(pair_data), chunk_size)]
with ThreadPoolExecutor(max_workers=6) as ex:
    feature_chunks = list(ex.map(feat_worker, chunks))

all_features = []
for fc in feature_chunks:
    all_features.extend(fc)
feat_arr = np.array(all_features, dtype=np.float32)
vetoes = (feat_arr[:, 24] == 1.0)
print(f"Vetoes applied: {np.sum(vetoes):,} ({np.sum(vetoes)/len(vetoes)*100:.1f}%)")

# Predict with LightGBM
model = lgb.Booster(model_file=r"c:\DEV\ML_challange\code\business_entity_resolution\models\lgbm_entity_resolution.txt")
preds = model.predict(feat_arr)

# Test multiple thresholds to see the exact curve
from src.train import compute_f05_per_entity, compute_macro_f05

for thresh in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.865, 0.90, 0.95]:
    pred_dict = {qid: set() for qid in s1_ids}
    for idx, (qid, cid, prob, is_v) in enumerate(zip(pair_s1, pair_cand, preds, vetoes)):
        if not is_v and prob >= thresh:
            pred_dict[qid].add(cid)
            
    scores = []
    p_scores = []
    r_scores = []
    zero_scores = 0
    perfect_scores = 0
    singleton_fp = 0
    singleton_total = 0
    
    for qid in s1_ids:
        true_set = gt_sample.get(qid, set())
        pred_set = pred_dict[qid]
        score = compute_f05_per_entity(true_set, pred_set)
        scores.append(score)
        if score == 1.0:
            perfect_scores += 1
        elif score == 0.0:
            zero_scores += 1
            
        if len(true_set) == 0:
            singleton_total += 1
            if len(pred_set) > 0:
                singleton_fp += 1
        else:
            tp = len(true_set & pred_set)
            if len(pred_set) > 0:
                p_scores.append(tp / len(pred_set))
            if len(true_set) > 0:
                r_scores.append(tp / len(true_set))
                
    macro_f05 = np.mean(scores)
    avg_p = np.mean(p_scores) if p_scores else 0
    avg_r = np.mean(r_scores) if r_scores else 0
    print(f"Threshold {thresh:.3f}: Macro F0.5 = {macro_f05:.4f} | Prec = {avg_p:.3f} | Rec = {avg_r:.3f} | Perfect = {perfect_scores/len(s1_ids)*100:.1f}% | Zero = {zero_scores/len(s1_ids)*100:.1f}% | Singleton FP = {singleton_fp}/{singleton_total}", flush=True)
