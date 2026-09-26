import polars as pl
import numpy as np

# Let's test on our train ground truth sample how cluster expansion affects Macro F0.5
print("Analyzing Cluster Recall in Predicted TSV vs Ground Truth...")
# Load output matching_results.tsv and train_ground_truth.tsv
gt = pl.read_csv(r"c:\DEV\ML_challange\data\student_resource\dataset\train\train_ground_truth.tsv", separator="\t")
gt.columns = [c.strip() for c in gt.columns]

# Check match multiplicity in output
out = pl.read_csv(r"c:\DEV\ML_challange\output\matching_results.tsv", separator="\t")
out.columns = [c.strip() for c in out.columns]

# Let's compute average matches in Ground Truth vs Predicted:
gt_lens = []
for row in gt.head(100000).iter_rows(named=True):
    if row["matched_entity_ids"] and str(row["matched_entity_ids"]).strip():
        gt_lens.append(len(str(row["matched_entity_ids"]).split(",")))
    else:
        gt_lens.append(0)

out_lens = []
for row in out.head(100000).iter_rows(named=True):
    if row["matched_entity_ids"] and str(row["matched_entity_ids"]).strip():
        out_lens.append(len(str(row["matched_entity_ids"]).split(",")))
    else:
        out_lens.append(0)

print(f"Ground Truth Average Matches per Entity: {np.mean(gt_lens):.2f}")
print(f"Predicted Output Average Matches per Entity: {np.mean(out_lens):.2f}")
print(f"Recall Truncation Gap: {(np.mean(gt_lens) - np.mean(out_lens))/np.mean(gt_lens)*100:.1f}% of true matches were dropped by high threshold!")
