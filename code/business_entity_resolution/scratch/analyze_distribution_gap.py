import polars as pl
import sys
sys.stdout.reconfigure(encoding='utf-8')

gt = pl.read_csv(r"c:\DEV\ML_challange\data\student_resource\dataset\train\train_ground_truth.tsv", separator="\t")
gt.columns = [c.strip() for c in gt.columns]

total = len(gt)
empty_matches = gt.filter(pl.col("matched_entity_ids").is_null() | (pl.col("matched_entity_ids") == "") | (pl.col("matched_entity_ids").str.strip_chars() == ""))
n_empty = len(empty_matches)
n_has_match = total - n_empty

print(f"Total Source 1 entities in train ground truth: {total:,}")
print(f"Entities with NO match (Singletons): {n_empty:,} ({n_empty/total*100:.2f}%)")
print(f"Entities WITH match: {n_has_match:,} ({n_has_match/total*100:.2f}%)")

non_empty = gt.filter(~(pl.col("matched_entity_ids").is_null() | (pl.col("matched_entity_ids") == "") | (pl.col("matched_entity_ids").str.strip_chars() == "")))
match_counts = non_empty["matched_entity_ids"].map_elements(lambda x: len(str(x).split(",")), return_dtype=pl.Int32)
print(f"\nGround Truth match counts: Mean={match_counts.mean():.2f}, Median={match_counts.median()}, Min={match_counts.min()}, Max={match_counts.max()}")

# Quantiles
for q in [0.25, 0.5, 0.75, 0.90, 0.95, 0.99]:
    print(f"  q{int(q*100)}: {match_counts.quantile(q)}")

out = pl.read_csv(r"c:\DEV\ML_challange\output\matching_results.tsv", separator="\t")
out.columns = [c.strip() for c in out.columns]
out_total = len(out)
out_empty = out.filter(pl.col("matched_entity_ids").is_null() | (pl.col("matched_entity_ids") == "") | (pl.col("matched_entity_ids").str.strip_chars() == ""))
out_n_empty = len(out_empty)
print(f"\nPredicted matching_results.tsv:")
print(f"Total: {out_total:,}")
print(f"Empty (predicted singletons): {out_n_empty:,} ({out_n_empty/out_total*100:.2f}%)")
print(f"Non-empty (predicted matches): {out_total - out_n_empty:,} ({(out_total - out_n_empty)/out_total*100:.2f}%)")

out_non_empty = out.filter(~(pl.col("matched_entity_ids").is_null() | (pl.col("matched_entity_ids") == "") | (pl.col("matched_entity_ids").str.strip_chars() == "")))
out_match_counts = out_non_empty["matched_entity_ids"].map_elements(lambda x: len(str(x).split(",")), return_dtype=pl.Int32)
print(f"Predicted match counts: Mean={out_match_counts.mean():.2f}, Median={out_match_counts.median()}, Min={out_match_counts.min()}, Max={out_match_counts.max()}")
for q in [0.25, 0.5, 0.75, 0.90, 0.95, 0.99]:
    print(f"  q{int(q*100)}: {out_match_counts.quantile(q)}")
