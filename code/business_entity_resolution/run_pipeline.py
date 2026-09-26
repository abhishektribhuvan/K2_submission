"""
run_pipeline.py - Full Training & Inference Pipeline Orchestrator
==================================================================
Runs:
1. Training with 15% sample (~330,000 S1 entities, ~3.8M labeled pairs)
   - Evaluates on stratified 20% validation split
   - Determines optimal decision threshold for macro F0.5
   - Saves model to models/lgbm_entity_resolution.txt
2. Test Inference
   - Preprocesses & caches test S2/S3 (including France)
   - Builds country-partitioned TF-IDF blocking index
   - Streams S1 in batches to generate candidate pairs & predictions
   - Writes output/candidate_pairs.tsv and output/matching_results.tsv
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.train import train_pipeline
from src.infer import inference_pipeline


def main():
    t0 = time.time()
    print("=" * 80)
    print("STARTING END-TO-END ENTITY RESOLUTION PIPELINE")
    print("=" * 80)

    # Step 1: Train model on 15% stratified sample (~3.8M training pairs)
    print("\n[PIPELINE] Phase 1: Training LightGBM Model (sample_fraction=0.15)...")
    t_train_start = time.time()
    model, threshold = train_pipeline(sample_fraction=0.15)
    t_train_elapsed = time.time() - t_train_start
    print(f"\n[PIPELINE] Training completed in {t_train_elapsed / 60:.1f} minutes.")
    print(f"[PIPELINE] Selected Optimal Threshold: {threshold:.4f}")

    # Step 2: Run inference on full test dataset
    print("\n[PIPELINE] Phase 2: Running Inference on Full Test Set...")
    t_infer_start = time.time()
    inference_pipeline()
    t_infer_elapsed = time.time() - t_infer_start
    print(f"\n[PIPELINE] Inference completed in {t_infer_elapsed / 60:.1f} minutes.")

    total_time = time.time() - t0
    print("=" * 80)
    print(f"PIPELINE FULLY COMPLETE IN {total_time / 60:.1f} MINUTES!")
    print("=" * 80)


if __name__ == "__main__":
    main()
