"""
main.py - Orchestrator Script
================================
End-to-end orchestrator that runs the full entity resolution pipeline:
1. Training: Preprocess → Block → Feature → Train → Threshold Sweep
2. Inference: Preprocess → Block → Feature → Predict → Output

Usage:
    # Full pipeline (train + infer):
    python -m src.main

    # Training only:
    python -m src.main --train-only

    # Inference only (requires pre-trained model):
    python -m src.main --infer-only

    # Quick test with 1% data sample:
    python -m src.main --sample 0.01

Command is run from: code/business_entity_resolution/
"""

import sys
import time
import argparse
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ensure_directories,
    validate_data_paths,
    OUTPUT_DIR,
    MODEL_PATH,
    THRESHOLD_PATH,
    MATCHING_RESULTS_FILE,
    CANDIDATE_PAIRS_FILE,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main                  # Full pipeline (train + infer)
  python -m src.main --train-only     # Training only
  python -m src.main --infer-only     # Inference only
  python -m src.main --sample 0.01    # Quick test with 1%% sample
        """
    )

    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Run only the training pipeline (no inference)"
    )
    parser.add_argument(
        "--infer-only",
        action="store_true",
        help="Run only the inference pipeline (requires pre-trained model)"
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=None,
        help="Fraction of data to sample for quick testing (e.g., 0.01 for 1%%)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override inference batch size (default: 50000)"
    )

    return parser.parse_args()


def main():
    """
    Main entry point for the entity resolution pipeline.

    Execution flow:
    1. Validate configuration and create directories
    2. Run training pipeline (if not --infer-only)
    3. Run inference pipeline (if not --train-only)
    4. Print final summary
    """
    args = parse_args()

    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 70)

    start_time = time.time()

    # ------------------------------------------------------------------
    # Step 0: Validate and setup
    # ------------------------------------------------------------------
    print("\n[MAIN] Step 0: Validating configuration...")
    validate_data_paths()
    ensure_directories()

    # ------------------------------------------------------------------
    # Step 1: Training
    # ------------------------------------------------------------------
    if not args.infer_only:
        print("\n" + "=" * 70)
        print("[MAIN] PHASE 1: TRAINING")
        print("=" * 70)

        train_start = time.time()

        from src.train import train_pipeline
        model, threshold = train_pipeline(sample_fraction=args.sample)

        train_elapsed = time.time() - train_start
        print(f"\n[MAIN] Training completed in {train_elapsed/60:.1f} minutes")
    else:
        print("\n[MAIN] Skipping training (--infer-only mode)")
        if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
            print("[MAIN] ERROR: No trained model found. Run training first.")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Inference
    # ------------------------------------------------------------------
    if not args.train_only:
        print("\n" + "=" * 70)
        print("[MAIN] PHASE 2: INFERENCE")
        print("=" * 70)

        infer_start = time.time()

        from src.infer import inference_pipeline

        batch_size = args.batch_size if args.batch_size else None
        if batch_size:
            inference_pipeline(batch_size=batch_size)
        else:
            inference_pipeline()

        infer_elapsed = time.time() - infer_start
        print(f"\n[MAIN] Inference completed in {infer_elapsed/60:.1f} minutes")
    else:
        print("\n[MAIN] Skipping inference (--train-only mode)")

    # ------------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------------
    total_elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n[MAIN] Total runtime: {total_elapsed/60:.1f} minutes")

    if MATCHING_RESULTS_FILE.exists():
        # Count output lines
        with open(str(MATCHING_RESULTS_FILE), 'r') as f:
            n_results = sum(1 for _ in f) - 1  # Subtract header
        print(f"[MAIN] matching_results.tsv: {n_results:,} entities")

    if CANDIDATE_PAIRS_FILE.exists():
        with open(str(CANDIDATE_PAIRS_FILE), 'r') as f:
            n_candidates = sum(1 for _ in f) - 1
        print(f"[MAIN] candidate_pairs.tsv: {n_candidates:,} entities")

    print(f"\n[MAIN] Output directory: {OUTPUT_DIR}")
    print("[MAIN] Done!")


if __name__ == "__main__":
    main()
