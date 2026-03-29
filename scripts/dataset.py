#!/usr/bin/env python
"""Preprocess raw keypoint data into NPZ format with validation and feature engineering."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from sklearn.model_selection import train_test_split

from handflow.data.loader import (
    get_config_hash,
    load_raw_data,
    save_processed_data,
)
from handflow.data.validator import SequenceValidator
from handflow.features import FeatureEngineer
from handflow.utils import load_config
from handflow.utils.logging import setup_logging, get_logger

logger = get_logger("handflow.dataset")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess HandFlow data")

    parser.add_argument(
        "--config",
        type=str,
        default=Path("config/config.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=None,
        help="Validation set fraction",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if cache is valid",
    )

    return parser.parse_args()


def validate_sequences(
    sequences: np.ndarray, 

    labels: np.ndarray,
    config: ValidationConfig | None = None,
    paths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate and filter sequences.
    
    Args:
        sequences: Raw keypoint sequences
        labels: Corresponding labels
        
    Returns:
        Tuple of (valid_sequences, valid_labels)
    """
    
    validator = SequenceValidator(config)
    
    valid_indices = []
    rejected_count = 0
    rejection_reasons = {}
    
    for i, seq in enumerate(sequences):
        result = validator.validate(seq)
        if result.is_valid:
            valid_indices.append(i)
        else:
            rejected_count += 1
            reason = result.reason.split(":")[0] if result.reason else "Unknown"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
     
    valid_sequences = sequences[valid_indices]
    valid_labels = labels[valid_indices]
    valid_paths = paths[valid_indices] if paths is not None else None

    
    if len(sequences) > 0:
        logger.info(f"✅ Valid: {len(valid_indices)}/{len(sequences)} ({len(valid_indices)/len(sequences):.1%})")
    else:
        logger.info(f"✅ Valid: 0/0 (0.0%)")

    if rejected_count > 0:
        logger.info(f"❌ Rejected: {rejected_count} ({rejection_reasons})")
    
    if paths is not None:
        return valid_sequences, valid_labels, valid_paths
    return valid_sequences, valid_labels


def apply_feature_engineering(
    sequences: np.ndarray, feature_engineer: FeatureEngineer
) -> np.ndarray:
    """Apply feature engineering to all sequences."""
    enhanced = []
    for seq in sequences:
        enhanced.append(feature_engineer.transform(seq))
    return np.array(enhanced)


def preprocess_hand(
    hand: str,
    config,
    validation_split: float,
    output_dir: Path,
) -> None:
    """
    Preprocess data for one hand.

    Args:
        hand: "right" or "left"
        config: Configuration object
        validation_split: Fraction for validation set
        output_dir: Directory to save processed files
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Preprocessing {hand.upper()} hand data")
    logger.info(f"{'='*60}")

    # Get paths and actions
    if hand == "right":
        data_path = Path(config.data.right_raw_path)
        actions = config.model.gestures
    else:
        data_path = Path(config.data.left_raw_path)
        actions = config.model.gestures

    logger.info(f"Data path: {data_path}")
    logger.info(f"ctions: {actions}")

    # Check if data exists
    if not data_path.exists():
        logger.info(f"⚠️ Data path not found, skipping: {data_path}")
        return

    logger.info("\n1. Loading raw data...")
    sequences, labels, paths = load_raw_data(
        data_path, actions, config.data.sequence_length, return_paths=True
    )
    logger.info(f"   Loaded {len(sequences)} sequences")
    logger.info(f"   Raw keypoints shape: {sequences.shape} (21 landmarks x 4 = 84 raw features)")

    logger.info("\n2. Validating data...")
    sequences, labels, paths = validate_sequences(
        sequences, labels, config, paths=paths
    )
    logger.info("\n3. Applying feature engineering...")
    feature_engineer = FeatureEngineer()
    logger.info(f"   FeatureEngineer output_dim: {feature_engineer.get_output_dim()}")
    X = apply_feature_engineering(sequences, feature_engineer)
    y = labels
    logger.info(f"   Engineered features shape: {X.shape}")
    logger.info(f"   Input dimension for model: {X.shape[-1]}")

    logger.info(f"\n4. Splitting data (val={validation_split:.0%})...")
    X_train, X_val, y_train, y_val, paths_train, paths_val = train_test_split(
        X, y, paths,
        test_size=validation_split,
        random_state=42,
        stratify=y.argmax(axis=1),
    )
    logger.info(f"   Training: {len(X_train)} samples")
    logger.info(f"   Validation: {len(X_val)} samples")

    config_hash = get_config_hash(config)

    logger.info("\n5. Saving processed data...")
    
    train_path = output_dir / f"{hand}_train.npz"
    save_processed_data(train_path, X_train, y_train, actions, config_hash, paths=paths_train)
    
    val_path = output_dir / f"{hand}_val.npz"
    save_processed_data(val_path, X_val, y_val, actions, config_hash, paths=paths_val)

    logger.info(f"\n✅ {hand.upper()} hand preprocessing complete!")

def main() -> None:
    """Main preprocessing function."""
    args = parse_args()

    log_file = "logs/dataset.log"
    setup_logging(level="INFO", log_file=log_file)

    logger = get_logger("handflow.dataset")

    # Load configuration
    config = load_config(args.config)

    if (args.validation_split): 
        config.training.validation_split = args.validation_split

    logger.info("🖐️ HandFlow Data Preprocessing")
    logger.info("=" * 60)

    logger.info(f"Config: {args.config or 'default'}")
    logger.info(f"Validation split: {config.training.validation_split:.0%}")

    # Output directory
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process requested hand(s)
    hands_to_process = ["right", "left"]

    for hand in hands_to_process:
        preprocess_hand(
            hand=hand,
            config=config,
            validation_split=config.training.validation_split,
            output_dir=output_dir,
        )

    logger.info("=" * 60)
    logger.info("✅ All preprocessing complete!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
