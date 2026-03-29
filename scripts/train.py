#!/usr/bin/env python
"""Train unified gesture recognition model. Supports resume from checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from tensorflow import keras

from handflow.data.loader import (
    check_processed_data_valid,
    load_processed_data,
)
from handflow.features import FeatureEngineer
from handflow.models import Trainer, build_model
from handflow.utils import load_config
from handflow.utils.logging import setup_logging, get_logger


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train HandFlow unified gesture model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train from scratch
    python scripts/train.py --architecture tcn --epochs 100

    # Resume training from a checkpoint with lower learning rate
    python scripts/train.py --resume models/hand_action.keras --epochs 50 --lr 0.00001

    # Resume from a specific checkpoint
    python scripts/train.py --resume models/checkpoints/tcn_20260203_best.h5 --lr 0.0001 --epochs 100
        """
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        choices=["lstm", "gru", "cnn1d", "transformer", "tcn"],
        help="Model architecture (ignored if --resume is used)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=Path("config/config.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model checkpoint to resume training from (.keras or .h5)",
    )
    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=None,
        dest="learning_rate",
        help="Starting learning rate (overrides config)",
    )
    return parser.parse_args()


def apply_feature_engineering(
    sequences: np.ndarray, feature_engineer: FeatureEngineer
) -> np.ndarray:
    """Apply feature engineering to all sequences."""
    enhanced = []
    for seq in sequences:
        enhanced.append(feature_engineer.transform(seq))
    return np.array(enhanced)


def load_training_data(
    config
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load processed data from cache for both hands and merge them.
    Uses flip canonicalization so both hands use the same model.

    Args:
        config: Configuration object

    Returns:
        Tuple of (x_train, x_val, y_train, y_val, actions)
    """
    processed_dir = Path("data/processed")
    
    # Paths for both hands
    right_train_path = processed_dir / "right_train.npz"
    right_val_path = processed_dir / "right_val.npz"
    left_train_path = processed_dir / "left_train.npz"
    left_val_path = processed_dir / "left_val.npz"

    actions = config.model.gestures

    # Check if all cached data files are valid
    all_paths = [right_train_path, right_val_path, left_train_path, left_val_path]
    all_valid = all(check_processed_data_valid(p, config) for p in all_paths)

    if not all_valid:
        raise FileNotFoundError(
            f"Processed data not found: {processed_dir}\n"
            "Run preprocessing first: python scripts/preprocess.py"
        )

    # Load data from both hands
    x_train_right, y_train_right, _ = load_processed_data(right_train_path)
    x_val_right, y_val_right, _ = load_processed_data(right_val_path)
    x_train_left, y_train_left, _ = load_processed_data(left_train_path)
    x_val_left, y_val_left, _ = load_processed_data(left_val_path)

    # Merge data from both hands
    x_train = np.concatenate([x_train_right, x_train_left], axis=0)
    y_train = np.concatenate([y_train_right, y_train_left], axis=0)
    x_val = np.concatenate([x_val_right, x_val_left], axis=0)
    y_val = np.concatenate([y_val_right, y_val_left], axis=0)

    return x_train, x_val, y_train, y_val, actions


def main() -> None:
    """Main training function."""
    args = parse_args()

    log_file = "logs/training.log"
    setup_logging(level="INFO", log_file=log_file)
    logger = get_logger("handflow.training")

    # Load configuration
    config = load_config(args.config)

    # Override config with CLI arguments
    if args.architecture and not args.resume:
        config.model.architecture = args.architecture
    if args.epochs:
        config.training.epochs = args.epochs
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate

    # Check resume path exists
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error(f"Resume model not found: {args.resume}")
            sys.exit(1)
        logger.info(f"Will resume training from: {args.resume}")

    output_path = Path(config.model.output_dir) / "hand_action.keras"

    logger.info(f"{'='*60}")
    if args.resume:
        logger.info(f"🔄 HandFlow Training - RESUME MODE")
    else:
        logger.info(f"🖐️ HandFlow Training - Unified Model (Both Hands)")
    logger.info(f"{'='*60}")
    if args.resume:
        logger.info(f"Resume from: {args.resume}")
    else:
        logger.info(f"Architecture: {config.model.architecture.upper()}")
    logger.info(f"Epochs: {config.training.epochs}")
    logger.info(f"Batch Size: {config.training.batch_size}")
    logger.info(f"Learning Rate: {config.training.learning_rate}")
    logger.info(f"Output: {output_path}")
    logger.info(f"{'='*60}")

    logger.info("\n Loading preprocessed data from cache (both hands)...")
    x_train, x_val, y_train, y_val, actions = load_training_data(config=config)
    logger.info("✅ Data is loaded and merged from both hands")
    logger.info(f"   Training: {len(x_train)} samples")
    logger.info(f"   Validation: {len(x_val)} samples")
    logger.info(f"   Data shape: {x_train.shape}")
    logger.info(f"   ✅ Input dimension for model: {x_train.shape[-1]} features")

    # Update input dim to config
    config.model.input_dim = x_train.shape[-1]

    if args.resume:
        logger.info(f"\n🔄 Loading model from {args.resume}...")
        model = keras.models.load_model(args.resume)
        logger.info("✅ Model loaded successfully")

        # Always recompile with fresh optimizer when resuming
        # (optimizer state from saved model is tied to old variable instances)
        lr = config.training.learning_rate
        logger.info(f"   Recompiling with learning rate: {lr}")
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        model.summary()
    else:
        logger.info(f"\n🔨 Building {config.model.architecture.upper()} model...")
        model = build_model(config)
        model.summary()

    experiment_name = "handflow-unified"
    trainer = Trainer(
        config=config,
        model=model,
        experiment_name=experiment_name,
        use_augmentation=config.augmentation.enabled,
    )

    # Train
    logger.info(f"\nStarting training...")
    history = trainer.train(x_train, y_train, x_val, y_val)

    # Evaluate
    logger.info("\n Final Evaluation:")
    metrics = trainer.evaluate(x_val, y_val)
    for name, value in metrics.items():
        logger.info(f"   {name}: {value:.4f}")

    # Save model
    logger.info(f"\n💾 Saving model to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save(output_path)

    logger.info("\n✅ Training complete!")
    logger.info(f"   Best validation accuracy: {max(history.history['val_accuracy']):.4f}")


if __name__ == "__main__":
    main()
