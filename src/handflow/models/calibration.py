# Copyright (c) 2026 Huynh Huy. All rights reserved.

"""
User Calibration Module
=======================

Fine-tune gesture model for individual users using transfer learning.

This module enables personalization by:
1. Freezing early layers (keep learned general features)
2. Training only later layers on user-specific data
3. Validating improvement before saving

Usage:
    calibrator = UserCalibrator(base_model_path="models/hand_action.keras")

    # Collect samples during calibration UI
    calibrator.add_sample(landmarks, gesture_name="swipe_left")

    # When enough samples collected
    success, metrics = calibrator.calibrate()

    if success:
        calibrator.save("models/user_profiles/user123.keras")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from tensorflow import keras

from handflow.utils.logging import get_logger


@dataclass
class CalibrationConfig:
    """Configuration for user calibration."""

    # Minimum samples required per gesture
    min_samples_per_gesture: int = 5

    # Recommended samples for good calibration
    recommended_samples_per_gesture: int = 10

    # Training settings (conservative for few-shot learning)
    learning_rate: float = 1e-5      # Very low to avoid destroying features
    epochs: int = 20                  # Few epochs
    batch_size: int = 4               # Small batch for small dataset
    validation_split: float = 0.2     # Hold out some for validation

    # Freeze strategy: "head_only", "last_block", "half"
    freeze_strategy: str = "last_block"

    # Minimum accuracy improvement to accept calibration
    min_accuracy_improvement: float = 0.0  # Accept if not worse

    # Early stopping
    early_stopping_patience: int = 5


@dataclass
class CalibrationResult:
    """Results from calibration."""
    success: bool
    base_accuracy: float
    calibrated_accuracy: float
    improvement: float
    samples_per_class: dict[str, int] = field(default_factory=dict)
    message: str = ""


class UserCalibrator:
    """
    Handles user-specific model calibration via transfer learning.

    Transfer Learning Strategy:
    ─────────────────────────────

    The pre-trained model has learned:
    - Early layers: General features (motion patterns, joint relationships)
    - Later layers: Task-specific mappings (gesture classification)

    For personalization:
    - FREEZE early layers (keep general knowledge)
    - TRAIN later layers (adapt to user's specific patterns)

    This works because:
    1. Early features transfer well across users
    2. Only classification needs adjustment
    3. Few samples are enough for final layers
    """

    def __init__(
        self,
        base_model_path: str | Path,
        class_names: list[str],
        config: Optional[CalibrationConfig] = None,
    ):
        """
        Initialize calibrator.

        Args:
            base_model_path: Path to pre-trained model
            class_names: List of gesture class names
            config: Calibration configuration
        """
        self.base_model_path = Path(base_model_path)
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.config = config or CalibrationConfig()
        self.logger = get_logger("handflow.calibration")

        # Load base model
        self.base_model = keras.models.load_model(str(self.base_model_path))
        self.calibrated_model: Optional[keras.Model] = None

        # Sample storage: {gesture_name: [samples]}
        self._samples: dict[str, list[np.ndarray]] = {name: [] for name in class_names}

        self.logger.info(f"Loaded base model from {self.base_model_path}")
        self.logger.info(f"Classes: {class_names}")

    def add_sample(self, sequence: np.ndarray, gesture_name: str) -> bool:
        """
        Add a calibration sample.

        Args:
            sequence: Landmark sequence (seq_len, features) e.g. (12, 84)
            gesture_name: Name of the gesture

        Returns:
            True if sample added successfully
        """
        if gesture_name not in self._samples:
            self.logger.warning(f"Unknown gesture: {gesture_name}")
            return False

        # Validate shape
        expected_shape = self.base_model.input_shape[1:]  # (seq_len, features)
        if sequence.shape != expected_shape:
            self.logger.warning(
                f"Invalid shape: {sequence.shape}, expected {expected_shape}"
            )
            return False

        self._samples[gesture_name].append(sequence.copy())
        return True

    def get_sample_counts(self) -> dict[str, int]:
        """Get number of samples collected per gesture."""
        return {name: len(samples) for name, samples in self._samples.items()}

    def is_ready(self) -> tuple[bool, str]:
        """
        Check if enough samples collected for calibration.

        Returns:
            (ready, message) tuple
        """
        counts = self.get_sample_counts()
        min_required = self.config.min_samples_per_gesture

        insufficient = [
            name for name, count in counts.items()
            if count < min_required
        ]

        if insufficient:
            return False, f"Need {min_required}+ samples for: {', '.join(insufficient)}"

        total = sum(counts.values())
        return True, f"Ready with {total} total samples"

    def clear_samples(self) -> None:
        """Clear all collected samples."""
        self._samples = {name: [] for name in self.class_names}
        self.logger.info("Cleared all samples")

    def calibrate(self) -> CalibrationResult:
        """
        Perform calibration using collected samples.

        Returns:
            CalibrationResult with success status and metrics
        """
        # Check readiness
        ready, msg = self.is_ready()
        if not ready:
            return CalibrationResult(
                success=False,
                base_accuracy=0.0,
                calibrated_accuracy=0.0,
                improvement=0.0,
                message=msg
            )

        # Prepare data
        X, y = self._prepare_data()
        sample_counts = self.get_sample_counts()

        self.logger.info(f"Calibrating with {len(X)} samples...")

        # Evaluate base model first
        base_accuracy = self._evaluate_model(self.base_model, X, y)
        self.logger.info(f"Base model accuracy: {base_accuracy:.2%}")

        # Create and configure calibration model
        self.calibrated_model = self._create_calibration_model()

        # Train
        self._train_calibration_model(X, y)

        # Evaluate calibrated model
        calibrated_accuracy = self._evaluate_model(self.calibrated_model, X, y)
        improvement = calibrated_accuracy - base_accuracy

        self.logger.info(f"Calibrated accuracy: {calibrated_accuracy:.2%}")
        self.logger.info(f"Improvement: {improvement:+.2%}")

        # Check if calibration improved things
        success = improvement >= self.config.min_accuracy_improvement

        if not success:
            self.logger.warning("Calibration did not improve accuracy, keeping base model")
            self.calibrated_model = None

        return CalibrationResult(
            success=success,
            base_accuracy=base_accuracy,
            calibrated_accuracy=calibrated_accuracy,
            improvement=improvement,
            samples_per_class=sample_counts,
            message="Calibration successful" if success else "No improvement"
        )

    def _prepare_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training data from collected samples."""
        X_list = []
        y_list = []

        for gesture_name, samples in self._samples.items():
            class_idx = self.class_names.index(gesture_name)
            for sample in samples:
                X_list.append(sample)
                y_list.append(class_idx)

        X = np.array(X_list)
        y = keras.utils.to_categorical(y_list, num_classes=self.num_classes)

        # Shuffle
        indices = np.random.permutation(len(X))
        return X[indices], y[indices]

    def _create_calibration_model(self) -> keras.Model:
        """
        Create calibration model with frozen layers.

        Freeze strategies:
        - "head_only": Freeze all except final Dense layer (safest, least adaptation)
        - "last_block": Freeze input + first 2 blocks (recommended)
        - "half": Freeze first half of layers (most adaptation, needs more data)
        """
        # Clone model architecture and weights
        model = keras.models.clone_model(self.base_model)
        model.set_weights(self.base_model.get_weights())

        strategy = self.config.freeze_strategy
        layer_names = [layer.name for layer in model.layers]

        self.logger.info(f"Freeze strategy: {strategy}")
        self.logger.info(f"Model layers: {layer_names}")

        if strategy == "head_only":
            # Freeze everything except last Dense layer
            for layer in model.layers[:-1]:
                layer.trainable = False
            trainable_count = 1

        elif strategy == "last_block":
            # Freeze input projection + first 2 residual blocks
            # Typically: input, conv, (block1 layers), (block2 layers), (block3 layers), pooling, dense, dense
            # We want to train: last block + dense layers

            # Find layers to freeze (roughly first 60% of layers)
            freeze_until = int(len(model.layers) * 0.6)
            for i, layer in enumerate(model.layers):
                if i < freeze_until:
                    layer.trainable = False
            trainable_count = len(model.layers) - freeze_until

        elif strategy == "half":
            # Freeze first half
            freeze_until = len(model.layers) // 2
            for i, layer in enumerate(model.layers):
                if i < freeze_until:
                    layer.trainable = False
            trainable_count = len(model.layers) - freeze_until

        else:
            raise ValueError(f"Unknown freeze strategy: {strategy}")

        # Log trainable status
        frozen = [l.name for l in model.layers if not l.trainable]
        trainable = [l.name for l in model.layers if l.trainable]
        self.logger.info(f"Frozen layers ({len(frozen)}): {frozen}")
        self.logger.info(f"Trainable layers ({len(trainable)}): {trainable}")

        # Compile with low learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    def _train_calibration_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the calibration model."""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
        ]

        self.calibrated_model.fit(
            X, y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            callbacks=callbacks,
            verbose=1
        )

    def _evaluate_model(self, model: keras.Model, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model accuracy."""
        _, accuracy = model.evaluate(X, y, verbose=0)
        return accuracy

    def get_model(self) -> keras.Model:
        """Get the calibrated model (or base if calibration failed)."""
        return self.calibrated_model if self.calibrated_model else self.base_model

    def save(self, path: str | Path) -> None:
        """
        Save the calibrated model.

        Args:
            path: Path to save model
        """
        model = self.get_model()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
        self.logger.info(f"Saved calibrated model to {path}")

    def reset(self) -> None:
        """Reset calibrator to initial state."""
        self.calibrated_model = None
        self.clear_samples()


def quick_calibrate(
    base_model_path: str,
    user_samples: np.ndarray,
    user_labels: np.ndarray,
    class_names: list[str],
    freeze_strategy: str = "last_block",
) -> tuple[keras.Model, CalibrationResult]:
    """
    Quick calibration function for simple use cases.

    Args:
        base_model_path: Path to pre-trained model
        user_samples: User's calibration samples (N, seq_len, features)
        user_labels: Labels as class indices (N,)
        class_names: List of class names
        freeze_strategy: "head_only", "last_block", or "half"

    Returns:
        (calibrated_model, result)

    Example:
        model, result = quick_calibrate(
            "models/hand_action.keras",
            user_samples,  # shape (100, 12, 84)
            user_labels,   # shape (100,) with values 0-10
            class_names=["none", "swipe", "touch", ...]
        )
        if result.success:
            model.save("models/user_model.keras")
    """
    config = CalibrationConfig(freeze_strategy=freeze_strategy)
    calibrator = UserCalibrator(base_model_path, class_names, config)

    # Add samples
    for sample, label in zip(user_samples, user_labels):
        gesture_name = class_names[label]
        calibrator.add_sample(sample, gesture_name)

    # Calibrate
    result = calibrator.calibrate()

    return calibrator.get_model(), result
