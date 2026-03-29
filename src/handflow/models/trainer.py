"""Training pipeline with on-the-fly augmentation, W&B/TensorBoard tracking, and class weighting."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from handflow.utils.logging import get_logger
from handflow.utils.experiment_tracker import (
    ExperimentTracker,
    TrackingConfig as TrackerConfig,
)

if TYPE_CHECKING:
    from handflow.data.augmentation import SequenceAugmenter
    from handflow.utils.config import Config


class AugmentedDataGenerator(keras.utils.Sequence):
    """Keras data generator that applies random augmentations each epoch."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        augmenter: SequenceAugmenter | None = None,
        shuffle: bool = True,
    ) -> None:
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.augmenter = augmenter
        self.shuffle = shuffle
        self.indices = np.arange(len(x))
        
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self) -> int:
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.x))
        batch_indices = self.indices[start:end]
        
        x_batch = self.x[batch_indices].copy()
        y_batch = self.y[batch_indices]
        
        # Apply augmentation if available
        if self.augmenter is not None:
            x_batch = self.augmenter.augment_batch(x_batch)
        
        return x_batch, y_batch
    
    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)


class Trainer:
    def __init__(
        self,
        config: Config,
        model: keras.Model,
        experiment_name: str = "handflow-training",
        use_augmentation: bool = True,
        use_class_weights: bool = True,
    ) -> None:
        self.config = config
        self.model = model
        self.experiment_name = experiment_name
        self.use_augmentation = use_augmentation
        self.use_class_weights = use_class_weights
        self.history: keras.callbacks.History | None = None
        self.augmenter: SequenceAugmenter | None = None
        self.logger = get_logger("handflow.trainer")

        # Setup experiment tracking using the unified tracker
        self.tracker = self._setup_tracking()
        self._setup_augmentation()

    def _setup_tracking(self) -> Optional[ExperimentTracker]:
        """Setup experiment tracking using the unified tracker.

        Args:
            use_wandb_override: CLI override for wandb (--use-wandb flag)

        Returns:
            Configured ExperimentTracker or None if tracking disabled
        """
        # Get tracking config from main config
        tracking_config = self.config.tracking 

        use_wandb = tracking_config.wandb.enabled 

        # Build tracker config
        tracker_config = TrackerConfig(
            enabled=True,
            wandb_enabled=use_wandb,
            wandb_project=tracking_config.wandb.project, 
            wandb_entity=tracking_config.wandb.entity,
            tensorboard_enabled=tracking_config.tensorboard.enabled,
            tensorboard_log_dir=tracking_config.tensorboard.log_dir,
            log_model=tracking_config.wandb.log_model,
        )

        return ExperimentTracker(tracker_config)

    def _setup_augmentation(self) -> None:
        """Setup data augmentation."""
        if self.use_augmentation:
            try:
                from handflow.data.augmentation import SequenceAugmenter
                self.augmenter = SequenceAugmenter(self.config)
                self.logger.info("On-the-fly augmentation enabled")
            except ImportError:
                self.logger.warning("Augmentation module not found. Skipping augmentation.")
                self.use_augmentation = False

    def _compute_class_weights(self, y_train: np.ndarray) -> dict[int, float]:
        """
        Compute class weights for imbalanced data.

        Uses 'balanced' mode which sets weights inversely proportional to
        class frequencies: n_samples / (n_classes * np.bincount(y))

        Args:
            y_train: One-hot encoded labels, shape (n_samples, n_classes)

        Returns:
            Dictionary mapping class index to weight
        """
        # Convert one-hot to class indices
        y_indices = np.argmax(y_train, axis=1)

        # Get unique classes present in training data
        classes = np.unique(y_indices)

        # Compute balanced class weights
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_indices
        )

        # Create dict mapping class index to weight
        class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}

        # Log class distribution and weights
        class_names = self.config.model.gestures
        self.logger.info("Class weights computed:")
        for cls_idx, weight in sorted(class_weight_dict.items()):
            count = np.sum(y_indices == cls_idx)
            name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
            self.logger.info(f"  {name}: {count} samples, weight={weight:.3f}")

        return class_weight_dict

    def _build_run_config(
        self,
        x_train: np.ndarray,
        x_val: Optional[np.ndarray]
    ) -> dict:
        """Build comprehensive configuration dict for experiment tracking.

        Logs all relevant hyperparameters: model, training, augmentation, and data info.

        Args:
            x_train: Training data (for shape/size info)
            x_val: Validation data (for shape/size info)

        Returns:
            Dictionary of all parameters to log
        """
        model_cfg = self.config.model
        train_cfg = self.config.training
        aug_cfg = self.config.augmentation

        run_config = {
            # Model parameters
            "model/architecture": model_cfg.architecture,
            "model/input_dim": model_cfg.input_dim,
            "model/hidden_units": model_cfg.hidden_units,
            "model/num_layers": model_cfg.num_layers,
            "model/dropout": model_cfg.dropout,
            "model/num_classes": model_cfg.num_classes,



            # Training parameters
            "training/epochs": train_cfg.epochs,
            "training/batch_size": train_cfg.batch_size,
            "training/learning_rate": train_cfg.learning_rate,
            "training/optimizer": train_cfg.optimizer,
            "training/loss": train_cfg.loss,
            "training/early_stopping_patience": train_cfg.early_stopping_patience,
            "training/validation_split": train_cfg.validation_split,

            # Learning rate scheduler
            "training/reduce_lr_enabled": train_cfg.reduce_lr.enabled,
            "training/reduce_lr_factor": train_cfg.reduce_lr.factor,
            "training/reduce_lr_patience": train_cfg.reduce_lr.patience,
            "training/reduce_lr_min": train_cfg.reduce_lr.min_lr,

            # Augmentation parameters
            "augmentation/enabled": self.use_augmentation,
            "augmentation/n_variants": aug_cfg.n_variants if self.use_augmentation else 0,
            "augmentation/noise_prob": aug_cfg.noise_prob if self.use_augmentation else 0,
            "augmentation/noise_std": aug_cfg.noise_std if self.use_augmentation else 0,
            "augmentation/time_warp_enabled": getattr(aug_cfg, 'time_warp_enabled', False) if self.use_augmentation else False,
            "augmentation/time_warp_prob": aug_cfg.time_warp_prob if self.use_augmentation else 0,
            "augmentation/dropout_prob": aug_cfg.dropout_prob if self.use_augmentation else 0,
            "augmentation/scale_prob": aug_cfg.scale_prob if self.use_augmentation else 0,
            "augmentation/rotation_prob": aug_cfg.rotation_prob if self.use_augmentation else 0,

            # Class weighting
            "training/class_weights_enabled": self.use_class_weights,

            # Data info
            "data/training_samples": len(x_train),
            "data/validation_samples": len(x_val) if x_val is not None else 0,
            "data/sequence_length": x_train.shape[1] if len(x_train.shape) > 1 else 0,
            "data/feature_dim": x_train.shape[2] if len(x_train.shape) > 2 else 0,
        }

        return run_config

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        run_name: str | None = None,
    ) -> keras.callbacks.History:
        """
        Train the model.

        Args:
            x_train: Training features, shape (n_samples, seq_len, n_features).
            y_train: Training labels, shape (n_samples, n_classes) one-hot encoded.
            x_val: Validation features (optional).
            y_val: Validation labels (optional).
            run_name: Name for this training run.

        Returns:
            Training history.
        """
        train_config = self.config.training

        if run_name is None:
            run_name = f"{self.config.model.architecture}_{datetime.now():%Y%m%d_%H%M%S}"

        # Prepare validation data
        validation_data = None
        if x_val is not None and y_val is not None:
            validation_data = (x_val, y_val)
        elif train_config.validation_split > 0:
            validation_data = None

        # Determine if validation metrics will be available
        # True if explicit validation data is provided OR validation_split is used
        has_validation_data = (
            validation_data is not None or train_config.validation_split > 0
        )

        # Compute class weights if enabled
        class_weight = None
        if self.use_class_weights:
            class_weight = self._compute_class_weights(y_train)
            self.logger.info("Class weighting enabled for training")

        # Build callbacks
        callbacks = self._build_callbacks(run_name, has_validation_data)

        # Start experiment tracking run with full configuration
        run_config = self._build_run_config(x_train, x_val)

        # Add class weight info to run config
        if class_weight is not None:
            class_names = self.config.model.gestures
            y_indices = np.argmax(y_train, axis=1)
            for cls_idx, weight in class_weight.items():
                name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
                run_config[f"class_weight/{name}"] = weight
                run_config[f"class_count/{name}"] = int(np.sum(y_indices == cls_idx))

        if self.tracker:
            self.tracker.start_run(run_name, run_config)
            # Add tracker callbacks (W&B and/or TensorBoard)
            callbacks.extend(self.tracker.get_keras_callbacks())

        try:
            # Use data generator with augmentation for training
            if self.use_augmentation and self.augmenter is not None:
                self.logger.info("Training with on-the-fly augmentation...")
                train_generator = AugmentedDataGenerator(
                    x_train,
                    y_train,
                    batch_size=train_config.batch_size,
                    augmenter=self.augmenter,
                    shuffle=True,
                )

                self.history = self.model.fit(
                    train_generator,
                    epochs=train_config.epochs,
                    validation_data=validation_data,
                    callbacks=callbacks,
                    class_weight=class_weight,
                    verbose=1,
                )
            else:
                # Standard training without augmentation
                self.history = self.model.fit(
                    x_train,
                    y_train,
                    batch_size=train_config.batch_size,
                    epochs=train_config.epochs,
                    validation_data=validation_data,
                    validation_split=(
                        train_config.validation_split if validation_data is None else 0
                    ),
                    callbacks=callbacks,
                    class_weight=class_weight,
                    verbose=1,
                )

            # Log final metrics
            self._log_final_metrics()

        finally:
            if self.tracker:
                self.tracker.finish()

        return self.history

    def _build_callbacks(
        self, run_name: str, has_validation_data: bool
    ) -> list[keras.callbacks.Callback]:
        """Build training callbacks (early stopping, checkpointing, LR scheduling).

        Args:
            run_name: Name for checkpoint files.
            has_validation_data: Whether validation data will be used during training.
                This determines whether to monitor val_loss/val_accuracy or loss/accuracy.
        """
        callbacks = []
        train_config = self.config.training

        # Determine which metrics to monitor based on actual validation data availability
        loss_metric = "val_loss" if has_validation_data else "loss"
        acc_metric = "val_accuracy" if has_validation_data else "accuracy"

        self.logger.info(f"Callbacks will monitor: {loss_metric} (loss), {acc_metric} (accuracy)")
        self.logger.info(f"Early stopping patience: {train_config.early_stopping_patience} epochs")

        # Early stopping
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=acc_metric,
                mode="max",
                patience=train_config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            )
        )

        # Model checkpoint
        checkpoint_dir = Path(self.config.model.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_dir / f"{run_name}_best.keras"),
                monitor=acc_metric,
                mode="max",
                save_best_only=True,
                verbose=1,
            )
        )

        # Learning rate reducer
        if train_config.reduce_lr.enabled:
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor=loss_metric,
                    factor=train_config.reduce_lr.factor,
                    patience=train_config.reduce_lr.patience,
                    min_lr=train_config.reduce_lr.min_lr,
                    verbose=1,
                )
            )

        return callbacks

    def _log_final_metrics(self) -> None:
        """Log final metrics to experiment tracker."""
        if self.history is None:
            return

        final_metrics = {
            "final_loss": self.history.history["loss"][-1],
            "final_accuracy": self.history.history["accuracy"][-1],
        }

        if "val_loss" in self.history.history:
            final_metrics["final_val_loss"] = self.history.history["val_loss"][-1]
            final_metrics["final_val_accuracy"] = self.history.history["val_accuracy"][-1]

        if self.tracker:
            self.tracker.log_metrics(final_metrics)

    def save(self, path: str | Path) -> None:
        """
        Save the trained model.

        Args:
            path: Path to save the model (.h5 or SavedModel directory).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        self.logger.info(f"Model saved to {path}")

    def evaluate(
        self,
        x_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[list[str]] = None,
        log_confusion_matrix: bool = True,
    ) -> dict[str, float]:
        """
        Evaluate the model on test data.

        Args:
            x_test: Test features.
            y_test: Test labels (one-hot encoded).
            class_names: List of class names for confusion matrix.
            log_confusion_matrix: Whether to log confusion matrix to W&B.

        Returns:
            Dictionary of metric names to values.
        """
        results = self.model.evaluate(x_test, y_test, verbose=0)
        metrics = dict(zip(self.model.metrics_names, results))

        # Log confusion matrix if tracker available and W&B enabled
        if log_confusion_matrix and self.tracker:
            # Get predictions
            y_pred_probs = self.model.predict(x_test, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = np.argmax(y_test, axis=1)

            # Use class names from config if not provided
            if class_names is None:
                class_names = self.config.model.gestures

            self.tracker.log_confusion_matrix(
                y_true=y_true.tolist(),
                y_pred=y_pred.tolist(),
                class_names=class_names,
                title="Confusion Matrix"
            )
            self.logger.info("Logged confusion matrix to W&B")

        return metrics
