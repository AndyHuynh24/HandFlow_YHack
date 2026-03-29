"""Unified experiment tracking with Weights & Biases and TensorBoard backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrackingConfig:
    """Configuration for experiment tracking.

    Attributes:
        enabled: Master switch for all tracking
        wandb_enabled: Enable Weights & Biases tracking
        wandb_project: W&B project name
        wandb_entity: W&B team/username (optional)
        tensorboard_enabled: Enable TensorBoard logging
        tensorboard_log_dir: Directory for TensorBoard logs
        log_model: Whether to log model artifacts
        log_gradients: Whether to log gradient histograms
    """
    enabled: bool = True
    wandb_enabled: bool = False
    wandb_project: str = "ml-experiments"
    wandb_entity: Optional[str] = None
    tensorboard_enabled: bool = True
    tensorboard_log_dir: str = "logs/tensorboard"
    log_model: bool = True
    log_gradients: bool = False


class BaseTracker(ABC):
    """Abstract base class for experiment trackers."""

    @abstractmethod
    def start_run(self, run_name: str, config: Dict[str, Any]) -> None:
        """Start a new tracking run."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        pass

    @abstractmethod
    def log_artifact(self, path: str, name: str, artifact_type: str = "model") -> None:
        """Log a file artifact (model, data, etc.)."""
        pass

    @abstractmethod
    def finish(self) -> None:
        """End the current run."""
        pass


class WandBTracker(BaseTracker):
    """Weights & Biases experiment tracker.

    W&B provides:
    - Real-time metrics visualization
    - Hyperparameter tracking and comparison
    - Model artifact versioning
    - Team collaboration features
    - Automatic system metrics (GPU, CPU, memory)
    """

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        log_model: bool = True,
    ) -> None:
        self.project = project
        self.entity = entity
        self.log_model = log_model
        self._run = None
        self._wandb = None

        # Import wandb lazily to avoid import errors if not installed
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb not installed. Install with: pip install wandb\n"
                "Then login with: wandb login"
            )

    def start_run(self, run_name: str, config: Dict[str, Any]) -> None:
        """Start a new W&B run.

        Args:
            run_name: Name for this run (appears in W&B dashboard)
            config: Hyperparameters and configuration to log
        """
        try:
            self._run = self._wandb.init(
                project=self.project,
                entity=self.entity,
                name=run_name,
                config=config,
                resume="allow",  # Replaces deprecated reinit=True
            )
        except self._wandb.errors.CommError as e:
            if "permission denied" in str(e).lower() or "403" in str(e):
                raise RuntimeError(
                    f"W&B permission denied for project '{self.project}'.\n"
                    f"Possible fixes:\n"
                    f"  1. Create the project first at: https://wandb.ai/new\n"
                    f"  2. Check project name is correct (no typos)\n"
                    f"  3. Re-login: wandb login --relogin\n"
                    f"  4. Check your team/entity permissions\n"
                    f"Original error: {e}"
                ) from e
            raise

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to W&B.

        Args:
            metrics: Dictionary of metric_name -> value
            step: Training step (epoch, batch, etc.)
        """
        if self._run is not None:
            if step is not None:
                self._wandb.log(metrics, step=step)
            else:
                self._wandb.log(metrics)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to W&B config.

        Args:
            params: Dictionary of param_name -> value
        """
        if self._run is not None:
            self._wandb.config.update(params)

    def log_artifact(self, path: str, name: str, artifact_type: str = "model") -> None:
        """Log a file as a W&B artifact.

        Args:
            path: Path to the file to upload
            name: Name for the artifact
            artifact_type: Type of artifact ("model", "dataset", etc.)
        """
        if self._run is not None and self.log_model:
            artifact = self._wandb.Artifact(name, type=artifact_type)
            artifact.add_file(path)
            self._run.log_artifact(artifact)

    def log_table(self, name: str, columns: List[str], data: List[List[Any]]) -> None:
        """Log a table to W&B.

        Args:
            name: Table name
            columns: Column headers
            data: Table rows
        """
        if self._run is not None:
            table = self._wandb.Table(columns=columns, data=data)
            self._wandb.log({name: table})

    def log_image(self, name: str, image: Any, caption: Optional[str] = None) -> None:
        """Log an image to W&B.

        Args:
            name: Image name/key
            image: Image data (numpy array, PIL Image, or file path)
            caption: Optional caption
        """
        if self._run is not None:
            self._wandb.log({name: self._wandb.Image(image, caption=caption)})

    def log_confusion_matrix(
        self,
        y_true: List,
        y_pred: List,
        class_names: List[str],
        title: str = "Confusion Matrix"
    ) -> None:
        """Log a confusion matrix to W&B.

        Args:
            y_true: Ground truth labels (class indices or names)
            y_pred: Predicted labels (class indices or names)
            class_names: List of class names
            title: Title for the confusion matrix plot
        """
        if self._run is not None:
            self._wandb.log({
                title: self._wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true,
                    preds=y_pred,
                    class_names=class_names
                )
            })

    def get_keras_callback(self):
        """Get Keras callback for automatic metric logging.

        Returns:
            WandbMetricsLogger callback for use with model.fit()
        """
        try:
            from wandb.integration.keras import WandbMetricsLogger
            return WandbMetricsLogger()
        except ImportError:
            pass
            return None

    def finish(self) -> None:
        """Finish the current W&B run."""
        if self._run is not None:
            self._run.finish()
            self._run = None


class TensorBoardTracker(BaseTracker):
    """TensorBoard experiment tracker.

    TensorBoard provides:
    - Scalar metrics visualization
    - Model graph visualization
    - Histograms (weights, gradients)
    - Profiling tools
    - Local-only (no cloud upload)
    """

    def __init__(self, log_dir: str = "logs/tensorboard") -> None:
        self.base_log_dir = Path(log_dir)
        self.log_dir: Optional[Path] = None
        self._writer = None
        self._step = 0

        try:
            import tensorflow as tf
            self._tf = tf
        except ImportError:
            raise ImportError("tensorflow not installed. Install with: pip install tensorflow")

    def start_run(self, run_name: str, config: Dict[str, Any]) -> None:
        """Start a new TensorBoard run.

        Args:
            run_name: Name for this run (creates subdirectory)
            config: Configuration to log as text
        """
        self.log_dir = self.base_log_dir / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = self._tf.summary.create_file_writer(str(self.log_dir))
        self._step = 0

        # Log config as text
        with self._writer.as_default():
            config_text = "\n".join(f"{k}: {v}" for k, v in config.items())
            self._tf.summary.text("config", config_text, step=0)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to TensorBoard.

        Args:
            metrics: Dictionary of metric_name -> value
            step: Training step
        """
        if self._writer is None:
            return

        current_step = step if step is not None else self._step

        with self._writer.as_default():
            for name, value in metrics.items():
                self._tf.summary.scalar(name, value, step=current_step)

        self._writer.flush()
        self._step = current_step + 1

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters as text.

        Args:
            params: Dictionary of param_name -> value
        """
        if self._writer is None:
            return

        with self._writer.as_default():
            params_text = "\n".join(f"{k}: {v}" for k, v in params.items())
            self._tf.summary.text("hyperparameters", params_text, step=0)
        self._writer.flush()

    def log_artifact(self, _path: str, _name: str, _artifact_type: str = "model") -> None:
        """TensorBoard doesn't support artifacts - this is a no-op."""
        pass

    def log_histogram(self, name: str, values: Any, step: Optional[int] = None) -> None:
        """Log a histogram to TensorBoard.

        Args:
            name: Histogram name
            values: Values to histogram
            step: Training step
        """
        if self._writer is None:
            return

        current_step = step if step is not None else self._step

        with self._writer.as_default():
            self._tf.summary.histogram(name, values, step=current_step)
        self._writer.flush()

    def get_keras_callback(self, histogram_freq: int = 1):
        """Get Keras TensorBoard callback.

        Args:
            histogram_freq: Frequency to log weight histograms

        Returns:
            TensorBoard callback for use with model.fit()
        """
        from tensorflow import keras
        return keras.callbacks.TensorBoard(
            log_dir=str(self.log_dir),
            histogram_freq=histogram_freq,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
        )

    def finish(self) -> None:
        """Close the TensorBoard writer."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class ExperimentTracker:
    """Unified experiment tracker supporting multiple backends.

    This is the main class you should use. It manages multiple tracking
    backends (W&B, TensorBoard) and provides a unified interface.

    Example:
        >>> config = TrackingConfig(
        ...     wandb_enabled=True,
        ...     wandb_project="my-project",
        ...     tensorboard_enabled=True
        ... )
        >>> tracker = ExperimentTracker(config)
        >>> tracker.start_run("run-1", {"lr": 0.001})
        >>> tracker.log_metrics({"loss": 0.5})
        >>> tracker.finish()
    """

    def __init__(self, config: Optional[TrackingConfig] = None) -> None:
        """Initialize experiment tracker.

        Args:
            config: Tracking configuration. If None, uses defaults.
        """
        self.config = config or TrackingConfig()
        self._trackers: List[BaseTracker] = []
        self._run_active = False

        if not self.config.enabled:
            return

        # Initialize enabled trackers
        if self.config.wandb_enabled:
            try:
                self._trackers.append(WandBTracker(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    log_model=self.config.log_model,
                ))
            except ImportError:
                pass

        if self.config.tensorboard_enabled:
            try:
                self._trackers.append(TensorBoardTracker(
                    log_dir=self.config.tensorboard_log_dir
                ))
            except ImportError:
                pass

    def __enter__(self) -> "ExperimentTracker":
        """Context manager entry."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        """Context manager exit - ensures run is finished."""
        self.finish()

    def start_run(
        self,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Start a new experiment run.

        Args:
            run_name: Name for this run. Auto-generated if None.
            config: Hyperparameters and configuration to log.
        """
        if self._run_active:
            self.finish()

        if run_name is None:
            run_name = f"run_{datetime.now():%Y%m%d_%H%M%S}"

        config = config or {}

        for tracker in self._trackers:
            tracker.start_run(run_name, config)

        self._run_active = True

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to all enabled backends.

        Args:
            metrics: Dictionary of metric_name -> value
            step: Training step (epoch, batch, etc.)
        """
        for tracker in self._trackers:
            tracker.log_metrics(metrics, step)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to all enabled backends.

        Args:
            params: Dictionary of param_name -> value
        """
        for tracker in self._trackers:
            tracker.log_params(params)

    def log_artifact(self, path: str, name: str, artifact_type: str = "model") -> None:
        """Log a file artifact to backends that support it.

        Args:
            path: Path to the file
            name: Artifact name
            artifact_type: Type of artifact
        """
        for tracker in self._trackers:
            tracker.log_artifact(path, name, artifact_type)

    def log_confusion_matrix(
        self,
        y_true: List,
        y_pred: List,
        class_names: List[str],
        title: str = "Confusion Matrix"
    ) -> None:
        """Log a confusion matrix (W&B only - TensorBoard doesn't support this).

        Args:
            y_true: Ground truth labels (class indices)
            y_pred: Predicted labels (class indices)
            class_names: List of class names
            title: Title for the plot
        """
        wandb_tracker = self.get_wandb_tracker()
        if wandb_tracker:
            wandb_tracker.log_confusion_matrix(y_true, y_pred, class_names, title)

    def get_keras_callbacks(self) -> List:
        """Get Keras callbacks for automatic logging.

        Returns:
            List of Keras callbacks for use with model.fit()
        """
        callbacks = []
        for tracker in self._trackers:
            if hasattr(tracker, "get_keras_callback"):
                cb = tracker.get_keras_callback()
                if cb is not None:
                    callbacks.append(cb)
        return callbacks

    def get_wandb_tracker(self) -> Optional[WandBTracker]:
        """Get the W&B tracker if enabled.

        Returns:
            WandBTracker instance or None
        """
        for tracker in self._trackers:
            if isinstance(tracker, WandBTracker):
                return tracker
        return None

    def get_tensorboard_tracker(self) -> Optional[TensorBoardTracker]:
        """Get the TensorBoard tracker if enabled.

        Returns:
            TensorBoardTracker instance or None
        """
        for tracker in self._trackers:
            if isinstance(tracker, TensorBoardTracker):
                return tracker
        return None

    def finish(self) -> None:
        """Finish the current run and cleanup."""
        if not self._run_active:
            return

        for tracker in self._trackers:
            tracker.finish()

        self._run_active = False


# Convenience function for quick setup
def create_tracker(
    project: str = "ml-experiments",
    use_wandb: bool = False,
    use_tensorboard: bool = True,
    tensorboard_dir: str = "logs/tensorboard",
    wandb_entity: Optional[str] = None,
) -> ExperimentTracker:
    """Create an experiment tracker with simplified configuration.

    Args:
        project: Project name for W&B
        use_wandb: Enable W&B tracking
        use_tensorboard: Enable TensorBoard tracking
        tensorboard_dir: Directory for TensorBoard logs
        wandb_entity: W&B team/username

    Returns:
        Configured ExperimentTracker instance

    Example:
        >>> tracker = create_tracker(
        ...     project="my-project",
        ...     use_wandb=True,
        ...     use_tensorboard=True
        ... )
    """
    config = TrackingConfig(
        enabled=True,
        wandb_enabled=use_wandb,
        wandb_project=project,
        wandb_entity=wandb_entity,
        tensorboard_enabled=use_tensorboard,
        tensorboard_log_dir=tensorboard_dir,
    )
    return ExperimentTracker(config)
