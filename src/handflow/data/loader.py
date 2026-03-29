"""Load raw and processed gesture data for training and inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from tensorflow import keras

if TYPE_CHECKING:
    from handflow.utils.config import Config

from handflow.utils import get_logger


def load_raw_data(
    data_path: str | Path,
    actions: list[str],
    sequence_length: int = 12,
    return_paths: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw training data from NPY files.

    Loads individual .npy frame files from the data collection format:
    data_path/
    ├── gesture1/
    │   ├── 0/          # sequence 0
    │   │   ├── 0.npy   # frame 0
    │   │   ├── 1.npy   # frame 1
    │   │   └── ...
    │   ├── 1/          # sequence 1
    │   └── ...
    └── gesture2/
        └── ...

    Args:
        data_path: Path to data directory (e.g., data/raw/MP_Data).
        actions: List of gesture/action names to load.
        sequence_length: Number of frames per sequence.

    Returns:
        Tuple of (sequences, labels) arrays.
        - sequences: Shape (n_samples, sequence_length, n_keypoints)
        - labels: One-hot encoded, shape (n_samples, n_classes)
    """
    logger = get_logger("handflow.data.loader")

    data_path = Path(data_path)
    sequences = []
    labels = []
    paths = []

    label_map = {action: idx for idx, action in enumerate(actions)}

    for action in actions:
        action_path = data_path / action
        if not action_path.exists():
            logger.info(f"⚠️ Warning: Action directory not found: {action_path}")
            continue

        # Get all sequence directories
        seq_dirs = sorted([d for d in action_path.iterdir() if d.is_dir()])
        
        for seq_dir in seq_dirs:
            # Load all frames in sequence
            window = []
            for frame_num in range(sequence_length):
                frame_path = seq_dir / f"{frame_num}.npy"
                if frame_path.exists():
                    frame_data = np.load(frame_path)
                    window.append(frame_data)

            if len(window) == sequence_length:
                sequences.append(window)
                labels.append(label_map[action])
                # Store relative path from data_path's parent 
                paths.append(str(seq_dir.relative_to(data_path.parent)))
        
        logger.info(f"✅ Loaded {len(seq_dirs)} sequences for action '{action}'")


    sequences = np.array(sequences)
    labels = np.array(labels)

    # One-hot encode labels
    num_classes = len(actions)
    labels_onehot = keras.utils.to_categorical(labels, num_classes)

    return sequences, labels_onehot, np.array(paths) if return_paths else (sequences, labels_onehot)


def load_processed_data(

    processed_path: str | Path,
    include_paths: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]] | tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """
    Load preprocessed data from NPZ file.

    The NPZ file should contain:
    - X: Feature array, shape (n_samples, sequence_length, n_features)
    - y: One-hot labels, shape (n_samples, n_classes)
    - actions: List of action names

    Args:
        processed_path: Path to the .npz file.

    Returns:
        Tuple of (X, y, actions).
    """
    processed_path = Path(processed_path)
    
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {processed_path}\n"
            "Run preprocessing first: python scripts/preprocess.py"
        )

    data = np.load(processed_path, allow_pickle=True)
    
    X = data["X"]
    y = data["y"]
    actions = data["actions"].tolist()

    if include_paths and "paths" in data:
        return X, y, actions, data["paths"]
    
    return X, y, actions


def save_processed_data(
    output_path: str | Path,
    X: np.ndarray,
    y: np.ndarray,
    actions: list[str],
    config_hash: str | None = None,
    paths: np.ndarray | None = None,
) -> None:

    """
    Save preprocessed data to NPZ file.

    Args:
        output_path: Path to save the .npz file.
        X: Feature array.
        y: Label array.
        actions: List of action names.
        config_hash: Optional hash of config for cache invalidation.

    Example:
        >>> save_processed_data("data/processed/right_train.npz", X, y, actions)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        actions=np.array(actions, dtype=object),
        config_hash=config_hash or "",
        paths=paths if paths is not None else np.array([]),
    )

    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"💾 Saved processed data: {output_path} ({size_mb:.2f} MB)")


def get_config_hash(config: Config) -> str:
    """
    Generate a hash of feature engineering config.
    
    Used to detect when preprocessing needs to be re-run.

    Args:
        config: Configuration object.

    Returns:
        MD5 hash string.
    """
    config_dict = {
        "sequence_length": config.data.sequence_length,
        "input_dim": config.model.input_dim,
        "architecture": config.model.architecture,
    }
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def check_processed_data_valid(
    processed_path: str | Path,
    config: Config,
) -> bool:
    """
    Check if processed data is still valid (config hasn't changed).

    Args:
        processed_path: Path to the .npz file.
        config: Current configuration.

    Returns:
        True if processed data is valid and can be used.
    """
    processed_path = Path(processed_path)
    
    if not processed_path.exists():
        return False

    try:
        data = np.load(processed_path, allow_pickle=True)
        stored_hash = str(data.get("config_hash", ""))
        current_hash = get_config_hash(config)
        return stored_hash == current_hash
    except Exception:
        return False
