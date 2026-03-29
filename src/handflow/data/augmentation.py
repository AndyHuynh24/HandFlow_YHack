"""Online geometric augmentation for hand gesture sequences: noise, time warp, rotation, and depth transforms."""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from handflow.utils.config import Config

# -------------------------------------------------
# Landmark definitions
# -------------------------------------------------
WRIST_IDX = 0

THUMB_INDICES  = [1, 2, 3, 4]
INDEX_INDICES  = [5, 6, 7, 8]
MIDDLE_INDICES = [9, 10, 11, 12]
RING_INDICES   = [13, 14, 15, 16]
PINKY_INDICES  = [17, 18, 19, 20]

FINGERTIP_INDICES = [4, 8, 12, 16, 20]
ALL_FINGERS = [
    THUMB_INDICES, INDEX_INDICES, MIDDLE_INDICES,
    RING_INDICES, PINKY_INDICES
]


class SequenceAugmenter:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def augment(self, sequence: np.ndarray) -> np.ndarray:
        if not self.config.augmentation.enabled:
            return sequence

        seq = sequence.copy()

        time_warp_enabled = getattr(self.config.augmentation, 'time_warp_enabled', False)
        if (
            time_warp_enabled
            and np.random.rand() < self.config.augmentation.time_warp_prob
            and len(seq) >= self.config.augmentation.min_timewarp_frames
        ):
            seq = self._time_warp(seq)

        if np.random.rand() < self.config.augmentation.scale_prob:
            seq = self._scale(seq)

        if np.random.rand() < self.config.augmentation.rotation_prob:
            seq = self._rotate_2d(seq)

        # Depth-related augmentations
        if np.random.rand() < self.config.augmentation.z_scale_prob:
            seq = self._z_scale(seq)

        if np.random.rand() < self.config.augmentation.z_shift_prob:
            seq = self._z_shift(seq)

        if np.random.rand() < self.config.augmentation.z_proportional_prob:
            seq = self._z_proportional(seq)

        if np.random.rand() < self.config.augmentation.z_finger_length_prob:
            seq = self._z_finger_length(seq)

        if np.random.rand() < self.config.augmentation.z_noise_prob:
            seq = self._z_noise(seq)

        if np.random.rand() < self.config.augmentation.hand_tilt_prob:
            seq = self._hand_tilt(seq)

        return seq

    def augment_batch(self, batch: np.ndarray) -> np.ndarray:
        return np.asarray([self.augment(seq) for seq in batch])

    # -------------------------------------------------
    # Core augmentations
    # -------------------------------------------------

    def _motion_energy(self, sequence: np.ndarray) -> float:
        """Estimate how much motion exists in the sequence."""
        diffs = np.diff(sequence[:, :63], axis=0)  # xyz only
        return np.mean(np.linalg.norm(diffs, axis=1))

    def _add_noise(self, sequence: np.ndarray) -> np.ndarray:
        std = self.config.augmentation.noise_std
        motion = self._motion_energy(sequence)

        # Reduce noise if hand is mostly static 
        scale = np.clip(
            motion * self.config.augmentation.motion_noise_scale, 0.3, 1.0
        )

        noise = np.random.normal(0, std * scale, sequence.shape)
        noise = np.clip(noise, -self.config.augmentation.max_frame_jitter,
                        self.config.augmentation.max_frame_jitter)  
        return sequence + noise

    def _time_warp(self, sequence: np.ndarray) -> np.ndarray:
        """
        Time warp augmentation - simulates faster/slower gesture performance.

        - warp > 1 (stretch): Gesture performed slower, interpolate more intermediate frames
        - warp < 1 (compress): Gesture performed faster, fewer intermediate frames

        Both cases resample back to original seq_len to maintain fixed input size.
        This simulates natural variation in gesture speed while preserving the gesture's
        characteristic motion pattern.
        """
        seq_len, feat_dim = sequence.shape
        warp = np.random.uniform(
            1 - self.config.augmentation.time_warp_factor,
            1 + self.config.augmentation.time_warp_factor
        )

        # Calculate warped length (can be smaller OR larger than seq_len)
        tgt_len = int(seq_len * warp)
        tgt_len = max(2, tgt_len)  # Ensure at least 2 frames for valid interpolation

        # Step 1: Resample original sequence to warped length
        # This simulates the gesture happening at a different speed
        src_indices = np.arange(seq_len)
        tgt_indices = np.linspace(0, seq_len - 1, tgt_len)

        warped = np.zeros((tgt_len, feat_dim))
        for d in range(feat_dim):
            warped[:, d] = np.interp(tgt_indices, src_indices, sequence[:, d])

        # Step 2: Resample warped sequence back to original length
        # This ensures consistent input size for the model
        final_indices = np.linspace(0, tgt_len - 1, seq_len)
        out = np.zeros_like(sequence)
        for d in range(feat_dim):
            out[:, d] = np.interp(final_indices, np.arange(tgt_len), warped[:, d])

        return out

    def _keypoint_dropout(self, sequence: np.ndarray) -> np.ndarray:
        """
        Drop landmarks instead of raw features.
        """
        result = sequence.copy()
        mode = self.config.augmentation.landmark_dropout_mode

        if mode == "fingertips":
            drop_pool = FINGERTIP_INDICES
        else:
            drop_pool = np.arange(min(21, sequence.shape[1] // 4))

        n_drop = max(1, int(len(drop_pool) * self.config.augmentation.dropout_rate))
        drop_ids = np.random.choice(drop_pool, n_drop, replace=False)

        for idx in drop_ids:
            start = idx * 4
            result[:, start:start + 4] = 0.0

        return result

    def _scale(self, sequence: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(*self.config.augmentation.scale_range)
        return sequence * scale

    def _rotate_2d(self, sequence: np.ndarray) -> np.ndarray:
        angle = np.deg2rad(np.random.uniform(*self.config.augmentation.rotation_range))
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        result = sequence.copy()
        wrist_x = sequence[:, WRIST_IDX * 4]
        wrist_y = sequence[:, WRIST_IDX * 4 + 1]

        for i in range(min(21, sequence.shape[1] // 4)):
            x_i, y_i = i * 4, i * 4 + 1
            x = sequence[:, x_i] - wrist_x
            y = sequence[:, y_i] - wrist_y
            result[:, x_i] = x * cos_a - y * sin_a + wrist_x
            result[:, y_i] = x * sin_a + y * cos_a + wrist_y

        return result

    # -------------------------------------------------
    # Depth augmentations
    # -------------------------------------------------

    def _z_scale(self, sequence: np.ndarray) -> np.ndarray:
        factor = np.random.uniform(*self.config.augmentation.z_scale_range)
        result = sequence.copy()
        for i in range(min(21, sequence.shape[1] // 4)):
            z = i * 4 + 2
            result[:, z] *= factor
        return np.clip(result, -self.config.augmentation.max_depth_delta,
                       self.config.augmentation.max_depth_delta) 

    def _z_shift(self, sequence: np.ndarray) -> np.ndarray:
        shift = np.random.uniform(*self.config.augmentation.z_shift_range)
        result = sequence.copy()
        for i in range(min(21, sequence.shape[1] // 4)):
            z = i * 4 + 2
            result[:, z] += shift
        return result

    def _z_proportional(self, sequence: np.ndarray) -> np.ndarray:
        ratio = np.random.uniform(*self.config.augmentation.z_proportional_range)
        result = sequence.copy()
        wrist_z = sequence[:, WRIST_IDX * 4 + 2]

        for idx in FINGERTIP_INDICES:
            z = idx * 4 + 2
            result[:, z] = wrist_z + (sequence[:, z] - wrist_z) * ratio

        return result

    def _z_finger_length(self, sequence: np.ndarray) -> np.ndarray:
        result = sequence.copy()

        for finger in ALL_FINGERS:
            base = finger[0]
            base_z = sequence[:, base * 4 + 2]
            mult = np.random.uniform(*self.config.augmentation.z_finger_length_range)

            for j, idx in enumerate(finger[1:], start=1):
                z = idx * 4 + 2
                progress = j / len(finger)
                local = 1 + (mult - 1) * progress
                result[:, z] = base_z + (sequence[:, z] - base_z) * local

        return result

    def _z_noise(self, sequence: np.ndarray) -> np.ndarray:
        std = self.config.augmentation.z_noise_std
        noise = np.random.normal(0, std, sequence.shape)
        noise = np.clip(noise, -self.config.augmentation.max_depth_delta,
                        self.config.augmentation.max_depth_delta)  
        return sequence + noise

    def _hand_tilt(self, sequence: np.ndarray) -> np.ndarray:
        angle = np.deg2rad(np.random.uniform(*self.config.augmentation.hand_tilt_range))
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        result = sequence.copy()
        wrist_y = sequence[:, WRIST_IDX * 4 + 1]
        wrist_z = sequence[:, WRIST_IDX * 4 + 2]

        for i in range(min(21, sequence.shape[1] // 4)):
            y = i * 4 + 1
            z = i * 4 + 2
            y_rel = sequence[:, y] - wrist_y
            z_rel = sequence[:, z] - wrist_z
            result[:, y] = y_rel * cos_a - z_rel * sin_a + wrist_y
            result[:, z] = y_rel * sin_a + z_rel * cos_a + wrist_z

        return result

    # -------------------------------------------------
    # Dataset expansion
    # -------------------------------------------------

    def generate_variants(self, sequence: np.ndarray, n_variants: int = 9) -> list[np.ndarray]:
        return [self.augment(sequence) for _ in range(n_variants)]
