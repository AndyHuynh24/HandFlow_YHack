"""Extracts 96 geometric features per frame from raw hand landmarks for gesture classification."""

from __future__ import annotations
import numpy as np


class FeatureEngineer:
    """Computes 96 features per frame: relative positions, distances, velocities, angles, and pinch dynamics.

    Velocity features are normalized by delta_time so the model behaves consistently across frame rates.
    """

    DEFAULT_REFERENCE_FPS = 20.0

    def __init__(self, reference_fps: float = None) -> None:
        if reference_fps is not None:
            self._reference_fps = reference_fps
        else:
            # Try to load from config
            try:
                from handflow.utils import load_config
                config = load_config("config/config.yaml")
                self._reference_fps = getattr(config.data, 'target_fps', self.DEFAULT_REFERENCE_FPS)
            except Exception:
                self._reference_fps = self.DEFAULT_REFERENCE_FPS

        self._reference_dt = 1.0 / self._reference_fps

    def transform(self, sequence: np.ndarray, delta_time: float = None) -> np.ndarray:
        """Transform raw keypoints (T, 84) into engineered features (T, 96)."""
        landmarks_xyz = sequence.reshape(-1, 21, 4)[:, :, :3]

        raw_thumb_tip = landmarks_xyz[:, 4, :]
        raw_index_mcp = landmarks_xyz[:, 5, :]
        raw_index_tip = landmarks_xyz[:, 8, :]

        # Frame-to-frame velocities (thumb velocity distinguishes touch from hover)
        vel_index_mcp = np.diff(raw_index_mcp, axis=0, prepend=raw_index_mcp[0:1])
        vel_index_tip = np.diff(raw_index_tip, axis=0, prepend=raw_index_tip[0:1])
        vel_thumb_tip = np.diff(raw_thumb_tip, axis=0, prepend=raw_thumb_tip[0:1])

        # Normalize velocities to reference FPS so model is frame-rate invariant
        if delta_time is not None and delta_time > 0:
            time_scale = self._reference_dt / delta_time
            vel_index_mcp = vel_index_mcp * time_scale
            vel_index_tip = vel_index_tip * time_scale
            vel_thumb_tip = vel_thumb_tip * time_scale

        # Wrist-relative positions for translation invariance
        wrist = landmarks_xyz[:, 0:1, :]
        relative_xyz = landmarks_xyz - wrist

        positions = relative_xyz.reshape(sequence.shape[0], -1)  # (T, 63)

        distances = self._compute_inter_finger_distances(sequence)
        angles = self._compute_finger_angles(sequence)

        # Pinch dynamics: aperture rate-of-change separates taps from hovers
        thumb_index_dist = distances[:, 0:1]
        pinch_aperture_delta = np.diff(thumb_index_dist, axis=0, prepend=thumb_index_dist[0:1])
        pinch_aperture_accel = np.diff(pinch_aperture_delta, axis=0, prepend=pinch_aperture_delta[0:1])
        thumb_index_z_diff = (raw_thumb_tip[:, 2:3] - raw_index_tip[:, 2:3])

        # Thumb posture: abduction angle and extension distance
        wrist_flat = landmarks_xyz[:, 0, :]
        index_mcp = landmarks_xyz[:, 5, :]
        v_thumb = raw_thumb_tip - wrist_flat
        v_palm = index_mcp - wrist_flat
        dot_prod = np.sum(v_thumb * v_palm, axis=1, keepdims=True)
        norm_thumb = np.linalg.norm(v_thumb, axis=1, keepdims=True) + 1e-8
        norm_palm = np.linalg.norm(v_palm, axis=1, keepdims=True) + 1e-8
        thumb_abduction = np.arccos(np.clip(dot_prod / (norm_thumb * norm_palm), -1, 1))
        thumb_wrist_dist = np.linalg.norm(raw_thumb_tip - wrist_flat, axis=1, keepdims=True)

        T = sequence.shape[0]
        output = np.empty((T, 96), dtype=np.float32)

        output[:, 0:63] = positions
        output[:, 63:68] = distances
        output[:, 68:71] = raw_thumb_tip
        output[:, 71:74] = raw_index_mcp
        output[:, 74:77] = raw_index_tip
        output[:, 77:80] = vel_index_mcp
        output[:, 80:83] = vel_index_tip
        output[:, 83:88] = angles
        output[:, 88:91] = vel_thumb_tip
        output[:, 91:92] = pinch_aperture_delta
        output[:, 92:93] = pinch_aperture_accel
        output[:, 93:94] = thumb_index_z_diff
        output[:, 94:95] = thumb_abduction
        output[:, 95:96] = thumb_wrist_dist

        return output

    def _compute_inter_finger_distances(self, sequence: np.ndarray) -> np.ndarray:
        """
        Compute distances between adjacent fingertips over time.
    
        Returns:
            distances: (T, 4)
        """
        num_frames = sequence.shape[0]
        distances = np.zeros((num_frames, 5))
    
        # Fingertip landmark indices: Thumb to Index, Mid, Ring, Pinky
        pairs = [(4, 8), (4, 12), (4, 16), (4, 20), (4, 6)]
    
        for i, (p1, p2) in enumerate(pairs):
            p1_coords = sequence[:, p1*4 : p1*4 + 3]
            p2_coords = sequence[:, p2*4 : p2*4 + 3]
            distances[:, i] = np.linalg.norm(p1_coords - p2_coords, axis=1)
    
        return distances

    def _compute_finger_angles(self, sequence: np.ndarray) -> np.ndarray:
        """
        Compute bending angles for all 5 fingers.
        Angle is computed at the middle joint (PIP for fingers, IP for thumb).
        Returns: (T, 5) where values are normalized radians (or degrees scaled).
        """
        # Indices for joints [Proxim, Mid, Distal]
        # Thumb: 2, 3, 4
        # Index: 5, 6, 7
        # Middle: 9, 10, 11
        # Ring: 13, 14, 15
        # Pinky: 17, 18, 19
        
        joints = [
            (2, 3, 4),   # Thumb
            (5, 6, 7),   # Index
            (9, 10, 11), # Middle
            (13, 14, 15),# Ring
            (17, 18, 19) # Pinky
        ]
        
        T = sequence.shape[0]
        angles = np.zeros((T, 5))
        
        for i, (a_idx, b_idx, c_idx) in enumerate(joints):
            # Extract points (T, 3)
            # Reshape raw features to (T, 21, 4) then take :3 => (T, 21, 3) if input is (T, 84)
            # But sequence input is (T, 84) flattened.
            # So:
            a = sequence[:, a_idx*4 : a_idx*4+3]
            b = sequence[:, b_idx*4 : b_idx*4+3]
            c = sequence[:, c_idx*4 : c_idx*4+3]
            
            # Vectors BA and BC
            ba = a - b
            bc = c - b
            
            # Normalize vectors
            # Add eps to avoid div by zero
            norm_ba = np.linalg.norm(ba, axis=1, keepdims=True) + 1e-8
            norm_bc = np.linalg.norm(bc, axis=1, keepdims=True) + 1e-8
            
            dot_prod = np.sum(ba * bc, axis=1, keepdims=True) / (norm_ba * norm_bc)
            
            # Clip for arccos stability
            dot_prod = np.clip(dot_prod, -1.0, 1.0)
            
            # Angle in radians
            angle_rad = np.arccos(dot_prod)
            
            # Store in output (flatten from (T,1) to (T,))
            angles[:, i] = angle_rad.flatten()
            
        return angles

    def get_output_dim(self) -> int:
        """
        Get the output dimension after feature engineering.

        Returns:
            Total feature count: 96
        """
        return 96


def add_velocity_features(
    keypoints: np.ndarray,
    delta_time: float = None,
    reference_fps: float = 20.0
) -> np.ndarray:
    """
    Standalone function to add velocity features with optional FPS normalization.

    Args:
        keypoints: Shape (num_frames, 84).
        delta_time: Time between frames in seconds. If provided, normalizes velocities.
        reference_fps: Reference FPS for normalization (default 20).

    Returns:
        Array with velocity appended, shape (num_frames, 168).
    """
    velocities = np.diff(keypoints, axis=0, prepend=keypoints[0:1])

    # Normalize if delta_time provided
    if delta_time is not None and delta_time > 0:
        reference_dt = 1.0 / reference_fps
        time_scale = reference_dt / delta_time
        velocities = velocities * time_scale

    return np.concatenate([keypoints, velocities], axis=-1)


def add_acceleration_features(
    keypoints: np.ndarray,
    delta_time: float = None,
    reference_fps: float = 20.0
) -> np.ndarray:
    """
    Standalone function to add acceleration features with optional FPS normalization.

    Args:
        keypoints: Shape (num_frames, 84).
        delta_time: Time between frames in seconds. If provided, normalizes.
        reference_fps: Reference FPS for normalization (default 20).

    Returns:
        Array with acceleration appended, shape (num_frames, 168).
    """
    velocity = np.diff(keypoints, axis=0, prepend=keypoints[0:1])
    acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])

    # Normalize if delta_time provided
    if delta_time is not None and delta_time > 0:
        reference_dt = 1.0 / reference_fps
        time_scale = reference_dt / delta_time
        # Acceleration scales with time^2
        velocity = velocity * time_scale
        acceleration = acceleration * (time_scale ** 2)

    return np.concatenate([keypoints, acceleration], axis=-1)
