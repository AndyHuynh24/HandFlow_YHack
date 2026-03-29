"""ArUco marker-based screen boundary detection and camera-to-screen coordinate mapping.

Four markers (IDs 0-3) at screen corners define a homography for finger-to-pixel mapping.
Handles partial occlusion (3+ markers visible) via affine/similarity estimation from cache.
"""

import cv2
import numpy as np
import yaml
import os
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field, asdict

from handflow.utils import get_logger


@dataclass
class CornerOffset:
    """Offset for a single corner as % of marker bbox width."""
    horizontal: float = 0.0  # Positive = inward (toward screen center)
    vertical: float = 0.0    # Positive = inward (toward screen center)


@dataclass
class ArUcoCalibration:
    """Calibration data for ArUco screen detection."""
    # Offsets for each corner (% of marker bbox width)
    # These adjust from marker CENTER to actual screen CORNER
    top_left: CornerOffset = field(default_factory=CornerOffset)
    top_right: CornerOffset = field(default_factory=CornerOffset)
    bottom_right: CornerOffset = field(default_factory=CornerOffset)
    bottom_left: CornerOffset = field(default_factory=CornerOffset)

    def to_dict(self) -> dict:
        return {
            'top_left': asdict(self.top_left),
            'top_right': asdict(self.top_right),
            'bottom_right': asdict(self.bottom_right),
            'bottom_left': asdict(self.bottom_left)
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ArUcoCalibration':
        return cls(
            top_left=CornerOffset(**data.get('top_left', {})),
            top_right=CornerOffset(**data.get('top_right', {})),
            bottom_right=CornerOffset(**data.get('bottom_right', {})),
            bottom_left=CornerOffset(**data.get('bottom_left', {}))
        )


class ArUcoScreenDetector:

    CONFIG_FILE = "config/aruco_calibration.yaml"

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        dictionary: int = cv2.aruco.DICT_4X4_50,
        config_file: Optional[str] = None
    ):
        """
        Args:
            screen_width: Target screen width in pixels
            screen_height: Target screen height in pixels
            dictionary: ArUco dictionary (must match printed markers)
            config_file: Path to calibration config file
        """
        self.logger = get_logger("handflow.arucoScreenDectector")

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.config_file = config_file or self.CONFIG_FILE

        # ArUco detector setup
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        # Optimized detector parameters for speed
        detector_params = cv2.aruco.DetectorParameters()

        # === Tuned for FAST detection during quick head movements ===

        # Adaptive thresholding - more windows = better detection in varying light
        detector_params = cv2.aruco.DetectorParameters()

        # --- Adaptive thresholding (avoid key-by-key binarization) ---
        detector_params.adaptiveThreshWinSizeMin = 7
        detector_params.adaptiveThreshWinSizeMax = 35
        detector_params.adaptiveThreshWinSizeStep = 6

        # --- Marker size sanity (kills keyboard keys) ---
        detector_params.minMarkerPerimeterRate = 0.05   # default 0.03
        detector_params.maxMarkerPerimeterRate = 4.0

        # --- Geometry strictness (still tolerant to blur) ---
        detector_params.polygonalApproxAccuracyRate = 0.05

        # --- Corner refinement (important for jitter) ---
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector_params.cornerRefinementWinSize = 7
        detector_params.cornerRefinementMaxIterations = 30

        # --- Marker separation (prevents grid confusion) ---
        detector_params.minCornerDistanceRate = 0.05
        detector_params.minMarkerDistanceRate = 0.05

        # --- Bit extraction (ID stability) ---
        detector_params.perspectiveRemovePixelPerCell = 6
        detector_params.perspectiveRemoveIgnoredMarginPerCell = 0.15

        # --- Error correction (most important) ---
        detector_params.errorCorrectionRate = 0.5
        self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, detector_params)

        # Calibration data
        self.calibration = ArUcoCalibration()
        self.load_calibration()

        # Detection state
        self._last_corners: Optional[np.ndarray] = None
        self._last_homography: Optional[np.ndarray] = None
        self._last_marker_width: float = 50.0  # Default marker width in pixels
        self._detection_valid = False

        # Marker position cache for occlusion handling
        # Stores last known position of each marker when all 4 were visible
        self._marker_cache: Dict[int, np.ndarray] = {}  # marker_id -> position
        self._marker_cache_frames: int = 0  # Frames since cache was fully updated
        self._cache_max_frames: int = 30  # Max frames to use cached data (~1 sec at 30fps)
        self._estimated_markers: set = set()  # Track which markers are currently estimated

        # Detection persistence - don't invalidate immediately on detection failure
        # This prevents jittering when detection fails briefly
        self._valid_grace_counter: int = 0
        self._valid_grace_max: int = 5  # Keep valid for N failed detections before invalidating

        # Detection mode tracking for debug display
        # "full" = 4 markers, "partial_3" = 3 markers + 1 estimated,
        # "partial_2" = 2 markers + 2 estimated, "grace" = using cached homography
        self._detection_mode: str = "none"

        # Temporal smoothing for corner positions (reduces jitter)
        # EMA smoothing factor: higher = more responsive, lower = smoother
        # 0.75 is more responsive for fast head movements
        self._smoothing_alpha: float = 0.75
        self._smoothed_corners: Optional[np.ndarray] = None

        # Outlier rejection - max allowed jump distance (pixels) between frames
        # If a marker jumps more than this, it's likely a false positive
        # Increased to 250 for fast head movements at lower resolutions
        self._max_jump_distance: float = 250.0
        self._last_marker_centers: Dict[int, np.ndarray] = {}  # marker_id -> last center

        # Side pairs for motion compensation
        # When one marker is blocked, use its pair on the same side
        self._side_pairs = {
            0: 3,  # TL pairs with BL (left side)
            3: 0,  # BL pairs with TL (left side)
            1: 2,  # TR pairs with BR (right side)
            2: 1,  # BR pairs with TR (right side)
        }
        # Horizontal pairs as fallback
        self._horizontal_pairs = {
            0: 1,  # TL pairs with TR (top)
            1: 0,  # TR pairs with TL (top)
            2: 3,  # BR pairs with BL (bottom)
            3: 2,  # BL pairs with BR (bottom)
        }

        # Screen corners in normalized coords (0-1)
        # Order: TL, TR, BR, BL
        self._screen_corners_norm = np.array([
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1]
        ], dtype=np.float32)

    def _validate_quadrilateral(self, corners: np.ndarray) -> bool:
        """
        Validate that corners form a reasonable convex quadrilateral.

        Rejects false positives from random text/noise detections by checking:
        1. Minimum area (not degenerate)
        2. Convexity (proper quadrilateral shape)
        3. Aspect ratio within reasonable bounds

        Args:
            corners: 4 corner points (TL, TR, BR, BL)

        Returns:
            True if valid quadrilateral, False otherwise
        """
        if len(corners) != 4:
            return False

        # Check minimum area (at least 400 sq pixels to avoid tiny false positives)
        # Reduced for smaller processing resolutions (480x270)
        area = cv2.contourArea(corners)
        if area < 400:
            return False

        # Check convexity
        if not cv2.isContourConvex(corners):
            return False

        # Check aspect ratio is reasonable (0.2 to 5.0)
        # Calculate bounding rect
        x, y, w, h = cv2.boundingRect(corners)
        if w == 0 or h == 0:
            return False
        aspect = w / h
        if aspect < 0.2 or aspect > 5.0:
            return False

        # Check that no two adjacent corners are too close (degenerate edge)
        # Reduced for smaller processing resolutions (480x270)
        min_edge_length = 15  # pixels
        for i in range(4):
            edge_len = np.linalg.norm(corners[i] - corners[(i + 1) % 4])
            if edge_len < min_edge_length:
                return False

        return True

    def load_calibration(self) -> bool:
        """Load calibration from config file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    data = yaml.safe_load(f)
                self.calibration = ArUcoCalibration.from_dict(data)
                self.logger.info(f"[ArUco] Loaded calibration from {self.config_file}")
                return True
        except Exception as e:
            self.logger.info(f"[ArUco] Could not load calibration: {e}")
        return False

    def save_calibration(self) -> bool:
        """Save calibration to config file."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                yaml.dump(self.calibration.to_dict(), f, default_flow_style=False)
            self.logger.info(f"[ArUco] Saved calibration to {self.config_file}")
            return True
        except Exception as e:
            self.logger.info(f"[ArUco] Could not save calibration: {e}")
            return False

    def detect(self, frame: np.ndarray) -> bool:
        """
        Detect ArUco markers and compute screen corners.
        Handles partial occlusion (3 markers) using cached positions with motion compensation.

        Args:
            frame: BGR camera frame

        Returns:
            True if detection valid (4 markers or 3 with valid cache), False otherwise
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        # Map detected marker IDs to their center positions and sizes
        marker_data: Dict[int, Tuple[np.ndarray, float]] = {}

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in [0, 1, 2, 3]:
                    marker_corners = corners[i][0]  # 4 corners of the marker
                    center = np.mean(marker_corners, axis=0)
                    # Marker width = distance between corner 0 and corner 1
                    width = np.linalg.norm(marker_corners[0] - marker_corners[1])

                    # Outlier rejection: check if marker jumped too far from last position
                    if marker_id in self._last_marker_centers:
                        last_center = self._last_marker_centers[marker_id]
                        jump_distance = np.linalg.norm(center - last_center)
                        if jump_distance > self._max_jump_distance:
                            # This detection is likely a false positive (text, noise, etc.)
                            # Skip it and let the estimation logic handle this marker
                            continue

                    marker_data[marker_id] = (center, width)

        detected_count = len(marker_data)

        # Case 1: All 4 markers detected - update cache and proceed
        if detected_count == 4:
            self._marker_cache_frames = 0
            # Update cache AFTER we have all 4 (no estimation needed)
            for marker_id, (center, _) in marker_data.items():
                self._marker_cache[marker_id] = center.copy()
                # Update last known centers for outlier rejection
                self._last_marker_centers[marker_id] = center.copy()

        # Case 2: 3 markers detected - estimate 1 missing
        elif detected_count == 3 and len(self._marker_cache) == 4:
            # Find the missing marker
            missing_ids = [mid for mid in [0, 1, 2, 3] if mid not in marker_data]

            if len(missing_ids) == 1:
                # FIRST: Estimate missing marker using OLD cache values
                estimated_pos = self._estimate_missing_marker(missing_ids[0], marker_data)
                if estimated_pos is not None:
                    avg_width = np.mean([w for _, w in marker_data.values()])
                    marker_data[missing_ids[0]] = (estimated_pos, avg_width)
                    # Update cache for ESTIMATED marker too (for next frame's delta calculation)
                    self._marker_cache[missing_ids[0]] = estimated_pos.copy()

            # THEN: Update cache for visible markers
            for marker_id in [0, 1, 2, 3]:
                if marker_id in marker_data and marker_id not in missing_ids:
                    self._marker_cache[marker_id] = marker_data[marker_id][0].copy()

        # Case 3: 2 markers detected - estimate 2 missing using geometry + cache
        elif detected_count == 2 and len(self._marker_cache) == 4:
            # Find missing markers
            missing_ids = [mid for mid in [0, 1, 2, 3] if mid not in marker_data]

            if len(missing_ids) == 2:
                # Estimate both missing markers using motion transform + cache
                estimated_positions = self._estimate_missing_markers_2(missing_ids, marker_data)
                if len(estimated_positions) == 2:
                    avg_width = np.mean([w for _, w in marker_data.values()])
                    for mid, pos in estimated_positions.items():
                        marker_data[mid] = (pos, avg_width)
                        # Update cache for estimated markers
                        self._marker_cache[mid] = pos.copy()

            # Update cache for visible markers
            for marker_id in [0, 1, 2, 3]:
                if marker_id in marker_data and marker_id not in missing_ids:
                    self._marker_cache[marker_id] = marker_data[marker_id][0].copy()

        # Case 4: Less than 2 markers - can't estimate reliably
        # (need at least 2 visible markers for motion compensation)

        # Track which markers are estimated (for debug drawing)
        self._estimated_markers = set()
        detected_ids = set(ids.flatten() if ids is not None else [])
        for mid in [0, 1, 2, 3]:
            if mid in marker_data and mid not in detected_ids:
                self._estimated_markers.add(mid)


        # Determine detection mode based on actual vs estimated markers
        num_estimated = len(self._estimated_markers)
        if len(marker_data) == 4:
            if num_estimated == 0:
                self._detection_mode = "full"
            elif num_estimated == 1:
                self._detection_mode = "partial_3"
            else:
                self._detection_mode = "partial_2"

        # Need at least 4 markers (detected or estimated) to proceed
        if len(marker_data) < 4:
            # Use grace period - don't invalidate immediately
            if self._detection_valid:
                self._valid_grace_counter += 1
                self._detection_mode = "grace"
                if self._valid_grace_counter >= self._valid_grace_max:
                    # Grace expired, actually invalidate
                    self._detection_valid = False
                    self._detection_mode = "none"
                    self._valid_grace_counter = 0
                # Keep using cached homography during grace period
            else:
                self._detection_mode = "none"
            return self._detection_valid

        # Average marker width for offset calculations
        self._last_marker_width = np.mean([w for _, w in marker_data.values()])

        # Apply calibration offsets to get actual screen corners
        screen_corners = self._apply_calibration_offsets(marker_data)

        raw_corners = np.array(screen_corners, dtype=np.float32)

        # Validate that corners form a reasonable quadrilateral
        if not self._validate_quadrilateral(raw_corners):
            # Invalid shape - likely false positive detections
            if self._detection_valid:
                self._valid_grace_counter += 1
                self._detection_mode = "grace"
                if self._valid_grace_counter >= self._valid_grace_max:
                    self._detection_valid = False
                    self._detection_mode = "none"
                    self._valid_grace_counter = 0
            return self._detection_valid

        # Apply temporal smoothing (EMA) to reduce jitter
        if self._smoothed_corners is None:
            self._smoothed_corners = raw_corners.copy()
        else:
            # Exponential moving average: new = alpha * raw + (1-alpha) * old
            self._smoothed_corners = (
                self._smoothing_alpha * raw_corners +
                (1 - self._smoothing_alpha) * self._smoothed_corners
            )

        self._last_corners = self._smoothed_corners.copy()

        # Compute homography: camera corners -> normalized screen coords
        self._last_homography, _ = cv2.findHomography(
            self._last_corners,
            self._screen_corners_norm
        )

        self._detection_valid = self._last_homography is not None
        # Reset grace counter on successful detection
        if self._detection_valid:
            self._valid_grace_counter = 0
        return self._detection_valid

    def _estimate_missing_marker(
        self,
        missing_id: int,
        marker_data: Dict[int, Tuple[np.ndarray, float]]
    ) -> Optional[np.ndarray]:

        if missing_id not in self._marker_cache:
            return None

        visible_ids = list(marker_data.keys())
        H = self._estimate_motion_transform(visible_ids, marker_data)

        cached_pos = self._marker_cache[missing_id]

        if H is not None:
            pt = np.array([[cached_pos]], dtype=np.float32)
            projected = cv2.perspectiveTransform(pt, H)
            return projected[0][0]

        return cached_pos

    def _estimate_missing_markers_2(
        self,
        missing_ids: list,
        marker_data: Dict[int, Tuple[np.ndarray, float]]
    ) -> Dict[int, np.ndarray]:

        estimated = {}
        visible_ids = list(marker_data.keys())
        H = self._estimate_motion_transform(visible_ids, marker_data)

        for mid in missing_ids:
            if mid not in self._marker_cache:
                continue

            cached_pos = self._marker_cache[mid]

            if H is not None:
                pt = np.array([[cached_pos]], dtype=np.float32)
                projected = cv2.perspectiveTransform(pt, H)
                estimated[mid] = projected[0][0]
            else:
                estimated[mid] = cached_pos

        return estimated

    def _apply_calibration_offsets(self, marker_data: Dict[int, Tuple[np.ndarray, float]]) -> list:
        """Apply calibration offsets to marker positions to get screen corners."""
        screen_corners = []

        # ID 0: Top-Left - offset inward means +x, +y
        center, _ = marker_data[0]
        offset_x = self.calibration.top_left.horizontal * self._last_marker_width / 100.0
        offset_y = self.calibration.top_left.vertical * self._last_marker_width / 100.0
        screen_corners.append(center + np.array([offset_x, offset_y]))

        # ID 1: Top-Right - offset inward means -x, +y
        center, _ = marker_data[1]
        offset_x = self.calibration.top_right.horizontal * self._last_marker_width / 100.0
        offset_y = self.calibration.top_right.vertical * self._last_marker_width / 100.0
        screen_corners.append(center + np.array([-offset_x, offset_y]))

        # ID 2: Bottom-Right - offset inward means -x, -y
        center, _ = marker_data[2]
        offset_x = self.calibration.bottom_right.horizontal * self._last_marker_width / 100.0
        offset_y = self.calibration.bottom_right.vertical * self._last_marker_width / 100.0
        screen_corners.append(center + np.array([-offset_x, -offset_y]))

        # ID 3: Bottom-Left - offset inward means +x, -y
        center, _ = marker_data[3]
        offset_x = self.calibration.bottom_left.horizontal * self._last_marker_width / 100.0
        offset_y = self.calibration.bottom_left.vertical * self._last_marker_width / 100.0
        screen_corners.append(center + np.array([offset_x, -offset_y]))

        return screen_corners

    def _estimate_motion_transform(
        self,
        visible_ids: list,
        marker_data: Dict[int, Tuple[np.ndarray, float]]
    ) -> Optional[np.ndarray]:
        """
        Estimate motion transform from cached marker positions to current ones.

        Uses:
        - Homography (>=4 points)
        - Affine (3 points)
        - Translation (2 points)

        Always returns a 3x3 matrix or None.
        """

        if len(visible_ids) < 2:
            return None

        src_pts = []
        dst_pts = []

        for mid in visible_ids:
            if mid in self._marker_cache:
                src_pts.append(self._marker_cache[mid])
                dst_pts.append(marker_data[mid][0])

        src_pts = np.asarray(src_pts, dtype=np.float32)
        dst_pts = np.asarray(dst_pts, dtype=np.float32)

        # -------------------------------
        # 4+ points → Homography
        # -------------------------------
        if len(src_pts) >= 4:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return H

        # -------------------------------
        # 3 points → Affine
        # -------------------------------
        if len(src_pts) == 3:
            A, _ = cv2.estimateAffine2D(src_pts, dst_pts)
            if A is None:
                return None
            H = np.eye(3, dtype=np.float32)
            H[:2, :] = A
            return H

        # -------------------------------
        # 2 points → Similarity transform (translation + rotation + uniform scale)
        # This is better than translation-only for camera movement
        # -------------------------------
        if len(src_pts) == 2:
            # Try estimating a similarity transform (4 DOF: tx, ty, scale, rotation)
            # This handles camera pan/tilt/zoom better than translation alone
            try:
                # Use estimateAffinePartial2D which estimates similarity transform
                A, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                if A is not None:
                    H = np.eye(3, dtype=np.float32)
                    H[:2, :] = A
                    return H
            except:
                pass

            # Fallback to translation only
            delta = np.mean(dst_pts - src_pts, axis=0)
            H = np.eye(3, dtype=np.float32)
            H[0, 2] = delta[0]
            H[1, 2] = delta[1]
            return H

        return None


    def transform_point(self, camera_point: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        """
        Transform a point from camera coordinates to screen coordinates.

        Args:
            camera_point: (x, y) in camera pixel coordinates

        Returns:
            (screen_x, screen_y) in screen pixel coordinates, or None if invalid
        """
        if not self._detection_valid or self._last_homography is None:
            return None

        # Apply homography
        pt = np.array([[camera_point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self._last_homography)

        if transformed is None:
            return None

        # Convert normalized (0-1) to screen pixels
        norm_x, norm_y = transformed[0][0]

        # Clamp to valid range
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        screen_x = int(norm_x * self.screen_width)
        screen_y = int(norm_y * self.screen_height)

        return (screen_x, screen_y)

    def is_point_in_screen(self, camera_point: Tuple[float, float]) -> bool:
        """Check if a camera point falls within the detected screen region."""
        if not self._detection_valid or self._last_corners is None:
            return False

        pt = np.array([[camera_point[0], camera_point[1]]], dtype=np.float32)
        result = cv2.pointPolygonTest(self._last_corners, (camera_point[0], camera_point[1]), False)
        return result >= 0

    @property
    def is_valid(self) -> bool:
        """Check if detection is currently valid."""
        return self._detection_valid

    @property
    def screen_corners(self) -> Optional[np.ndarray]:
        """Get the last detected screen corners (camera coordinates)."""
        return self._last_corners if self._detection_valid else None

    @property
    def marker_width(self) -> float:
        """Get the average marker width (for offset calculations)."""
        return self._last_marker_width

    @property
    def detection_mode(self) -> str:
        """Get current detection mode: 'full', 'partial_3', 'partial_2', 'grace', or 'none'."""
        return self._detection_mode

    def draw_debug(self, frame: np.ndarray, finger_pos: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Draw debug visualization on frame.

        Args:
            frame: BGR frame to draw on
            finger_pos: Optional finger position in camera coords

        Returns:
            Frame with debug overlay
        """
        output = frame.copy()

        # Draw detected markers
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(output, corners, ids)

        if self._detection_valid and self._last_corners is not None:
            # Determine boundary color based on detection mode
            # Green = full (4 markers), Yellow = partial_3, Orange = partial_2, Cyan = grace
            mode_colors = {
                "full": (0, 255, 0),       # Green - best quality
                "partial_3": (0, 255, 255), # Yellow - good quality
                "partial_2": (0, 165, 255), # Orange - acceptable
                "grace": (255, 255, 0),     # Cyan - using cached
            }
            boundary_color = mode_colors.get(self._detection_mode, (0, 255, 0))

            # Draw screen boundary with mode-appropriate color
            pts = self._last_corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(output, [pts], True, boundary_color, 2)

            # Draw detection mode indicator
            mode_text = {
                "full": "4/4 Full",
                "partial_3": "3/4 Est",
                "partial_2": "2/4 Est",
                "grace": "Grace",
            }
            detected_now = len(ids) if ids is not None else 0
            mode_label = mode_text.get(self._detection_mode, self._detection_mode)
            cv2.putText(output, f"Mode: {mode_label} ({detected_now} visible)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, boundary_color, 1)

            # Draw corner labels
            # Yellow = detected, Orange = estimated from cache
            labels = ['TL', 'TR', 'BR', 'BL']
            marker_ids = [0, 1, 2, 3]
            for i, (corner, label, mid) in enumerate(zip(self._last_corners, labels, marker_ids)):
                is_estimated = mid in self._estimated_markers
                # Orange for estimated, Yellow for detected
                color = (0, 165, 255) if is_estimated else (0, 255, 255)
                cv2.circle(output, tuple(corner.astype(int)), 5, color, -1)
                label_text = f"{label}*" if is_estimated else label
                cv2.putText(output, label_text, tuple((corner + [5, -5]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Show occlusion status
            if self._estimated_markers:
                cv2.putText(output, f"Estimated: {list(self._estimated_markers)}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

            # Draw finger position and screen mapping
            if finger_pos is not None:
                fx, fy = int(finger_pos[0]), int(finger_pos[1])
                cv2.circle(output, (fx, fy), 8, (255, 0, 255), 2)

                screen_pos = self.transform_point(finger_pos)
                if screen_pos:
                    in_bounds = self.is_point_in_screen(finger_pos)
                    color = (0, 255, 0) if in_bounds else (0, 0, 255)
                    cv2.putText(output, f"Screen: {screen_pos}", (fx + 10, fy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        else:
            # Detection status - not valid
            detected = len(ids) if ids is not None else 0
            cv2.putText(output, f"Markers: {detected}/4 (need 2+)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return output


class ArUcoCalibrationUI:
    """
    Calibration window for adjusting ArUco corner offsets.

    Provides sliders to adjust the offset from marker centers
    to actual screen corners. Offsets are stored as % of marker
    bbox width for scale invariance.
    """

    def __init__(self, detector: ArUcoScreenDetector, camera_id: int = 0, settings=None):
        """
        Args:
            detector: ArUcoScreenDetector instance to calibrate
            camera_id: Camera to use for preview
            setting
        """
        self.detector = detector
        self.camera_id = camera_id
        self.settings = settings
        self._running = False
        self._cap = None
        self._after_id = None  # Track scheduled callback for cleanup

        # Slider ranges (% of marker width)
        self._min_offset = -200
        self._max_offset = 200

        # Performance settings
        self._update_interval = 33  # ~30 FPS

    def run(self):
        """Run the calibration UI (blocking)."""
        import tkinter as tk

        self._running = True

        # Create main window
        self.root = tk.Tk()
        self.root.title("ArUco Screen Calibration")
        self.root.geometry("400x550")

        # Handle window close FIRST (before anything else)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Instructions
        tk.Label(self.root, text="Adjust offsets to align green box with screen edges",
                wraplength=380, justify="center", font=("Arial", 11)).pack(pady=10)

        tk.Label(self.root, text="Values are % of marker width (positive = inward)",
                font=("Arial", 9, "italic")).pack(pady=(0, 10))

        # Slider container
        slider_frame = tk.Frame(self.root)
        slider_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Store references to sliders
        self._sliders = {}

        corners = [
            ("Top-Left", "top_left"),
            ("Top-Right", "top_right"),
            ("Bottom-Right", "bottom_right"),
            ("Bottom-Left", "bottom_left")
        ]

        for label, key in corners:
            self._create_corner_sliders(slider_frame, label, key)

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", pady=15, padx=20)

        tk.Button(btn_frame, text="Reset All", command=self._reset_all, width=12).pack(side="left", padx=10)
        tk.Button(btn_frame, text="Save & Close", command=self._save_and_close,
                 bg="#90EE90", width=12).pack(side="right", padx=10)

        # Initialize Camera
        self._init_camera()

        # Start update loop
        self._schedule_update()

        # Run main loop (blocking)
        self.root.mainloop()

        # Cleanup after mainloop exits
        self._cleanup()

    def _create_corner_sliders(self, parent, label: str, key: str):
        """Create H and V sliders for a corner."""
        import tkinter as tk

        corner_offset = getattr(self.detector.calibration, key)

        group = tk.LabelFrame(parent, text=label, font=("Arial", 9, "bold"), padx=5, pady=3)
        group.pack(fill="x", pady=4)

        # Horizontal slider
        h_frame = tk.Frame(group)
        h_frame.pack(fill="x")
        tk.Label(h_frame, text="H:", width=2).pack(side="left")

        h_var = tk.DoubleVar(value=corner_offset.horizontal)
        h_scale = tk.Scale(h_frame, from_=self._min_offset, to=self._max_offset,
                          variable=h_var, orient="horizontal", length=250, showvalue=True,
                          resolution=0.1,
                          command=lambda val, k=key: self._on_slider_change(k, 'h', val))
        h_scale.pack(side="left", fill="x", expand=True)

        # Vertical slider
        v_frame = tk.Frame(group)
        v_frame.pack(fill="x")
        tk.Label(v_frame, text="V:", width=2).pack(side="left")

        v_var = tk.DoubleVar(value=corner_offset.vertical)
        v_scale = tk.Scale(v_frame, from_=self._min_offset, to=self._max_offset,
                          variable=v_var, orient="horizontal", length=250, showvalue=True,
                          resolution=0.1,
                          command=lambda val, k=key: self._on_slider_change(k, 'v', val))
        v_scale.pack(side="left", fill="x", expand=True)

        self._sliders[key] = {'h': h_scale, 'v': v_scale, 'h_var': h_var, 'v_var': v_var}

    def _on_slider_change(self, corner_key: str, axis: str, value):
        """Called when any slider changes - update calibration immediately."""
        corner = getattr(self.detector.calibration, corner_key)
        val = float(value)
        if axis == 'h':
            corner.horizontal = val
        else:
            corner.vertical = val

    def _reset_all(self):
        """Reset all sliders to zero."""
        for key, sliders in self._sliders.items():
            sliders['h'].set(0)
            sliders['v'].set(0)
            corner = getattr(self.detector.calibration, key)
            corner.horizontal = 0
            corner.vertical = 0

    def _save_and_close(self):
        """Save calibration and close."""
        self.detector.save_calibration()
        self._running = False
        self.root.quit()  # Exit mainloop

    def _on_close(self):
        """Handle window close button."""
        self._running = False
        self.root.quit()  # Exit mainloop

    def _cleanup(self):
        """Clean up all resources."""
        # Cancel any pending after callback
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except:
                pass
            self._after_id = None

        # Release camera
        if self._cap is not None:
            try:
                self._cap.release()
            except:
                pass
            self._cap = None

        # Destroy OpenCV windows
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Process the destroy
        except:
            pass

        # Destroy tkinter window
        try:
            self.root.destroy()
        except:
            pass

    def _init_camera(self):
        """Initialize camera for preview."""
        try:
            self._cap = cv2.VideoCapture(self.camera_id)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._cap.set(cv2.CAP_PROP_FPS, 30)

            # Create window
            cv2.namedWindow("ArUco Calibration Preview", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ArUco Calibration Preview", 960, 540)

            # Flush initial buffer
            for _ in range(3):
                self._cap.read()
        except Exception as e:
            self.logger.info(f"[Calibration] Camera init error: {e}")

    def _schedule_update(self):
        """Schedule the next update."""
        if self._running:
            self._after_id = self.root.after(self._update_interval, self._update_loop)

    def _update_loop(self):
        """Update loop for camera preview."""
        if not self._running:
            return

        try:
            if self._cap and self._cap.isOpened():
                # Grab latest frame
                self._cap.grab()
                ret, frame = self._cap.retrieve()

                if ret:
                    # Apply flips
                    if self.settings:
                        if self.settings.camera.flip_horizontal:
                            frame = cv2.flip(frame, 1)
                        if self.settings.camera.flip_vertical:
                            frame = cv2.flip(frame, 0)
                    else:
                        frame = cv2.flip(frame, 1)

                    # Process ArUco detection (this uses current calibration values)
                    self.detector.detect(frame)
                    frame = self.detector.draw_debug(frame)

                    # Status overlay
                    h, w = frame.shape[:2]
                    status = "SCREEN DETECTED" if self.detector.is_valid else "Searching for markers..."
                    color = (0, 255, 0) if self.detector.is_valid else (0, 100, 255)

                    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
                    cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    if self.detector.is_valid:
                        info = f"Marker: {self.detector.marker_width:.0f}px | Adjust sliders to align green box"
                        cv2.putText(frame, info, (300, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    else:
                        cv2.putText(frame, "Place 4 ArUco markers (ID 0-3) at screen corners",
                                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    cv2.imshow("ArUco Calibration Preview", frame)
                    cv2.waitKey(1)

        except Exception as e:
            self.logger.info(f"[Calibration] Update error: {e}")

        # Schedule next update
        self._schedule_update()


# Convenience function for standalone testing
def run_calibration(camera_id: int = 0, screen_width: int = None, screen_height: int = None, settings=None):
    """Run calibration UI with auto-detected screen size."""
    import pyautogui

    if screen_width is None or screen_height is None:
        sw, sh = pyautogui.size()
        screen_width = screen_width or sw
        screen_height = screen_height or sh

    detector = ArUcoScreenDetector(
        screen_width=screen_width,
        screen_height=screen_height
    )

    ui = ArUcoCalibrationUI(detector, camera_id, settings=settings)
    ui.run()

    return detector


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ArUco Screen Calibration")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID")
    parser.add_argument("--width", type=int, help="Screen width")
    parser.add_argument("--height", type=int, help="Screen height")
    args = parser.parse_args()

    run_calibration(args.camera, args.width, args.height)
