"""Spatial hand tracker that stabilizes MediaPipe's noisy handedness labels via centroid
continuity tracking and majority voting over a sliding window.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

from handflow.utils import get_logger


@dataclass
class TrackedHand:
    """A hand being tracked with label voting history."""
    slot: str                                    # "slot_0" or "slot_1" - tracking identity
    centroid: Tuple[float, float]                # Current (x, y) normalized
    landmarks: Optional[np.ndarray] = None       # (21, 4) array

    # Label voting system
    label_history: deque = field(default_factory=lambda: deque(maxlen=15))
    stable_label: str = "Unknown"                # Current stable label after voting

    # Tracking state
    frames_visible: int = 0
    frames_missing: int = 0

    # MediaPipe confidence for this detection
    mp_confidence: float = 0.5


class HandTracker:
    """
    Robust hand tracker combining spatial tracking with MediaPipe predictions.
    """

    # Spatial matching
    MATCH_THRESHOLD = 0.25          # Max distance to match detection to tracked hand

    # Memory
    MEMORY_FRAMES = 10              # Frames to remember a missing hand

    # Phantom filtering
    MIN_HAND_SEPARATION = 0.12      # Min distance between two real hands

    # Label voting
    STABILITY_WINDOW = 15           # Frames of history for voting
    VOTE_THRESHOLD = 0.6            # 60% agreement needed to change label
    CONFIDENCE_THRESHOLD = 0.65     # MediaPipe confidence to trust its label

    def __init__(self):
        self.logger = get_logger("handflow.HandTracker")

        # Two tracking slots (max 2 hands)
        self._slot_0: Optional[TrackedHand] = None
        self._slot_1: Optional[TrackedHand] = None

        # Settings
        self._flip_h = True
        self._frame_count = 0

    def update(
        self,
        results,
        flip_h: bool,
        swap_hands: bool
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Process MediaPipe results and return (right_kp, left_kp).
        """
        self._frame_count += 1
        self._flip_h = flip_h

        # Extract detections with MediaPipe labels
        detections = self._extract_detections(results)

        # Filter phantom detections (same hand detected twice)
        detections = self._filter_phantoms(detections)

        # Match detections to tracked slots
        self._match_and_update(detections)

        # Update label votes and compute stable labels
        self._update_stable_labels()

        # Cleanup missing hands
        self._cleanup_stale()

        # Build result
        return self._build_result(swap_hands)

    def _extract_detections(self, results) -> List[dict]:
        """Extract detections with MediaPipe's handedness predictions."""
        if not results or not results.multi_hand_landmarks:
            return []

        detections = []

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Centroid from stable landmarks
            centroid = self._compute_centroid(hand_landmarks)

            # Full landmarks
            landmarks = np.array([
                [lm.x, lm.y, lm.z, getattr(lm, "visibility", 0.0)]
                for lm in hand_landmarks.landmark
            ])

            # MediaPipe's handedness prediction
            mp_label = None
            mp_confidence = 0.5
            if results.multi_handedness and i < len(results.multi_handedness):
                handedness = results.multi_handedness[i]
                if handedness.classification:
                    mp_label = handedness.classification[0].label
                    mp_confidence = handedness.classification[0].score

            detections.append({
                'centroid': centroid,
                'landmarks': landmarks,
                'mp_label': mp_label,
                'mp_confidence': mp_confidence
            })

        return detections

    def _compute_centroid(self, hand_landmarks) -> Tuple[float, float]:
        """Compute centroid from wrist, middle MCP, pinky MCP."""
        points = np.array([
            [hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y]
            for i in [0, 9, 17]
        ])
        c = points.mean(axis=0)
        return (float(c[0]), float(c[1]))

    def _filter_phantoms(self, detections: List[dict]) -> List[dict]:
        """Filter out phantom double-detections of same hand."""
        if len(detections) < 2:
            return detections

        d0, d1 = detections[0], detections[1]
        dist = self._distance(d0['centroid'], d1['centroid'])

        if dist < self.MIN_HAND_SEPARATION:
            # Too close - same hand detected twice, keep higher confidence
            keep = d0 if d0['mp_confidence'] >= d1['mp_confidence'] else d1
            self.logger.debug(f"[Tracker] Filtered phantom (dist={dist:.3f})")
            return [keep]

        return detections

    def _match_and_update(self, detections: List[dict]):
        """Match detections to tracking slots and update."""

        if not detections:
            # No hands - increment missing counters
            if self._slot_0:
                self._slot_0.frames_missing += 1
                self._slot_0.frames_visible = 0
            if self._slot_1:
                self._slot_1.frames_missing += 1
                self._slot_1.frames_visible = 0
            return

        # Get current slots
        slots = []
        if self._slot_0:
            slots.append(('slot_0', self._slot_0))
        if self._slot_1:
            slots.append(('slot_1', self._slot_1))

        if not slots:
            # No tracked hands - create new ones
            self._create_hands_from_detections(detections)
            return

        # Match by proximity
        used_detections = set()
        used_slots = set()

        # Build distance matrix
        matches = []
        for det_idx, det in enumerate(detections):
            for slot_name, slot in slots:
                dist = self._distance(det['centroid'], slot.centroid)
                if dist < self.MATCH_THRESHOLD:
                    matches.append((dist, det_idx, slot_name, slot))

        # Greedy matching by distance
        matches.sort(key=lambda x: x[0])

        for dist, det_idx, slot_name, slot in matches:
            if det_idx in used_detections or slot_name in used_slots:
                continue

            # Update this slot with detection
            det = detections[det_idx]
            self._update_slot(slot, det)
            used_detections.add(det_idx)
            used_slots.add(slot_name)

        # Handle unmatched detections (new hands)
        for det_idx, det in enumerate(detections):
            if det_idx not in used_detections:
                self._create_new_hand(det)

        # Handle unmatched slots (hands went missing)
        for slot_name, slot in slots:
            if slot_name not in used_slots:
                slot.frames_missing += 1
                slot.frames_visible = 0

    def _update_slot(self, slot: TrackedHand, det: dict):
        """Update a tracking slot with new detection."""
        slot.centroid = det['centroid']
        slot.landmarks = det['landmarks']
        slot.frames_visible += 1
        slot.frames_missing = 0
        slot.mp_confidence = det['mp_confidence']

        # Add MediaPipe's label vote (weighted by confidence)
        mp_label = det.get('mp_label')
        if mp_label and det['mp_confidence'] > self.CONFIDENCE_THRESHOLD:
            # High confidence - add vote
            slot.label_history.append(mp_label)
        elif mp_label:
            # Lower confidence - still add but it will be outvoted if wrong
            slot.label_history.append(mp_label)

    def _create_hands_from_detections(self, detections: List[dict]):
        """Create tracked hands from detections when no hands are tracked."""
        if len(detections) == 1:
            det = detections[0]
            hand = self._make_tracked_hand('slot_0', det)
            self._slot_0 = hand
            self.logger.info(f"[Tracker] New hand slot_0 at ({det['centroid'][0]:.2f}, {det['centroid'][1]:.2f})")

        elif len(detections) >= 2:
            # Sort by X position
            dets = sorted(detections[:2], key=lambda d: d['centroid'][0])

            self._slot_0 = self._make_tracked_hand('slot_0', dets[0])
            self._slot_1 = self._make_tracked_hand('slot_1', dets[1])

            self.logger.info(f"[Tracker] New hands: slot_0 at x={dets[0]['centroid'][0]:.2f}, slot_1 at x={dets[1]['centroid'][0]:.2f}")

    def _create_new_hand(self, det: dict):
        """Create a new tracked hand in an empty slot."""
        if self._slot_0 is None:
            self._slot_0 = self._make_tracked_hand('slot_0', det)
            self.logger.info(f"[Tracker] New hand slot_0")
        elif self._slot_1 is None:
            self._slot_1 = self._make_tracked_hand('slot_1', det)
            self.logger.info(f"[Tracker] New hand slot_1")
        # else: already have 2 hands, ignore

    def _make_tracked_hand(self, slot: str, det: dict) -> TrackedHand:
        """Create a TrackedHand from detection."""
        hand = TrackedHand(
            slot=slot,
            centroid=det['centroid'],
            landmarks=det['landmarks'],
            label_history=deque(maxlen=self.STABILITY_WINDOW),
            stable_label="Unknown",
            frames_visible=1,
            frames_missing=0,
            mp_confidence=det['mp_confidence']
        )

        # Seed label history with initial votes
        mp_label = det.get('mp_label')
        if mp_label:
            # Add initial votes
            for _ in range(3):
                hand.label_history.append(mp_label)
        else:
            # No MediaPipe label - use position as initial guess
            x = det['centroid'][0]
            if self._flip_h:
                guess = "Left" if x < 0.5 else "Right"
            else:
                guess = "Right" if x < 0.5 else "Left"
            for _ in range(3):
                hand.label_history.append(guess)

        return hand

    def _update_stable_labels(self):
        """Update stable labels based on voting history."""
        hands = [h for h in [self._slot_0, self._slot_1] if h is not None]

        for hand in hands:
            if len(hand.label_history) < 3:
                continue

            # Count votes
            votes = list(hand.label_history)
            right_votes = sum(1 for v in votes if v == "Right")
            left_votes = sum(1 for v in votes if v == "Left")
            total = len(votes)

            # Determine winning label with threshold
            if right_votes / total >= self.VOTE_THRESHOLD:
                new_label = "Right"
            elif left_votes / total >= self.VOTE_THRESHOLD:
                new_label = "Left"
            else:
                # No clear winner - keep current or use simple majority
                new_label = "Right" if right_votes > left_votes else "Left"

            # Update if changed
            if hand.stable_label != new_label:
                old = hand.stable_label
                hand.stable_label = new_label
                self.logger.debug(f"[Tracker] {hand.slot} label: {old} -> {new_label} (R:{right_votes}/L:{left_votes})")

        # Ensure unique labels when 2 hands present
        self._ensure_unique_labels()

    def _ensure_unique_labels(self):
        """Ensure two hands don't have the same label."""
        if self._slot_0 is None or self._slot_1 is None:
            return

        if self._slot_0.stable_label == self._slot_1.stable_label:
            # Conflict! Resolve by position (more reliable than random)
            # In mirror mode: left screen = left hand
            if self._slot_0.centroid[0] < self._slot_1.centroid[0]:
                left_hand, right_hand = self._slot_0, self._slot_1
            else:
                left_hand, right_hand = self._slot_1, self._slot_0

            if self._flip_h:
                left_hand.stable_label = "Left"
                right_hand.stable_label = "Right"
            else:
                left_hand.stable_label = "Right"
                right_hand.stable_label = "Left"

            self.logger.debug(f"[Tracker] Resolved label conflict by position")

    def _cleanup_stale(self):
        """Remove hands that have been missing too long."""
        if self._slot_0 and self._slot_0.frames_missing > self.MEMORY_FRAMES:
            self.logger.info(f"[Tracker] Lost slot_0 ({self._slot_0.stable_label})")
            self._slot_0 = None

        if self._slot_1 and self._slot_1.frames_missing > self.MEMORY_FRAMES:
            self.logger.info(f"[Tracker] Lost slot_1 ({self._slot_1.stable_label})")
            self._slot_1 = None

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _build_result(
        self,
        swap_hands: bool
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Build (right_kp, left_kp) result."""
        right_kp = None
        left_kp = None

        for hand in [self._slot_0, self._slot_1]:
            if hand is None or hand.landmarks is None:
                continue
            if hand.frames_missing > 0:
                continue  # Don't return stale data

            # Get effective label
            label = hand.stable_label
            if swap_hands:
                label = "Left" if label == "Right" else "Right"

            kp = hand.landmarks.copy()

            if label == "Right":
                right_kp = kp.flatten()
            else:
                # X-flip for left hand
                kp[:, 0] = 1.0 - kp[:, 0]
                left_kp = kp.flatten()

        return right_kp, left_kp

    def get_hand_labels(self, swap_hands: bool = False) -> Dict[int, Tuple[str, Tuple[float, float]]]:
        """Get current hand labels and positions for display."""
        result = {}
        idx = 0

        for hand in [self._slot_0, self._slot_1]:
            if hand is None or hand.frames_missing > 0:
                continue

            label = hand.stable_label
            if swap_hands:
                label = "Left" if label == "Right" else "Right"

            result[idx] = (label, hand.centroid)
            idx += 1

        return result

    def get_finger_tips(self, swap_hands: bool = False) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """Get index finger tip positions."""
        right_tip = None
        left_tip = None

        for hand in [self._slot_0, self._slot_1]:
            if hand is None or hand.landmarks is None or hand.frames_missing > 0:
                continue

            label = hand.stable_label
            if swap_hands:
                label = "Left" if label == "Right" else "Right"

            tip = (float(hand.landmarks[8, 0]), float(hand.landmarks[8, 1]))

            if label == "Right":
                right_tip = tip
            else:
                left_tip = tip

        return right_tip, left_tip

    def reset(self):
        """Reset all tracking state."""
        self._slot_0 = None
        self._slot_1 = None
        self._frame_count = 0
        self.logger.info("[Tracker] Reset")
