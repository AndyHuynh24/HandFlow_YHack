"""Validate and clean gesture sequence data before feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from handflow.utils.logging import get_logger
logger = get_logger("handflow.data")


if TYPE_CHECKING:
    from handflow.utils.config import Config

@dataclass
class ValidationResult:
    """Result of validating a sequence."""
    is_valid: bool
    reason: str | None = None
    stats: dict | None = None

class SequenceValidator:
    """
    Validate gesture sequences before processing.
    
    Checks for:
    - NaN/Inf values
    - Low confidence detections
    - Outlier keypoints
    - Sequence length consistency
    
    """
    
    def __init__(self, config: Config | None = None) -> None:
        """
        Initialize validator.
        
        Args:
            config: Validation configuration. Uses defaults if None.
        """
        self.config = config
    
    def validate(self, sequence: np.ndarray) -> ValidationResult:
        """
        Validate a single sequence.
        
        Args:
            sequence: Shape (num_frames, 84) - raw keypoints
            
        Returns:
            ValidationResult with is_valid flag and reason if invalid.
        """
        stats = {}
        
        # Check shape
        if len(sequence.shape) != 2:
            return ValidationResult(False, f"Invalid shape: {sequence.shape}")
        
        num_frames, num_features = sequence.shape
        stats["num_frames"] = num_frames
        stats["num_features"] = num_features
        
        # Check sequence length
        if self.config.data.validator.check_sequence_length:
            if num_frames != self.config.data.sequence_length:
                return ValidationResult(
                    False, 
                    f"Wrong length: {num_frames} != {self.config.data.sequence_length}",
                    stats
                )
        
        # Check for NaN
        if self.config.data.validator.check_nan:
            nan_count = np.sum(np.isnan(sequence))
            stats["nan_count"] = int(nan_count)
            if nan_count > 0:
                return ValidationResult(False, f"Contains {nan_count} NaN values", stats)
        
        # Check for Inf
        if self.config.data.validator.check_inf:
            inf_count = np.sum(np.isinf(sequence))
            stats["inf_count"] = int(inf_count)
            if inf_count > 0:
                return ValidationResult(False, f"Contains {inf_count} Inf values", stats)
        

        
        # Check for outliers
        if self.config.data.validator.check_outliers:
            # Compute z-scores
            mean = np.mean(sequence)
            std = np.std(sequence)
            if std > 0:
                z_scores = np.abs((sequence - mean) / std)
                max_z = np.max(z_scores)
                stats["max_z_score"] = float(max_z)
                
                if max_z > self.config.data.validator.outlier_std_threshold:
                    return ValidationResult(
                        False,
                        f"Outlier detected: z-score {max_z:.1f} > {self.config.data.validator.outlier_std_threshold}",
                        stats
                    )
        
        return ValidationResult(True, None, stats)
    
    def validate_batch(
        self, 
        sequences: np.ndarray, 
        return_valid_only: bool = True
    ) -> tuple[np.ndarray, list[int], list[ValidationResult]]:
        """
        Validate a batch of sequences.
        
        Args:
            sequences: Shape (n_samples, num_frames, num_features)
            return_valid_only: If True, return only valid sequences
            
        Returns:
            Tuple of (valid_sequences, valid_indices, all_results)
        """
        results = []
        valid_indices = []
        
        for i, seq in enumerate(sequences):
            result = self.validate(seq)
            results.append(result)
            if result.is_valid:
                valid_indices.append(i)
        
        if return_valid_only:
            valid_sequences = sequences[valid_indices]
        else:
            valid_sequences = sequences
        
        return valid_sequences, valid_indices, results
    
    def get_summary(self, results: list[ValidationResult]) -> dict:
        """
        Get summary statistics from validation results.
        
        Args:
            results: List of ValidationResult objects.
            
        Returns:
            Summary dictionary.
        """
        total = len(results)
        valid = sum(1 for r in results if r.is_valid)
        invalid = total - valid
        
        # Count reasons
        reasons = {}
        for r in results:
            if not r.is_valid and r.reason:
                reason_type = r.reason.split(":")[0]
                reasons[reason_type] = reasons.get(reason_type, 0) + 1
        
        return {
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "valid_ratio": valid / total if total > 0 else 0,
            "reasons": reasons,
        }



