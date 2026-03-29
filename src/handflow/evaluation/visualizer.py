"""Visualization tools for gesture sequences and engineered features."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.animation import FuncAnimation

# MediaPipe Hand Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),    # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),    # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]


class FeatureVisualizer:
    """
    Visualization helper for HandFlow features.
    """

    @staticmethod
    def reconstruct_landmarks(feature_seq: np.ndarray) -> np.ndarray:
        """
        Reconstruct 3D landmarks from feature sequence.
        
        Assumes the first 63 features are the flattened (x, y, z) coordinates
        of the 21 hand landmarks, normalized relative to the wrist.
        
        Args:
            feature_seq: (T, D) feature array.
            
        Returns:
            (T, 21, 3) landmarks.
        """
        # Take first 63 dims
        positions = feature_seq[:, :63]
        return positions.reshape(-1, 21, 3)

    @staticmethod
    def create_gesture_animation(
        feature_seq: np.ndarray, 
        title: str = "Gesture Animation"
    ) -> FuncAnimation:
        """
        Create a Matplotlib animation of the hand gesture.
        
        Args:
            feature_seq: (T, D) feature array.
            title: Plot title.
            
        Returns:
            Matplotlib FuncAnimation object.
        """
        landmarks_seq = FeatureVisualizer.reconstruct_landmarks(feature_seq)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Determine fixed bounds for stability
        # Use min/max of entire sequence to keep scale constant
        all_x = landmarks_seq[:, :, 0].flatten()
        all_y = landmarks_seq[:, :, 1].flatten()
        
        # Add some padding
        pad = 0.05
        x_min, x_max = all_x.min() - pad, all_x.max() + pad
        y_min, y_max = all_y.min() - pad, all_y.max() + pad
        
        # Center around wrist (0,0) usually
        limit = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
        limit = max(limit, 0.2) # Minimum size
        
        def plot_frame(i):
            ax.clear()
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.invert_yaxis() # Match screen coords
            ax.set_aspect('equal')
            ax.set_title(f"{title}\nFrame {i+1}/{len(landmarks_seq)}")
            ax.grid(True, alpha=0.3)
            
            landmarks = landmarks_seq[i]
            x = landmarks[:, 0]
            y = landmarks[:, 1]
            
            # Connections
            for start, end in HAND_CONNECTIONS:
                ax.plot([x[start], x[end]], [y[start], y[end]], 'b-', linewidth=1.5, alpha=0.6)
            
            # Points
            colors = ['red'] + ['orange']*4 + ['green']*4 + ['blue']*4 + ['purple']*4 + ['pink']*4
            ax.scatter(x, y, c=colors, s=30, zorder=5)
            
            # Wrist annotation
            ax.annotate('Wrist', (x[0], y[0]), xytext=(5, 5), textcoords='offset points', fontsize=8)

        anim = FuncAnimation(
            fig, 
            plot_frame, 
            frames=len(landmarks_seq), 
            interval=100,
            blit=False
        )
        plt.close() # Prevent double display in notebooks
        return anim

    @staticmethod
    def create_feature_plot(
        feature_seq: np.ndarray, 
        feature_names: list[str] | None = None
    ) -> go.Figure:
        """
        Create an interactive Plotly line graph of features (subplots).
        
        Args:
            feature_seq: (T, 88) feature array.
            feature_names: Ignored, generated automatically based on schema.
            
        Returns:
            Plotly Figure object.
        """
        T, D = feature_seq.shape
        x = np.arange(T)
        
        # Define Feature Groups based on FeatureEngineer.transform
        # 1. Distances (5): Indices 63-67
        # 2. Keypoints (9): Indices 68-76 (ThumbTip, IndexMCP, IndexTip)
        # 3. Velocities (6): Indices 77-82
        # 4. Angles (5): Indices 83-87
        
        groups = [
            {
                "name": "Inter-finger Distances",
                "indices": range(63, 68),
                "labels": ["Thumb-Index", "Thumb-Mid", "Thumb-Ring", "Thumb-Pinky", "Thumb-IndexPIP"]
            },
            {
                "name": "Specific Keypoints (Raw)",
                "indices": range(68, 77),
                "labels": [
                    "ThumbTip_X", "ThumbTip_Y", "ThumbTip_Z",
                    "IndexMCP_X", "IndexMCP_Y", "IndexMCP_Z",
                    "IndexTip_X", "IndexTip_Y", "IndexTip_Z"
                ]
            },
            {
                "name": "Velocities",
                "indices": range(77, 83),
                "labels": [
                    "Vel_IndMCP_X", "Vel_IndMCP_Y", "Vel_IndMCP_Z",
                    "Vel_IndTip_X", "Vel_IndTip_Y", "Vel_IndTip_Z"
                ]
            },
            {
                "name": "Finger Angles",
                "indices": range(83, 88),
                "labels": ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            }
        ]
        
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[g["name"] for g in groups]
        )
        
        for row_idx, group in enumerate(groups, start=1):
            if group["indices"].stop > D:
                continue # Skip if features missing
                
            for i, feat_idx in enumerate(group["indices"]):
                label = group["labels"][i]
                fig.add_trace(
                    go.Scatter(
                        x=x, 
                        y=feature_seq[:, feat_idx],
                        mode='lines',
                        name=label,
                        legendgroup=str(row_idx),
                        hovertemplate=f"<b>{label}</b><br>Frame: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>"
                    ),
                    row=row_idx, col=1
                )

        fig.update_layout(
            title="Engineered Features Analysis",
            height=1000,
            template="plotly_dark",
            hovermode="x unified",
            showlegend=True
        )
        
        return fig
