"""ActionPathBuilder for synthesizing camera poses from WASD control input."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ActionPathBuilder:
    """Builds action_path folders with poses.npy and intrinsics.npy from WASD input."""
    
    def __init__(self):
        self.temp_dir = None
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def cleanup(self):
        """Clean up temporary directory."""
        if self.temp_dir:
            try:
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir: {e}")
            self.temp_dir = None
    
    def build_from_ctrl_input(
        self,
        ctrl_input: str,
        frame_num: int = 81,
        translation_speed: float = 0.1,
        rotation_speed: float = 0.05,
        focal_length: float = 500.0,
        image_size: tuple[int, int] = (480, 832),
    ) -> str:
        """Build action_path folder from W3C KeyW/A/S/D control input.
        
        Args:
            ctrl_input: String containing WASD keys (e.g., "WASDWWSS")
            frame_num: Number of frames to generate
            translation_speed: Speed for forward/backward movement
            rotation_speed: Speed for left/right rotation
            focal_length: Camera focal length
            image_size: (height, width) of images
            
        Returns:
            Path to directory containing poses.npy and intrinsics.npy
        """
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="action_path_")
        
        # Initialize camera pose (camera-to-world matrix)
        pose = np.eye(4)
        pose[2, 3] = -1.0  # Start 1 unit back from origin
        
        poses = []
        poses.append(pose.copy())
        
        # Process each control input character
        if not ctrl_input:
            # If no control input, generate smooth forward motion
            ctrl_input = "W" * frame_num
        
        # Ensure we have enough control inputs
        while len(ctrl_input) < frame_num - 1:
            ctrl_input += ctrl_input[-1] if ctrl_input else "W"
        
        for i in range(1, frame_num):
            key = ctrl_input[min(i-1, len(ctrl_input)-1)].upper()
            
            if key == 'W':  # Forward
                pose[2, 3] -= translation_speed
            elif key == 'S':  # Backward
                pose[2, 3] += translation_speed
            elif key == 'A':  # Left (rotate)
                angle = rotation_speed
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rotation = np.array([
                    [cos_a, 0, sin_a, 0],
                    [0, 1, 0, 0],
                    [-sin_a, 0, cos_a, 0],
                    [0, 0, 0, 1]
                ])
                pose = rotation @ pose
            elif key == 'D':  # Right (rotate)
                angle = -rotation_speed
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                rotation = np.array([
                    [cos_a, 0, sin_a, 0],
                    [0, 1, 0, 0],
                    [-sin_a, 0, cos_a, 0],
                    [0, 0, 0, 1]
                ])
                pose = rotation @ pose
            
            poses.append(pose.copy())
        
        # Convert poses to numpy array
        poses_array = np.array(poses)  # Shape: (frame_num, 4, 4)
        
        # Create intrinsics matrix
        h, w = image_size
        intrinsics = np.array([
            [focal_length, 0, w / 2.0],
            [0, focal_length, h / 2.0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Save to files
        poses_path = os.path.join(self.temp_dir, "poses.npy")
        intrinsics_path = os.path.join(self.temp_dir, "intrinsics.npy")
        
        np.save(poses_path, poses_array)
        np.save(intrinsics_path, intrinsics)
        
        logger.info(f"Generated action_path: {self.temp_dir}")
        logger.info(f"Poses shape: {poses_array.shape}")
        logger.info(f"Intrinsics shape: {intrinsics.shape}")
        
        return self.temp_dir
    
    def reset_cache(self):
        """Reset cache functionality."""
        self.cleanup()
        self.temp_dir = None