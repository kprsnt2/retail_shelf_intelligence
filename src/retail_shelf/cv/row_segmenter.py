"""Shelf Row Segmentation Module.

Segments retail shelf images into horizontal shelf tiers (Top, Eye-Level, Reach, Bottom)
using horizontal edge projection and contrast analysis.
"""
from typing import List, Tuple
import numpy as np
from PIL import Image
from retail_shelf.models.shelf import RowSegment, RowLevel

class ShelfRowSegmenter:
    def __init__(self, expected_rows: int = 4):
        self.expected_rows = expected_rows

    def segment_rows(self, image: Image.Image) -> List[RowSegment]:
        """Detect horizontal shelf lips and return bounded row segments."""
        img_gray = np.array(image.convert("L"))
        height, width = img_gray.shape
        
        # Calculate horizontal derivative / gradient across image rows
        # Shelf edges have high contrast with product bodies and shelf channels
        dy = np.abs(np.diff(img_gray.astype(np.float32), axis=0))
        # Average gradient across the center 70% width to avoid outer frame uprights
        center_x1, center_x2 = int(width * 0.15), int(width * 0.85)
        row_energy = np.mean(dy[:, center_x1:center_x2], axis=1)
        
        # Find candidate shelf boundary lines
        # Window size roughly height / (expected_rows * 2)
        step = height // self.expected_rows
        boundaries = []
        
        for i in range(1, self.expected_rows):
            target_region_start = int((i * step) - (step * 0.35))
            target_region_end = int((i * step) + (step * 0.35))
            target_region_start = max(0, target_region_start)
            target_region_end = min(height - 1, target_region_end)
            
            # Find peak gradient in expected transition zone
            sub_energy = row_energy[target_region_start:target_region_end]
            if len(sub_energy) > 0:
                peak_idx = target_region_start + int(np.argmax(sub_energy))
                boundaries.append(peak_idx)
            else:
                boundaries.append(i * step)
        
        # Build row intervals
        row_cuts = [40] + sorted(boundaries) + [height - 40]
        
        tiers: List[RowLevel] = ["top", "eye_level", "reach", "bottom"]
        if self.expected_rows != 4:
            tiers = ["top"] + ["reach"] * (self.expected_rows - 2) + ["bottom"]
        
        segments: List[RowSegment] = []
        for r_idx in range(len(row_cuts) - 1):
            y_min = int(row_cuts[r_idx])
            y_max = int(row_cuts[r_idx + 1])
            tier = tiers[r_idx] if r_idx < len(tiers) else "reach"
            
            segments.append(RowSegment(
                row_index=r_idx,
                row_level=tier,
                y_min=y_min,
                y_max=y_max,
                total_slots=0,
                occupied_slots=0,
                oos_slots=0,
                facings=[]
            ))
            
        return segments
