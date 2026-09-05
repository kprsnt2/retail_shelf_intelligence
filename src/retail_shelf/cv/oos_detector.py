"""Out-of-Stock (OOS) Void Detector.

Detects empty shelf slots and voids between occupied facings and boundary uprights.
"""
from typing import List
from PIL import Image
import numpy as np

from retail_shelf.models.shelf import Facing, BoundingBox, RowSegment

class OOSVoidDetector:
    def __init__(self, min_void_width_px: int = 55):
        self.min_void_width_px = min_void_width_px

    def detect_voids(self, image: Image.Image, row: RowSegment, detected_facings: List[Facing]) -> List[Facing]:
        """Identify missing facings/empty voids in a shelf row segment."""
        img_np = np.array(image)
        height, width, _ = img_np.shape
        
        y1 = row.y_min + int((row.y_max - row.y_min) * 0.15)
        y2 = row.y_max - 24 # above shelf lip
        
        left_bound = 80
        right_bound = width - 80
        
        # Sort detected facings horizontally
        sorted_facings = sorted(detected_facings, key=lambda f: f.bbox.x1)
        
        voids: List[Facing] = []
        void_counter = 0
        
        # Check gap between left frame and first facing
        if sorted_facings:
            first_x1 = sorted_facings[0].bbox.x1
            if (first_x1 - left_bound) >= self.min_void_width_px:
                void_w = first_x1 - left_bound
                num_slots = max(1, round(void_w / 140.0))
                slot_w = void_w / num_slots
                for i in range(num_slots):
                    vx1 = int(left_bound + i * slot_w)
                    vx2 = int(left_bound + (i + 1) * slot_w)
                    voids.append(self._create_void_facing(row.row_index, row.row_level, vx1, y1, vx2, y2, void_counter))
                    void_counter += 1
                    
        # Check gaps between consecutive facings
        for i in range(len(sorted_facings) - 1):
            curr_x2 = sorted_facings[i].bbox.x2
            next_x1 = sorted_facings[i + 1].bbox.x1
            gap = next_x1 - curr_x2
            
            if gap >= self.min_void_width_px:
                num_slots = max(1, round(gap / 140.0))
                slot_w = gap / num_slots
                for s in range(num_slots):
                    vx1 = int(curr_x2 + s * slot_w) + 2
                    vx2 = int(curr_x2 + (s + 1) * slot_w) - 2
                    voids.append(self._create_void_facing(row.row_index, row.row_level, vx1, y1, vx2, y2, void_counter))
                    void_counter += 1
                    
        # Check gap between last facing and right frame
        if sorted_facings:
            last_x2 = sorted_facings[-1].bbox.x2
            if (right_bound - last_x2) >= self.min_void_width_px:
                void_w = right_bound - last_x2
                num_slots = max(1, round(void_w / 140.0))
                slot_w = void_w / num_slots
                for i in range(num_slots):
                    vx1 = int(last_x2 + i * slot_w)
                    vx2 = int(last_x2 + (i + 1) * slot_w)
                    voids.append(self._create_void_facing(row.row_index, row.row_level, vx1, y1, vx2, y2, void_counter))
                    void_counter += 1
                    
        # If no facings detected at all, row is completely empty (dark shelf!)
        if not sorted_facings:
            total_w = right_bound - left_bound
            num_slots = max(1, round(total_w / 140.0))
            slot_w = total_w / num_slots
            for i in range(num_slots):
                vx1 = int(left_bound + i * slot_w)
                vx2 = int(left_bound + (i + 1) * slot_w)
                voids.append(self._create_void_facing(row.row_index, row.row_level, vx1, y1, vx2, y2, void_counter))
                void_counter += 1
                
        return voids

    def _create_void_facing(self, row_idx: int, level: str, x1: int, y1: int, x2: int, y2: int, counter: int) -> Facing:
        return Facing(
            id=f"VOID_R{row_idx}_{counter}",
            sku_id=None,
            brand="Out of Stock",
            product_name="Empty Facing (OOS)",
            bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.95),
            row_index=row_idx,
            row_level=level,
            facing_type="void_oos",
            detected_price=None
        )
