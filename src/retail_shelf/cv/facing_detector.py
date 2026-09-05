"""Product Facing Detector Module.

Locates individual product facings within shelf row segments and matches
detected visual signatures against the store catalog.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
from retail_shelf.models.shelf import Facing, BoundingBox, RowSegment

class FacingDetector:
    def __init__(self, catalog: List[Dict[str, Any]]):
        self.catalog = catalog
        self.sku_map = {item["sku_id"]: item for item in catalog}
        # Precompute typical brand color signatures for quick on-prem classification
        self.signatures = {
            "SKU-BEV-001": {"brand": "Red Bull", "rgb": (20, 50, 130), "accent": (220, 30, 30)},
            "SKU-BEV-002": {"brand": "Monster", "rgb": (30, 30, 30), "accent": (50, 205, 50)},
            "SKU-BEV-003": {"brand": "Celsius", "rgb": (230, 130, 40), "accent": (255, 255, 255)},
            "SKU-BEV-004": {"brand": "Coca-Cola", "rgb": (210, 30, 30), "accent": (255, 255, 255)},
            "SKU-BEV-005": {"brand": "Diet Coke", "rgb": (160, 160, 160), "accent": (210, 30, 30)},
            "SKU-BEV-006": {"brand": "PepsiCo", "rgb": (10, 80, 170), "accent": (210, 30, 30)},
            "SKU-BEV-007": {"brand": "San Pellegrino", "rgb": (40, 110, 180), "accent": (255, 255, 255)},
            "SKU-BEV-008": {"brand": "StoreBrand", "rgb": (240, 230, 210), "accent": (50, 50, 50)}
        }

    def detect_facings_in_row(self, image: Image.Image, row: RowSegment) -> List[Facing]:
        """Detect product facings within a row segment."""
        img_np = np.array(image)
        height, width, _ = img_np.shape
        
        y1 = max(0, row.y_min)
        y2 = min(height, row.y_max - 24) # Exclude shelf front lip channel
        if y2 <= y1:
            return []
        
        # Crop row slice
        row_slice = img_np[y1:y2, 80:width - 80]
        slice_h, slice_w, _ = row_slice.shape
        if slice_w == 0 or slice_h == 0:
            return []
        
        # Compute vertical column profile
        # Contrast of row items vs background
        gray_slice = np.mean(row_slice, axis=2)
        # Background wall is light off-white (mean ~ 230-240)
        is_foreground = (gray_slice < 215)
        fg_cols = np.mean(is_foreground, axis=0) > 0.25
        
        # Group contiguous foreground column runs
        facings: List[Facing] = []
        in_facing = False
        start_x = 0
        min_facing_width = 45
        
        runs = []
        for x, active in enumerate(fg_cols):
            if active and not in_facing:
                in_facing = True
                start_x = x
            elif not active and in_facing:
                in_facing = False
                runs.append((start_x, x))
        if in_facing:
            runs.append((start_x, slice_w))
            
        counter = 0
        for rx1, rx2 in runs:
            w = rx2 - rx1
            if w < min_facing_width:
                continue
                
            # If width is a wide block of multiple contiguous cans of same color, subdivide
            num_subfacings = max(1, round(w / 140.0))
            sub_w = w / num_subfacings
            
            for sub_i in range(num_subfacings):
                fx1 = 80 + int(rx1 + sub_i * sub_w) + 4
                fx2 = 80 + int(rx1 + (sub_i + 1) * sub_w) - 4
                fy1 = y1 + int((y2 - y1) * 0.12)
                fy2 = y2
                
                # Extract patch for visual signature classification
                patch = img_np[fy1:fy2, fx1:fx2]
                sku_match, brand_match, conf = self._classify_patch(patch)
                
                sku_info = self.sku_map.get(sku_match, {})
                f_id = f"F_R{row.row_index}_{counter}"
                facings.append(Facing(
                    id=f_id,
                    sku_id=sku_match,
                    brand=brand_match or sku_info.get("brand", "Unknown"),
                    product_name=sku_info.get("name", "Retail Item"),
                    bbox=BoundingBox(x1=fx1, y1=fy1, x2=fx2, y2=fy2, confidence=conf),
                    row_index=row.row_index,
                    row_level=row.row_level,
                    facing_type="product",
                    detected_price=sku_info.get("pos_price")
                ))
                counter += 1
                
        return facings

    def _classify_patch(self, patch: np.ndarray) -> Tuple[Optional[str], Optional[str], float]:
        if patch.size == 0:
            return None, None, 0.0
        
        # Mean RGB of the product body
        mean_rgb = np.mean(patch, axis=(0, 1))[:3]
        
        best_sku = None
        best_dist = float("inf")
        
        for sku_id, sig in self.signatures.items():
            dist = np.linalg.norm(mean_rgb - np.array(sig["rgb"]))
            if dist < best_dist:
                best_dist = dist
                best_sku = sku_id
                
        if best_sku:
            # Map distance to confidence (dist < 40 is very high match)
            confidence = max(0.5, min(0.98, 1.0 - (best_dist / 180.0)))
            brand = self.signatures[best_sku]["brand"]
            return best_sku, brand, float(confidence)
            
        return None, None, 0.5
