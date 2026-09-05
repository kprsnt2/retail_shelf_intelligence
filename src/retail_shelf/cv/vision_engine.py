"""Unified Vision Engine for On-Prem Shelf Analysis.

Coordinates row segmentation, facing recognition, OOS void detection,
and price tag reading into a single high-throughput on-prem execution pipeline.
"""
from typing import List, Dict, Any, Optional
import time
from pathlib import Path
from PIL import Image

from retail_shelf.config import config, CATALOG_PATH
from retail_shelf.models.shelf import ShelfDetectionResult, RowSegment, Facing
from retail_shelf.cv.row_segmenter import ShelfRowSegmenter
from retail_shelf.cv.facing_detector import FacingDetector
from retail_shelf.cv.oos_detector import OOSVoidDetector
from retail_shelf.cv.tag_reader import ShelfTagReader
import json

class UnifiedVisionEngine:
    def __init__(self, catalog: Optional[List[Dict[str, Any]]] = None):
        if catalog is None:
            if CATALOG_PATH.exists():
                with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                    catalog = json.load(f)["skus"]
            else:
                catalog = []
                
        self.catalog = catalog
        self.segmenter = ShelfRowSegmenter(expected_rows=config.row_segmentation_bands)
        self.facing_detector = FacingDetector(catalog=self.catalog)
        self.void_detector = OOSVoidDetector(min_void_width_px=config.min_oos_void_width_px)
        self.tag_reader = ShelfTagReader()

    def analyze_shelf_image(self, image: Image.Image, image_id: str = "shelf_scan_01") -> ShelfDetectionResult:
        """Run complete on-prem computer vision pipeline on a shelf image."""
        start_time = time.perf_counter()
        
        orig_width, orig_height = image.size
        CANONICAL_W, CANONICAL_H = 1200, 900
        needs_scaling = (orig_width != CANONICAL_W or orig_height != CANONICAL_H)
        
        proc_image = image.resize((CANONICAL_W, CANONICAL_H), Image.Resampling.BILINEAR) if needs_scaling else image
        
        # 1. Segment shelf into horizontal rows
        rows: List[RowSegment] = self.segmenter.segment_rows(proc_image)
        
        total_occupied = 0
        total_voids = 0
        
        # 2. Process each row for facings, voids, and price tags
        for row in rows:
            # Detect product facings
            facings = self.facing_detector.detect_facings_in_row(proc_image, row)
            # Detect price tags
            self.tag_reader.detect_tags_in_row(proc_image, row, facings)
            # Detect empty slots (OOS voids)
            voids = self.void_detector.detect_voids(proc_image, row, facings)
            
            all_slots = facings + voids
            # Sort all slots left-to-right
            all_slots.sort(key=lambda item: item.bbox.x1)
            
            row.facings = all_slots
            row.occupied_slots = len(facings)
            row.oos_slots = len(voids)
            row.total_slots = len(all_slots)
            
            total_occupied += len(facings)
            total_voids += len(voids)
            
        # 3. Project coordinates back to original image scale if normalized
        if needs_scaling:
            scale_x = orig_width / float(CANONICAL_W)
            scale_y = orig_height / float(CANONICAL_H)
            for row in rows:
                row.y_min = int(row.y_min * scale_y)
                row.y_max = int(row.y_max * scale_y)
                for item in row.facings:
                    item.bbox.x1 = int(item.bbox.x1 * scale_x)
                    item.bbox.y1 = int(item.bbox.y1 * scale_y)
                    item.bbox.x2 = int(item.bbox.x2 * scale_x)
                    item.bbox.y2 = int(item.bbox.y2 * scale_y)
                    if item.tag_bbox:
                        item.tag_bbox.x1 = int(item.tag_bbox.x1 * scale_x)
                        item.tag_bbox.y1 = int(item.tag_bbox.y1 * scale_y)
                        item.tag_bbox.x2 = int(item.tag_bbox.x2 * scale_x)
                        item.tag_bbox.y2 = int(item.tag_bbox.y2 * scale_y)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        return ShelfDetectionResult(
            image_id=image_id,
            image_width=orig_width,
            image_height=orig_height,
            rows=rows,
            total_facings=total_occupied + total_voids,
            occupied_facings=total_occupied,
            oos_voids=total_voids,
            execution_time_ms=round(elapsed_ms, 2),
            inference_engine="on_prem_cv_v1"
        )
