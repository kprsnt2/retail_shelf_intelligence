"""Planogram Target and Compliance Manager."""
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

from retail_shelf.config import PLANOGRAMS_PATH
from retail_shelf.models.pos import PlanogramTarget, PlanogramRow, PlanogramSlot
from retail_shelf.models.shelf import ShelfDetectionResult

class PlanogramManager:
    def __init__(self, planograms_path: Optional[Path] = None):
        self.planograms_path = planograms_path or PLANOGRAMS_PATH
        self.planograms: Dict[str, PlanogramTarget] = {}
        self._load()

    def _load(self):
        if not self.planograms_path.exists():
            return
        with open(self.planograms_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data.get("planograms", []):
                pog = PlanogramTarget(**item)
                self.planograms[pog.planogram_id] = pog

    def get(self, planogram_id: str) -> Optional[PlanogramTarget]:
        return self.planograms.get(planogram_id)

    def evaluate_compliance(self, detection: ShelfDetectionResult, planogram_id: str) -> Dict[str, Any]:
        """Compute planogram compliance score and slot matching."""
        pog = self.get(planogram_id)
        if not pog:
            return {
                "compliance_score_pct": 100.0,
                "total_target_facings": detection.total_facings,
                "compliant_facings": detection.occupied_facings,
                "missing_facings": 0,
                "row_compliance": []
            }
            
        total_target_facings = 0
        matched_facings = 0
        missing_facings = 0
        row_compliance = []
        
        # Build count of target facings per SKU per row
        for pog_row in pog.rows:
            r_idx = pog_row.row_index
            target_sku_counts: Dict[str, int] = {}
            for slot in pog_row.slots:
                target_sku_counts[slot.expected_sku_id] = target_sku_counts.get(slot.expected_sku_id, 0) + slot.expected_facings
                total_target_facings += slot.expected_facings
                
            # Find detected facings in this row
            detected_row = next((r for r in detection.rows if r.row_index == r_idx), None)
            detected_sku_counts: Dict[str, int] = {}
            if detected_row:
                for f in detected_row.facings:
                    if f.facing_type == "product" and f.sku_id:
                        detected_sku_counts[f.sku_id] = detected_sku_counts.get(f.sku_id, 0) + 1
                        
            row_matched = 0
            row_target = sum(target_sku_counts.values())
            for sku_id, target_count in target_sku_counts.items():
                actual_count = detected_sku_counts.get(sku_id, 0)
                row_matched += min(target_count, actual_count)
                if actual_count < target_count:
                    missing_facings += (target_count - actual_count)
                    
            matched_facings += row_matched
            row_pct = (row_matched / row_target * 100.0) if row_target > 0 else 100.0
            row_compliance.append({
                "row_index": r_idx,
                "row_level": pog_row.row_level,
                "target_facings": row_target,
                "matched_facings": row_matched,
                "compliance_pct": round(row_pct, 1)
            })
            
        compliance_pct = (matched_facings / total_target_facings * 100.0) if total_target_facings > 0 else 100.0
        
        return {
            "compliance_score_pct": round(compliance_pct, 1),
            "total_target_facings": total_target_facings,
            "compliant_facings": matched_facings,
            "missing_facings": missing_facings,
            "row_compliance": row_compliance
        }
