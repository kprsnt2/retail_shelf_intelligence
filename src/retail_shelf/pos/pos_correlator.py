"""POS Analytics and Financial Correlation Engine.

Ties detected shelf facings to POS sales velocity, calculating Revenue at Risk,
Share of Shelf, Eye-Level Efficiency, and Price Tag Discrepancies.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime

from retail_shelf.config import config
from retail_shelf.models.shelf import ShelfDetectionResult
from retail_shelf.models.pos import SKUMetadata
from retail_shelf.models.analytics import (
    ShelfAnalyticsReport,
    RevenueAtRiskItem,
    ShareOfShelf,
    EyeLevelEfficiency,
    PriceTagAudit
)
from retail_shelf.pos.catalog import StoreCatalog
from retail_shelf.pos.planogram import PlanogramManager

class POSAnalyticsCorrelator:
    def __init__(self, catalog: Optional[StoreCatalog] = None, planogram_mgr: Optional[PlanogramManager] = None):
        self.catalog = catalog or StoreCatalog()
        self.planogram_mgr = planogram_mgr or PlanogramManager()

    def generate_report(self, detection: ShelfDetectionResult, planogram_id: str = "POG-BEV-COOLER-01") -> ShelfAnalyticsReport:
        """Correlate physical shelf scan with POS velocities and financial risks."""
        compliance_data = self.planogram_mgr.evaluate_compliance(detection, planogram_id)
        pog = self.planogram_mgr.get(planogram_id)
        
        # 1. On-Shelf Availability (OSA %)
        total_slots = detection.occupied_facings + detection.oos_voids
        osa_pct = (detection.occupied_facings / total_slots * 100.0) if total_slots > 0 else 100.0
        
        # 2. Share of Shelf (SoS)
        brand_counts: Dict[str, int] = {}
        for row in detection.rows:
            for f in row.facings:
                if f.facing_type == "product" and f.brand:
                    brand_counts[f.brand] = brand_counts.get(f.brand, 0) + 1
                    
        sos_list: List[ShareOfShelf] = []
        for brand, count in sorted(brand_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / detection.occupied_facings * 100.0) if detection.occupied_facings > 0 else 0.0
            sos_list.append(ShareOfShelf(
                brand=brand,
                facing_count=count,
                percentage=round(pct, 1),
                is_house_brand=(brand.lower() == "storebrand")
            ))
            
        # 3. Revenue at Risk Computation
        revenue_at_risk_items: List[RevenueAtRiskItem] = []
        total_daily_loss = 0.0
        
        # Map detected counts per row
        detected_per_row_sku: Dict[int, Dict[str, int]] = {}
        for row in detection.rows:
            detected_per_row_sku[row.row_index] = {}
            for f in row.facings:
                if f.facing_type == "product" and f.sku_id:
                    detected_per_row_sku[row.row_index][f.sku_id] = detected_per_row_sku[row.row_index].get(f.sku_id, 0) + 1
                    
        if pog:
            for pog_row in pog.rows:
                r_idx = pog_row.row_index
                row_detected = detected_per_row_sku.get(r_idx, {})
                
                for slot in pog_row.slots:
                    sku_info = self.catalog.get(slot.expected_sku_id)
                    if not sku_info:
                        continue
                        
                    actual_count = row_detected.get(slot.expected_sku_id, 0)
                    if actual_count < slot.expected_facings:
                        missing = slot.expected_facings - actual_count
                        # Fractional daily velocity lost
                        velocity_lost = (sku_info.avg_daily_velocity / slot.expected_facings) * missing
                        daily_loss = round(velocity_lost * sku_info.pos_price * config.default_stockout_penalty_multiplier, 2)
                        weekly_loss = round(daily_loss * 7, 2)
                        
                        risk_desc = "Stockout on Eye-Level Top Mover" if pog_row.row_level == "eye_level" else f"Empty Facing on {pog_row.row_level.title()} Shelf"
                        
                        revenue_at_risk_items.append(RevenueAtRiskItem(
                            sku_id=sku_info.sku_id,
                            product_name=sku_info.name,
                            brand=sku_info.brand,
                            row_index=r_idx,
                            row_level=pog_row.row_level,
                            missing_facings=missing,
                            unit_price=sku_info.pos_price,
                            unit_margin=round(sku_info.gross_margin, 2),
                            daily_velocity=sku_info.avg_daily_velocity,
                            daily_revenue_at_risk=daily_loss,
                            weekly_revenue_at_risk=weekly_loss,
                            risk_factor=risk_desc
                        ))
                        total_daily_loss += daily_loss
                        
        # Sort risk items by highest revenue loss first
        revenue_at_risk_items.sort(key=lambda item: item.daily_revenue_at_risk, reverse=True)
        total_weekly_loss = round(total_daily_loss * 7, 2)
        
        # 4. Eye-Level Efficiency Analysis
        eye_level_row = next((r for r in detection.rows if r.row_level == "eye_level"), None)
        eye_level_slots = eye_level_row.total_slots if eye_level_row else 8
        top_performers = 0
        underperformers = 0
        
        if eye_level_row:
            for f in eye_level_row.facings:
                if f.facing_type == "product" and f.sku_id:
                    sku = self.catalog.get(f.sku_id)
                    if sku and sku.avg_daily_velocity >= 30.0:
                        top_performers += 1
                    else:
                        underperformers += 1
                elif f.facing_type == "void_oos":
                    underperformers += 1 # Wasted eye-level space!
                    
        eff_score = (top_performers / eye_level_slots * 100.0) if eye_level_slots > 0 else 100.0
        opp_gap = round(underperformers * 22.50, 2) # estimated daily lift lost per wasted eye slot
        
        eye_metrics = EyeLevelEfficiency(
            eye_level_slots=eye_level_slots,
            top_performers_at_eye_level=top_performers,
            underperformers_at_eye_level=underperformers,
            efficiency_score_pct=round(eff_score, 1),
            opportunity_gap_daily=opp_gap
        )
        
        # 5. Price Tag Auditing
        price_tag_audits: List[PriceTagAudit] = []
        for row in detection.rows:
            for f in row.facings:
                if f.facing_type == "product" and f.sku_id and f.detected_price is not None:
                    sku = self.catalog.get(f.sku_id)
                    if sku:
                        diff = round(f.detected_price - sku.pos_price, 2)
                        status = "match"
                        if diff < -0.05:
                            status = "underpriced_on_shelf"
                        elif diff > 0.05:
                            status = "overpriced_on_shelf"
                            
                        price_tag_audits.append(PriceTagAudit(
                            sku_id=sku.sku_id,
                            product_name=sku.name,
                            row_index=row.row_index,
                            pos_price=sku.pos_price,
                            detected_shelf_price=f.detected_price,
                            discrepancy=diff,
                            status=status
                        ))
                        
        return ShelfAnalyticsReport(
            shelf_id=detection.image_id,
            scan_timestamp=datetime.utcnow().isoformat(),
            on_shelf_availability_pct=round(osa_pct, 1),
            planogram_compliance_pct=compliance_data["compliance_score_pct"],
            total_facings_detected=total_slots,
            total_occupied_facings=detection.occupied_facings,
            total_oos_voids=detection.oos_voids,
            total_daily_revenue_at_risk=round(total_daily_loss, 2),
            total_weekly_revenue_at_risk=total_weekly_loss,
            share_of_shelf=sos_list,
            eye_level_efficiency=eye_metrics,
            revenue_at_risk_items=revenue_at_risk_items,
            price_tag_audits=price_tag_audits
        )
