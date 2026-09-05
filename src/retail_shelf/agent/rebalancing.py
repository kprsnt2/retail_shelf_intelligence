"""Shelf Rebalancing Engine.

Identifies misallocated shelf real estate, eye-level waste,
and high-yield SKU relocation opportunities.
"""
from typing import List, Optional, Dict, Any

from retail_shelf.models.shelf import ShelfDetectionResult
from retail_shelf.models.pos import SKUMetadata
from retail_shelf.models.agent import StoreOpsActionItem
from retail_shelf.pos.catalog import StoreCatalog

class ShelfRebalancer:
    def __init__(self, catalog: Optional[StoreCatalog] = None):
        self.catalog = catalog or StoreCatalog()

    def identify_rebalancing_opportunities(self, detection: ShelfDetectionResult) -> List[StoreOpsActionItem]:
        """Detect misallocated eye-level slots and generate swap recommendations."""
        actions: List[StoreOpsActionItem] = []
        
        eye_row = next((r for r in detection.rows if r.row_level == "eye_level"), None)
        bottom_row = next((r for r in detection.rows if r.row_level == "bottom"), None)
        
        if not eye_row or not bottom_row:
            return actions
            
        # Find underperformers at eye level (velocity < 20)
        underperformers = []
        for f in eye_row.facings:
            if f.facing_type == "product" and f.sku_id:
                sku = self.catalog.get(f.sku_id)
                if sku and sku.avg_daily_velocity < 20.0:
                    underperformers.append((f, sku))
                    
        # Find high velocity items trapped on bottom shelf (velocity > 35)
        trapped_stars = []
        for f in bottom_row.facings:
            if f.facing_type == "product" and f.sku_id:
                sku = self.catalog.get(f.sku_id)
                if sku and sku.avg_daily_velocity > 35.0:
                    trapped_stars.append((f, sku))
                    
        # If we have matches, recommend swaps
        for i, ((eye_f, eye_sku), (bot_f, bot_sku)) in enumerate(zip(underperformers, trapped_stars)):
            # Lift estimate: ~30% velocity increase on bottom star when moved to eye-level
            est_daily_lift = round(bot_sku.avg_daily_velocity * 0.30 * bot_sku.gross_margin, 2)
            actions.append(StoreOpsActionItem(
                action_id=f"ACT-SWAP-{i+1}",
                title=f"Rebalance Facing: Promote {bot_sku.brand} to Eye-Level",
                action_type="planogram_swap",
                priority="P2_MEDIUM",
                location=f"Swap Row {eye_row.row_index} Slot ({eye_sku.brand}) with Row {bottom_row.row_index} ({bot_sku.brand})",
                sku_id=bot_sku.sku_id,
                product_name=bot_sku.name,
                revenue_impact_daily=est_daily_lift,
                revenue_impact_weekly=round(est_daily_lift * 7, 2),
                instructions=(
                    f"Re-allocate eye-level real estate: move high-velocity '{bot_sku.name}' "
                    f"(velocity: {bot_sku.avg_daily_velocity}/day) up from bottom shelf. Demote "
                    f"'{eye_sku.name}' (velocity: {eye_sku.avg_daily_velocity}/day) to lower tier."
                ),
                suggested_sku_swap={
                    "promote_sku": bot_sku.sku_id,
                    "demote_sku": eye_sku.sku_id,
                    "estimated_daily_lift": est_daily_lift
                },
                estimated_resolution_time_min=8
            ))
            
        return actions
