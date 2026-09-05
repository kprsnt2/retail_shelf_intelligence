"""Store Operations Prioritization Agent.

Generates "The Store, Prioritized" morning worklist ranked by revenue at risk,
with step-by-step instructions for floor staff and managers.
"""
from typing import List, Optional
from datetime import datetime

from retail_shelf.models.shelf import ShelfDetectionResult
from retail_shelf.models.analytics import ShelfAnalyticsReport
from retail_shelf.models.agent import StoreOpsWorklist, StoreOpsActionItem
from retail_shelf.agent.rebalancing import ShelfRebalancer
from retail_shelf.pos.catalog import StoreCatalog

class StoreOpsAgent:
    def __init__(self, catalog: Optional[StoreCatalog] = None):
        self.catalog = catalog or StoreCatalog()
        self.rebalancer = ShelfRebalancer(catalog=self.catalog)

    def generate_worklist(self, detection: ShelfDetectionResult, analytics: ShelfAnalyticsReport) -> StoreOpsWorklist:
        """Synthesize shelf analytics into prioritized, revenue-ranked store ops worklist."""
        actions: List[StoreOpsActionItem] = []
        action_counter = 1
        
        # 1. P0 Critical Stockout Restock Actions (Ranked by highest daily loss)
        for risk_item in analytics.revenue_at_risk_items:
            sku = self.catalog.get(risk_item.sku_id)
            backroom_stock = sku.stock_on_hand if sku else 0
            
            # Prioritize P0 if eye level or loss > $100/day
            priority = "P0_CRITICAL" if (risk_item.row_level == "eye_level" or risk_item.daily_revenue_at_risk >= 100.0) else "P1_HIGH"
            
            stock_msg = f"Backroom stock: {backroom_stock} units available." if backroom_stock > 0 else "BACKROOM EXHAUSTED: Expedited DC replenishment required."
            
            actions.append(StoreOpsActionItem(
                action_id=f"ACT-OOS-{action_counter:03d}",
                title=f"Restock {risk_item.brand} ({risk_item.missing_facings} empty facings)",
                action_type="urgent_restock",
                priority=priority,
                location=f"Row {risk_item.row_index} ({risk_item.row_level.title()})",
                sku_id=risk_item.sku_id,
                product_name=risk_item.product_name,
                revenue_impact_daily=risk_item.daily_revenue_at_risk,
                revenue_impact_weekly=risk_item.weekly_revenue_at_risk,
                instructions=(
                    f"Immediate facing replenish required. Missing {risk_item.missing_facings} facings on "
                    f"Row {risk_item.row_index} ({risk_item.row_level.title()}). {stock_msg} "
                    f"Recovering this facing prevents ~${risk_item.daily_revenue_at_risk}/day in lost floor sales."
                ),
                estimated_resolution_time_min=5
            ))
            action_counter += 1
            
        # 2. P1 Price Tag Discrepancies
        for audit in analytics.price_tag_audits:
            if audit.status != "match":
                actions.append(StoreOpsActionItem(
                    action_id=f"ACT-TAG-{action_counter:03d}",
                    title=f"Fix Price Tag: {audit.product_name} (Shelf: ${audit.detected_shelf_price:.2f} vs POS: ${audit.pos_price:.2f})",
                    action_type="price_correction",
                    priority="P1_HIGH",
                    location=f"Row {audit.row_index} Shelf Edge Channel",
                    sku_id=audit.sku_id,
                    product_name=audit.product_name,
                    revenue_impact_daily=round(abs(audit.discrepancy) * 15.0, 2), # estimated margin leakage
                    revenue_impact_weekly=round(abs(audit.discrepancy) * 15.0 * 7, 2),
                    instructions=(
                        f"Price mismatch detected on physical shelf label! Shelf lip tag displays "
                        f"${audit.detected_shelf_price:.2f} but POS database is configured to ${audit.pos_price:.2f}. "
                        f"Print and slide new electronic/paper shelf label immediately to prevent margin leakage."
                    ),
                    estimated_resolution_time_min=3
                ))
                action_counter += 1
                
        # 3. P2 Shelf Rebalancing Recommendations
        swap_actions = self.rebalancer.identify_rebalancing_opportunities(detection)
        actions.extend(swap_actions)
        
        # Sort actions strictly by revenue impact descending
        actions.sort(key=lambda a: (0 if a.priority == "P0_CRITICAL" else 1, -a.revenue_impact_daily))
        
        total_recovery = sum(a.revenue_impact_daily for a in actions)
        p0_count = sum(1 for a in actions if a.priority == "P0_CRITICAL")
        
        # Build executive brief
        brief = (
            f"Morning Store Audit ({datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}): "
            f"Shelf '{analytics.shelf_id}' has {len(actions)} operational action items with "
            f"${total_recovery:,.2f}/day (${total_recovery * 7:,.2f}/wk) in recoverable revenue. "
            f"{p0_count} critical stockout(s) on top-selling lines (notably {analytics.revenue_at_risk_items[0].product_name if analytics.revenue_at_risk_items else 'high-velocity items'}) "
            f"require immediate floor replenishment."
        )
        
        return StoreOpsWorklist(
            shelf_id=analytics.shelf_id,
            generated_at=datetime.utcnow().isoformat(),
            total_actions=len(actions),
            p0_critical_count=p0_count,
            total_actionable_daily_recovery=round(total_recovery, 2),
            total_actionable_weekly_recovery=round(total_recovery * 7, 2),
            executive_brief=brief,
            actions=actions
        )
