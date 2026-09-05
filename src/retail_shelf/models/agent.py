"""Pydantic schemas for Agentic Store Ops Worklist and Actions."""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

ActionType = Literal["urgent_restock", "planogram_swap", "price_correction", "dark_shelf_alert"]
PriorityLevel = Literal["P0_CRITICAL", "P1_HIGH", "P2_MEDIUM", "P3_LOW"]

class StoreOpsActionItem(BaseModel):
    action_id: str
    title: str
    action_type: ActionType
    priority: PriorityLevel
    location: str
    sku_id: Optional[str] = None
    product_name: Optional[str] = None
    revenue_impact_daily: float = 0.0
    revenue_impact_weekly: float = 0.0
    instructions: str
    suggested_sku_swap: Optional[dict] = None
    estimated_resolution_time_min: int = 5

class StoreOpsWorklist(BaseModel):
    shelf_id: str
    generated_at: str
    total_actions: int
    p0_critical_count: int
    total_actionable_daily_recovery: float
    total_actionable_weekly_recovery: float
    executive_brief: str
    actions: List[StoreOpsActionItem] = Field(default_factory=list)
