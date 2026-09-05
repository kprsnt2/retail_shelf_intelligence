"""Pydantic schemas for Retail Analytics and Financial Scoring."""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class RevenueAtRiskItem(BaseModel):
    sku_id: str
    product_name: str
    brand: str
    row_index: int
    row_level: str
    missing_facings: int
    unit_price: float
    unit_margin: float
    daily_velocity: float
    daily_revenue_at_risk: float
    weekly_revenue_at_risk: float
    risk_factor: str = Field(..., description="e.g. Empty Slot on Top-Seller, Misplaced SKU")

class ShareOfShelf(BaseModel):
    brand: str
    facing_count: int
    percentage: float
    is_house_brand: bool = False

class EyeLevelEfficiency(BaseModel):
    eye_level_slots: int
    top_performers_at_eye_level: int
    underperformers_at_eye_level: int
    efficiency_score_pct: float
    opportunity_gap_daily: float

class PriceTagAudit(BaseModel):
    sku_id: str
    product_name: str
    row_index: int
    pos_price: float
    detected_shelf_price: float
    discrepancy: float
    status: Literal["match", "underpriced_on_shelf", "overpriced_on_shelf"]

class ShelfAnalyticsReport(BaseModel):
    shelf_id: str
    scan_timestamp: str
    on_shelf_availability_pct: float
    planogram_compliance_pct: float
    total_facings_detected: int
    total_occupied_facings: int
    total_oos_voids: int
    total_daily_revenue_at_risk: float
    total_weekly_revenue_at_risk: float
    share_of_shelf: List[ShareOfShelf] = Field(default_factory=list)
    eye_level_efficiency: EyeLevelEfficiency
    revenue_at_risk_items: List[RevenueAtRiskItem] = Field(default_factory=list)
    price_tag_audits: List[PriceTagAudit] = Field(default_factory=list)
