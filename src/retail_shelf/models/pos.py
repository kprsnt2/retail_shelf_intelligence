"""Pydantic schemas for Point of Sale (POS) and Planogram Data."""
from typing import List, Optional
from pydantic import BaseModel, Field
from retail_shelf.models.shelf import RowLevel

class SKUMetadata(BaseModel):
    sku_id: str
    name: str
    brand: str
    category: str
    barcode: str
    pos_price: float = Field(..., description="Current POS system retail price")
    unit_cost: float = Field(..., description="Wholesale cost per unit")
    avg_daily_velocity: float = Field(..., description="Average units sold per day on this shelf")
    stock_on_hand: int = Field(..., description="Current backroom + shelf inventory count")
    target_facings: int = Field(1, description="Planogram allocated facings")

    @property
    def gross_margin(self) -> float:
        return max(0.0, self.pos_price - self.unit_cost)

    @property
    def margin_percentage(self) -> float:
        return (self.gross_margin / self.pos_price) * 100 if self.pos_price > 0 else 0.0

class PlanogramSlot(BaseModel):
    slot_index: int
    expected_sku_id: str
    expected_facings: int = 1

class PlanogramRow(BaseModel):
    row_index: int
    row_level: RowLevel
    slots: List[PlanogramSlot] = Field(default_factory=list)

class PlanogramTarget(BaseModel):
    planogram_id: str
    shelf_name: str
    category: str
    rows: List[PlanogramRow] = Field(default_factory=list)
