"""Pydantic schemas for shelf computer vision detections."""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime

RowLevel = Literal["top", "eye_level", "reach", "bottom"]
FacingType = Literal["product", "void_oos", "tag"]

class BoundingBox(BaseModel):
    x1: int = Field(..., description="Left coordinate in pixels")
    y1: int = Field(..., description="Top coordinate in pixels")
    x2: int = Field(..., description="Right coordinate in pixels")
    y2: int = Field(..., description="Bottom coordinate in pixels")
    confidence: float = Field(1.0, ge=0.0, le=1.0)

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

class Facing(BaseModel):
    id: str = Field(..., description="Facing unique identifier (e.g. F_R1_03)")
    sku_id: Optional[str] = Field(None, description="Matched SKU identifier from catalog")
    brand: Optional[str] = Field(None, description="Detected brand name")
    product_name: Optional[str] = Field(None, description="Readable product title")
    bbox: BoundingBox
    row_index: int = Field(..., description="Row index from top (0 to N-1)")
    row_level: RowLevel = Field(..., description="Physical shelf tier")
    facing_type: FacingType = Field("product", description="Product, empty void, or tag")
    detected_price: Optional[float] = Field(None, description="Price detected via OCR on shelf tag")
    tag_bbox: Optional[BoundingBox] = Field(None, description="Price tag bounding box if detected")

class RowSegment(BaseModel):
    row_index: int
    row_level: RowLevel
    y_min: int
    y_max: int
    total_slots: int = 0
    occupied_slots: int = 0
    oos_slots: int = 0
    facings: List[Facing] = Field(default_factory=list)

class ShelfDetectionResult(BaseModel):
    image_id: str
    image_width: int
    image_height: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    rows: List[RowSegment] = Field(default_factory=list)
    total_facings: int = 0
    occupied_facings: int = 0
    oos_voids: int = 0
    execution_time_ms: float = 0.0
    inference_engine: str = "on_prem_cv_v1"
