"""FastAPI Endpoints for Store POS and Planograms."""
from fastapi import APIRouter
from typing import List, Dict, Any

from retail_shelf.pos.catalog import StoreCatalog
from retail_shelf.pos.planogram import PlanogramManager

router = APIRouter(prefix="/api/pos", tags=["Store POS & Catalog"])

catalog = StoreCatalog()
planogram_mgr = PlanogramManager()

@router.get("/catalog")
def get_catalog() -> List[Dict[str, Any]]:
    return [sku.model_dump() for sku in catalog.list_all()]

@router.get("/planograms")
def get_planograms() -> List[Dict[str, Any]]:
    return [pog.model_dump() for pog in planogram_mgr.planograms.values()]
