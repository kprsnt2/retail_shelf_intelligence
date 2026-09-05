"""Store SKU Catalog and Inventory Manager."""
from typing import Dict, List, Optional
import json
from pathlib import Path

from retail_shelf.config import CATALOG_PATH
from retail_shelf.models.pos import SKUMetadata

class StoreCatalog:
    def __init__(self, catalog_path: Optional[Path] = None):
        self.catalog_path = catalog_path or CATALOG_PATH
        self.skus: Dict[str, SKUMetadata] = {}
        self._load()

    def _load(self):
        if not self.catalog_path.exists():
            return
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data.get("skus", []):
                sku = SKUMetadata(**item)
                self.skus[sku.sku_id] = sku

    def get(self, sku_id: str) -> Optional[SKUMetadata]:
        return self.skus.get(sku_id)

    def list_all(self) -> List[SKUMetadata]:
        return list(self.skus.values())

    def get_by_brand(self, brand: str) -> List[SKUMetadata]:
        return [s for s in self.skus.values() if s.brand.lower() == brand.lower()]
