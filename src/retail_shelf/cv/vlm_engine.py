"""Multimodal Vision-Language Model (VLM) Engine for Real-World Retail Shelves.

Uses Vision-Language Models (GPT-4o-mini or Gemini 2.0/1.5 Flash) to detect
actual physical shelf rows, individual product facings, brand names, and
out-of-stock (OOS) voids on arbitrary, complex, wide-angle retail photos.
"""
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import base64
import io
import time
from PIL import Image
import httpx

from retail_shelf.models.shelf import (
    ShelfDetectionResult,
    RowSegment,
    Facing,
    BoundingBox,
    RowLevel
)

VLM_PROMPT = """You are an expert on-prem retail shelf intelligence computer vision system.
Analyze this supermarket shelf / beverage cooler photograph in detail.

Tasks:
1. Locate the actual physical shelf tiers from top to bottom (row 0 = top, row 1 = eye level, row 2 = reach, row 3 = bottom).
   Only consider the actual shelving / display rack area; ignore ceilings, overhead lights, floors, or distant background.
2. For each detected product facing (bottle, can, or packaged item):
   - Provide a bounding box [ymin, xmin, ymax, xmax] normalized on a 0-1000 integer scale.
   - Identify the brand (e.g. "Monster", "Red Bull", "Coca-Cola", "PepsiCo", "Celsius", "Diet Coke", "San Pellegrino", or the visible brand).
   - Provide a concise product title.
   - Specify which row index (0-3) it belongs to.
3. For each out-of-stock (OOS) empty void / gap on the shelf where a product should be:
   - Provide a bounding box [ymin, xmin, ymax, xmax] normalized on a 0-1000 integer scale.
   - Specify which row index (0-3) it belongs to.

Return strictly valid JSON with this schema (no markdown fences, no explanatory text):
{
  "rows": [
    {"row_index": 0, "row_level": "top", "ymin": 300, "ymax": 480},
    {"row_index": 1, "row_level": "eye_level", "ymin": 480, "ymax": 640},
    {"row_index": 2, "row_level": "reach", "ymin": 640, "ymax": 800},
    {"row_index": 3, "row_level": "bottom", "ymin": 800, "ymax": 960}
  ],
  "facings": [
    {"row_index": 0, "bbox": [ymin, xmin, ymax, xmax], "brand": "Monster", "product_name": "Monster Energy 500ml", "confidence": 0.95}
  ],
  "voids": [
    {"row_index": 1, "bbox": [ymin, xmin, ymax, xmax]}
  ]
}
"""

class MultimodalVLMEngine:
    def __init__(
        self,
        provider: str = "openai", # "openai" or "gemini"
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY" if self.provider == "openai" else "GEMINI_API_KEY", "")
        if not model_name:
            self.model_name = "gpt-4o-mini" if self.provider == "openai" else "gemini-2.0-flash"
        else:
            self.model_name = model_name

    def analyze_shelf_image(
        self,
        image: Image.Image,
        image_id: str = "vlm_shelf_scan"
    ) -> ShelfDetectionResult:
        """Call VLM to locate real shelf rows, product facings, and voids."""
        start_time = time.perf_counter()
        width, height = image.size

        # Resize image for fast transmission while preserving detail
        max_dim = 1280
        scale = min(1.0, max_dim / max(width, height))
        if scale < 1.0:
            tx_w, tx_h = int(width * scale), int(height * scale)
            tx_img = image.resize((tx_w, tx_h), Image.Resampling.BILINEAR)
        else:
            tx_img = image

        # Encode to JPEG
        buf = io.BytesIO()
        tx_img.save(buf, format="JPEG", quality=85)
        b64_image = base64.b64encode(buf.getvalue()).decode("utf-8")

        if self.provider == "openai":
            raw_data = self._call_openai(b64_image)
        elif self.provider == "gemini":
            raw_data = self._call_gemini(b64_image)
        else:
            raise ValueError(f"Unsupported VLM provider: {self.provider}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return self._convert_to_shelf_result(raw_data, width, height, image_id, elapsed_ms)

    def _call_openai(self, b64_image: str) -> Dict[str, Any]:
        """Send image to OpenAI gpt-4o-mini endpoint."""
        if not self.api_key:
            raise ValueError("OpenAI API key is missing. Please provide an API key in the UI or set OPENAI_API_KEY.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VLM_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.1
        }

        with httpx.Client(timeout=45.0) as client:
            resp = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI API Error ({resp.status_code}): {resp.text}")
            result_json = resp.json()
            content = result_json["choices"][0]["message"]["content"]
            return json.loads(content)

    def _call_gemini(self, b64_image: str) -> Dict[str, Any]:
        """Send image to Google Gemini Flash endpoint."""
        if not self.api_key:
            raise ValueError("Gemini API key is missing. Please provide a key in the UI or set GEMINI_API_KEY.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": VLM_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_image
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1,
                "max_output_tokens": 4096
            }
        }

        with httpx.Client(timeout=45.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Gemini API Error ({resp.status_code}): {resp.text}")
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError(f"No candidates returned by Gemini: {data}")
            text_content = candidates[0]["content"]["parts"][0]["text"]
            return json.loads(text_content)

    def _convert_to_shelf_result(
        self,
        raw_data: Dict[str, Any],
        width: int,
        height: int,
        image_id: str,
        elapsed_ms: float
    ) -> ShelfDetectionResult:
        """Convert VLM JSON detection into ShelfDetectionResult."""
        rows_data = raw_data.get("rows", [])
        facings_data = raw_data.get("facings", [])
        voids_data = raw_data.get("voids", [])

        # Default tiers if missing
        tier_names: List[RowLevel] = ["top", "eye_level", "reach", "bottom"]
        row_segments: List[RowSegment] = []

        if not rows_data:
            # Create default 4 rows
            for i in range(4):
                y_min = int(height * (i / 4.0))
                y_max = int(height * ((i + 1) / 4.0))
                row_segments.append(RowSegment(
                    row_index=i,
                    row_level=tier_names[i],
                    y_min=y_min,
                    y_max=y_max,
                    facings=[]
                ))
        else:
            for r in rows_data:
                r_idx = int(r.get("row_index", len(row_segments)))
                r_tier = r.get("row_level", tier_names[min(r_idx, 3)])
                ymin_px = int((r.get("ymin", 0) / 1000.0) * height)
                ymax_px = int((r.get("ymax", 1000) / 1000.0) * height)
                row_segments.append(RowSegment(
                    row_index=r_idx,
                    row_level=r_tier,
                    y_min=ymin_px,
                    y_max=ymax_px,
                    facings=[]
                ))

        # Sort rows by vertical position
        row_segments.sort(key=lambda r: r.y_min)

        # Map facings into rows
        total_occupied = 0
        for idx, f in enumerate(facings_data):
            bbox_norm = f.get("bbox", [0, 0, 100, 100])
            ymin, xmin, ymax, xmax = bbox_norm
            x1 = max(0, int((xmin / 1000.0) * width))
            y1 = max(0, int((ymin / 1000.0) * height))
            x2 = min(width, int((xmax / 1000.0) * width))
            y2 = min(height, int((ymax / 1000.0) * height))
            
            mid_y = (y1 + y2) / 2.0
            r_target = f.get("row_index")
            if r_target is None or r_target >= len(row_segments):
                # Find matching row by mid_y
                r_target = 0
                for r_i, r_obj in enumerate(row_segments):
                    if r_obj.y_min <= mid_y <= r_obj.y_max:
                        r_target = r_i
                        break

            conf = float(f.get("confidence", 0.90))
            brand = f.get("brand", "Unknown")
            product_name = f.get("product_name", f"{brand} Beverage")

            facing_obj = Facing(
                id=f"VLM_F_{r_target}_{idx:03d}",
                sku_id=f"SKU-{brand.upper()[:3]}-001",
                brand=brand,
                product_name=product_name,
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf),
                row_index=r_target,
                row_level=row_segments[r_target].row_level,
                facing_type="product"
            )
            row_segments[r_target].facings.append(facing_obj)
            total_occupied += 1

        # Map voids into rows
        total_voids = 0
        for idx, v in enumerate(voids_data):
            bbox_norm = v.get("bbox", [0, 0, 100, 100])
            ymin, xmin, ymax, xmax = bbox_norm
            x1 = max(0, int((xmin / 1000.0) * width))
            y1 = max(0, int((ymin / 1000.0) * height))
            x2 = min(width, int((xmax / 1000.0) * width))
            y2 = min(height, int((ymax / 1000.0) * height))

            mid_y = (y1 + y2) / 2.0
            r_target = v.get("row_index")
            if r_target is None or r_target >= len(row_segments):
                r_target = 0
                for r_i, r_obj in enumerate(row_segments):
                    if r_obj.y_min <= mid_y <= r_obj.y_max:
                        r_target = r_i
                        break

            void_obj = Facing(
                id=f"VLM_VOID_{r_target}_{idx:03d}",
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.92),
                row_index=r_target,
                row_level=row_segments[r_target].row_level,
                facing_type="void_oos"
            )
            row_segments[r_target].facings.append(void_obj)
            total_voids += 1

        # Sort slots left-to-right in each row
        for row in row_segments:
            row.facings.sort(key=lambda item: item.bbox.x1)
            row.occupied_slots = sum(1 for f in row.facings if f.facing_type == "product")
            row.oos_slots = sum(1 for f in row.facings if f.facing_type == "void_oos")
            row.total_slots = len(row.facings)

        engine_tag = f"cloud_vlm_{self.provider}_{self.model_name}"
        return ShelfDetectionResult(
            image_id=image_id,
            image_width=width,
            image_height=height,
            rows=row_segments,
            total_facings=total_occupied + total_voids,
            occupied_facings=total_occupied,
            oos_voids=total_voids,
            execution_time_ms=round(elapsed_ms, 2),
            inference_engine=engine_tag
        )
