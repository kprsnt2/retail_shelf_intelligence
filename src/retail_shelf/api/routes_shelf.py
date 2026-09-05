"""FastAPI Endpoints for Shelf Analysis and Sample Scenes."""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from typing import Optional, Dict, Any, List
from pathlib import Path
from PIL import Image
import io
import json

from retail_shelf.config import SAMPLE_DIR, DATA_DIR
from retail_shelf.cv.vision_engine import UnifiedVisionEngine
from retail_shelf.pos.pos_correlator import POSAnalyticsCorrelator
from retail_shelf.agent.ops_agent import StoreOpsAgent

router = APIRouter(prefix="/api/shelf", tags=["Shelf Intelligence"])

engine = UnifiedVisionEngine()
correlator = POSAnalyticsCorrelator()
agent = StoreOpsAgent()

@router.get("/samples")
def list_sample_scenes() -> List[Dict[str, Any]]:
    """Return pre-packaged retail shelf scenes for walkthrough demonstrations."""
    gt_file = SAMPLE_DIR / "ground_truth.json"
    if not gt_file.exists():
        return []
    with open(gt_file, "r", encoding="utf-8") as f:
        scenes = json.load(f)["scenes"]
    return [
        {
            "shelf_id": s["shelf_id"],
            "title": s["title"],
            "filename": s["filename"],
            "image_url": f"/api/shelf/samples/{s['filename']}",
            "planogram_id": s["planogram_id"]
        }
        for s in scenes
    ]

@router.get("/samples/{filename}")
def get_sample_image(filename: str):
    """Serve sample shelf image."""
    img_path = SAMPLE_DIR / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(img_path, media_type="image/png")

@router.post("/analyze")
async def analyze_shelf(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    planogram_id: Optional[str] = Form("POG-BEV-COOLER-01")
) -> Dict[str, Any]:
    """Run full on-prem retail intelligence pipeline on an uploaded or sample shelf photo."""
    image: Optional[Image.Image] = None
    image_name = "custom_upload"

    if file:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
        image_name = file.filename or "uploaded_shelf.png"
    elif sample_id:
        img_path = SAMPLE_DIR / sample_id
        if not img_path.exists():
            # Try appending .png
            img_path = SAMPLE_DIR / f"{sample_id}.png"
        if not img_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample scene '{sample_id}' not found")
        image = Image.open(img_path).convert("RGB")
        image_name = sample_id
    else:
        # Default fallback to first benchmark scene
        img_path = SAMPLE_DIR / "beverages_shelf_01.png"
        image = Image.open(img_path).convert("RGB")
        image_name = "beverages_shelf_01.png"

    # Step 1: CV Detection
    detections = engine.analyze_shelf_image(image, image_id=image_name)

    # Step 2: POS Financial Correlation
    analytics = correlator.generate_report(detections, planogram_id=planogram_id or "POG-BEV-COOLER-01")

    # Step 3: Agentic Store Ops Worklist
    worklist = agent.generate_worklist(detections, analytics)

    return {
        "shelf_id": image_name,
        "image_width": detections.image_width,
        "image_height": detections.image_height,
        "execution_time_ms": detections.execution_time_ms,
        "inference_engine": detections.inference_engine,
        "detections": detections.model_dump(),
        "analytics": analytics.model_dump(),
        "worklist": worklist.model_dump()
    }
