"""FastAPI Endpoints for Store Ops Agent & Report Generation."""
from fastapi import APIRouter, Response
from typing import Dict, Any
from datetime import datetime
from PIL import Image

from retail_shelf.config import SAMPLE_DIR
from retail_shelf.cv.vision_engine import UnifiedVisionEngine
from retail_shelf.pos.pos_correlator import POSAnalyticsCorrelator
from retail_shelf.agent.ops_agent import StoreOpsAgent

router = APIRouter(prefix="/api/agent", tags=["Store Ops Agent"])

engine = UnifiedVisionEngine()
correlator = POSAnalyticsCorrelator()
agent = StoreOpsAgent()

@router.get("/report/export")
def export_store_report(sample_id: str = "beverages_shelf_01.png"):
    """Export one-click downloadable store audit report."""
    img_path = SAMPLE_DIR / sample_id
    if not img_path.exists():
        img_path = SAMPLE_DIR / "beverages_shelf_01.png"
        
    image = Image.open(img_path).convert("RGB")
    detections = engine.analyze_shelf_image(image, image_id=sample_id)
    analytics = correlator.generate_report(detections)
    worklist = agent.generate_worklist(detections, analytics)
    
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    
    report_md = f"""# DEEPWORK RETAIL INTELLIGENCE — STORE OPS AUDIT REPORT
Generated: {timestamp} | Mode: On-Premise Local Inference | Shelf: {sample_id}

================================================================================
EXECUTIVE SUMMARY
================================================================================
{worklist.executive_brief}

--------------------------------------------------------------------------------
CORE METRICS
--------------------------------------------------------------------------------
- On-Shelf Availability (OSA): {analytics.on_shelf_availability_pct}%
- Planogram Compliance:        {analytics.planogram_compliance_pct}%
- Total Facings Detected:     {analytics.total_facings_detected} ({analytics.total_occupied_facings} Occupied, {analytics.total_oos_voids} Voids)
- Daily Revenue At Risk:       ${analytics.total_daily_revenue_at_risk:,.2f}
- Weekly Revenue At Risk:      ${analytics.total_weekly_revenue_at_risk:,.2f}
- Scan Processing Latency:     {detections.execution_time_ms} ms

--------------------------------------------------------------------------------
PRIORITIZED ACTION WORKLIST (RANKED BY REVENUE IMPACT)
--------------------------------------------------------------------------------
"""
    for idx, act in enumerate(worklist.actions, 1):
        report_md += f"""[{act.priority}] #{idx}: {act.title}
  Location:       {act.location}
  Revenue Impact: ${act.revenue_impact_daily:.2f}/day (${act.revenue_impact_weekly:.2f}/week)
  Est. Time:      {act.estimated_resolution_time_min} mins
  Instructions:   {act.instructions}
--------------------------------------------------------------------------------
"""

    report_md += """
================================================================================
DATA INTEGRITY & SOVEREIGNTY NOTICE
This report was generated entirely on local store hardware. Zero imagery or POS
financial records were transmitted to any third-party cloud.
================================================================================
"""
    return Response(content=report_md, media_type="text/plain", headers={
        "Content-Disposition": f"attachment; filename=shelf_audit_{sample_id.replace('.png','')}.txt"
    })
