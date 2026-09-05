"""End-to-End System Test Suite for Retail Shelf Intelligence."""
from starlette.testclient import TestClient
from retail_shelf.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["on_prem_mode"] is True

def test_list_samples():
    response = client.get("/api/shelf/samples")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) >= 3
    assert any(s["filename"] == "beverages_shelf_01.png" for s in samples)

def test_analyze_shelf_sample():
    response = client.post("/api/shelf/analyze", data={"sample_id": "beverages_shelf_01.png"})
    assert response.status_code == 200
    payload = response.json()
    
    # Check detections
    assert "detections" in payload
    assert payload["detections"]["occupied_facings"] > 15
    assert payload["detections"]["oos_voids"] >= 1
    assert payload["execution_time_ms"] < 500 # sub-500ms on CPU!
    
    # Check POS analytics
    analytics = payload["analytics"]
    assert analytics["on_shelf_availability_pct"] > 0
    assert analytics["total_daily_revenue_at_risk"] > 0
    assert len(analytics["share_of_shelf"]) > 0
    
    # Check agent worklist
    worklist = payload["worklist"]
    assert worklist["total_actions"] > 0
    assert worklist["p0_critical_count"] >= 1
    assert len(worklist["actions"]) > 0

def test_eval_benchmark_endpoint():
    response = client.post("/api/eval/run?runs=5")
    assert response.status_code == 200
    report = response.json()
    assert "overall_metrics" in report
    assert report["overall_metrics"]["facing_mAP_50"] >= 0.85
    assert report["overall_metrics"]["oos_void_recall"] >= 0.85
    assert report["latency_benchmarks_ms"]["mean"] < 300

def test_export_report_endpoint():
    response = client.get("/api/agent/report/export?sample_id=beverages_shelf_01.png")
    assert response.status_code == 200
    assert "DEEPWORK RETAIL INTELLIGENCE" in response.text
    assert "PRIORITIZED ACTION WORKLIST" in response.text
