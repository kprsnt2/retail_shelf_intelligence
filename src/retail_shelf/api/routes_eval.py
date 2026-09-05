"""FastAPI Endpoints for Model Benchmarking and Evals."""
from fastapi import APIRouter
from typing import Dict, Any
from pathlib import Path

from retail_shelf.config import BASE_DIR
from retail_shelf.evals.runner import BenchmarkRunner

router = APIRouter(prefix="/api/eval", tags=["Model Evals & Benchmarks"])

runner = BenchmarkRunner()

@router.post("/run")
def trigger_benchmark(runs: int = 15) -> Dict[str, Any]:
    """Execute benchmark evaluation and return fresh metrics."""
    return runner.run_benchmark(latency_runs=runs)

@router.get("/report")
def get_benchmark_report() -> Dict[str, Any]:
    """Fetch latest benchmark report markdown."""
    rep_path = BASE_DIR / "BENCHMARK_REPORT.md"
    content = ""
    if rep_path.exists():
        with open(rep_path, "r", encoding="utf-8") as f:
            content = f.read()
    return {"markdown": content}
