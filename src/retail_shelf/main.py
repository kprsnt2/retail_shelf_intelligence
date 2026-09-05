"""Main FastAPI Application Entrypoint for Retail Shelf Intelligence."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path

from retail_shelf.config import config, BASE_DIR
from retail_shelf.api import routes_shelf, routes_pos, routes_agent, routes_eval

app = FastAPI(
    title=config.app_name,
    version=config.version,
    description="On-prem retail intelligence: computer vision and analytics that read any shelf photo into per-row, per-product sales."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(routes_shelf.router)
app.include_router(routes_pos.router)
app.include_router(routes_agent.router)
app.include_router(routes_eval.router)

# Mount Static assets
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "app": config.app_name,
        "version": config.version,
        "on_prem_mode": config.on_prem_mode,
        "inference_engine": "on_prem_cv_v1"
    }

@app.get("/")
def serve_dashboard():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Retail Shelf Intelligence API is running. Access /docs for API schema."}

def main():
    import uvicorn
    uvicorn.run("retail_shelf.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
