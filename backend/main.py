from fastapi import FastAPI

from backend.config import settings

app = FastAPI(
    title="Saarthi Backend",
    description="Backend API for Saarthi - An AI Vision Companion",
    version=settings.saarthi_version,
)


@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "ok",
        "service": settings.saarthi_service_name,
        "version": settings.saarthi_version,
    }
