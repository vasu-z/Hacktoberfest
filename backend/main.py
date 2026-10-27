from fastapi import FastAPI

from backend.config import settings
from backend.routers.objects import router as objects_router
from backend.routers.vision import router as vision_router

app = FastAPI(
    title="Saarthi Backend",
    description="Backend API for Saarthi - An AI Vision Companion",
    version=settings.saarthi_version,
)

app.include_router(vision_router)
app.include_router(objects_router)


@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "ok",
        "service": settings.saarthi_service_name,
        "version": settings.saarthi_version,
    }
