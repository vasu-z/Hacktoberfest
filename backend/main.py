from fastapi import FastAPI

app = FastAPI(
    title="Saarthi Backend",
    description="Backend API for Saarthi - An AI Vision Companion",
    version="0.1.0",
)


@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "ok",
        "service": "saarthi-backend",
        "version": "0.1.0",
    }
