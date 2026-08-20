from fastapi import FastAPI

from src.api.v1.routes.health import router as health_router
from src.api.v1.routes.query import router as query_router
from src.api.v1.routes.upload import router as upload_router

app = FastAPI(
    title="Smart Banking Assistant",
    description="Capstone Project 2 - LangGraph based Smart Banking Assistant",
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(query_router, prefix="/api/v1")
app.include_router(upload_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "application": "Smart Banking Assistant",
        "status": "running",
        "version": "0.1.0",
    }
