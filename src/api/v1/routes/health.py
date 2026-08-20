from fastapi import APIRouter

from src.core.db import check_database

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/api/v1/db-check")
async def db_check():
    return await check_database()
