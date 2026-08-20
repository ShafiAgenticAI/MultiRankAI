from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
)

from src.services.ingestion_service import (
    ingest_pdf,
)


router = APIRouter(
    tags=["Ingestion"]
)


@router.post("/upload")
async def upload(
    file: UploadFile = File(...)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="File name is required.",
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    try:

        file_bytes = await file.read()

        result = ingest_pdf(
            file_bytes=file_bytes,
            file_name=file.filename,
        )

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        print(
            f"Upload ingestion failed: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed.",
        )