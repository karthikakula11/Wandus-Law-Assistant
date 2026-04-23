from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.pdf_extract import bytes_to_document_text
from app.schemas import IngestBody, IngestResponse
from app.services.ingest import ingest_document

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
async def ingest_json(
    body: IngestBody,
    session: AsyncSession = Depends(get_session),
):
    doc_id, n = await ingest_document(
        session,
        title=body.title,
        text=body.text,
        source_uri=body.source_uri,
    )
    return IngestResponse(document_id=doc_id, chunks_created=n)


@router.post("/file", response_model=IngestResponse)
async def ingest_file(
    title: str = Form(..., min_length=1, max_length=512),
    source_uri: str | None = Form(None),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    name = (file.filename or "").strip().lower()
    if not (name.endswith(".pdf") or name.endswith(".txt")):
        raise HTTPException(
            status_code=400,
            detail="Only PDF or plain UTF-8 .txt uploads are supported.",
        )
    raw = await file.read()
    try:
        text = bytes_to_document_text(file.filename, raw)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read file: {e!s}",
        ) from e
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No extractable text in file (empty PDF or unsupported content).",
        )
    doc_id, n = await ingest_document(
        session, title=title, text=text, source_uri=source_uri
    )
    return IngestResponse(document_id=doc_id, chunks_created=n)
