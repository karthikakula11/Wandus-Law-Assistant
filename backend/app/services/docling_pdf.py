"""
PDF text extraction using **Docling**, same stack as jamwithai/production-agentic-rag-course.

Reference: ``reference/production-agentic-rag-course/src/services/pdf_parser/docling.py``
(``DoclingParser.parse_pdf`` → ``DocumentConverter`` → ``document.export_to_text()``).

Optional: set ``USE_DOCLING_FOR_PDF=false`` or omit the ``docling`` package to skip.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text_with_docling(
    pdf_bytes: bytes,
    *,
    max_pages: int = 100,
    max_file_size_mb: int = 50,
) -> str | None:
    """
    Return plain text from PDF bytes using Docling. Returns None if unavailable or on failure.
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError:
        logger.info("docling not installed; skipping Docling extraction (pip install docling)")
        return None
    except Exception as e:
        # NumPy 2 + old torch wheels, torch<2.4, etc. break imports before we run; PyMuPDF handles PDF.
        logger.warning(
            "docling import failed (%s). Fix: pip install -r requirements.txt (numpy<2, torch>=2.4). Using PyMuPDF.",
            e,
        )
        return None

    max_bytes = max_file_size_mb * 1024 * 1024
    if len(pdf_bytes) > max_bytes:
        logger.warning("PDF exceeds max size for Docling (%s MB)", max_file_size_mb)
        return None

    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            path = Path(tmp.name)

        if len(pdf_bytes) >= 4 and pdf_bytes[:4] != b"%PDF":
            logger.warning("Bytes do not start with %%PDF magic; Docling may fail")

        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            do_ocr=False,
        )
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        try:
            result = converter.convert(
                str(path),
                max_num_pages=max_pages,
                max_file_size=max_bytes,
            )
        except TypeError:
            result = converter.convert(str(path))
        doc = result.document
        text = doc.export_to_text()
        text = (text or "").strip()
        if not text:
            return None
        logger.info("docling_extract: ok, len=%s", len(text))
        return text
    except Exception as e:
        # Expected on many PDFs; ``pdf_extract`` always tries PyMuPDF next — not a fatal error.
        logger.info(
            "Docling extraction failed (%s); PyMuPDF will be used for text extraction.",
            e,
        )
        return None
    finally:
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
