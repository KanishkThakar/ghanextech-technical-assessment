"""PDF ingestion.

`pypdf` is used because it is pure Python (no system libraries, so it installs
anywhere) and fast enough for the handful of files a user uploads.

Every awkward case becomes a typed error or a warning the UI can show, rather
than a stack trace:

  not a PDF            -> IngestError
  password-protected   -> IngestError (owner-password files that open with an
                          empty password are accepted)
  no text at all       -> IngestError ("looks scanned, no OCR")
  some pages no text   -> IngestResult.warning naming how many pages were skipped
  corrupt / truncated  -> IngestError
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import List

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class IngestError(ValueError):
    """A file that cannot be used, with a message meant for the user."""


@dataclass(frozen=True)
class PageText:
    """The text of one page."""

    doc_name: str
    page: int  # 1-based, as a reader counts pages
    text: str


@dataclass
class IngestResult:
    """What came out of one file."""

    pages: List[PageText] = field(default_factory=list)
    warning: str | None = None


def ingest_pdf(data: bytes, doc_name: str) -> IngestResult:
    """Read a PDF's text page by page."""
    if not data:
        raise IngestError(f"{doc_name} is empty.")
    if not data.lstrip()[:5].startswith(b"%PDF"):
        raise IngestError(f"{doc_name} is not a PDF file.")

    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
    except PdfReadError as exc:
        raise IngestError(f"{doc_name} could not be read as a PDF ({exc}).") from exc

    if reader.is_encrypted:
        # Many PDFs are "encrypted" with an empty owner password and open fine.
        try:
            opened = reader.decrypt("")
        except Exception as exc:  # pypdf raises several types here
            raise IngestError(
                f"{doc_name} is password-protected. Remove the password and upload it again."
            ) from exc
        if opened == 0:
            raise IngestError(
                f"{doc_name} is password-protected. Remove the password and upload it again."
            )

    pages: List[PageText] = []
    empty_pages = 0

    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # a single bad page must not lose the whole document
            text = ""
        text = text.strip()
        if not text:
            empty_pages += 1
            continue
        pages.append(PageText(doc_name=doc_name, page=number, text=text))

    if not pages:
        raise IngestError(
            f"{doc_name} has no extractable text. It looks like a scanned document; "
            "this app does not run OCR."
        )

    warning = None
    if empty_pages:
        warning = (
            f"{doc_name}: {empty_pages} page(s) had no text layer and were skipped "
            "(likely scanned images)."
        )

    return IngestResult(pages=pages, warning=warning)
