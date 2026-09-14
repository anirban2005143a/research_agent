"""Document handler package for supported research file types."""

from .document_handler import DocumentHandler, SUPPORTED_EXTENSIONS
from .pdf_handler import PDFDocumentHandler

BaseDocumentHandler = DocumentHandler

__all__ = [
    "BaseDocumentHandler",
    "DocumentHandler",
    "PDFDocumentHandler",
    "SUPPORTED_EXTENSIONS",
]
