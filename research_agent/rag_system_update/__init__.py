"""Research-document RAG components."""

from .document_handler import BaseDocumentHandler, DocumentHandler, PDFDocumentHandler, SUPPORTED_EXTENSIONS

__all__ = [
    "BaseDocumentHandler",
    "DocumentHandler",
    "PDFDocumentHandler",
    "SUPPORTED_EXTENSIONS",
]
