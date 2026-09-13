"""Document loading, lightweight metadata extraction, and chunking."""

import csv
import json
import re
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from openpyxl import load_workbook
from pptx import Presentation
from pypdf import PdfReader

from ..config import settings

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".json",
    ".html",
    ".htm",
    ".py",
    ".xml",
    ".xlsx",
    ".pptx",
}


def _clean_text(text: str) -> str:
    """Remove layout-only whitespace without changing the actual wording."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_heading(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip("# -:;\t\n").strip()


def _extract_pdf_title_from_metadata(document: Document) -> str:
    """Prefer actual PDF metadata fields before any fallback heuristics."""
    for key in ("title", "Title", "document_title", "Document Title", "pdf_title"):
        value = document.metadata.get(key)
        if value:
            cleaned = _clean_heading(str(value))
            if cleaned:
                return cleaned
    return ""


def _extract_pdf_authors_from_metadata(document: Document) -> list[str]:
    """Use the real PDF author metadata when available."""
    authors_value = document.metadata.get("author") or document.metadata.get("Author") or document.metadata.get("authors")
    if not authors_value:
        return []

    if isinstance(authors_value, (list, tuple, set)):
        values = [str(item).strip() for item in authors_value if str(item).strip()]
    else:
        values = [part.strip() for part in str(authors_value).split(";") if part.strip()]

    cleaned = []
    for value in values:
        norm = _clean_heading(value)
        if norm and norm not in cleaned:
            cleaned.append(norm)
    return cleaned


def _extract_heading_candidates(text: str) -> list[str]:
    """Extract explicit heading markers from the document content itself."""
    headings: list[str] = []
    seen: set[str] = set()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        html_match = re.match(r"<h([1-6])[^>]*>\s*(.+?)\s*</h\1>", line, flags=re.IGNORECASE | re.DOTALL)
        if html_match:
            value = _clean_heading(html_match.group(2))
            if value and value not in seen:
                headings.append(value)
                seen.add(value)
            continue

        markdown_match = re.match(r"^(?:#{1,6})\s+(.+)$", line)
        if markdown_match:
            value = _clean_heading(markdown_match.group(1))
            if value and value.lower() not in {"abstract", "introduction"} and value not in seen:
                headings.append(value)
                seen.add(value)
            continue

        section_match = re.match(
            r"^(?:Section|Chapter|Part)\s+[0-9IVXLC]+\s*[:.-]?\s*(.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if section_match:
            value = _clean_heading(section_match.group(1))
            if value and value not in seen:
                headings.append(value)
                seen.add(value)
            continue

        if 2 <= len(line) <= 120 and line.endswith(":"):
            value = _clean_heading(line.rstrip(":"))
            if value and value not in seen and value[0].isupper():
                headings.append(value)
                seen.add(value)

    return headings


def _extract_title_from_text(document: Document) -> str:
    """Return the title from explicit document structure, not filename heuristics."""
    text = document.page_content
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        html_match = re.match(r"<h1[^>]*>\s*(.+?)\s*</h1>", line, flags=re.IGNORECASE | re.DOTALL)
        if html_match:
            value = _clean_heading(html_match.group(1))
            if value:
                return value

        markdown_match = re.match(r"^#\s+(.+)$", line)
        if markdown_match:
            value = _clean_heading(markdown_match.group(1))
            if value:
                return value

        if line and len(line) <= 180 and line[0].isupper() and len(line.split()) <= 15:
            if not line.endswith(".") and not line.endswith("?"):
                return _clean_heading(line)

    return ""


def _find_heading(document: Document) -> str:
    """Use the actual heading structure from the document when available."""
    metadata_title = _extract_pdf_title_from_metadata(document)
    if metadata_title:
        return metadata_title

    title_from_text = _extract_title_from_text(document)
    if title_from_text:
        return title_from_text

    heading_candidates = _extract_heading_candidates(document.page_content)
    if heading_candidates:
        return heading_candidates[0]
    return ""


def _make_metadata(document: Document, file_path: str | Path) -> dict[str, str | list[str]]:
    """Create a concise set of PDF-relevant metadata fields for retrieval."""
    page_number = document.metadata.get("page")
    pdf_title = _extract_pdf_title_from_metadata(document)
    title_from_text = _extract_title_from_text(document)
    heading_candidates = _extract_heading_candidates(document.page_content)

    title = pdf_title or title_from_text or (heading_candidates[0] if heading_candidates else "")
    section_heading = heading_candidates[0] if heading_candidates else ""
    topic_name = title or ""
    authors = _extract_pdf_authors_from_metadata(document)

    metadata: dict[str, str | list[str]] = {
        "filename": Path(file_path).name,
        "title": title,
        "section_heading": section_heading,
        "topic_name": topic_name,
    }

    if authors:
        metadata["authors"] = authors

    if page_number is not None:
        metadata["page_number"] = str(page_number)

    if document.metadata.get("source"):
        metadata["source"] = str(document.metadata["source"])

    return metadata


def _load_text_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [Document(page_content=json.dumps(payload, ensure_ascii=False, indent=2))]

    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = csv.reader(handle)
            text = "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)
        return [Document(page_content=text)]

    if suffix in {".html", ".htm"}:
        splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=[
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
            ]
        )
        return splitter.split_text(path.read_text(encoding="utf-8", errors="ignore"))

    if suffix == ".xlsx":
        workbook = load_workbook(path, read_only=True, data_only=True)
        documents: list[Document] = []
        for sheet in workbook.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                rows.append(" | ".join(str(value or "") for value in row))
            documents.append(Document(page_content="\n".join(rows)))
        return documents

    if suffix == ".pptx":
        presentation = Presentation(path)
        return [
            Document(
                page_content="\n".join(
                    shape.text for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()
                )
            )
            for slide in presentation.slides
        ]

    return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()


def _extract_pdf_page_headings(page_text: str) -> list[str]:
    """Return explicit section headings from a single PDF page."""
    return _extract_heading_candidates(page_text)


def _load_pdf_file(path: Path) -> list[Document]:
    """Parse PDFs using pypdf so we can preserve actual PDF metadata and page numbers."""
    reader = PdfReader(str(path))
    pdf_title = ""
    if reader.metadata:
        raw_title = getattr(reader.metadata, "/Title", None) or getattr(reader.metadata, "title", None)
        if raw_title:
            pdf_title = _clean_heading(str(raw_title))

    documents: list[Document] = []
    reader_authors = []
    if reader.metadata:
        raw_authors = getattr(reader.metadata, "/Author", None) or getattr(reader.metadata, "author", None)
        if raw_authors:
            reader_authors = [part.strip() for part in str(raw_authors).split(";") if part.strip()]

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = _clean_text(text)
        if not cleaned:
            continue

        headings = _extract_pdf_page_headings(cleaned)
        page_metadata = {
            "source": path.name,
            "page": page_number,
            "page_number": page_number,
            "title": pdf_title,
            "topic_name": pdf_title,
            "section_heading": headings[0] if headings else "",
        }
        if reader_authors:
            page_metadata["authors"] = reader_authors
        documents.append(Document(page_content=cleaned, metadata=page_metadata))

    if not documents and reader.metadata:
        fallback = _clean_heading(str(reader.metadata.get("/Title") or reader.metadata.get("title") or ""))
        if fallback:
            documents.append(Document(page_content=fallback, metadata={"source": path.name, "page": 1, "page_number": 1, "title": fallback, "topic_name": fallback, "section_heading": fallback, "heading": fallback}))

    return documents


def _load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported document type: {suffix}. Supported types: {supported}")

    if suffix == ".pdf":
        return _load_pdf_file(path)
    if suffix == ".docx":
        return Docx2txtLoader(str(path)).load()
    return _load_text_file(path)


class DocumentHandler:
    """Own document files and turn them into retrieval-ready chunks."""

    def __init__(self, storage_dir: str | Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(self, file_path: str | Path, content: bytes) -> Path:
        """Persist a file inside the session folder, replacing any prior copy with the same basename."""
        path = Path(file_path)
        if path.is_dir():
            raise ValueError("save_upload expects a file path, not a directory.")
        destination = self.storage_dir / path.name
        if destination.exists():
            destination.unlink()
            print(f"[RAG][FILES] Replaced existing file: {destination.name}")

        destination.write_bytes(content)
        print(f"[RAG][FILES] Saved: {destination.name}")
        return destination

    def remove_file(self, file_path: str | Path) -> bool:
        """Delete the stored copy for a file path in the current session folder."""
        path = Path(file_path)
        destination = self.storage_dir / path.name
        if not destination.exists():
            return False
        destination.unlink()
        print(f"[RAG][FILES] Removed: {destination.name}")
        return True

    def file_exists(self, file_path: str | Path) -> bool:
        return (self.storage_dir / Path(file_path).name).exists()

    def prepare_file(self, file_path: str | Path) -> list[Document]:
        """Load, clean, annotate, and split one file path only."""
        path = Path(file_path)
        if path.is_dir():
            raise ValueError(
                "DocumentHandler.prepare_file expects a single file path, not a directory. "
                "Directory iteration should happen in the test entrypoint."
            )
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")

        source_name = path.name
        print(f"[RAG][DOCUMENT] Reading: {source_name}")

        source_documents = _load_file(path)
        prepared: list[Document] = []

        for document in source_documents:
            content = _clean_text(document.page_content)
            if not content:
                continue
            prepared.append(
                Document(
                    page_content=content,
                    metadata=_make_metadata(document, file_path),
                )
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=getattr(settings, "rag_chunk_size", 900),
            chunk_overlap=getattr(settings, "rag_chunk_overlap", 140),
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
            add_start_index=False,
        )
        chunks = splitter.split_documents(prepared)

        for chunk in chunks:
            chunk.page_content = _clean_text(chunk.page_content)

        print(f"[RAG][CHUNKING] Created {len(chunks)} chunks from {source_name}")
        return [chunk for chunk in chunks if chunk.page_content]

    def read_stored_file(self, file_path: str | Path, max_chars: int = 12000) -> str:
        """Read the original stored file for non-RAG file inspection."""
        path = self.storage_dir / Path(file_path).name
        if not path.exists():
            return f"File not found: {file_path}"

        documents = _load_file(path)
        return "\n\n".join(document.page_content for document in documents)[:max_chars]
