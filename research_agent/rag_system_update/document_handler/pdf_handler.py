"""PDF-specific document handler with metadata extraction for title and section information."""

import re
from pathlib import Path

import pymupdf
import pymupdf4llm
from langchain_core.documents import Document
from pypdf import PdfReader

from .document_handler import DocumentHandler, _clean_text


class PDFDocumentHandler(DocumentHandler):
    """PDF-aware document handler for loading page-level documents and splitting them."""

    def load_document(self, path: Path) -> list[Document]:
        """Read a PDF and return one LangChain document per page."""
        markdown_source = self._extract_markdown_source(path)
        pdf_reader = PdfReader(str(path))
        pdf_title = self._extract_title_from_metadata(pdf_reader.metadata)
        outline_headings = self._build_outline_heading_map(path)
        page_documents: list[Document] = []

        for page_number, page in enumerate(pdf_reader.pages, start=1):
            page_text = _clean_text(page.extract_text() or "")
            if not page_text:
                continue

            page_heading = self._extract_heading(
                markdown_text=markdown_source,
                fallback_text=page_text,
                pdf_title=pdf_title,
                outline_headings=outline_headings,
                page_number=page_number,
            )

            page_documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": path.name,
                        "page": page_number,
                        "page_number": page_number,
                        "title": pdf_title,
                        "heading": page_heading,
                        "section_heading": page_heading,
                    },
                )
            )

        if not page_documents and pdf_reader.metadata:
            fallback_title = self._extract_title_from_metadata(pdf_reader.metadata)
            if fallback_title:
                page_documents.append(
                    Document(
                        page_content=fallback_title,
                        metadata={
                            "source": path.name,
                            "page": 1,
                            "page_number": 1,
                            "title": fallback_title,
                            "heading": fallback_title,
                            "section_heading": fallback_title,
                        },
                    )
                )

        if markdown_source and page_documents:
            page_documents[0].metadata["markdown_source"] = markdown_source[:25000]

        return page_documents

    def prepare_document(self, path: Path) -> list[Document]:
        """Load PDF pages, clean them, and produce chunked result documents."""
        source_name = path.name
        print(f"[RAG][DOCUMENT] Reading: {source_name}")

        page_documents = self.load_document(path)
        prepared_documents: list[Document] = []

        for page_document in page_documents:
            content = _clean_text(page_document.page_content)
            if not content:
                continue

            prepared_documents.append(
                Document(
                    page_content=content,
                    metadata=self._extract_page_metadata(page_document, path),
                )
            )

        return self.generate_chunks(prepared_documents, source_name)

    def _extract_markdown_source(self, path: Path) -> str:
        """Convert the PDF to markdown text when the markdown export library is available."""
        if pymupdf4llm is None:
            return ""

        try:
            return pymupdf4llm.to_markdown(str(path), use_ocr=False) or ""
        except Exception:
            return ""

    def _extract_title_from_metadata(self, metadata: object) -> str:
        """Read the document title from PDF metadata, using the most reliable available field."""
        if not metadata:
            return ""

        for key in ("/Title", "title", "Title", "document_title", "Document Title", "pdf_title"):
            value = getattr(metadata, key, None)
            if value is None and isinstance(metadata, dict):
                value = metadata.get(key)
            if value:
                cleaned = re.sub(r"\s+", " ", str(value)).strip("# -:;\t\n").strip()
                if cleaned:
                    return cleaned
        return ""

    def _extract_authors_from_metadata(self, metadata: object) -> list[str]:
        """Extract author names from PDF metadata and normalize them into a clean list."""
        if not metadata:
            return []

        authors_value = getattr(metadata, "/Author", None)
        if authors_value is None:
            authors_value = getattr(metadata, "author", None)
        if authors_value is None and isinstance(metadata, dict):
            authors_value = metadata.get("/Author") or metadata.get("author") or metadata.get("authors")
        if not authors_value:
            return []

        if isinstance(authors_value, (list, tuple, set)):
            values = [str(item).strip() for item in authors_value if str(item).strip()]
        else:
            values = [part.strip() for part in str(authors_value).split(";") if part.strip()]

        cleaned_authors: list[str] = []
        for value in values:
            cleaned_value = re.sub(r"\s+", " ", value).strip("# -:;\t\n").strip()
            if cleaned_value and cleaned_value not in cleaned_authors:
                cleaned_authors.append(cleaned_value)
        return cleaned_authors

    def _is_generic_heading(self, value: str) -> bool:
        """Return True when the heading is a generic section label like introduction or references."""
        normalized = value.lower().strip().rstrip(":")
        generic_headings = {
            "abstract",
            "introduction",
            "background",
            "motivation",
            "related work",
            "method",
            "methods",
            "results",
            "discussion",
            "conclusion",
            "limitations",
            "acknowledgments",
            "references",
            "appendix",
        }
        return normalized in generic_headings or normalized.startswith("section ") or normalized.startswith("chapter ") or normalized.startswith("part ")

    def _extract_heading_candidates(self, text: str) -> list[str]:
        """Scan text for likely real headings while ignoring generic title blocks and boilerplate."""
        candidates: list[str] = []
        seen: set[str] = set()

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            markdown_match = re.match(r"^(?:#{1,6})\s+(.+?)\s*$", line)
            if markdown_match:
                value = re.sub(r"\s+", " ", markdown_match.group(1)).strip("# -:;\t\n").strip()
                if value and not self._is_generic_heading(value) and value not in seen:
                    candidates.append(value)
                    seen.add(value)
                continue

            numbered_match = re.match(r"^(?:\d+|[IVXLC]+)[\.,\)]?\s+(.+)$", line, flags=re.IGNORECASE)
            if numbered_match:
                value = re.sub(r"\s+", " ", numbered_match.group(1)).strip("# -:;\t\n").strip()
                if (
                    value
                    and not self._is_generic_heading(value)
                    and value not in seen
                    and value[0].isupper()
                    and len(value.split()) <= 12
                ):
                    candidates.append(value)
                    seen.add(value)

        return candidates

    def _extract_heading(
        self,
        markdown_text: str,
        fallback_text: str = "",
        pdf_title: str = "",
        outline_headings: dict[int, str] | None = None,
        page_number: int | None = None,
    ) -> str:
        """Choose the best heading for a PDF page using the TOC, markdown text, and fallback heuristics."""
        if outline_headings and page_number is not None and page_number in outline_headings:
            return outline_headings[page_number]

        candidates = self._extract_heading_candidates(markdown_text or fallback_text)
        if pdf_title and not self._is_generic_heading(pdf_title):
            for candidate in candidates:
                if candidate.lower() != pdf_title.lower():
                    return candidate
            return pdf_title

        for candidate in candidates:
            if candidate:
                return candidate

        if pdf_title:
            return pdf_title

        for possible_heading in self._extract_heading_candidates(fallback_text):
            if possible_heading:
                return possible_heading

        return ""

    def _build_outline_heading_map(self, path: Path) -> dict[int, str]:
        """Read the PDF outline and map each page number to the latest heading in the document TOC."""
        try:
            pdf_document = pymupdf.open(str(path))
        except Exception:
            return {}

        headings_by_page: dict[int, str] = {}
        try:
            table_of_contents = pdf_document.get_toc(simple=False)
        except Exception:
            return {}

        for item in table_of_contents:
            if len(item) < 3:
                continue
            heading_title = str(item[1]).strip()
            page_number = int(item[2]) if str(item[2]).isdigit() else 0
            if not heading_title or page_number <= 0:
                continue
            headings_by_page[page_number] = heading_title

        ordered_pages = sorted(headings_by_page)
        resolved: dict[int, str] = {}
        current_heading = ""
        for page_number in range(1, pdf_document.page_count + 1):
            for outline_page in ordered_pages:
                if outline_page <= page_number:
                    current_heading = headings_by_page[outline_page]
                else:
                    break
            if current_heading:
                resolved[page_number] = current_heading

        return resolved

    def _extract_page_metadata(self, page_document: Document, file_path: str | Path) -> dict[str, str | list[str]]:
        """Build metadata for a single page document after the page has been cleaned and prepared."""
        page_number = page_document.metadata.get("page")
        pdf_title = self._extract_title_from_metadata(page_document.metadata)
        page_heading = str(page_document.metadata.get("heading") or self._extract_heading(
            markdown_text=str(page_document.metadata.get("markdown_source") or ""),
            fallback_text=page_document.page_content,
            pdf_title=pdf_title,
            page_number=page_number,
        ))

        metadata: dict[str, str | list[str]] = {
            "filename": Path(file_path).name,
            "title": pdf_title,
            "heading": page_heading,
            "section_heading": page_heading,
            "source": Path(file_path).name,
        }

        authors = self._extract_authors_from_metadata(page_document.metadata)
        if authors:
            metadata["authors"] = authors

        if page_number is not None:
            metadata["page_number"] = str(page_number)

        return metadata
