"""PDF to Markdown and section-wise chunking for RAG."""

import re
from pathlib import Path
import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pypdf import PdfReader

from .document_handler import DocumentHandler
from ...utils import log


class PDFDocumentHandler(DocumentHandler):
    """Convert PDFs to Markdown and create section-aware RAG chunks."""

    def __init__(
        self,
        max_chars: int = 4000,
        chunk_overlap: int = 400,
    ):
        self.max_chars = max_chars
        self.chunk_overlap = chunk_overlap

        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
                ("#####", "h5"),
                ("######", "h6"),
            ],
            strip_headers=False,
        )

        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                " ",
                "",
            ],
        )

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

    def prepare_document(self, path: Path) -> list[Document]:
        """
        Convert PDF to Markdown, split it into semantic sections,
        and recursively split only sections exceeding max_chars.
        """

        log(f"rag.document.reading | source={path.name}")

        # ---------------------------------------------------------
        # 1. PDF metadata
        # ---------------------------------------------------------
        pdf_reader = PdfReader(str(path))
        pdf_metadata = pdf_reader.metadata or {}

        title = self._extract_title_from_metadata(pdf_metadata)

        authors = self._extract_authors_from_metadata(pdf_metadata) 

        # ---------------------------------------------------------
        # 2. PDF -> Markdown
        # ---------------------------------------------------------
        markdown = pymupdf4llm.to_markdown(str(path))

        if not markdown.strip():
            return []

        # ---------------------------------------------------------
        # 3. Markdown -> semantic sections
        # ---------------------------------------------------------
        sections = self.markdown_splitter.split_text(markdown)

        chunks: list[Document] = []

        for section in sections:

            # -----------------------------------------------------
            # 1. Extract headings from MarkdownHeaderTextSplitter
            # -----------------------------------------------------
            section_headings = [
                value
                for key, value in sorted(
                    section.metadata.items(),
                    key=lambda item: int(item[0][1:]),
                )
                if key.startswith("h") and key[1:].isdigit() and value
            ]

            # -----------------------------------------------------
            # 2. Build metadata
            # -----------------------------------------------------
            section_metadata = {
                "source": path.name,
                "title": title,
            }
            if authors:
                section_metadata["author"] = authors
            if section_headings:
                section_metadata["section_headings"] = section_headings

            # -----------------------------------------------------
            # 3. Build semantic prefix
            #
            # Example:
            #
            # Book Title
            # Introduction
            # Background
            #
            # -----------------------------------------------------
            prefix_parts = []

            if title:
                prefix_parts.append(title)

            prefix_parts.extend(section_headings)

            heading_prefix = "\n".join(prefix_parts)

            if heading_prefix:
                heading_prefix += "\n\n"

            # -----------------------------------------------------
            # 4. Split the ORIGINAL section content
            #
            # RecursiveCharacterTextSplitter will return the
            # section as a single chunk if it is already small
            # enough.
            # -----------------------------------------------------
            sub_chunks = self.recursive_splitter.split_text(
                section.page_content
            )

            # -----------------------------------------------------
            # 5. Add title + headings to EVERY chunk
            # -----------------------------------------------------
            for chunk in sub_chunks:

                chunk_content = heading_prefix + chunk

                chunks.append(
                    Document(
                        page_content=chunk_content,
                        metadata=section_metadata
                    )
                )


        return chunks
