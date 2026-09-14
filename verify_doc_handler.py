from pathlib import Path
from research_agent.rag_system_update.document_handler import DocumentHandler, PDFDocumentHandler, SUPPORTED_EXTENSIONS
assert SUPPORTED_EXTENSIONS == {'.pdf', ' .md', ' .ppt', ' .pptx', ' .docx', ' .txt'}
handler = DocumentHandler(Path('tmp_doc_session'))
assert handler._get_document_handler('sample.pdf').__class__ is PDFDocumentHandler
assert handler._get_document_handler('notes.txt').__class__ is DocumentHandler
print('OK', sorted(SUPPORTED_EXTENSIONS))
