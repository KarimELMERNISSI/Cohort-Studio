"""
High-Performance PDF Document Loader.

Replaces deprecated `langchain_community.document_loaders` with native `pypdf` extraction,
improving performance, eliminating deprecation warnings, and resiliently handling errors.
"""

import logging
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFDocumentLoader:
    """
    Lightweight, native PDF document loader using pypdf.
    
    Converts PDF pages into standardized `langchain_core.documents.Document`
    objects with complete metadata (source, filename, page, total_pages).
    """

    def __init__(self, file_or_dir_path: str | Path):
        """
        Initialize the loader with a file or directory path.
        
        Args:
            file_or_dir_path: Path to a single PDF or a directory containing PDFs.
        """
        self.path = Path(file_or_dir_path)

    def load_single_pdf(self, file_path: str | Path) -> list[Document]:
        """
        Extract text from an individual PDF file page by page.
        
        Args:
            file_path: Absolute or relative path to the PDF file.
            
        Returns:
            List of Document objects representing non-empty pages.
        """
        path = Path(file_path)
        if not path.is_file() or path.suffix.lower() != ".pdf":
            logger.warning(f"Skipping non-PDF file: {path}")
            return []

        documents: list[Document] = []
        try:
            reader = PdfReader(str(path))
            total_pages = len(reader.pages)

            for page_idx, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                    # Normalize whitespace but keep paragraphs
                    cleaned_text = text.strip()
                    if cleaned_text:
                        metadata = {
                            "source": str(path.resolve()),
                            "file_name": path.name,
                            "page": page_idx,
                            "total_pages": total_pages,
                        }
                        documents.append(Document(page_content=cleaned_text, metadata=metadata))
                except Exception as page_err:
                    logger.warning(f"Error extracting page {page_idx} from {path.name}: {page_err}")
                    continue

        except Exception as e:
            logger.error(f"Failed to read PDF file '{path}': {e}")
            return []

        return documents

    def load(self, recursive: bool = True) -> list[Document]:
        """
        Load all PDF documents from the initialized path.
        
        Args:
            recursive: If True and path is a directory, searches subdirectories.
            
        Returns:
            Aggregated list of Document objects from all discovered PDFs.
        """
        if not self.path.exists():
            logger.warning(f"Specified path does not exist: {self.path}")
            return []

        if self.path.is_file():
            return self.load_single_pdf(self.path)

        documents: list[Document] = []
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = sorted(self.path.glob(pattern))

        for pdf_file in pdf_files:
            file_docs = self.load_single_pdf(pdf_file)
            documents.extend(file_docs)

        return documents
