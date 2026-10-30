import os
import glob
from typing import List
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_file(data_dir: str) -> List[Document]:
    """
    Load all PDF files from the specified directory using pypdf and
    return a list of LangChain Document objects.
    """
    documents: List[Document] = []
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))

    for file_path in pdf_files:
        reader = PdfReader(file_path)
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file_path, "page": page_idx},
                    )
                )

    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list containing
    only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src},
            )
        )
    return minimal_docs


def text_split(
    extracted_data: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 20,
) -> List[Document]:
    """Split extracted documents into smaller chunks for vector indexing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(extracted_data)

