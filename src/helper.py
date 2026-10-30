"""
Backward-compatibility facade for src.helper.
Re-exports document loaders and embeddings from their respective modular files.
"""

from src.document_loader import load_pdf_file, filter_to_minimal_docs, text_split
from src.embeddings import (
    DirectGeminiEmbeddings,
    PortkeyGeminiEmbeddings,
    get_query_embeddings,
    get_portkey_embeddings,
)

# Alias for legacy compatibility
get_embeddings = get_portkey_embeddings

__all__ = [
    "load_pdf_file",
    "filter_to_minimal_docs",
    "text_split",
    "DirectGeminiEmbeddings",
    "PortkeyGeminiEmbeddings",
    "get_query_embeddings",
    "get_portkey_embeddings",
    "get_embeddings",
]
