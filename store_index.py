import logging
from src.config import config
from src.document_loader import load_pdf_file, filter_to_minimal_docs, text_split
from src.embeddings import get_portkey_embeddings
from src.database import get_weaviate_client, close_weaviate
from langchain_weaviate import WeaviateVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def run_indexing(data_dir: str = "data/"):
    """Extracts, chunks, embeds, and indexes PDF medical knowledge into Weaviate Cloud."""

    # 1. Document Extraction & Chunking
    logger.info("Loading PDF documents from %s ...", data_dir)
    extracted_data = load_pdf_file(data_dir)
    filtered_data = filter_to_minimal_docs(extracted_data)
    text_chunks = text_split(filtered_data)
    logger.info("Total text chunks prepared: %d", len(text_chunks))

    # 2. Embedding Model Setup (Portkey Gemini)
    logger.info("Initializing Portkey embedding model (%s)...", config.embedding_model)
    embeddings = get_portkey_embeddings()

    # 3. Weaviate Cloud Ingestion
    client = get_weaviate_client()
    index_name = config.index_name

    if client.collections.exists(index_name):
        logger.info("Collection '%s' already exists. Recreating collection...", index_name)
        client.collections.delete(index_name)

    logger.info("Creating fresh collection '%s'...", index_name)
    client.collections.create(name=index_name, vectorizer_config=None)

    try:
        logger.info("Indexing %d chunks into Weaviate Cloud collection '%s'...", len(text_chunks), index_name)
        docsearch = WeaviateVectorStore(
            client=client,
            index_name=index_name,
            embedding=embeddings,
            text_key="text",
        )
        docsearch.add_documents(text_chunks)
        logger.info("🎉 Vector indexing into Weaviate Cloud completed successfully!")
    finally:
        close_weaviate()


if __name__ == "__main__":
    run_indexing()
