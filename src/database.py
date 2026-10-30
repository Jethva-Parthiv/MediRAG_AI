import atexit
import logging
from typing import Optional
import weaviate
from weaviate.auth import Auth
from langchain_weaviate import WeaviateVectorStore

from src.config import config
from src.embeddings import get_query_embeddings

logger = logging.getLogger(__name__)

_client: Optional[weaviate.WeaviateClient] = None


def get_weaviate_client() -> weaviate.WeaviateClient:
    """Singleton connection provider for Weaviate Cloud."""
    global _client
    if _client is None or not _client.is_connected():
        _client = weaviate.connect_to_weaviate_cloud(
            cluster_url=config.weaviate_url,
            auth_credentials=Auth.api_key(config.weaviate_api_key),
            skip_init_checks=True,
        )
        if not _client.is_connected():
            raise ConnectionError("Unable to establish Weaviate Cloud connection.")
        logger.info("Connected to Weaviate Cloud.")
    return _client


def close_weaviate():
    """Gracefully close the Weaviate client connection."""
    global _client
    if _client is not None:
        try:
            if _client.is_connected():
                _client.close()
                logger.info("Weaviate connection closed gracefully.")
        except Exception:
            logger.exception("Error while closing Weaviate connection.")


atexit.register(close_weaviate)


def get_vectorstore(client: Optional[weaviate.WeaviateClient] = None, embeddings=None) -> WeaviateVectorStore:
    """Factory to obtain the configured WeaviateVectorStore."""
    if client is None:
        client = get_weaviate_client()
    if embeddings is None:
        embeddings = get_query_embeddings()

    return WeaviateVectorStore(
        client=client,
        index_name=config.index_name,
        text_key="text",
        embedding=embeddings,
    )


def get_retriever(top_k: Optional[int] = None):
    """Factory to get the vectorstore retriever configured with top_k."""
    k = top_k if top_k is not None else config.top_k
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

