import os
import time
from typing import List
import httpx
from langchain_core.embeddings import Embeddings
from portkey_ai import Portkey, PORTKEY_GATEWAY_URL

from src.config import config


# 1. Direct Google Gemini Embeddings (High Performance for Chat Query)
class DirectGeminiEmbeddings(Embeddings):
    """Direct Google AI Studio client for fast query embeddings with HTTP keep-alive."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-2",
    ):
        self.api_key = api_key
        self.model = model
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent?key={self.api_key}"
        self.client = httpx.Client(
            timeout=15.0,
            limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=60.0),
        )

    def embed_query(self, text: str) -> List[float]:
        t0 = time.perf_counter()
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
        }
        resp = self.client.post(self.url, json=payload)
        resp.raise_for_status()
        vec = resp.json()["embedding"]["values"]
        elapsed = time.perf_counter() - t0
        print(f"    [Direct Google AI] Query embedding generated in: {elapsed:.3f}s")
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]


def get_query_embeddings() -> DirectGeminiEmbeddings:
    """Factory to get the direct high-speed embedding model for user queries."""
    if not config.google_api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY must be configured in .env for fast query embeddings."
        )
    return DirectGeminiEmbeddings(
        api_key=config.google_api_key,
        model=config.embedding_model,
    )


# 2. Portkey Gemini Embeddings (Used for document batch ingestion)
class PortkeyGeminiEmbeddings(Embeddings):
    """Portkey AI Gateway client for managed batch ingestion."""

    def __init__(
        self,
        api_key: str,
        config_id: str,
        model: str = "gemini-embedding-2",
    ):
        self.model = model
        self.client = Portkey(
            api_key=api_key,
            config=config_id,
            base_url=PORTKEY_GATEWAY_URL,
        )

    def embed_query(self, text: str) -> List[float]:
        t0 = time.perf_counter()
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
        )
        elapsed = time.perf_counter() - t0
        print(f"    [Portkey API] Query embedding generated in: {elapsed:.3f}s")
        return response.data[0].embedding

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]


def get_portkey_embeddings() -> PortkeyGeminiEmbeddings:
    """Factory to get the Portkey-managed embedding model for batch indexing."""
    api_key = os.getenv("PORTKEY_API_KEY")
    config_id = os.getenv("PORTKEY_EMBEDDING_CONFIG_ID")

    if not api_key or not config_id:
        raise RuntimeError(
            "PORTKEY_API_KEY and PORTKEY_EMBEDDING_CONFIG_ID must be configured."
        )

    return PortkeyGeminiEmbeddings(
        api_key=api_key,
        config_id=config_id,
        model=config.embedding_model,
    )

