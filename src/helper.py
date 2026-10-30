import os
import time
import glob
from typing import List
import httpx
from pypdf import PdfReader
from portkey_ai import Portkey, PORTKEY_GATEWAY_URL
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Extract Data From the PDF Files in a directory
def load_pdf_file(data: str) -> List[Document]:
    """
    Load all PDF files from the specified directory using pypdf and
    return a list of LangChain Document objects.
    """
    documents: List[Document] = []
    pdf_files = glob.glob(os.path.join(data, "*.pdf"))

    for file_path in pdf_files:
        reader = PdfReader(file_path)
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file_path, "page": page_idx}
                    )
                )

    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs


# Split the Data into Text Chunks
def text_split(extracted_data: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Direct Google Gemini Embeddings (High Performance for Chat Query)
class DirectGeminiEmbeddings(Embeddings):
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
            limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=60.0)
        )

    def embed_query(self, text: str) -> List[float]:
        t0 = time.perf_counter()
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]}
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
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not google_api_key:
        raise RuntimeError("GOOGLE_API_KEY must be configured in .env for fast query embeddings.")
    model = os.getenv("PORTKEY_EMBEDDING_MODEL", "gemini-embedding-2")
    return DirectGeminiEmbeddings(api_key=google_api_key, model=model)


# Portkey Gemini Embeddings Implementation
class PortkeyGeminiEmbeddings(Embeddings):
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


def get_embeddings() -> PortkeyGeminiEmbeddings:
    api_key = os.getenv("PORTKEY_API_KEY")
    config_id = os.getenv("PORTKEY_EMBEDDING_CONFIG_ID")
    model = os.getenv(
        "PORTKEY_EMBEDDING_MODEL",
        "gemini-embedding-2",
    )

    if not api_key or not config_id:
        raise RuntimeError(
            "PORTKEY_API_KEY and PORTKEY_EMBEDDING_CONFIG_ID "
            "must be configured."
        )

    return PortkeyGeminiEmbeddings(
        api_key=api_key,
        config_id=config_id,
        model=model,
    )