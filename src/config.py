import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Application and external service configuration."""

    # Weaviate Cloud Configuration
    weaviate_url: str = os.getenv("WEAVIATE_URL", "")
    weaviate_api_key: str = os.getenv("WEAVIATE_API_KEY", "")
    index_name: str = os.getenv("WEAVIATE_INDEX_NAME", "MediragAiDb")

    # LLM & Embedding Models
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    groq_temperature: float = float(os.getenv("GROQ_TEMPERATURE", "0.4"))

    google_api_key: str = os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    embedding_model: str = os.getenv("PORTKEY_EMBEDDING_MODEL", "gemini-embedding-2")

    # Retrieval Configuration
    top_k: int = int(os.getenv("RETRIEVER_TOP_K", "3"))

    # Server Configuration
    host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    port: int = int(os.getenv("SERVER_PORT", "8080"))
    debug: bool = os.getenv("FLASK_DEBUG", "0").lower() in ("1", "true")

    def __post_init__(self):
        # Validate critical credentials
        if not all([self.weaviate_url, self.weaviate_api_key, self.groq_api_key]):
            raise EnvironmentError(
                "Missing essential environment variables: WEAVIATE_URL, WEAVIATE_API_KEY, or GROQ_API_KEY."
            )

        # Normalize URL scheme
        if self.weaviate_url and not self.weaviate_url.startswith(("http://", "https://")):
            object.__setattr__(self, "weaviate_url", f"https://{self.weaviate_url}")


# Global config instance
config = AppConfig()

