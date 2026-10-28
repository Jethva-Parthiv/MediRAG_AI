import os
from dotenv import load_dotenv
import weaviate
from weaviate.auth import Auth
from langchain_weaviate import WeaviateVectorStore
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings

load_dotenv()

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "").strip()
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY", "").strip()

if not WEAVIATE_URL or not WEAVIATE_API_KEY:
    raise ValueError("Missing WEAVIATE_URL or WEAVIATE_API_KEY in .env")

if not WEAVIATE_URL.startswith("http"):
    WEAVIATE_URL = f"https://{WEAVIATE_URL}"

# 1. Extract and chunk PDF documents
print("Loading PDF documents from data/ ...")
extracted_data = load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)
print(f"Total text chunks created: {len(text_chunks)}")

# 2. Load HuggingFace Embeddings model (384 dimensions)
print("Loading HuggingFace embedding model...")
embeddings = download_hugging_face_embeddings()

# 3. Connect to Weaviate Cloud
print(f"Connecting to Weaviate Cloud at {WEAVIATE_URL}...")
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY)
)

index_name = "MediragAiDb"

# 4. Recreate collection allowing server default vector index (hfresh)
if client.collections.exists(index_name):
    print(f"Collection '{index_name}' already exists. Deleting old instance...")
    client.collections.delete(index_name)

print(f"Creating collection '{index_name}'...")
client.collections.create(
    name=index_name,
    vectorizer_config=None
)

try:
    print(f"Indexing {len(text_chunks)} chunks into Weaviate collection '{index_name}'...")
    docsearch = WeaviateVectorStore(
        client=client,
        index_name=index_name,
        embedding=embeddings,
        text_key="text"
    )
    docsearch.add_documents(text_chunks)
    print("🎉 Vector indexing into Weaviate Cloud completed successfully!")
finally:
    client.close()