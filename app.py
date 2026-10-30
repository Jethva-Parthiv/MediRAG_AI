import os
import time
import atexit
import logging
import warnings

# Suppress internal LangChain / Pydantic deprecation warning noise in logs
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*allowed_objects.*")

import weaviate
from weaviate.auth import Auth
from dotenv import load_dotenv
from flask import Flask, render_template, request
from langchain_weaviate import WeaviateVectorStore

from src.helper import get_query_embeddings
from src.graph import create_medical_graph

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Environment configuration
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not all([WEAVIATE_URL, WEAVIATE_API_KEY, GROQ_API_KEY]):
    raise EnvironmentError(
        "Missing WEAVIATE_URL, WEAVIATE_API_KEY, or GROQ_API_KEY in .env"
    )

if not WEAVIATE_URL.startswith(("http://", "https://")):
    WEAVIATE_URL = f"https://{WEAVIATE_URL}"


# Weaviate Cloud connection
client = None


def close_weaviate():
    global client
    if client is not None:
        try:
            if client.is_connected():
                client.close()
                logger.info("Weaviate connection closed.")
        except Exception:
            logger.exception("Error while closing Weaviate connection.")


def initialize_weaviate():
    global client
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        skip_init_checks=True,
    )
    if not client.is_connected():
        raise ConnectionError("Unable to establish Weaviate connection.")
    logger.info("Connected to Weaviate Cloud.")


atexit.register(close_weaviate)


# Initialize embeddings
embeddings = get_query_embeddings()

# Initialize vector store
INDEX_NAME = "MediragAiDb"
initialize_weaviate()

docsearch = WeaviateVectorStore(
    client=client,
    index_name=INDEX_NAME,
    text_key="text",
    embedding=embeddings,
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Initialize compiled LangGraph workflow
rag_graph = create_medical_graph(retriever=retriever)


# Flask routes
@app.route("/", methods=["GET"])
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()

    if not msg:
        return "Please enter a question.", 400

    try:
        start_time = time.perf_counter()

        # Invoke LangGraph state graph (visualized as a full graph DAG in LangSmith)
        final_state = rag_graph.invoke(
            {"question": msg},
            config={"run_name": "MediRAG_Graph_Execution"}
        )

        answer = final_state.get("answer", "No answer generated.")
        total_time = time.perf_counter() - start_time

        logger.info(
            "Query: %s | End-to-end graph latency: %.3fs",
            msg[:100],
            total_time,
        )

        return answer

    except Exception:
        logger.exception("LangGraph pipeline execution failed.")
        return (
            "Sorry, something went wrong while processing your question.",
            500,
        )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=False,
    )
