import time
import logging
import warnings

# Suppress internal LangChain / Pydantic deprecation warning noise in logs
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*allowed_objects.*")

from flask import Flask, render_template, request

from src.config import config
from src.database import get_retriever
from src.graph import create_medical_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask Application
app = Flask(__name__)

# Initialize compiled LangGraph workflow with database retriever
rag_graph = create_medical_graph(
    retriever=get_retriever(),
    model_name=config.groq_model,
    temperature=config.groq_temperature,
)


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

        # Invoke LangGraph state graph (visualized as a full DAG in LangSmith)
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
        host=config.host,
        port=config.port,
        debug=config.debug,
    )
