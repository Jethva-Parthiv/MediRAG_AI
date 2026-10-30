import os
import time
import atexit
import weaviate
from weaviate.auth import Auth
from dotenv import load_dotenv
from flask import Flask, render_template, request
from langchain_weaviate import WeaviateVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import get_query_embeddings
from src.prompt import system_prompt

app = Flask(__name__)

load_dotenv()

WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not WEAVIATE_URL or not WEAVIATE_API_KEY or not GROQ_API_KEY:
    raise ValueError("Missing WEAVIATE_URL, WEAVIATE_API_KEY, or GROQ_API_KEY in .env")

if not WEAVIATE_URL.startswith("http"):
    WEAVIATE_URL = f"https://{WEAVIATE_URL}"

# 1. Connect to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    skip_init_checks=True,
)


# Gracefully close the Weaviate client when worker/process terminates
def _close_weaviate():
    try:
        if client and client.is_connected():
            client.close()
    except Exception:
        pass


atexit.register(_close_weaviate)

# 2. Fast direct Gemini embeddings model for user queries
embeddings = get_query_embeddings()

index_name = "MediragAiDb"

# 3. Connect to existing Weaviate vector store
docsearch = WeaviateVectorStore(
    client=client,
    index_name=index_name,
    text_key="text",
    embedding=embeddings,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 4. Groq LLM and Prompt setup
chatModel = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    try:
        t_total_start = time.perf_counter()

        # Step 1: Retrieval (Portkey Query Embedding + Weaviate Vector Search)
        t_retrieval_start = time.perf_counter()
        docs = retriever.invoke(msg)
        t_retrieval = time.perf_counter() - t_retrieval_start

        # Step 2: Generation (Groq LLM Synthesis)
        t_llm_start = time.perf_counter()
        answer = question_answer_chain.invoke({"input": msg, "context": docs})
        t_llm = time.perf_counter() - t_llm_start

        t_total = time.perf_counter() - t_total_start

        print(f"\n⏱️ --- Latency Breakdown for: '{msg[:40]}...' ---")
        print(f"  🔍 Retrieval (Direct Google AI Embed + Weaviate Search): {t_retrieval:.3f}s")
        print(f"  🧠 LLM Generation (Groq Cloud):                         {t_llm:.3f}s")
        print(f"  ⚡ Total Turnaround Time:                               {t_total:.3f}s")
        print(f"Response: {answer}\n")

        return str(answer)
    except Exception as e:
        print("Error:", repr(e))
        return f"Sorry, something went wrong: {e}", 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)