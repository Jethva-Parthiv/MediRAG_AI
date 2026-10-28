import os
import weaviate
from weaviate.auth import Auth
from dotenv import load_dotenv
from flask import Flask, render_template, request
from langchain_weaviate import WeaviateVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import download_hugging_face_embeddings
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
)

# 2. Download embeddings model
embeddings = download_hugging_face_embeddings()

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
        response = rag_chain.invoke({"input": msg})
        print("Response: ", response["answer"])
        return str(response["answer"])
    except Exception as e:
        print("Error:", repr(e))
        return f"Sorry, something went wrong: {e}", 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)