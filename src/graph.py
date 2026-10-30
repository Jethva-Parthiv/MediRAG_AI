import os
import logging
from typing import TypedDict, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from src.prompt import system_prompt

logger = logging.getLogger(__name__)


# 1. Define explicit State Schema for the RAG pipeline
class MedicalRagState(TypedDict):
    question: str
    documents: List[Document]
    context_text: str
    answer: str


def create_medical_graph(retriever, model_name: str = "openai/gpt-oss-120b", temperature: float = 0.4):
    """
    Build and compile a LangGraph state graph for the MediRAG chatbot.
    Nodes:
      - retrieve: fetches relevant chunks from Weaviate using query embeddings
      - format_context: converts retrieved documents into structured context text
      - generate: invokes the Groq LLM with system prompt and context
    """
    # LLM and Prompt initialization
    chat_model = ChatGroq(
        model=model_name,
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Node 1: Vector Retrieval
    def retrieve_node(state: MedicalRagState) -> dict:
        question = state["question"]
        logger.info("Executing Graph Node [retrieve] for: '%s...'", question[:50])
        docs = retriever.invoke(question)
        return {"documents": docs}

    # Node 2: Context Formatting
    def format_context_node(state: MedicalRagState) -> dict:
        docs = state.get("documents", [])
        logger.info("Executing Graph Node [format_context] with %d docs", len(docs))
        formatted_context = "\n\n".join(doc.page_content for doc in docs)
        return {"context_text": formatted_context}

    # Node 3: LLM Generation
    def generate_node(state: MedicalRagState) -> dict:
        question = state["question"]
        context_text = state.get("context_text", "")
        logger.info("Executing Graph Node [generate] with context length: %d chars", len(context_text))

        messages = prompt.format_messages(input=question, context=context_text)
        response = chat_model.invoke(messages)
        return {"answer": str(response.content)}

    # Assemble the State Graph
    workflow = StateGraph(MedicalRagState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("format_context", format_context_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "format_context")
    workflow.add_edge("format_context", "generate")
    workflow.add_edge("generate", END)

    # Compile the graph
    app_graph = workflow.compile()
    return app_graph

