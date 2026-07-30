
from langchain_core.runnables import RunnableSerializable
from functools import partial
from langchain_chroma import Chroma
from langgraph.graph import END, START, StateGraph

from langchain_section.graphs.nodes import decide_retrieval_path, node_analyze, node_generate, node_retrieve
from langchain_section.graphs.states import RAGAgentState


def build_rag_agent(vectorstore: Chroma) -> RunnableSerializable:
    """Construir y compilar RAG agéntico

    Args:
        vectorstore (Chroma): Almacén de vectores.

    Returns:
        RunnableSerializable: Grafo compilado.
    """
    builder = StateGraph(RAGAgentState)

    node_retrieve_with_vs = partial(node_retrieve, vectorstore=vectorstore)

    builder.add_node("analyze", node_analyze)
    builder.add_node("retrieve", node_retrieve_with_vs)
    builder.add_node("generate", node_generate)

    builder.add_edge(START, "analyze")

    builder.add_conditional_edges(
        "analyze",
        decide_retrieval_path,
        {
            "retrieve": "retrieve",
            "generate": "generate"
        }

    )

    builder.add_edge("retrieve", "generate")

    builder.add_edge("generate", END)

    return builder.compile()
