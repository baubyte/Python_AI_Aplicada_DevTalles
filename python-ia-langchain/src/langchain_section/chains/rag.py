
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from langchain_section.core.llm import get_llm
from langchain_section.config.settings import settings


def format_docs(docs: list[Document]) -> str:
    """Convierte lista de documents en texto para el prompt

    Args:
        docs (list[Document]): Lista de documents

    Returns:
        str: Texto formateado para el prompt
    """
    return "\n\n---\n\n".join([
        f"[Fuente: {doc.metadata.get('source', 'desconocida')},"
        f"Página: {doc.metadata.get('page', "N/A")}]\n {doc.page_content}"
        for doc in docs
    ])


def build_rag_chain(vectorstore: Chroma) -> tuple[Runnable, object]:
    """Pipeline RAG con LCEL (LangChain Expression Language)

    Args:
        vectorstore (Chroma): Vector store

    Returns:
        tuple[Runnable, object]: Runnable y retriever
    """

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.TOP_K_RESULTS}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un asistente que responde preguntas
basándote ÚNICAMENTE en el contexto proporcionado.

Contexto recuperado de los documentos:
{context}

Instrucciones:
- Si la respuesta está en el contexto, respóndela con precisión.
- Si no está, di: "No encontré esa información en los documentos."
- Cita la fuente cuando sea posible.
- No inventes ni supongas información."""),
        ("human", "{question}")
    ])
    llm = get_llm(temperature=settings.LOW_TEMPERATURE)
    parser = StrOutputParser()

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        } | prompt | llm | parser
    )

    return rag_chain, retriever
