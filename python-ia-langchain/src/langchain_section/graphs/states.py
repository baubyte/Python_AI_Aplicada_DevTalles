
from typing import Annotated, TypedDict

from langgraph.graph import add_messages


class RAGAgentState(TypedDict):
    """Estados

        - messages: Lista de mensajes del chat.
        - question: Pregunta del usuario.
        - retrieved_docs: Documentos recuperados del almacén de vectores.
        - response: Respuesta del agente.
        - needs_retrieval: Booleano que indica si se necesita recuperar documentos.
        - sources: Fuentes de los documentos recuperados.
    """
    messages: Annotated[list, add_messages]
    question: str
    retrieved_docs: list[str]
    response: str
    needs_retrieval: bool
    sources: list[dict]
