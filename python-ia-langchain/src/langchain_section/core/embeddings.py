
import os

from langchain_chroma import Chroma

from langchain_section.core.llm import get_embeddings
from langchain_section.config.settings import settings


def get_or_create_vectorstore(
    collection_name: str,
    persist_path: str = None
) -> Chroma:
    """Obtener un vector store existente o crearlo si no existe

    Args:
        collection_name (str): Nombre de la colección
        persist_path (str, optional): Path donde se encuentra el vector store.
        Defaults a settings.CHROMA_PATH.

    Returns:
        Chroma: Vector store
    """
    path = persist_path or settings.CHROMA_PATH
    os.makedirs(path, exist_ok=True)

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=path,
        collection_metadata={"hnsw:space": "cosine"},
    )
