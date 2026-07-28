"""Cliente LLM centralizado"""
from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_section.config.settings import settings


@lru_cache(maxsize=1)
def get_llm(temperature: float = None) -> ChatOpenAI:
    """
    Obtiene una instancia en cache del modelo de lenguaje.

    Args:
        temperature (float, optional): Temperatura del modelo. Defaults to None.

    Returns:
        ChatOpenAI: Instancia del modelo de lenguaje.
    """
    return ChatOpenAI(
        model=settings.CHAT_MODEL,
        temperature=temperature or settings.DEFAULT_TEMPERATURE,
        max_retries=settings.MAX_RETRIES,
        base_url=settings.OPENAI_BASE_URL
    )


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """
    Obtiene una instancia en cache del cliente de embeddings.
    Returns:
        OpenAIEmbeddings: Instancia del cliente de embeddings.
    """
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL, base_url=settings.OPENAI_BASE_URL)
