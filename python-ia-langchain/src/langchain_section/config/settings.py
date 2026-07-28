"""Configuración general"""
import os
from dotenv import load_dotenv

load_dotenv()


class LangChainSettings:
    """
    """
    # Modelos
    CHAT_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str = os.getenv(
        "OPENAI_MODEL_EMBEDDING", "text-embedding-3-small")

    # Parámetros LLM
    DEFAULT_TEMPERATURE: float = float(
        os.getenv("LLM_DEFAULT_TEMPERATURE", 0.7))
    LOW_TEMPERATURE: float = float(os.getenv("LLM_LOW_TEMPERATURE", 0.1))
    MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", 3))

    # Rutas de persistencia
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "data/chat_history.db")
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./data/langchain_chroma")

    # RAG
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", 3))

    # Costos aproximados
    COST_INPUT_PER_MILLION: float = float(
        os.getenv("COST_INPUT_PER_MILLION", 0.15))
    COST_OUTPUT_PER_MILLION: float = float(
        os.getenv("COST_OUTPUT_PER_MILLION", 0.60))

    @classmethod
    def validate(cls) -> None:
        """
        Valida que las variables críticas estén configuradas.

        Raises:
            ValueError: Si la API key no está configurada.
        """

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY no está configurada",
                "Crea un archivo .env y pega tu APIKey"
            )


settings = LangChainSettings()
