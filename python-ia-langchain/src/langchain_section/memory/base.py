
from abc import ABC, abstractmethod
from langchain_core.chat_history import BaseChatMessageHistory


class BaseMemoryBackend(ABC):
    """
    Interfaz común que deben implementar todos los backends de memoria.

    Define los métodos que deben estar disponibles en cualquier implementación de memoria:
    - obtener historial
    - limpiar historial
    - listar sesiones
    """

    @abstractmethod
    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        """Retorna el historial de mensaje para una sesión específica.

        Args:
            session_id (str): ID de la sesión

        Returns:
            BaseChatMessageHistory: Historial de mensaje para una sesión específica
        """

    @abstractmethod
    def clear_history(self, session_id: str) -> None:
        """Elimina el historial de una sesión

        Args:
            session_id (str): ID de la sesión

        Returns:
            None
        """

    @abstractmethod
    def list_sessions(self) -> list[str]:
        """Lista todos los session_ids disponibles

        Returns:
            list[str]: Lista de session_ids disponibles
        """
