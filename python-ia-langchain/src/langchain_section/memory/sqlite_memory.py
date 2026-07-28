

import os

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from sqlalchemy import column, create_engine, select, table
from sqlalchemy.exc import OperationalError

from langchain_section.config.settings import settings
from langchain_section.memory.base import BaseMemoryBackend


class SQLiteMemoryBackend(BaseMemoryBackend):
    """Backend de memoria persistente usando SQLite

    Args:
        BaseMemoryBackend (BaseMemoryBackend): base memory backend
    """

    def __init__(self, db_path: str = None):
        """Inicializa el backend de memoria SQLite

        Args:
            db_path (str, optional): Ruta de la base de datos.
                                    por defecto usa el valor de
                                    settings.SQLITE_DB_PATH.
        """
        self.db_path = db_path or settings.SQLITE_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._connection_string = f"sqlite:///{self.db_path}"
        self.engine = create_engine(self._connection_string)

    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        """Retorna el historial de mensajes para una sesión

        Args:
            session_id (str): ID de la sesión

        Returns:
            BaseChatMessageHistory: Historial de mensajes para una sesión específica
        """
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=self._connection_string
        )

    def clear_history(self, session_id: str) -> None:
        """Elimina el historial de una sesión

        Args:
            session_id (str): ID de la sesión

        Returns:
            None
        """
        history = self.get_history(session_id)
        history.clear()

    def list_sessions(self) -> list[str]:
        """Lista todos los session_ids disponibles

        Returns:
            list[str]: Lista de session_ids disponibles
        """
        message_store_table = table("message_store", column("session_id"))
        stmt = select(message_store_table.c.session_id).distinct()

        try:
            with self.engine.connect() as conn:
                result = conn.execute(stmt)
                return [row[0] for row in result]
        except OperationalError:
            return []

