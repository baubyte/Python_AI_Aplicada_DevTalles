

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from langchain_section.core.llm import get_llm


def build_assistant_chain(system_prompt: str = None) -> Runnable:
    """Construye una cadena base del asistente con soporte en el historial

    Args:
        system_prompt (str, optional): Descripción del rol del asistente.
            Defaults to None (usa el default_system).

    Returns:
        Runnable: Cadena LangChain lista para usar
    """
    default_system = """Eres un asistente técnico experto en Python e IA.
Tienes acceso al historial completo de esta conversación.
Úsalo para dar respuestas contextuales y coherentes.
Si el usuario hace referencia a algo anterior, recuérdalo."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt or default_system),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    llm = get_llm()
    parser = StrOutputParser()

    return prompt | llm | parser
