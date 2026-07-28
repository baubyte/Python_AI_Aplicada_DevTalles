"""Demo LCEL"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_section.core.llm import get_llm
# from src.langchain_section.config.settings import settings


llm = get_llm(0.1)


# código aquí
def demo_simple_chain() -> None:
    """
    Demuestra el funcionamiento de una cadena simple con LCEL (LangChain Expression Language).

    Returns:
        None
    """

    # ChatPromptTemplate (role, contenido)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Eres un experto en {tema}. Responde de forma concisa, máximo 3 oraciones"),
        ("human", "{pregunta}")
    ])

    # StrOutParser
    parser = StrOutputParser()

    # Cadena (Chain)->prompt -> llm -> parser
    chain = prompt | llm | parser

    # Invocamos la cadena con invoke
    response = chain.invoke({
        "tema": "Python",
        "pregunta": "¿Qué es un decorador?"
    })

    print("Simple Chain")
    print(response)
    print()


def demo_steps_inspection() -> None:
    """
    Demuestra el funcionamiento de cada componente de la cadena.

    Returns:
        None
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente técnico."),
        ("human", "{pregunta}")
    ])

    parser = StrOutputParser()

    input_message = {"pregunta": "¿Qué es una API REST?"}

    # Paso 1 prompt
    messages = prompt.invoke(input_message)
    print("Paso 1 - Prompt output")
    print(f" Tipo: {type(messages).__name__}")
    print(f" Mensajes: {messages.messages}")

    # Paso 2 LLM -> AIMessage
    ai_message = llm.invoke(messages)
    print("Paso 2 - LLM output")
    print(f" Tipo: {type(ai_message).__name__}")
    print(f" AIMessage: {ai_message} ")
    print(f" Contenido: {ai_message.content[:100]}...")

    # Paso 3: Parser extraer texto
    text = parser.invoke(ai_message)
    print("Paso 3 - Parser output")
    print(f" Tipo: {type(text).__name__}")
    print(f" Texto: {text[:100]}...")
    print()


def demo_batch() -> None:
    """
    Demuestra el funcionamiento del método batch.

    Returns:
        None
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Clasifica el sentimiento. Responde SOLO: POSITIVO, NEGATIVO o NEUTRO"),
        ("human", "{texto}")
    ])

    parser = StrOutputParser()

    # chain = prompt -> llm -> parser
    chain = prompt | llm | parser

    inputs = [
        {"texto": "Me encanta este framework, es increíble."},
        {"texto": "El servidor estuvo caído 3 horas!! es inaceptable"},
        {"texto": "La versión 2.0 ya está disponible"},
        {"texto": "Perdí todos mis datos por un bug crítico."},
        {"texto": "La documentación es bastante clara."},
    ]

    results = chain.batch(inputs)

    print("BATCH PROCESSING:")
    for input_message, result in zip(inputs, results):
        print(f" [{result}] {input_message['texto'][:50]}...")
    print()


def demo_steaming() -> None:
    """
    Demuestra el funcionamiento del método stream.

    Returns:
        None
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Explica conceptos técnicos de forma clara."),
        ("human", "Explica que es {concepto} en 2 párrafos.")
    ])

    parser = StrOutputParser()

    # chain = prompt -> llm -> parser
    chain = prompt | llm | parser

    print("STREAMING EN TIEMPO REAL.")
    print("IA: ", end="", flush=True)

    for chunk in chain.stream({"concepto": "La ventana de contexto en LLMs"}):
        print(chunk, end="", flush=True)

    print("\n")


def demo_passthrough() -> None:
    """Passthrough"""
    # Simula retriever
    def search_context(question: str) -> str:
        contexts = {
            "python": "Python fue creado por Guido Van Rossum en 1991.",
            "langchain": "LangChain es un framework para aplicaciones con LLMs.",
            "devtalles": "Una plataforma muy cool con instructores guapos.",
        }

        for keyword, ctx in contexts.items():
            if keyword.lower() in question.lower():
                return ctx
        return "No se encontró contexto relevante."

    retriever = RunnableLambda(search_context)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Responde usando este contexto: \n{context}"),
        ("human", "{question}")
    ])

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke("¿Qué es Devtalles?")
    print("PASSTHROUGH DEMO: ")
    print(response)
    print()


if __name__ == "__main__":
    print("="*60)
    print("LangChain LCEL - Fundamentos")
    # demo_simple_chain()
    # demo_steps_inspection()
    # demo_batch()
    # demo_steaming()
    demo_passthrough()
    print("="*60)
