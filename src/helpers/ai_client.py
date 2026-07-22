"""Helper para llamar a la IA y manejar errores"""
from openai import omit
from multiprocessing import AuthenticationError
from openai import OpenAI, APIConnectionError, RateLimitError, APIConnectionError
from dotenv import load_dotenv

load_dotenv()
# base_url="https://router.bynara.id/v1"
base_url = "https://openrouter.ai/api/v1"
client = OpenAI(base_url=base_url)


def call_ai(messages: list, temperature: float = 0.1, response_format: str = "text") -> str:
    """
        Llama a la IA y maneja errores
        Args:
            messages: Lista de mensajes a enviar a la IA
            temperature: Temperatura de la IA
            response_format: Formato de respuesta
        Returns:
            Respuesta de la IA
        Raises:
            AuthenticationError: Si la API Key es inválida
            RateLimitError: Si se excede el límite de requests
            APIConnectionError: Si hay un error de conexión
            Exception: Si hay cualquier otro error
    """
    try:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=messages,
            temperature=temperature,
            response_format={"type": response_format}
        )
        return response.choices[0].message.content
    except AuthenticationError:
        print("Revisá la API Key")
        raise SystemExit(1)
    except RateLimitError as e:
        print("Llegaste al límite de requests")
        raise
    except APIConnectionError:
        print("Error de conexión. Revisá la red")
        raise
    except Exception as e:
        print(f"Error inesperado: {e}")
        raise


def call_ai_tools(messages: list, temperature: float = 0.1, response_format: str = "text", tools: list = omit, tool_choice: str = omit) -> str:
    """
    Llama a la IA con herramientas y maneja errores
    Args:
        messages: Lista de mensajes a enviar a la IA
        temperature: Temperatura de la IA
        response_format: Formato de respuesta
        tools: Herramientas a usar
        tool_choice: Elección de herramientas
    Returns:
        Respuesta de la IA
    Raises:
        AuthenticationError: Si la API Key es inválida
        RateLimitError: Si se excede el límite de requests
        APIConnectionError: Si hay un error de conexión
        Exception: Si hay cualquier otro error
    """
    try:
        response = client.chat.completions.create(
            model="openrouter/free",
            messages=messages,
            temperature=temperature,
            response_format={"type": response_format},
            tools=tools,
            tool_choice=tool_choice
        )
    except AuthenticationError:
        print("Revisá la API Key")
        raise SystemExit(1)
    except RateLimitError as e:
        print("Llegaste al límite de requests")
        raise
    except APIConnectionError:
        print("Error de conexión. Revisá la red")
        raise
    except Exception as e:
        print(f"Error inesperado: {e}")
        raise

    return response.choices[0].message
