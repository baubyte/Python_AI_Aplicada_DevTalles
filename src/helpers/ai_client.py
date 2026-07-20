"""Helper para llamar a la IA y manejar errores"""
from multiprocessing import AuthenticationError
from openai import OpenAI, APIConnectionError, RateLimitError, APIConnectionError
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
def call_ai(messages: list, temperature: float = 0.1) -> str:
    """
        Llama a la IA y maneja errores
        Args:
            messages: Lista de mensajes a enviar a la IA
            temperature: Temperatura de la IA
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
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature
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