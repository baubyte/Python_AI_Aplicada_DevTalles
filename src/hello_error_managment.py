from multiprocessing import AuthenticationError
from openai import OpenAI, APIConnectionError, RateLimitError, APIConnectionError
from dotenv import load_dotenv

load_dotenv()

def call_ai(question: str) -> str:
    """Llama a la IA y maneja errores"""
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": question
                },
            ],
            max_tokens=500,
            temperature=0.7, #0= Determinístico, 1 = Creativo, 2= Caótico
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


if __name__ == "__main__":
    response = call_ai("¿Cuál es la capital de Argentina?")
    print(f"AI: {response}")