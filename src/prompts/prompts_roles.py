from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def show_roles():
    """Obtiene los roles de asistente predefinidos y los muestra"""
    print("=" * 50)
    print("Role: User, (sin rol system)")
    print("=" * 50)

    response_1 = client.chat.completions.create(
       model="gpt-4o-mini",
       messages=[
           {"role": "user", "content": "¿Cuanto es 2 + 2?"}
       ]
    )
    print(f"Respuesta: {response_1.choices[0].message.content}\n")

    # Rol System
    print("=" * 50)
    print("Role: System")
    print("=" * 50)

    response_2 = client.chat.completions.create(
       model="gpt-4o-mini",
       messages=[
           {
               "role": "system",
               "content": """Eres un matemático gruñón que contesta preguntas simples con desdén o enojo pero posición absoluta. Siempre incluyes un comentario sobre lo básico que es la pregunta."""
           },
           {
               "role": "user",
               "content": "¿Cuanto es 2 + 2?"
        }
       ]
    )
    print(f"Respuesta: {response_2.choices[0].message.content}\n")

    # Rol asistente
    print("=" * 50)
    print("Role: Assistant")
    print("=" * 50)

    response_3 = client.chat.completions.create(
       model="gpt-4o-mini",
       messages=[
           {"role": "system", "content": """Eres un clasificador de sentimientos, tus únicas respuestas válidas son: positivo, negativo o neutral, no agregues nada más."""},
           {"role": "user", "content": "Me enojé mucho cuando me enteré que el servicio al cliente se había equivocado de nuevo."},
           {"role": "assistant", "content": "Negativo"},
           {"role": "user", "content": "Hoy es mi cumpleaños y me regalaron una PS5"},
           {"role": "assistant", "content": "Positivo"},
           {"role": "user", "content": "Los ingredientes principales de esta receta son: 1 kg de carne, 500 g de papas, 1 cebolla, 2 dientes de ajo, 1 pimiento verde, 2 tomates maduros, 200 ml de vino blanco, 50 ml de aceite de oliva, sal y pimienta al gusto."},
           {"role": "assistant", "content": "Neutral"},
           #{"role": "user", "content": "El gobierno anunció medidas económicas que podrían afectar a los pequeños empresarios."},
           {"role":"user","content":"Detesto los días de calor, odio el verano, el sol me hace daño"},
       ]
    )
    print(f"Sentimiento: {response_3.choices[0].message.content}\n")

if __name__ == "__main__":
    show_roles()