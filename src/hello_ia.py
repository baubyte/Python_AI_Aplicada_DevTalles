"""
Autor: Baubyte
Fecha: 2026-07-20
Descripción: Ejemplo de uso de la API de OpenAI para generar texto.
"""
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

messages = [
    {"role": "user", "content": "Explica que es una API en una sola oración."},
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)
text_response = response.choices[0].message.content
print(text_response)
