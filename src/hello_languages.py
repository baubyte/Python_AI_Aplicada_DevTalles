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
    {"role": "user", "content": "Dime hola en tres idiomas distintos."},
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
)
text_response = response.choices[0].message.content
print(text_response)
print(f"\n---Uso de Tokens---\n")
print(f"Tokens de entrada: {response.usage.prompt_tokens}")
print(f"Tokens de salida: {response.usage.completion_tokens}")
print(f"Total de tokens: {response.usage.total_tokens}")
cost_input = (response.usage.prompt_tokens / 1_000_000) * 0.15
cost_output = (response.usage.completion_tokens / 1_000_000) * 0.60
cost_total = cost_input + cost_output
print(f"Costo Estimado de entrada: ${cost_input:,.8f} USD")
print(f"Costo Estimado de salida: ${cost_output:,.8f} USD")
print(f"Costo Total Estimado: ${cost_total:,.8f} USD")
print(f"\nID de la respuesta: {response.id}")
print(f"\nModelo: {response.model}")
