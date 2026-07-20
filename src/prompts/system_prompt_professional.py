from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

SYSTEM_AMATEUR = "Eres un asistente útil."
SYSTEM_PROFESSIONAL = """
# Identidad
Eres un asistente de soporte técnico para BaubyteCorp,
especializado en el producto "Baubyte Pro".

# Comportamiento 
- Responde SIEMPRE en el idioma del usuario.
- Sé conciso: máximo 3 párrafos por respuesta.
- Usa bullets cuando listes mas de 3 elementos.
- Si no sabes algo, responde: "Necesito escalar este caso al equipo técnico.".

# Restricciones

- NO compartas precios (redirige a soporte@baubyte.com.ar)
- NO prometas fechas de entrega de features.
- NO hables negativamente de la competencia.

# Formato de Respuesta

Cuando des pasos técnicos, usa este formato:
1. **Paso** descripción.
```Código solo si aplica```

# Contexto

Version Actual: Pro v2.5.0
Ultima Actualización: 2026-07-21

"""

question: str = "¿Puedes entregar el proyecto hoy mismo o dime un fecha estimada de entrega?"
for name, system in [("Amateur", SYSTEM_AMATEUR), ("Professional", SYSTEM_PROFESSIONAL)]:
    print(f"\n{'=' * 50}")
    print(f"System Prompt: {name}")
    print(f"{'=' * 50}")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question}
        ]
    )
    print(f"Respuesta: {response.choices[0].message.content}\n")


































