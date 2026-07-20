"""
Proyecto: CLI Chatbot
"""
from openai.types.chat import ChatCompletion
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuracion
MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """Eres un asistente técnico experto en Python e IA.
Eres directo, usas ejemplos de código cuando es relevante,
y respondes en el mismo idioma que el usuario.
Si no sabes algo, lo dices honestamente."""

# Costos de la API en USD
INPUT_COST_PER_MILLION = 0.15
OUTPUT_COST_PER_MILLION = 0.60

def main():
    """Función principal"""
    print("╔══════════════════════════════════════╗")
    print("║      Python IA Aplicada - Chatbot    ║")
    print("║  Escribe 'quit' o Ctrl+C para salir  ║")
    print("╚══════════════════════════════════════╝\n")

    bot = ChatBot()
    try:
        while True:
            try:
                user_input = input("👤 Tú: ")
            except EOFError:
                print("\n👋 ¡Adiós!")
                break

            if user_input.lower() in ["quit", "exit", "salir", "adiós"]:
                break
            if user_input.lower() == "/stats":
                bot.show_stats()
                continue
            if user_input.lower() == "/reset":
                bot.history = [bot.history[0]]
                print("🧹 Historial reiniciado")
                continue

            print(f"🤖 IA: ", end="", flush=True)
            try:
                response = bot.chat(user_input)
                print(f" {response}")
                print(f"\nTokens usado: {bot.total_tokens} | Costo: ${bot.total_cost}")
            except Exception as e:
                print(f"\n❌ Error: {e}")
    except KeyboardInterrupt:
        print("\n👋 ¡Adiós!")
    finally:
        bot.show_stats()

class ChatBot:
    def __init__(self, system_prompt:str=SYSTEM_PROMPT):
        self.client = OpenAI()
        self.model = MODEL
        self.system_prompt = system_prompt
        self.history:list[dict] = [
            {"role": "system", "content": system_prompt}
        ]
        self.total_tokens = 0;
        self.total_cost = 0.0;

    def chat(self, user_message:str):
        """Genera una respuesta usando OpenAI
            y actualiza el historial y costos.
            Args:
                user_message (str): Mensaje del usuario.
            Returns:
                str: Respuesta del chatbot.
        """
        self.history.append({
            "role": "user",
            "content": user_message
        })

        response:ChatCompletion = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            max_tokens= 1000,
            temperature= 0.7
        )

        ai_response = response.choices[0].message.content
        self.history.append({
            "role": "assistant",
            "content": ai_response
        })
        self._update_cost(response.usage)
        return ai_response

    def _update_cost(self, usage:dict) -> None:
        """Calcula los costos de tokens de una respuesta."""
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        input_cost = (prompt_tokens / 1_000_000) * INPUT_COST_PER_MILLION
        output_cost = (completion_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
        total_cost = input_cost + output_cost

        self.total_tokens += total_tokens
        self.total_cost += total_cost

        return None

    def show_stats(self):
        """Imprime el resumen de tokens"""
        print(f"\n{'-' * 40}")
        print("📊 Estadísticas de uso:")
        print(f"Tokens totales: {self.total_tokens}")
        print(f"Costo total: ${self.total_cost:.6f}")
        print(f"Turnos: {len(self.history) // 2}")
        print(f"{'-' * 40}")





if __name__ == "__main__":
    main()