
from src.prompts.news_extractors import run_news_extractor
from rich.console import Console
from rich.panel import Panel

from src.prompts.zero_few_shot import run_zero_few_shot
from src.prompts.cot_prompts import run_chain_of_thought
from src.prompts.prompt_template import run_prompt_templates
from src.prompts.json_mode import run_json_mode
from src.prompts.function_calling import run_chat_with_tools

console = Console()


def main():
    console.print(
        Panel.fit(
            "[bold cyan]Técnicas de prompting\n"
        )
    )

    #run_zero_few_shot()
    #run_chain_of_thought()
    #run_prompt_templates()
    #run_json_mode()
    #run_news_extractor()
    run_chat_with_tools("¿Qué clima hace en Posadas?")
    run_chat_with_tools("¿Qué clima hace en Corrientes?")
    run_chat_with_tools("¿Qué clima hace en Buenos Aires?")
    run_chat_with_tools("¿Cual es la capital de Misiones?")

    console.print("\n[bold green]Ejecución completada\n")


if __name__ == "__main__":
    main()