
from rich.console import Console
from rich.panel import Panel

from src.prompts.zero_few_shot import run_zero_few_shot
from src.prompts.cot_prompts import run_chain_of_thought

console = Console()


def main():
    console.print(
        Panel.fit(
            "[bold cyan]Técnicas de prompting\n"
        )
    )

   #run_zero_few_shot()
    run_chain_of_thought()

    console.print("\n[bold green]Ejecución completada\n")


if __name__ == "__main__":
    main()