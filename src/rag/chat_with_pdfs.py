import sys

from rich.panel import Panel
from rich.table import Table
from src.rag.pipeline_rag import RAGPipeline
from pathlib import Path
from pypdf import PdfReader
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()
console = Console()


class PDFProcessor:
    """
    Clase para extraer texto de un PDF y obtener una lista de PDFs
    """

    @staticmethod
    def extract_text(pdf_path: Path) -> str:
        """Extrae el texto de un PDF y lo devuelve

        Args:
            pdf_path (Path): Ruta del PDF

        Returns:
            str: Texto extraído del PDF
        """
        try:
            reader = PdfReader(str(pdf_path))
            pages_text = []

            for page_number, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()

                if page_text and page_text.strip():
                    pages_text.append(f"[Página {page_number}]\n{page_text}")

            return "\n\n".join(pages_text)

        except Exception as e:
            console.print(f"[red]❌ Error leyendo {pdf_path.name}: {e}[/red]")
            return ""

    @staticmethod
    def get_pdfs(folder: Path) -> list[Path]:
        """
        Obtiene una lista de archivos PDF de una carpeta

        Args:
            folder (Path): Ruta de la carpeta

        Returns:
            list[Path]: Lista de archivos PDF
        """
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)

        return list(folder.glob("*.pdf"))


class IndexRegistry:
    """
    Clase para registrar pdf indexados
    """

    def __init__(self, registry_path: Path):
        """
        Inicializa el registro de PDFs indexados

        Args:
            registry_path (Path): Ruta del registro de PDFs indexados
        """
        self.path = registry_path
        self.registry: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        """
        Carga el registro de PDFs indexados desde un archivo
        a la memoria, guardando el nombre del archivo y su tamaño en bytes
        """
        if not self.path.exists():
            return

        with open(self.path, "r") as file:
            for line in file.read().splitlines():
                if not line.strip():
                    continue

                parts = line.rsplit(":", 1)  # manual.pdf:204700

                if len(parts) == 2:
                    name, size = parts
                    self.registry[name] = int(size)

    def save(self) -> None:
        """
        Guarda el registro de PDFs indexados en un archivo
        de memoria a disco, para persistencia de datos.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as file:
            for name, size in self.registry.items():
                file.write(f"{name}:{size}\n")  # manual.pdf:204700

    def is_indexed(self, pdf_path: Path) -> bool:
        """
        Comprueba si un PDF está indexado

        Args:
            pdf_path (Path): Ruta del PDF

        Returns:
            bool: True si el PDF está indexado, False en caso contrario
        """
        if pdf_path.name not in self.registry:
            return False

        current_size = pdf_path.stat().st_size
        return self.registry[pdf_path.name] == current_size

    def mark_indexed(self, pdf_path: Path) -> None:
        """
        Marca un PDF como indexado

        Args:
            pdf_path (Path): Ruta del PDF
        """
        self.registry[pdf_path.name] = pdf_path.stat().st_size

    @property
    def indexed_names(self) -> list[str]:
        """Obtiene los nombres de los PDFs indexados de forma ordenada por nombre"""
        return sorted(self.registry.keys())

    @property
    def count(self) -> int:
        """Obtiene la cantidad de PDFs indexados"""
        return len(self.registry)


class ChatWithPDFs:
    """
    Clase para chatear con PDFs
    """

    def __init__(self, pdf_folder: str = "data/files/pdfs") -> None:
        """
        Inicializa el chat con PDFs

        Args:
            pdf_folder (str): Ruta de la carpeta de PDFs
        """
        self.pdf_folder = Path(pdf_folder)
        self.processor = PDFProcessor()
        self.registry = IndexRegistry(Path("data/pdfs_indexados.txt"))

        self.rag = RAGPipeline(
            collection_name="mis_pdfs",
            db_path="./data/chromadb_pdfs"
        )

    def index_new_pdfs(self) -> int:
        """
        Indexa los nuevos PDFs

        Returns:
            int: Número de PDFs indexados
        """
        pdfs = self.processor.get_pdfs(self.pdf_folder)

        if not pdfs:
            console.print(
                f"\n[yellow]⚠️ No hay PDFs en '{self.pdf_folder}'[/yellow]\n"
                f"[dim]📁 Coloca archivos .pdf ahí y escribe 'reindexar'[/dim]"
            )
            return 0

        news_pdfs = [pdf for pdf in pdfs if not self.registry.is_indexed(pdf)]

        if not news_pdfs:
            console.print(
                f"[green]✅ Todos los PDFs ya están indexados"
                f"({len(pdfs)} archivos) [/green]"
            )

            return 0

        console.print(
            f"\n[bold cyan]🔄 Indexando {len(news_pdfs)} PDF(s) nuevo(s)...[/bold cyan]")

        indexed_count = 0

        for pdf_path in news_pdfs:
            console.print(
                f"⚙️ Procesando: [bold]{pdf_path.name}[/bold]", end=" ")
            text = self.processor.extract_text(pdf_path)

            if not text.strip():
                console.print(
                    "[red]❌ Sin texto extraíble[/red]\n"
                    "[dim](puede ser un PDF escaneado - necesitaría OCR)[/dim]"
                )
                continue

            num_chunks = self.rag.index_chunks(
                long_text=text,
                chunk_size=400,
                overlap=40,
                base_metadata={
                    "fuente": pdf_path.name,
                    "tipo": "pdf"
                }
            )

            console.print(f"[green] ✅ {num_chunks} fragmentos[/green]")
            self.registry.mark_indexed(pdf_path)
            indexed_count += 1

        self.registry.save()
        return indexed_count

    def show_status(self) -> None:
        """
        Muestra el estado del sistema
        """
        table = Table(title="📊 Estado del Sistema RAG", show_header=True)
        table.add_column("Métrica", style="cyan")
        table.add_column("Valor", style="green")

        table.add_row("📄 PDFs indexados", str(self.registry.count))
        table.add_row("🧩 Fragmentos en ChromaDB",
                      str(self.rag.collection.count()))
        table.add_row("📁 Carpeta de PDFs", str(self.pdf_folder))
        table.add_row("🧠 Modelo embeddings", "nvidia/nemotron-3-embed-1b:free")
        table.add_row("🤖 Modelo respuesta", "openrouter/free")

        console.print(table)

        if self.registry.indexed_names:
            console.print("\n[bold]📚 PDFs en la base de conocimiento:[/bold]")
            for name in self.registry.indexed_names:
                console.print(f" 📄 {name}")

    def chat(self) -> None:
        """
        Loop principal de chat interactivo.
        """

        console.print(Panel.fit(
            "[bold cyan]💬 Chat con tus PDFs 💡[/bold cyan]\n"
            "[dim]Comandos disponibles: 'estado' | 'reindexar' | 'salir'[/dim]",
            border_style="cyan"
        ))

        if self.rag.collection.count() == 0:
            console.print(
                "\n[yellow]⚠️  La base de conocimiento está vacía.[/yellow]\n"
                f"Coloca PDFs en [bold]{self.pdf_folder}[/bold] "
                "y escribe 'reindexar'\n"
            )

        while True:
            try:
                console.print()
                question = console.input(
                    "[bold green]👤 Tú: [/bold green]").strip()

                if not question:
                    continue

                if question.lower() in ("salir", "exit", "quit"):
                    console.print("[dim]👋 ¡Hasta luego![/dim]")
                    break

                if question.lower() == "estado":
                    self.show_status()
                    continue

                if question.lower() == "reindexar":
                    self.index_new_pdfs()
                    continue

                if self.rag.collection.count() == 0:
                    console.print(
                        "[red]No hay documentos indexados"
                        "Agrega PDFs y escribe 'reindexar'[/red]"
                    )
                    continue

                # RAG
                with console.status("[dim]🔍 Buscando en tus documentos...[/dim]"):
                    result = self.rag.answer(
                        question=question,
                        n_fragments=3,
                        verbose=False
                    )
                # Respuesta
                console.print()
                console.print(Panel(
                    result["respuesta"],
                    title="[bold blue]🤖 Asistente[/bold blue]",
                    border_style="blue"
                ))

                if result.get("fragmentos_usados"):
                    console.print("[dim]📚 Fuentes consultadas:[/dim]")
                    seen = set()

                    for fragment in result["fragmentos_usados"]:
                        source = fragment["metadata"].get(
                            "fuente", "desconocida")

                        if source not in seen:
                            console.print(
                                f"[dim]📌 {source}"
                                f"(similitud: {fragment['similitud']})[/dim]"
                            )
                            seen.add(source)

            except KeyboardInterrupt:
                console.print("[dim]👋 ¡Hasta luego![/dim]")
                break

            except Exception as e:
                console.print(f"[red]❌ Error inesperado: {e}[/red]")


def main():
    """
    Función principal
    """
    Path("data/files/pdfs").mkdir(parents=True, exist_ok=True)
    console.print("\n[bold]🚀 Iniciando Chat con PDFs...[/bold]\n")

    system = ChatWithPDFs(pdf_folder="data/files/pdfs")
    system.index_new_pdfs()
    system.show_status()
    system.chat()


if __name__ == "__main__":
    main()
