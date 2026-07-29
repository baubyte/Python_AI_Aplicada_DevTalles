
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_section.config.settings import settings

LOADERS = {
    ".pdf": lambda path: PyPDFLoader(str(path)),
    ".txt": lambda path: TextLoader(str(path), encoding="utf-8"),
}


def get_supported_extensions() -> list[str]:
    """Retorna la lista de extensiones soportadas basadas en los loaders disponibles

    Returns:
        list[str]: Lista de extensiones soportadas
    """
    return list(LOADERS.keys())


def load_file(file_path: Path) -> list[Document]:
    """Carga un archivo y retorna una lista de Documents de Langchain

    Args:
        file_path (Path): Path del archivo

    Returns:
        list[Document]: Lista de documents

    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si la extensión del archivo no es soportada
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

    extension = file_path.suffix.lower()

    if extension not in LOADERS:
        supported_str = ", ".join(get_supported_extensions())
        raise ValueError(
            f"Extensión '{extension}' no soportada. Usar: {supported_str}"
        )

    print(f" Cargando: {file_path.name} -> ", end="")

    loader_factory = LOADERS[extension]
    loader = loader_factory(file_path)
    docs = loader.load()

    for doc in docs:
        doc.metadata.update(
            {
                "source": str(file_path),
                "file_name": file_path.name,
                "file_type": extension.lstrip("."),
            }
        )

    print(f"{len(docs)} sección/es")

    return docs


def load_directory(directory_path: Path) -> list[Document]:
    """Carga todos los archivos soportados en una carpeta

    Args:
        directory_path (Path): Path del directorio

    Returns:
        list[Document]: Lista de documents
    """
    supported_extensions = get_supported_extensions()
    supported_str = ", ".join(supported_extensions)
    all_files = []

    directory_path.mkdir(parents=True, exist_ok=True)

    for extension in supported_extensions:
        all_files.extend(directory_path.glob(f"*{extension}"))
        all_files.extend(directory_path.glob(f"*{extension.upper()}"))

    all_files = list(set(all_files))

    if not all_files:
        print(f" No se encontraron archivos en: {directory_path}")
        print(f"Agrega archivos ({supported_str}) y vuelve a ejecutar")
        return []

    print(f"Archivos encontrados en {directory_path} : ")
    all_docs = []
    errors = []

    for file_path in sorted(all_files):
        try:
            docs = load_file(file_path)
            all_docs.extend(docs)
        except Exception as e:
            errors.append((file_path.name, str(e)))
            print(f"Error cargando {file_path.name}: {e}")

    if errors:
        print(
            f"\n ❌{len(errors)} archivos(s) con error, ✅{len(all_docs)} documento(s) cargados.")
    else:
        print(
            f"\n ✅{len(all_files)} archivo(s) cargados -> {len(all_docs)} sección/es totales")

    return all_docs


def split_documents(
    docs: list[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[Document]:
    """Divide los documentos en chunks para indexación

    Args:
        docs (list[Document]): Lista de documents
        chunk_size (int, optional): Tamaño de los chunks. Defaults a settings.CHUNK_SIZE.
        chunk_overlap (int, optional): Overlap entre chunks. Defaults a settings.CHUNK_OVERLAP.

    Returns:
        list[Document]: Lista de chunks
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )

    chunks = splitter.split_documents(docs)

    print(f"{len(docs)} sección/es -> {len(chunks)} chunks"
          f"(tamaño: ~{chunk_size or settings.CHUNK_SIZE} chars, "
          f"Overlap: {chunk_overlap or settings.CHUNK_OVERLAP})")

    return chunks
