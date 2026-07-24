from openai import OpenAI
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import os

load_dotenv()


def create_chroma_client(persist: bool = True) -> chromadb.Client:
    """Crea un cliente de ChromaDB

    Args:
        persist (bool, optional): Si es True, el cliente persistirá en disco. Defaults to True.

    Returns:
        chromadb.Client: Cliente de  ChromaDB
    """
    if persist:
        return chromadb.PersistentClient(path="./data/chromadb")
    else:
        return chromadb.EphemeralClient()


class OpenRouterEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "nvidia/nemotron-3-embed-1b:free"):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENAI_API_KEY") or os.getenv(
                "OPENROUTER_API_KEY")
        )

    def __call__(self, input: Documents) -> Embeddings:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=input,
            encoding_format="float"
        )
        return [data.embedding for data in response.data]


def create_collection(client, name: str) -> chromadb.Collection:
    """Crea una colección en ChromaDB

    Args:
        client (chromadb.Client): Cliente de ChromaDB
        name (str): Nombre de la colección

    Returns:
        chromadb.Collection: Colección de ChromaDB
    """
    embedding_function = OpenRouterEmbeddingFunction(
        model_name="nvidia/nemotron-3-embed-1b:free"
    )

    collection = client.get_or_create_collection(
        name=name,
        embedding_function=embedding_function,
        metadata={"description": "Base de conocimiento del curso"}
    )

    return collection


def add_documents(collection: chromadb.Collection, documents: list[dict]) -> None:
    """
    Agrega documentos a la colección

    Args:
        collection (chromadb.Collection): Colección de ChromaDB
        documents (list[dict]): Lista de documentos a agregar

    """
    collection.add(
        ids=[doc["id"] for doc in documents],
        documents=[doc["texto"] for doc in documents],
        metadatas=[doc["metadata"] for doc in documents],
    )

    print(f"💾 OK {len(documents)} documentos agregados a ChromaDB")


def search_similar(collection: chromadb.Collection, question: str, n_results: int = 3) -> list[dict]:
    """
    Busca los documentos más relevantes en la colección

    Args:
        collection (chromadb.Collection): Colección de ChromaDB
        question (str): Pregunta a buscar
        n_results (int, optional): Número de resultados a buscar. Defaults to 3.

    Returns:
        list[dict]: Lista de documentos más relevantes
    """
    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    formatted_docs = []

    for i in range(len(results["documents"][0])):
        formatted_docs.append({
            "texto": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "similitud": round(1-(results["distances"][0][i]), 3)
        })

    return formatted_docs


KNOWLEDGE_BASE = [
    {
        "id": "doc_001",
        "texto": "Para reiniciar el servidor Nginx en Ubuntu ejecuta: sudo systemctl restart nginx. Verifica el estado con: sudo systemctl status nginx.",
        "metadata": {"fuente": "manual_ops.pdf", "seccion": "Servidores", "pagina": 12}
    },
    {
        "id": "doc_002",
        "texto": "Las variables de entorno se configuran en el archivo .env en la raíz del proyecto. Nunca subas el archivo .env a Git. Usa .env.example como plantilla.",
        "metadata": {"fuente": "guia_dev.pdf", "seccion": "Configuración", "pagina": 3}
    },
    {
        "id": "doc_003",
        "texto": "El límite de rate en nuestra API es de 1000 requests por minuto por usuario. Si lo superas recibirás un error 429. Implementa exponential backoff en el cliente.",
        "metadata": {"fuente": "api_docs.pdf", "seccion": "Rate Limits", "pagina": 8}
    },
    {
        "id": "doc_004",
        "texto": "Para hacer deploy a producción: 1) Corre los tests con pytest, 2) Build la imagen Docker, 3) Push al registry, 4) Aplica el helm chart con kubectl.",
        "metadata": {"fuente": "deploy_guide.pdf", "seccion": "DevOps", "pagina": 22}
    },
    {
        "id": "doc_005",
        "texto": "La base de datos PostgreSQL corre en el puerto 5432. Las credenciales están en Vault bajo el path secret/prod/postgres. Nunca uses las credenciales de prod en local.",
        "metadata": {"fuente": "infra_docs.pdf", "seccion": "Base de Datos", "pagina": 5}
    },
    {
        "id": "doc_006",
        "texto": "Para restaurar un backup de la base de datos: pg_restore -U postgres -d mydb backup.dump. Los backups se generan automáticamente cada noche a las 2am UTC.",
        "metadata": {"fuente": "infra_docs.pdf", "seccion": "Base de Datos", "pagina": 7}
    },
]


if __name__ == "__main__":
    print("="*40)
    print("🤖 RAG con ChromaDB")
    print("="*40)

    # 1. Crear cliente
    client = create_chroma_client()
    collection = create_collection(client, "base_conocimientos_baubyte")

    # 2. Agregar documentos
    if collection.count() == 0:
        print("\n🔵 Primera ejecución: Indexando documentos...")
        add_documents(collection, KNOWLEDGE_BASE)
    else:
        print(f"\n🔵 Colección existente con {collection.count()} documentos")

    # 3. Buscar respuestas
    test_questions = [
        # "¿Cómo reinicio el servidor web?",
        # "¿Dónde están las credenciales de la base de datos?",
        # "¿Cómo hago deploy a producción?",
        # "¿Qué pasa si hago demasiadas llamadas a la API?",
        "Mi web app dejó de responder",
        "Olvidé dónde guardamos los passwords",
        "Quiero publicar mi código en vivo",
    ]

    print("\n")
    print("="*40)
    print("🔍 Búsqueda semántica")

    for question in test_questions:
        print(f"\n👤 Pregunta: {question}")

        results = search_similar(collection, question)

        for i, doc in enumerate(results, 1):
            print(
                f"\n🤖 #{i} Similitud: {doc['similitud']}"
                f"\n📄 Fuente: {doc['metadata']['fuente']}"
                f"\n📋 (pág. {doc['metadata']['pagina']})"
            )

            print(f"{doc['texto'][:120]}...")
