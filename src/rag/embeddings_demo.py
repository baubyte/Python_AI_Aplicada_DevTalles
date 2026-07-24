import math
import os
import pathlib
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
base_url = "https://openrouter.ai/api/v1"
clientOpenAI = OpenAI(base_url=base_url)


def get_embedding(text: str) -> list[float]:
    """
    Genera un embedding para el texto de entrada utilizando el modelo nvidia/llama-nemotron-embed-vl-1b-v2:free.

    Args:
        text (str): Texto a codificar.

    Returns:
        list[float]: Embedding del texto en formato float32.
    """
    response = clientOpenAI.embeddings.create(
        model='nvidia/nemotron-3-embed-1b:free',
        input=text,
        encoding_format='float',
    )
    return response.data[0].embedding


def cosine_similarity(embedding_1: list[float], embedding_2: list[float]) -> float:
    """
    Calcula la similitud del coseno entre dos embeddings.

    Args:
        embedding_1 (list[float]): Primer embedding.
        embedding_2 (list[float]): Segundo embedding.

    Returns:
        float: Similitud del coseno entre los dos embeddings.
    """
    dot_product = sum(e1 * e2 for e1, e2 in zip(embedding_1, embedding_2))
    magnitude_1 = math.sqrt(sum(e1**2 for e1 in embedding_1))
    magnitude_2 = math.sqrt(sum(e2**2 for e2 in embedding_2))
    if magnitude_1 == 0 or magnitude_2 == 0:
        return 0.0
    return dot_product / (magnitude_1 * magnitude_2)


def demonstrate_semantic_similarity():
    """
    Muestra la similitud semántica entre una frase y una lista de documentos.

    """

    # Pregunta
    base_phrase = input("👤 User: ")

    # Documentos
    candidates = [
        "Para reiniciar el servidor ejecuta: sudo systemctl restart nginx",
        "Puedes reboot el proceso con el comando service stop/start",
        "The server restart procedure is documented in section 4.2",
        "La pizza margarita lleva tomate, mozzarella y albahaca",
        "Los gatos domésticos duermen un promedio de 16 horas al día",
        "Para apagar el servidor usa: sudo shutdown -h now",
    ]

    print("🤖 Procesando...")

    base_embedding = get_embedding(base_phrase)
    results = []

    for phrase in candidates:
        candidate_embedding = get_embedding(phrase)
        similarity = cosine_similarity(base_embedding, candidate_embedding)
        results.append((similarity, phrase))

    results.sort(reverse=True)

    print(f"\n❓ Pregunta: {base_phrase}")
    print("\n📊 Resultados ordenados por similitud: ")
    print("="*40)

    for similarity, phrase in results:
        bar = "🟩" * int(similarity*30)
        relevance = "✅ RELEVANTE" if similarity > 0.5 else "❌ IRRELEVANTE"

        print(f"{similarity:.3f} {bar}")
        print(f"{relevance}: {phrase[:60]}...")


if __name__ == "__main__":
    print("="*60)
    print("Embeddings - Búsqueda por Vectores")
    print("="*60)
    # embedding_1 = get_embedding("Café")
    # embedding_2 = get_embedding("Té")
    # similarity = cosine_similarity(embedding_1, embedding_2)
    # print("Dimensión 1: ", len(embedding_1))
    # print("Dimensión 2: ", len(embedding_2))
    # print("Similitud: ", similarity)
    demonstrate_semantic_similarity()
