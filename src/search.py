import os
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()


def get_store() -> PGVector:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001")
    )
    return PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PGVECTOR_COLLECTION"),
        connection=os.getenv("PGVECTOR_URL"),
        use_jsonb=True,
    )


def search(query: str, k: int = 10) -> list:
    store = get_store()
    return store.similarity_search_with_score(query, k=k)


if __name__ == "__main__":
    query = input("Digite sua busca: ").strip()
    results = search(query)
    for i, (doc, score) in enumerate(results, start=1):
        print("=" * 50)
        print(f"Resultado {i} (score: {score:.4f}):")
        print("=" * 50)
        print(doc.page_content.strip())
        print()
