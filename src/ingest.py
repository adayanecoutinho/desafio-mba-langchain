import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

for var in ("GOOGLE_API_KEY", "PGVECTOR_URL", "PGVECTOR_COLLECTION"):
    if not os.getenv(var):
        raise RuntimeError(f"Variável de ambiente '{var}' não configurada. Verifique o arquivo .env")

pdf_path = Path(__file__).parent.parent / "document.pdf"
if not pdf_path.exists():
    raise FileNotFoundError(f"PDF não encontrado: {pdf_path}\nColoque o arquivo document.pdf na raiz do projeto.")

print(f"Carregando PDF: {pdf_path}")
docs = PyPDFLoader(str(pdf_path)).load()
print(f"Páginas carregadas: {len(docs)}")

chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
).split_documents(docs)
print(f"Chunks gerados: {len(chunks)}")

embeddings = GoogleGenerativeAIEmbeddings(
    model=os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001")
)

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)

store.add_documents(documents=chunks)
print("Ingestão concluída com sucesso!")
