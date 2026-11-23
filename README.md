# Document-grounded Q&A over a PDF

[Português](README.pt-BR.md)

**Author:** Adayane Coutinho

A command-line RAG (retrieval-augmented generation) app built for an AI Engineering MBA challenge. It loads one PDF, stores chunk embeddings in PostgreSQL with pgvector, and answers questions from those chunks. The grounding rules live in the prompt in `src/chat.py`: answer only from the retrieved text, and otherwise reply with the fixed sentence `Não tenho informações necessárias para responder sua pergunta.`

The corpus checked into this repository is `document.pdf`, *The Complete Guide to Building Skills for Claude* (33 pages). Ingestion always reads that path. Replacing the file and running ingestion again points the same pipeline at a different document.

This repository does not include evaluation scores, latency numbers, or a saved chat transcript.

## Problem

General-purpose chat models answer from parametric memory and will fill gaps. This project is the small, local version of the constraint an applied AI engineer is usually asked to enforce: a user asks a question, the system may use only the customer's document, and it must say when the document does not contain the answer.

The implementation is a synchronous CLI with three entry points:

| Script | Role |
| --- | --- |
| `src/ingest.py` | Load the PDF, split it, embed the chunks, write them to pgvector |
| `src/search.py` | Embed a query and print the nearest chunks with the distance pgvector returns |
| `src/chat.py` | Retrieve 10 chunks, build one grounded prompt, print the model reply |

There is no HTTP API, no user accounts, and no conversation store. Each question is handled on its own.

## RAG pipeline

### Ingestion

`src/ingest.py` loads `.env`, then refuses to start unless `GOOGLE_API_KEY`, `PGVECTOR_URL`, and `PGVECTOR_COLLECTION` are set. It also refuses to start if `document.pdf` is missing from the repository root.

1. `PyPDFLoader` reads `document.pdf` and the script prints how many pages came back.
2. `RecursiveCharacterTextSplitter` splits those pages with `chunk_size=1000` and `chunk_overlap=150`. The splitter's default length function counts characters. The script prints how many chunks it produced.
3. `GoogleGenerativeAIEmbeddings` embeds every chunk.
4. `PGVector.add_documents` writes them into the collection named by `PGVECTOR_COLLECTION`.

The script does not delete or upsert existing rows first. Running it again appends another copy of the chunks.

### Embeddings

Both ingestion and search construct:

```python
GoogleGenerativeAIEmbeddings(model=os.getenv("GEMINI_EMBED_MODEL", "models/embedding-001"))
```

The model id is whatever `GEMINI_EMBED_MODEL` is set to. `.env.example` sets `models/text-embedding-004`. If the variable is missing, the code uses `models/embedding-001`. The string `models/gemini-embedding-001` appears in the header of the original Portuguese README and is not read by the Python.

The Gemini client takes the API key from `GOOGLE_API_KEY` in the environment. The scripts do not pass the key on the command line.

### Vector store

`docker-compose.yml` starts two containers from `pgvector/pgvector:pg17`:

- `postgres` (`container_name: postgres_rag`) serves database `rag` on port `5432`, with user and password `postgres`. Those are the local Compose credentials. A named volume `postgres_data` keeps the data, and a `pg_isready` healthcheck gates the next step.
- `bootstrap_vector_ext` waits until Postgres is healthy, runs `CREATE EXTENSION IF NOT EXISTS vector;`, and exits.

`PGVector` is constructed with `use_jsonb=True` and no explicit distance strategy. In `langchain-postgres==0.0.15` that default is cosine distance, and `use_jsonb=False` is rejected by the library. Rows land in the library tables `langchain_pg_collection` and `langchain_pg_embedding`. The connection string in `.env.example` is the SQLAlchemy URL for this Compose file:

`postgresql+psycopg://postgres:postgres@localhost:5432/rag`

### Retrieval

`src/search.py` opens the same store and calls `similarity_search_with_score(query, k=10)`. The call embeds the query with the same embedding model, orders by cosine distance, and returns `(Document, distance)` pairs. Lower distance is the closer match. The standalone CLI prints each distance and the chunk text.

`search.py` does not repeat the environment checks from `ingest.py`. A missing variable fails later, inside the client or the driver.

No metadata filter is applied. `k` is fixed at 10 in the chat path.

### Generation

`src/chat.py` calls `search(question, k=10)`, joins `doc.page_content` with blank lines, and discards the distances. It fills a single Portuguese prompt (`PROMPT_TEMPLATE`) that contains:

- the retrieved context
- the rules: use only that context, do not invent, do not use outside knowledge, do not add opinions
- three worked examples of questions the model must refuse
- the user question

The prompt is sent as one `HumanMessage` to `ChatGoogleGenerativeAI`. The model id comes from `GEMINI_LLM_MODEL`, defaulting to `gemini-2.5-flash-lite`, which is also the value in `.env.example`. The script does not set temperature and does not send previous turns.

The CLI prints `PERGUNTA:` / `RESPOSTA:` until the input is `sair`, `exit`, or `quit`.

## Architecture

```mermaid
flowchart TD
  pdf["document.pdf"]
  ingest["src/ingest.py"]
  loader["PyPDFLoader"]
  split["RecursiveCharacterTextSplitter\n1000 characters, overlap 150"]
  embed["GoogleGenerativeAIEmbeddings\nGEMINI_EMBED_MODEL"]
  pg[("PostgreSQL 17 + pgvector\ncosine distance, JSONB metadata")]
  question["CLI question"]
  retrieve["similarity_search_with_score\nk = 10"]
  prompt["Portuguese grounded prompt\nchunk text only"]
  llm["ChatGoogleGenerativeAI\nGEMINI_LLM_MODEL"]
  answer["Printed answer or the fixed refusal"]

  pdf --> ingest --> loader --> split --> embed --> pg
  question --> retrieve
  retrieve --> embed
  retrieve --> pg
  retrieve --> prompt --> llm --> answer
```

## Stack

Pinned in `requirements.txt` unless noted:

| Piece | Used for |
| --- | --- |
| Python 3.10+ | Runtime requested by the original setup notes |
| `langchain==0.3.27` | Document and message types used by the other packages |
| `langchain-community==0.3.27` | `PyPDFLoader` |
| `langchain-text-splitters==0.3.9` | `RecursiveCharacterTextSplitter` |
| `langchain-google-genai==2.1.9` | Gemini embeddings and chat |
| `langchain-postgres==0.0.15` | `PGVector` |
| `psycopg[binary]` | Postgres driver behind `postgresql+psycopg` (unpinned) |
| `pypdf` | PDF parsing used by the loader (unpinned) |
| `python-dotenv` | Loads `.env` (unpinned) |
| `pgvector/pgvector:pg17` | Postgres 17 with the `vector` extension |
| Google Gemini | `GEMINI_EMBED_MODEL` for vectors, `gemini-2.5-flash-lite` for answers when using `.env.example` |

## Setup and run

Prerequisites: Python 3.10 or newer, Docker with Docker Compose, and a Google Gemini API key.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

Edit `.env` and set `GOOGLE_API_KEY`. Get a key from the [Gemini API key docs](https://ai.google.dev/gemini-api/docs/api-key). Leave the other values as in the example if you are using the Compose database.

```bash
docker compose up -d
docker compose ps                  # wait until bootstrap_vector_ext has exited successfully
python src/ingest.py
python src/search.py               # optional: inspect raw neighbors before calling the LLM
python src/chat.py
```

`ingest.py` prints the PDF path, `Páginas carregadas: <n>`, `Chunks gerados: <n>`, and `Ingestão concluída com sucesso!`. The page and chunk counts depend on the file. This README does not record a run, so it does not quote them.

Stop the database with `docker compose down`. `docker compose down -v` also removes the `postgres_data` volume.

## Environment variables

Copy `.env.example`. The values below are placeholders and the local database URL. Do not commit a filled-in `.env`.

```bash
GOOGLE_API_KEY=<your-google-api-key>
GEMINI_EMBED_MODEL=models/text-embedding-004
GEMINI_LLM_MODEL=gemini-2.5-flash-lite
PGVECTOR_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PGVECTOR_COLLECTION=pdf_collection
```

| Variable | Required to start `ingest.py` | If unset in `search.py` / `chat.py` |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Yes | The Gemini client reads it from the environment and fails without it |
| `PGVECTOR_URL` | Yes | No default; the driver fails when the store opens |
| `PGVECTOR_COLLECTION` | Yes | No default; the store is opened with `None` |
| `GEMINI_EMBED_MODEL` | No | Falls back to `models/embedding-001` |
| `GEMINI_LLM_MODEL` | Not read by ingest or search | Falls back to `gemini-2.5-flash-lite` |

`PGVECTOR_URL` uses the Compose user and password (`postgres` / `postgres`) so a local `docker compose up` works without extra secrets. That password is a development default published in `docker-compose.yml`. It is not a production credential.

## Example questions

There is no captured answer from a run against `document.pdf` in this repository. The only question/answer pairs checked into the code are the refusal examples inside `PROMPT_TEMPLATE`:

```text
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."
```

A live session follows the loop in `src/chat.py`:

```text
Chat iniciado. Digite 'sair' para encerrar.

PERGUNTA: <your question>
RESPOSTA: <model reply>

PERGUNTA: sair
Encerrando chat.
```

The Portuguese README also shows an exchange about the revenue of a company called SuperTechIABrazil. That exchange is an illustration in the original write-up. It is not a logged run, and that company is not in `document.pdf`.

`src/search.py` asks `Digite sua busca:` and prints each neighbor as `Resultado <i> (score: <distance>)` followed by the chunk text. The value is the cosine distance from pgvector.

## What I would improve

These are the gaps in the code as it is checked in. None of them have been measured here.

- **Make ingestion idempotent.** `add_documents` does not set ids and nothing clears the collection, so a second ingest duplicates every chunk and silently changes what retrieval returns.
- **Use one embedding model name.** `.env.example` and the code default disagree (`models/text-embedding-004` vs `models/embedding-001`). Re-ingesting after switching models mixes vectors that are not comparable.
- **Let the distance affect the answer.** The chat path throws away the cosine distance and always passes 10 chunks in. A cutoff, or at least showing the distance to the model, would make the refusal rule depend on retrieval instead of only on the prompt.
- **Cite the page.** The loader returns one document per page, but the prompt receives chunk text only. Answers should point at the page they came from.
- **Align the prompt with the corpus.** The rules, the refusal sentence, and the CLI are in Portuguese. The committed PDF is in English. The prompt should either follow the user's language or state that choice on purpose.
- **Fail fast everywhere.** `ingest.py` checks required variables. `search.py` and `chat.py` do not.
- **Keep secrets out of git by default.** There is no `.gitignore`. A local `.env` is one `git add -A` away from the history. Ignore `.env`, virtualenvs, and `__pycache__`.
- **Pin the unpinned dependencies.** `psycopg`, `pypdf`, and `python-dotenv` float, unlike the LangChain packages.
- **Add a regression set before tuning.** A short list of in-document questions and expected refusals, plus a test that a second ingest does not double the row count, is the next engineering step. This repo has no tests and no measured retrieval or answer quality, so chunk size, `k`, and model choice are still uncompared.
- **Add memory only if a product needs it.** The loop does not send prior turns. That is a good fit for independent lookups and a poor fit for follow-up questions.

## Layout

```text
├── docker-compose.yml    # Postgres 17 + pgvector, and a one-shot CREATE EXTENSION
├── requirements.txt
├── .env.example          # placeholders and the local database URL
├── document.pdf          # committed corpus (33-page Claude skills guide)
├── src/
│   ├── ingest.py
│   ├── search.py
│   └── chat.py
├── README.md             # this file
└── README.pt-BR.md       # original Portuguese README
```
