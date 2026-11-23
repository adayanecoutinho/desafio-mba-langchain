# Ingestão e Busca Semântica com LangChain e Postgres

Sistema RAG (Retrieval-Augmented Generation) que lê um PDF, armazena seu conteúdo como vetores
no PostgreSQL com pgVector e responde perguntas via CLI com base exclusivamente no conteúdo do documento.

**Provedor:** Google Gemini
**Embeddings:** `models/gemini-embedding-001`
**LLM:** `gemini-2.5-flash-lite`

---

## Estrutura do projeto

```
├── docker-compose.yml   # Banco PostgreSQL + pgVector
├── requirements.txt     # Dependências Python
├── .env.example         # Template de variáveis de ambiente
├── src/
│   ├── ingest.py        # Ingestão do PDF no banco vetorial
│   ├── search.py        # Busca por similaridade (pode rodar isolado)
│   └── chat.py          # Chat via CLI
├── document.pdf         # PDF a ser ingerido (adicione você mesmo)
└── README.md
```

---

## Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- API Key do Google Gemini

---

## Configuração

### 1. Ambiente virtual

```bash
# Ativar o venv existente (Windows)
venv\Scripts\activate

# Se preferir criar um novo:
python -m venv venv
venv\Scripts\activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Variáveis de ambiente

Copie o template e preencha com sua API Key:

```bash
copy .env.example .env
```

Edite o arquivo `.env`:

```
GOOGLE_API_KEY=sua_chave_aqui
GEMINI_EMBED_MODEL=models/embedding-001
GEMINI_LLM_MODEL=gemini-2.5-flash-lite
PGVECTOR_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PGVECTOR_COLLECTION=pdf_collection
```

Obtenha sua chave em: https://ai.google.dev/gemini-api/docs/api-key

### 4. Adicionar o PDF

Coloque o arquivo PDF que deseja usar na raiz do projeto com o nome `document.pdf`.

---

## Execução

### Passo 1 — Subir o banco de dados

```bash
docker compose up -d
```

Aguarde o container inicializar (cerca de 10 segundos).

### Passo 2 — Ingerir o PDF

```bash
python src/ingest.py
```

Saída esperada:
```
Carregando PDF: .../document.pdf
Páginas carregadas: 10
Chunks gerados: 45
Ingestão concluída com sucesso!
```

### Passo 3 — Rodar o chat

```bash
python src/chat.py
```

Exemplo de uso:
```
Chat iniciado. Digite 'sair' para encerrar.

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

PERGUNTA: sair
Encerrando chat.
```

---

## Teste isolado da busca

Para verificar os resultados brutos da busca vetorial antes de enviar para o LLM:

```bash
python src/search.py
```

---

## Parar o banco de dados

```bash
docker compose down
```

Para remover também os dados:

```bash
docker compose down -v
```
