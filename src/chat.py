import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).parent))
from search import search

load_dotenv()

PROMPT_TEMPLATE = """CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{question}

RESPONDA A "PERGUNTA DO USUÁRIO\""""

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash-lite")
)


def answer(question: str) -> str:
    results = search(question, k=10)
    context = "\n\n".join(doc.page_content for doc, _ in results)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def main():
    print("Chat iniciado. Digite 'sair' para encerrar.\n")
    while True:
        question = input("PERGUNTA: ").strip()
        if not question:
            continue
        if question.lower() in ("sair", "exit", "quit"):
            print("Encerrando chat.")
            break
        print(f"RESPOSTA: {answer(question)}\n")


if __name__ == "__main__":
    main()
