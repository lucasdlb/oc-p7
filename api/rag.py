"""RAG chain logic: vector store loading, LLM chain construction, and queries."""

import subprocess
import sys

from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from config import CONFIG, SETTINGS

# Module-level singleton — chargé une fois au démarrage via init_vectorstore()
_vectorstore: FAISS | None = None


PROMPT_TEMPLATE = (
    "Tu es un assistant culturel spécialisé dans les événements des Bouches-du-Rhône.\n"
    "Réponds uniquement en te basant sur les événements fournis dans le contexte.\n"
    "Si tu ne trouves pas d'événement pertinent, dis-le clairement.\n"
    "Réponds toujours en français.\n\n"
    "Contexte:\n{context}\n\nQuestion: {question}\n\nRéponse:"
)


def init_vectorstore() -> None:
    """Charge le vector store depuis le disque et le stocke en mémoire.

    À appeler une seule fois au démarrage (lifespan FastAPI). Toutes les
    requêtes suivantes réutilisent l'instance en mémoire via _vectorstore.
    """
    global _vectorstore
    _vectorstore = load_vectorstore()


def load_vectorstore() -> FAISS:
    """Load the FAISS vector store from disk.

    Reads the index directory from the active configuration
    (controlled by RUN_MODE: config.toml → vector_store/,
    debug.toml → vector_store_debug/).

    Returns:
        FAISS vector store with the embedded event chunks loaded.
    """
    config = CONFIG
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=SETTINGS.mistral_api_key,
    )
    vectorstore = FAISS.load_local(
        str(config.vectorisation.index_dir),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def build_chain(vectorstore: FAISS) -> RetrievalQA:
    """Build a LangChain RetrievalQA chain for the given vector store.

    Args:
        vectorstore: FAISS vector store to use as the retriever.

    Returns:
        RetrievalQA chain configured with mistral-large-latest (temperature 0.3)
        and a custom French prompt.
    """
    llm = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.3,
        mistral_api_key=SETTINGS.mistral_api_key,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain


def ask(question: str) -> dict[str, object]:
    """Answer a question about cultural events using the RAG pipeline.

    Loads the vector store, builds the QA chain, invokes it with the question,
    and formats the result.

    Args:
        question: Natural language question about events in Bouches-du-Rhône.

    Returns:
        Dict with keys:
            - question: original question
            - answer: LLM-generated answer string
            - sources: list of dicts with title, city, date
    """
    vs = _vectorstore or load_vectorstore()
    chain = build_chain(vs)
    result = chain.invoke({"query": question})
    sources: list[dict[str, str]] = []
    for doc in result.get("source_documents", []):
        metadata = doc.metadata
        sources.append(
            {
                "title": str(metadata.get("title", "")),
                "city": str(metadata.get("city", "")),
                "date": str(metadata.get("date", "")),
                "content": str(doc.page_content),
            }
        )
    return {
        "question": question,
        "answer": result["result"],
        "sources": sources,
    }


def rebuild_index() -> dict[str, object]:
    """Rebuild the entire FAISS index from scratch.

    Runs the full pipeline in sequence:
        1. fetch_events.py — fetch raw events from OpenDataSoft
        2. clean_events.py — clean and validate events → CSV
        3. build_index.py — chunk, embed, and save FAISS index

    Each step is run as a subprocess. If any step fails the error is
    propagated with its stderr output.

    Returns:
        Dict with keys:
            - status: "ok" on success
            - events_indexed: number of events in the clean CSV (header row excluded)
    """
    result = subprocess.run(
        [sys.executable, "scripts/fetch_events.py"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"fetch failed: {result.stderr}")

    result = subprocess.run(
        [sys.executable, "scripts/clean_events.py"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"clean failed: {result.stderr}")

    result = subprocess.run(
        [sys.executable, "scripts/build_index.py"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"build_index failed: {result.stderr}")

    config = CONFIG
    with open(config.paths.clean_file, newline="", encoding="utf-8") as f:
        events_indexed = sum(1 for _ in f) - 1

    return {"status": "ok", "events_indexed": events_indexed}
