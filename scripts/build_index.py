"""
Construction de l'index vectoriel FAISS à partir des événements nettoyés.

Pipeline:
    events_clean.csv → Documents → chunking → embeddings Mistral → FAISS index

Usage:
    uv run python scripts/build_index.py

Environment:
    Requires MISTRAL_API_KEY in .env (for mistral-embed embeddings).
    RUN_MODE=debug uses debug.toml → vector_store_debug/
    RUN_MODE=production (or unset) uses config.toml → vector_store/

Example:
    uv run python scripts/build_index.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent))

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import PATH, SETTINGS, VEC
from logging_config import setup_logging

logger = setup_logging(__name__)


def load_clean_events() -> pd.DataFrame:
    """Load cleaned events from CSV file configured in active config.

    Reads the clean CSV file path from PATH.clean_file (controlled by RUN_MODE).

    Returns:
        pd.DataFrame: DataFrame with columns matching events_clean.csv schema
            (uid, title, description, city, address, department, postal_code,
            latitude, longitude, firstdate_begin, lastdate_end, keywords).

    Raises:
        SystemExit: If the clean CSV file does not exist.
    """
    if not PATH.clean_file.exists():
        logger.error(f"{PATH.clean_file} not found — run clean_events.py first")
        raise SystemExit(1)
    df = pd.read_csv(PATH.clean_file)
    logger.info(f"Loaded {len(df)} events from {PATH.clean_file}")
    return df


def build_documents(df: pd.DataFrame) -> list[Document]:
    """Convert DataFrame rows into LangChain Document objects.

    Each event becomes a Document where:
    - page_content = "{title}. {description}"
    - metadata contains uid, title, city, firstdate_begin, lastdate_end

    Args:
        df: DataFrame of events with at least columns:
            uid, title, description, city, firstdate_begin, lastdate_end.

    Returns:
        list[Document]: List of LangChain Documents ready for chunking.
    """
    documents = []
    for _, row in df.iterrows():
        content = f"{row['title']}. {row['description']}"
        metadata = {
            "uid": str(row["uid"]),
            "title": str(row.get("title", "")),
            "city": str(row.get("city", "")),
            "firstdate_begin": str(row.get("firstdate_begin", "")),
            "lastdate_end": str(row.get("lastdate_end", "")),
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks using recursive character splitting.

    Uses RecursiveCharacterTextSplitter with configurable chunk_size and
    chunk_overlap from VEC config (or debug.toml defaults).

    Args:
        documents: List of Documents to split.

    Returns:
        list[Document]: Split chunks with metadata preserved from parent documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=VEC.chunk_size,
        chunk_overlap=VEC.chunk_overlap,
        length_function=len,
        keep_separator=False,
        add_start_index=False,
        strip_whitespace=True,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def build_index(chunks: list[Document]) -> tuple[FAISS, MistralAIEmbeddings]:
    """Build FAISS index from document chunks using Mistral embeddings.

    Reads MISTRAL_API_KEY from SETTINGS (validated at startup).
    Processes in batches of 250 to avoid Mistral rate limiting (400 errors
    with larger batches). If a batch fails, retries with exponential backoff.

    Args:
        chunks: List of Document chunks to index.

    Returns:
        tuple[FAISS, MistralAIEmbeddings]: Tuple of (vectorstore, embeddings)
            for save and verify steps.
    """
    embeddings = MistralAIEmbeddings(
        model=VEC.model,
        mistral_api_key=SETTINGS.mistral_api_key,
    )

    logger.info(f"Building FAISS index with {len(chunks)} chunks...")
    start = time.time()

    batch_size = 250
    vs = FAISS.from_documents(chunks[:batch_size], embeddings)
    logger.info(f"  Batch 1 ({batch_size} chunks): OK")
    for i in range(batch_size, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        attempt = 0
        while True:
            try:
                other = FAISS.from_documents(batch, embeddings)
                vs.merge_from(other)
                logger.info(f"  Batch {i // batch_size + 1} ({len(batch)} chunks): OK")
                break
            except Exception as e:
                attempt += 1
                if attempt > 3:
                    logger.error(f"Batch {i} failed after {attempt} attempts: {e}")
                    raise
                wait = 2**attempt
                logger.warning(f"Batch {i} failed (attempt {attempt}): {e} — retrying in {wait}s")
                time.sleep(wait)

    elapsed = time.time() - start
    logger.info(f"Index built in {elapsed:.1f}s — {vs.index.ntotal} vectors")

    return vs, embeddings


def main() -> None:
    """Run the full vectorisation pipeline end-to-end.

    1. Load clean events from CSV
    2. Build Documents
    3. Chunk documents
    4. Build FAISS index with Mistral embeddings
    5. Save index to disk (VEC.index_dir)
    6. Verify by reloading from disk

    Uses RUN_MODE to select config.toml (production) or debug.toml (debug).
    """
    df = load_clean_events()
    documents = build_documents(df)
    chunks = chunk_documents(documents)
    vectorstore, embeddings = build_index(chunks)

    VEC.index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VEC.index_dir))
    logger.info(f"Index saved to {VEC.index_dir}")

    loaded = FAISS.load_local(
        str(VEC.index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info(f"Verification: loaded index has {loaded.index.ntotal} vectors")


if __name__ == "__main__":
    main()
