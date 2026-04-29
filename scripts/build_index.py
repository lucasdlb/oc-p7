"""
Construction de l'index vectoriel FAISS à partir des événements nettoyés.

Pipeline: events_clean.csv → chunking → embeddings Mistral → FAISS index

Usage:
    uv run python scripts/build_index.py
"""

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent))

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import PATH, VEC

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_clean_events() -> pd.DataFrame:
    if not PATH.clean_file.exists():
        logger.error(f"{PATH.clean_file} not found — run clean_events.py first")
        raise SystemExit(1)
    df = pd.read_csv(PATH.clean_file)
    logger.info(f"Loaded {len(df)} events from {PATH.clean_file}")
    return df


def build_documents(df: pd.DataFrame) -> list:
    from langchain_core.documents import Document

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


def chunk_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=VEC.chunk_size,
        chunk_overlap=VEC.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def build_index(chunks: list) -> tuple[FAISS, MistralAIEmbeddings]:
    import os

    from dotenv import load_dotenv
    from pydantic import SecretStr

    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        logger.error("MISTRAL_API_KEY not found in environment — set it in .env")
        raise SystemExit(1)

    embeddings = MistralAIEmbeddings(model=VEC.model, mistral_api_key=SecretStr(api_key))

    logger.info(f"Building FAISS index with {len(chunks)} chunks...")
    start = time.time()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    elapsed = time.time() - start
    logger.info(f"Index built in {elapsed:.1f}s — {vectorstore.index.ntotal} vectors")

    return vectorstore, embeddings


def main():
    df = load_clean_events()
    documents = build_documents(df)
    chunks = chunk_documents(documents)
    vectorstore, embeddings = build_index(chunks)

    VEC.index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VEC.index_dir))
    logger.info(f"Index saved to {VEC.index_dir}")

    loaded = FAISS.load_local(str(VEC.index_dir), embeddings, allow_dangerous_deserialization=True)
    logger.info(f"Verification: loaded index has {loaded.index.ntotal} vectors")


if __name__ == "__main__":
    main()
