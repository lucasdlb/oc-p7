"""FastAPI application for the cultural events RAG system."""

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, cast

from fastapi import FastAPI, HTTPException

from api import rag
from api.models import (
    AskRequest,
    AskResponse,
    HealthResponse,
    RebuildResponse,
    Source,
)

_index_loaded: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the FAISS vector index at application startup.

    Attempts to load the index synchronously. If loading fails the
    application still starts but the /health endpoint will report
    index_loaded=False.
    """
    global _index_loaded
    try:
        rag.load_vectorstore()
        _index_loaded = True
    except Exception:
        _index_loaded = False
    yield


app = FastAPI(title="RAG Cultural Events API", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return the health status of the service and its dependencies."""
    return HealthResponse(status="ok", index_loaded=_index_loaded)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Answer a user question about cultural events in Bouches-du-Rhône.

    Retrieves the most relevant chunks from the FAISS vector index and
    generates a grounded answer using mistral-large-latest.

    Raises:
        HTTPException 400: If the question is empty or whitespace-only.
        HTTPException 500: If the RAG pipeline or the vector store fails.
    """
    if not req.stripped_question:
        raise HTTPException(status_code=400, detail="question cannot be empty")
    try:
        result = rag.ask(req.question)
        result_any: Any = result
        q: str = str(result_any["question"])
        a: str = str(result_any["answer"])
        src_list: list[dict[str, str]] = cast("list[dict[str, str]]", result_any["sources"])
        sources = [Source(**s) for s in src_list]
        return AskResponse(question=q, answer=a, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/rebuild", response_model=RebuildResponse)
def rebuild() -> RebuildResponse:
    """Rebuild the FAISS index from scratch by re-running the full pipeline.

    Runs fetch_events.py → clean_events.py → build_index.py in sequence.
    Returns the number of events indexed in the new index.

    Raises:
        HTTPException 500: If any step in the pipeline fails.
    """
    try:
        result = rag.rebuild_index()
        result_any: Any = result
        status_str: str = cast("str", result_any["status"])
        events_int: int = cast("int", result_any["events_indexed"])
        return RebuildResponse(status=status_str, events_indexed=events_int)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
