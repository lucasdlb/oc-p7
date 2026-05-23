"""Pydantic schemas for the FastAPI REST API."""

from pydantic import BaseModel


class AskRequest(BaseModel):
    """Request body for the /ask endpoint.

    Attributes:
        question: The user's natural language question about cultural events
            in Bouches-du-Rhône (dept. 13).
    """

    question: str

    @property
    def stripped_question(self) -> str:
        """Return the question stripped of leading/trailing whitespace."""
        return self.question.strip()


class Source(BaseModel):
    """A single event source referenced in an RAG answer.

    Attributes:
        title: Title of the event.
        city: City where the event takes place.
        date: Event date or date range as a string.
    """

    title: str
    city: str
    date: str
    content: str = ""


class AskResponse(BaseModel):
    """Response body for the /ask endpoint.

    Attributes:
        question: The original question asked by the user.
        answer: The RAG-generated answer, grounded in the retrieved context.
        sources: List of event sources used to ground the answer.
    """

    question: str
    answer: str
    sources: list[Source]


class RebuildResponse(BaseModel):
    """Response body for the /rebuild endpoint.

    Attributes:
        status: Overall status of the rebuild operation (e.g., "ok").
        events_indexed: Number of events indexed in the rebuilt FAISS index.
    """

    status: str
    events_indexed: int


class HealthResponse(BaseModel):
    """Response body for the /health endpoint.

    Attributes:
        status: Overall health status (e.g., "ok").
        index_loaded: Whether the FAISS vector index was successfully loaded
            at application startup.
    """

    status: str
    index_loaded: bool
