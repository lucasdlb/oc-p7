import sys
from pathlib import Path

sys.path.insert(
    0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()))
)


def test_ask_request_model_validation():
    from api.models import AskRequest

    req = AskRequest(question="Concerts à Marseille")
    assert req.question == "Concerts à Marseille"


def test_ask_request_accepts_whitespace_string():
    from api.models import AskRequest

    req = AskRequest(question="   ")
    assert req.question == "   "
    assert req.stripped_question == ""


def test_health_response_model():
    from api.models import HealthResponse

    resp = HealthResponse(status="ok", index_loaded=True)
    assert resp.status == "ok"
    assert resp.index_loaded is True


def test_rebuild_response_model():
    from api.models import RebuildResponse

    resp = RebuildResponse(status="ok", events_indexed=100)
    assert resp.status == "ok"
    assert resp.events_indexed == 100


def test_source_model():
    from api.models import Source

    src = Source(title="Jazz Festival", city="Marseille", date="2026-07-15")
    assert src.title == "Jazz Festival"
    assert src.city == "Marseille"


def test_ask_response_model():
    from api.models import AskResponse, Source

    resp = AskResponse(
        question="Concert à Marseille",
        answer="Il y a un festival de jazz",
        sources=[Source(title="Jazz Fest", city="Marseille", date="2026-07")],
    )
    assert resp.question == "Concert à Marseille"
    assert len(resp.sources) == 1
