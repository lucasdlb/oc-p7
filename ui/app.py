"""Gradio UI for the RAG cultural events API.

Connects to the FastAPI backend (default: http://localhost:8000).
Override with the API_BASE_URL environment variable.

Usage:
    uv run --group ui python ui/app.py

Requirements:
    The FastAPI backend must be running:
        uv run uvicorn api.main:app --reload
"""

import os

import gradio as gr
import httpx

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")


# ---------------------------------------------------------------------------
# Backend calls
# ---------------------------------------------------------------------------


def call_ask(question: str) -> tuple[str, str]:
    """Call POST /ask and return (answer_md, sources_md)."""
    question = question.strip()
    if not question:
        return "Veuillez saisir une question.", ""
    try:
        resp = httpx.post(f"{API_BASE}/ask", json={"question": question}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "")
        sources = data.get("sources", [])
        if sources:
            rows = "\n".join(f"| {s['title']} | {s['city']} | {s['date']} |" for s in sources)
            sources_md = f"| Titre | Ville | Date |\n|---|---|---|\n{rows}"
        else:
            sources_md = "_Aucune source retournée._"
        return answer, sources_md
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e))
        return f"Erreur {e.response.status_code} : {detail}", ""
    except httpx.RequestError as e:
        return f"Impossible de joindre l'API ({API_BASE}) : {e}", ""


def call_health() -> str:
    """Call GET /health and return a status string."""
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "?")
        index = data.get("index_loaded", False)
        index_icon = "✅" if index else "❌"
        return f"**Status:** {status}\n\n**Index chargé:** {index_icon}"
    except httpx.RequestError as e:
        return f"Impossible de joindre l'API ({API_BASE}) : {e}"


def call_rebuild() -> str:
    """Call POST /rebuild and return a result string."""
    try:
        resp = httpx.post(f"{API_BASE}/rebuild", timeout=600)
        resp.raise_for_status()
        data = resp.json()
        n = data.get("events_indexed", "?")
        return f"✅ Index reconstruit — **{n} événements** indexés."
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e))
        return f"Erreur {e.response.status_code} : {detail}"
    except httpx.RequestError as e:
        return f"Impossible de joindre l'API ({API_BASE}) : {e}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="RAG Événements Culturels — Bouches-du-Rhône") as demo:
    gr.Markdown(f"# RAG Événements Culturels\n**Bouches-du-Rhône (dept. 13)** — API : `{API_BASE}`")

    with gr.Tab("Poser une question"):
        question_input = gr.Textbox(
            label="Votre question",
            placeholder="Ex : Quels concerts sont prévus à Marseille en juin ?",
            lines=2,
        )
        ask_btn = gr.Button("Envoyer", variant="primary")
        answer_output = gr.Markdown(label="Réponse")
        sources_output = gr.Markdown(label="Sources")

        ask_btn.click(
            fn=call_ask,
            inputs=question_input,
            outputs=[answer_output, sources_output],
        )
        question_input.submit(
            fn=call_ask,
            inputs=question_input,
            outputs=[answer_output, sources_output],
        )

    with gr.Tab("Santé de l'API"):
        health_btn = gr.Button("Vérifier", variant="secondary")
        health_output = gr.Markdown()
        health_btn.click(fn=call_health, inputs=[], outputs=health_output)

    with gr.Tab("Reconstruire l'index"):
        gr.Markdown(
            "> **Attention** : cette opération relance le fetch OpenDataSoft, "
            "le nettoyage et la vectorisation complète. Elle peut prendre plusieurs minutes."
        )
        rebuild_btn = gr.Button("Reconstruire", variant="stop")
        rebuild_output = gr.Markdown()
        rebuild_btn.click(fn=call_rebuild, inputs=[], outputs=rebuild_output)


if __name__ == "__main__":
    demo.launch()
