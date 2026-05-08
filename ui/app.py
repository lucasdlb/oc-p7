"""Gradio chatbot UI for the RAG cultural events API.

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

# Suggested questions shown as quick-start examples
EXAMPLES = [
    "Quels concerts sont prévus à Marseille ?",
    "Y a-t-il des expositions d'art dans les Bouches-du-Rhône ?",
    "Je cherche des activités pour enfants dans le 13.",
    "Des événements autour de la formation professionnelle ?",
    "Que faire un week-end à Aix-en-Provence ?",
]


# ---------------------------------------------------------------------------
# Backend calls
# ---------------------------------------------------------------------------


def call_ask(question: str) -> tuple[str, str]:
    """Call POST /ask. Returns (answer, sources_md)."""
    try:
        resp = httpx.post(f"{API_BASE}/ask", json={"question": question}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        answer: str = data.get("answer", "")
        sources: list[dict] = data.get("sources", [])
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
    """Call GET /health. Returns a markdown status string."""
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "?")
        index = data.get("index_loaded", False)
        index_icon = "✅" if index else "❌"
        return f"**Status :** {status}\n\n**Index chargé :** {index_icon}"
    except httpx.RequestError as e:
        return f"❌ Impossible de joindre l'API ({API_BASE}) : {e}"


def call_rebuild() -> str:
    """Call POST /rebuild. Returns a result string."""
    try:
        resp = httpx.post(f"{API_BASE}/rebuild", timeout=600)
        resp.raise_for_status()
        data = resp.json()
        n = data.get("events_indexed", "?")
        return f"✅ Index reconstruit — **{n} événements** indexés."
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e))
        return f"❌ Erreur {e.response.status_code} : {detail}"
    except httpx.RequestError as e:
        return f"❌ Impossible de joindre l'API ({API_BASE}) : {e}"


# ---------------------------------------------------------------------------
# Chat handler
# ---------------------------------------------------------------------------


def chat(
    message: str,
    history: list[dict],
) -> tuple[list[dict], str, str]:
    """Send a message to /ask and append the exchange to history.

    Args:
        message: User input from the textbox.
        history: Current gr.Chatbot history (list of {role, content} dicts).

    Returns:
        (updated_history, sources_md, cleared_input)
    """
    message = message.strip()
    if not message:
        return history, "", ""

    answer, sources_md = call_ask(message)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return history, sources_md, ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Assistant Événements Culturels — Bouches-du-Rhône") as demo:
    gr.Markdown(
        f"# 🎭 Assistant Événements Culturels\n**Bouches-du-Rhône (dept. 13)** — API : `{API_BASE}`"
    )

    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(
            label="Conversation",
            height=480,
            buttons=["copy_all"],
            avatar_images=(
                None,
                "https://upload.wikimedia.org/wikipedia/fr/4/4b/Logo_Mistral_AI.svg",
            ),
        )
        sources_display = gr.Markdown(label="Sources des événements cités")

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Posez une question sur les événements culturels...",
                show_label=False,
                scale=9,
                autofocus=True,
            )
            send_btn = gr.Button("Envoyer", variant="primary", scale=1)

        gr.Examples(
            examples=EXAMPLES,
            inputs=msg_input,
            label="Suggestions de questions",
        )

        clear_btn = gr.Button("Effacer la conversation", variant="secondary", size="sm")

        # Wire up interactions
        send_btn.click(
            fn=chat,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, sources_display, msg_input],
        )
        msg_input.submit(
            fn=chat,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, sources_display, msg_input],
        )
        clear_btn.click(
            fn=lambda: ([], "", ""),
            outputs=[chatbot, sources_display, msg_input],
        )

    with gr.Tab("Santé de l'API"):
        health_btn = gr.Button("Vérifier", variant="secondary")
        health_output = gr.Markdown()
        health_btn.click(fn=call_health, inputs=[], outputs=health_output)

    with gr.Tab("Reconstruire l'index"):
        gr.Markdown(
            "> ⚠️ **Attention** : cette opération relance le fetch OpenDataSoft, "
            "le nettoyage et la vectorisation complète. "
            "Elle peut prendre **plusieurs minutes**."
        )
        rebuild_btn = gr.Button("Reconstruire", variant="stop")
        rebuild_output = gr.Markdown()
        rebuild_btn.click(fn=call_rebuild, inputs=[], outputs=rebuild_output)


if __name__ == "__main__":
    demo.launch()
