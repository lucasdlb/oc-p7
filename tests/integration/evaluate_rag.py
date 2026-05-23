# pyright: ignore[reportArgumentType]

"""
Évaluation du système RAG avec Ragas.

Charge le jeu de test annoté (docs/test_set.json), exécute chaque question
via la chaîne RAG, puis calcule les métriques faithfulness, answer_relevancy
et context_recall.

Usage:
    uv run python tests/evaluate_rag.py

Environment:
    MISTRAL_API_KEY doit être défini dans .env
    RUN_MODE=production (défaut) — utilise l'index vector_store/
    RUN_MODE=debug — utilise l'index vector_store_debug/
"""

import json
import os
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd
from langchain_mistralai import MistralAIEmbeddings
from openai import OpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import llm_factory
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness
from ragas.run_config import RunConfig

from api.rag import ask as rag_ask
from config import PROJECT_ROOT, SETTINGS
from logging_config import setup_logging

logger = setup_logging(__name__)

MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


def get_mistral_llm():
    """Crée un LLM Mistral pour Ragas via llm_factory.

    Utilise mistral-large-latest avec température 0 (déterministe)
    pour des évaluations cohérentes. Passe par l'API OpenAI-compatible
    de Mistral avec instructor comme adapter (provider="openai").

    Returns:
        InstructorBaseRagasLLM configuré avec mistral-large-latest.
    """
    client = OpenAI(
        api_key=SETTINGS.mistral_api_key.get_secret_value(),
        base_url=MISTRAL_BASE_URL,
    )
    return llm_factory("mistral-small-latest", provider="openai", client=client, max_tokens=2048)


def get_mistral_embeddings():
    """Crée un embeddings Mistral pour Ragas.

    Utilise MistralAIEmbeddings (LangChain) directement car AnswerRelevancy
    requiert l'interface LangChain (.embed_query / .embed_documents).

    Returns:
        MistralAIEmbeddings configuré avec mistral-embed.
    """
    return MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=SETTINGS.mistral_api_key,
    )


def load_test_set(path: Path) -> list[dict]:
    """Charge le jeu de test annoté depuis un fichier JSON.

    Args:
        path: Chemin vers le fichier test_set.json.

    Returns:
        Liste de dictionnaires contenant les clés 'question', 'ground_truth',
        et optionnellement 'context_uids'.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} test cases from {path}")
    return data


def build_dataset(test_cases: list[dict]) -> EvaluationDataset:
    """Construit un Dataset Ragas à partir des cas de test.

    Pour chaque cas de test, interroge la chaîne RAG pour obtenir
    la réponse générée et les contextes retrievés.

    Args:
        test_cases: Liste de dictionnaires avec clés 'question' et 'ground_truth'.

    Returns:
        EvaluationDataset Ragas contenant les colonnes user_input, response,
        retrieved_contexts, et reference.
    """
    rows: list[dict] = []
    for tc in test_cases:
        question = tc["question"]
        logger.info(f"Asking: {question[:60]}...")

        try:
            result_any: Any = rag_ask(question)
        except Exception as e:
            logger.warning(f"Failed to get answer for '{question[:40]}...': {e}")
            continue

        sources_any: Any = result_any["sources"]
        cast_list: list[Any] = cast("list[Any]", sources_any)
        retrieved_contexts: list[str] = [str(doc["content"]) for doc in cast_list]
        if not retrieved_contexts:
            retrieved_contexts = ["Aucun contexte retrouvé"]

        rows.append(
            {
                "user_input": question,
                "response": str(result_any["answer"]),
                "retrieved_contexts": retrieved_contexts,
                "reference": tc["ground_truth"],
            }
        )
        time.sleep(2)

    df = pd.DataFrame(rows)
    dataset = EvaluationDataset.from_pandas(df)
    logger.info(f"Dataset built with {len(rows)} rows")
    return dataset


def main() -> object:  # pyright: ignore[reportReturnType]
    """Point d'entrée : charge le test set, exécute l'évaluation Ragas, affiche les résultats."""
    test_set_path = PROJECT_ROOT / "docs" / "test_set.json"
    if not test_set_path.exists():
        logger.error(f"Test set not found: {test_set_path}")
        raise SystemExit(1)

    test_cases = load_test_set(test_set_path)[:2]
    dataset = build_dataset(test_cases)

    llm = get_mistral_llm()
    embeddings = get_mistral_embeddings()

    logger.info("Running Ragas evaluation (this may take several minutes)...")
    eval_result = evaluate(  # type: ignore
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=1, max_wait=30, max_retries=6),
        raise_exceptions=False,
    )

    raw_scores = getattr(eval_result, "scores", [])
    if isinstance(raw_scores, list) and len(raw_scores) > 0:
        scores = raw_scores[0]
    else:
        scores = {}

    logger.info(f"\n=== RAGAS EVALUATION RESULTS === in Mode:{os.getenv('RUN_MODE')}")
    logger.info(f"Faithfulness:    {scores.get('faithfulness', 'N/A')}")
    logger.info(f"Answer Relevancy: {scores.get('answer_relevancy', 'N/A')}")
    logger.info(f"Context Recall:  {scores.get('context_recall', 'N/A')}")

    if hasattr(eval_result, "to_pandas"):
        results_df = eval_result.to_pandas()  # type: ignore
        output_path = PROJECT_ROOT / "docs" / "evaluation_results.csv"
        results_df.to_csv(output_path, index=False)  # type: ignore[union-attr]
        logger.info(f"\nDetailed results saved to {output_path}")

    return eval_result


if __name__ == "__main__":
    main()
