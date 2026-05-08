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
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Any, cast

import pandas as pd
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from pydantic import SecretStr
from ragas import evaluate
from ragas.dataset import Dataset
from ragas.metrics.collections import AnswerRelevancy, ContextRecall, Faithfulness

from api.rag import ask as rag_ask

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)
logger = logging.getLogger(__name__)


def get_mistral_llm():
    """Crée un LLM Mistral wrapped pour Ragas.

    Utilise mistral-large-latest avec température 0 (déterministe)
    pour des évaluations cohérentes.

    Returns:
        LangchainLLMWrapper configuré avec ChatMistralAI.
    """
    from ragas.integrations.langchain import LangchainLLMWrapper

    api_key = os.getenv("MISTRAL_API_KEY", "")
    llm = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0.0,
        mistral_api_key=SecretStr(api_key),
    )
    return LangchainLLMWrapper(llm)


def get_mistral_embeddings():
    """Crée un embeddings Mistral wrapped pour Ragas.

    Returns:
        LangchainEmbeddingsWrapper configuré avec mistral-embed.
    """
    from ragas.integrations.langchain import LangchainEmbeddingsWrapper

    emb = MistralAIEmbeddings(model="mistral-embed")
    return LangchainEmbeddingsWrapper(emb)


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


def build_dataset(test_cases: list[dict]) -> Dataset:
    """Construit un Dataset Ragas à partir des cas de test.

    Pour chaque cas de test, interroge la chaîne RAG pour obtenir
    la réponse générée et les contextes retrievés.

    Args:
        test_cases: Liste de dictionnaires avec clés 'question' et 'ground_truth'.

    Returns:
        Dataset Ragas contenant les colonnes user_input, response,
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
        retrieved_contexts: list[str] = [str(doc["title"]) for doc in cast_list]
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

    df = pd.DataFrame(rows)
    dataset = Dataset.from_pandas(df, name="rag_eval", backend="inmemory")
    logger.info(f"Dataset built with {len(rows)} rows")
    return dataset


def main() -> object:  # pyright: ignore[reportReturnType]
    """Point d'entrée : charge le test set, exécute l'évaluation Ragas, affiche les résultats."""
    load_dotenv()

    test_set_path = Path(__file__).resolve().parents[1] / "docs" / "test_set.json"
    if not test_set_path.exists():
        logger.error(f"Test set not found: {test_set_path}")
        raise SystemExit(1)

    test_cases = load_test_set(test_set_path)
    dataset = build_dataset(test_cases)

    llm = get_mistral_llm()
    embeddings = get_mistral_embeddings()

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextRecall(llm=llm),
    ]

    logger.info("Running Ragas evaluation (this may take several minutes)...")
    eval_result = evaluate(dataset, metrics=metrics, return_executor=False, raise_exceptions=False)  # type: ignore

    raw_scores = getattr(eval_result, "scores", [])
    if isinstance(raw_scores, list) and len(raw_scores) > 0:
        scores = raw_scores[0]
    else:
        scores = {}

    logger.info("\n=== RAGAS EVALUATION RESULTS ===")
    logger.info(f"Faithfulness:    {scores.get('faithfulness', 'N/A')}")
    logger.info(f"Answer Relevancy: {scores.get('answer_relevancy', 'N/A')}")
    logger.info(f"Context Recall:  {scores.get('context_recall', 'N/A')}")

    if hasattr(eval_result, "to_pandas"):
        results_df = eval_result.to_pandas()  # type: ignore
        output_path = Path(__file__).resolve().parents[1] / "docs" / "evaluation_results.csv"
        results_df.to_csv(output_path, index=False)  # type: ignore[union-attr]
        logger.info(f"\nDetailed results saved to {output_path}")

    return eval_result


if __name__ == "__main__":
    main()
