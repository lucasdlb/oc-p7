"""Benchmark des performances du pipeline RAG.

Mesure les timings de chaque étape de la chaîne RAG via des callbacks LangChain :
  - embed_ms     : vectorisation de la question (Mistral embed)
  - retriever_ms : recherche de similarité FAISS (embed + similarity_search)
  - llm_ms       : appel au LLM Mistral (génération de la réponse)
  - total_ms     : temps total end-to-end (question → réponse complète)

Les questions proviennent par défaut de docs/test_set.json (jeu de test annoté).

Usage:
    uv run python scripts/benchmark_rag.py
    uv run python scripts/benchmark_rag.py --n 3
    uv run python scripts/benchmark_rag.py --no-csv
    uv run python scripts/benchmark_rag.py --questions "Q1?" "Q2?"

Environment:
    MISTRAL_API_KEY doit être défini dans .env
    RUN_MODE=production (défaut) — utilise l'index vector_store/
    RUN_MODE=debug — utilise l'index vector_store_debug/
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from api.rag import build_chain, load_vectorstore
from config import PROJECT_ROOT
from logging_config import setup_logging

logger = setup_logging(__name__)

# Largeur des colonnes pour l'affichage terminal
_COL_QUESTION = 45
_COL_NUM = 12


class RAGTimingCallback(BaseCallbackHandler):
    """Callback LangChain qui mesure les timings du retriever et du LLM."""

    def __init__(self) -> None:
        super().__init__()
        self._retriever_start: float = 0.0
        self._llm_start: float = 0.0
        self.retriever_ms: float = 0.0
        self.llm_ms: float = 0.0

    def reset(self) -> None:
        """Réinitialise les mesures pour une nouvelle invocation."""
        self._retriever_start = 0.0
        self._llm_start = 0.0
        self.retriever_ms = 0.0
        self.llm_ms = 0.0

    # --- Retriever ---

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._retriever_start = time.perf_counter()

    def on_retriever_end(
        self,
        documents: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if self._retriever_start:
            self.retriever_ms = (time.perf_counter() - self._retriever_start) * 1000

    # --- LLM ---

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._llm_start = time.perf_counter()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if self._llm_start:
            self.llm_ms = (time.perf_counter() - self._llm_start) * 1000


def load_default_questions(path: Path) -> list[str]:
    """Charge les questions depuis le jeu de test annoté.

    Args:
        path: Chemin vers docs/test_set.json.

    Returns:
        Liste des questions extraites du fichier.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [item["question"] for item in data]


def benchmark_question(
    question: str,
    chain: Any,
    callback: RAGTimingCallback,
    vectorstore: Any,
    n_runs: int,
) -> list[dict[str, float]]:
    """Exécute la chaîne RAG N fois sur une question et retourne les timings.

    Mesure séparément :
    - embed_ms : temps d'embed_query seul (vectorisation de la question)
    - retriever_ms : temps total du retriever (embed + similarity_search)
    - llm_ms : temps de génération LLM
    - total_ms : temps end-to-end de chain.invoke

    Args:
        question: Question à poser au RAG.
        chain: Chaîne RetrievalQA LangChain.
        callback: Instance RAGTimingCallback associée à la chaîne.
        vectorstore: FAISS vectorstore (pour mesure d'embed isolée).
        n_runs: Nombre d'exécutions.

    Returns:
        Liste de dicts {embed_ms, retriever_ms, llm_ms, total_ms} par run.
    """
    runs: list[dict[str, float]] = []

    for _ in range(n_runs):
        # Mesure de l'embedding seul (avant l'invoke complet)
        embed_start = time.perf_counter()
        vectorstore.embeddings.embed_query(question)
        embed_ms = (time.perf_counter() - embed_start) * 1000

        # Réinitialisation des timings du callback
        callback.reset()

        # Mesure du temps total de la chaîne
        total_start = time.perf_counter()
        chain.invoke({"query": question}, config={"callbacks": [callback]})
        total_ms = (time.perf_counter() - total_start) * 1000

        runs.append(
            {
                "embed_ms": embed_ms,
                "retriever_ms": callback.retriever_ms,
                "llm_ms": callback.llm_ms,
                "total_ms": total_ms,
            }
        )

    return runs


def aggregate_runs(runs: list[dict[str, float]]) -> dict[str, float]:
    """Calcule les statistiques (mean, min, max, p95) sur plusieurs runs.

    Args:
        runs: Liste de dicts de timings par run.

    Returns:
        Dict contenant mean, min, max, p95 pour chaque métrique.
    """
    metrics = ["embed_ms", "retriever_ms", "llm_ms", "total_ms"]
    agg: dict[str, float] = {}
    for m in metrics:
        values = [r[m] for r in runs]
        agg[f"{m}_mean"] = statistics.mean(values)
        agg[f"{m}_min"] = min(values)
        agg[f"{m}_max"] = max(values)
        sorted_vals = sorted(values)
        p95_idx = max(0, int(len(sorted_vals) * 0.95) - 1)
        agg[f"{m}_p95"] = sorted_vals[p95_idx]
    return agg


def _trunc(text: str, width: int) -> str:
    """Tronque une chaîne à la largeur donnée avec ellipse."""
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def print_results(
    results: list[dict[str, Any]],
    n_runs: int,
) -> None:
    """Affiche les résultats de benchmark sous forme de tableau terminal.

    Args:
        results: Liste de dicts par question avec les timings agrégés.
        n_runs: Nombre de runs effectués par question.
    """
    cols = ["embed_ms", "retriever_ms", "llm_ms", "total_ms"]

    # En-tête
    header = f"{'Question':<{_COL_QUESTION}}"
    for col in cols:
        header += f"  {col:>{_COL_NUM}}"
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print(f"  RAG BENCHMARK  —  {n_runs} run(s) par question  (toutes valeurs en ms)")
    print(f"{'=' * len(header)}")

    # Lignes par question (on affiche la moyenne)
    print(
        f"\n{'Question':<{_COL_QUESTION}}"
        f"  {'embed_ms':>{_COL_NUM}}"
        f"  {'retriever_ms':>{_COL_NUM}}"
        f"  {'llm_ms':>{_COL_NUM}}"
        f"  {'total_ms':>{_COL_NUM}}"
    )
    print(sep)

    for r in results:
        q = _trunc(r["question"], _COL_QUESTION)
        embed = r["embed_ms_mean"]
        retriever = r["retriever_ms_mean"]
        llm = r["llm_ms_mean"]
        total = r["total_ms_mean"]
        print(
            f"{q:<{_COL_QUESTION}}"
            f"  {embed:>{_COL_NUM}.0f}"
            f"  {retriever:>{_COL_NUM}.0f}"
            f"  {llm:>{_COL_NUM}.0f}"
            f"  {total:>{_COL_NUM}.0f}"
        )

    # Statistiques globales
    print(f"\n{'=' * len(header)}")
    print(f"  STATISTIQUES GLOBALES  (sur {len(results)} question(s) × {n_runs} run(s))")
    print(f"{'=' * len(header)}")

    print(
        f"\n{'Métrique':<{_COL_QUESTION}}"
        f"  {'embed_ms':>{_COL_NUM}}"
        f"  {'retriever_ms':>{_COL_NUM}}"
        f"  {'llm_ms':>{_COL_NUM}}"
        f"  {'total_ms':>{_COL_NUM}}"
    )
    print(sep)

    # Calcul des stats globales sur toutes les questions
    global_embed = [r["embed_ms_mean"] for r in results]
    global_retriever = [r["retriever_ms_mean"] for r in results]
    global_llm = [r["llm_ms_mean"] for r in results]
    global_total = [r["total_ms_mean"] for r in results]

    def _stat_row(label: str, fn: Any) -> str:
        row = f"{label:<{_COL_QUESTION}}"
        for vals in [global_embed, global_retriever, global_llm, global_total]:
            row += f"  {fn(vals):>{_COL_NUM}.0f}"
        return row

    print(_stat_row("mean", statistics.mean))
    print(_stat_row("min", min))
    print(_stat_row("max", max))

    def _p95(vals: list[float]) -> float:
        s = sorted(vals)
        return s[max(0, int(len(s) * 0.95) - 1)]

    print(
        f"{'p95':<{_COL_QUESTION}}"
        f"  {_p95(global_embed):>{_COL_NUM}.0f}"
        f"  {_p95(global_retriever):>{_COL_NUM}.0f}"
        f"  {_p95(global_llm):>{_COL_NUM}.0f}"
        f"  {_p95(global_total):>{_COL_NUM}.0f}"
    )
    print(f"{'=' * len(header)}\n")

    # Récapitulatif clair des étapes
    mean_embed = statistics.mean(global_embed)
    mean_retriever = statistics.mean(global_retriever)
    mean_llm = statistics.mean(global_llm)
    mean_total = statistics.mean(global_total)

    overhead = mean_total - mean_retriever - mean_llm
    print("  Décomposition du temps moyen total :")
    print(f"    Embedding question  : {mean_embed:6.0f} ms  (inclus dans retriever)")
    pct_r = 100 * mean_retriever / mean_total
    pct_l = 100 * mean_llm / mean_total
    pct_o = 100 * overhead / mean_total
    print(f"    Retriever (FAISS)   : {mean_retriever:6.0f} ms  ({pct_r:.1f}%)")
    print(f"    LLM Mistral         : {mean_llm:6.0f} ms  ({pct_l:.1f}%)")
    print(f"    Overhead chaîne     : {overhead:6.0f} ms  ({pct_o:.1f}%)")
    print("    ────────────────────────────────")
    print(f"    Total end-to-end    : {mean_total:6.0f} ms\n")


def save_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    """Sauvegarde les résultats détaillés dans un fichier CSV.

    Args:
        results: Liste de dicts par question avec tous les timings.
        output_path: Chemin de sortie du fichier CSV.
    """
    if not results:
        return

    fieldnames = list(results[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Résultats sauvegardés dans {output_path}")
    print(f"  CSV sauvegardé : {output_path}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark du pipeline RAG — mesure les timings par étape."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        metavar="N",
        help="Nombre de runs par question (défaut: 1)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Ne pas sauvegarder les résultats en CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "benchmark_results.csv",
        metavar="PATH",
        help="Chemin du fichier CSV de sortie (défaut: docs/benchmark_results.csv)",
    )
    parser.add_argument(
        "--questions",
        nargs="+",
        metavar="Q",
        help="Questions à poser (défaut: chargées depuis docs/test_set.json)",
    )
    return parser.parse_args()


def main() -> None:
    """Point d'entrée du benchmark RAG."""
    args = parse_args()

    # Chargement des questions
    if args.questions:
        questions = args.questions
        logger.info(f"Questions fournies en ligne de commande : {len(questions)}")
    else:
        test_set_path = PROJECT_ROOT / "docs" / "test_set.json"
        if not test_set_path.exists():
            logger.error(f"Jeu de test introuvable : {test_set_path}")
            raise SystemExit(1)
        questions = load_default_questions(test_set_path)[:3]
        logger.info(f"Questions chargées depuis {test_set_path} : {len(questions)} (limitées à 3)")

    # Initialisation du vector store et de la chaîne
    logger.info("Chargement du vector store...")
    vectorstore = load_vectorstore()
    callback = RAGTimingCallback()
    chain = build_chain(vectorstore)

    print(f"\nBenchmark RAG — {len(questions)} question(s), {args.n} run(s) chacune")
    print("Chargement de l'index FAISS... OK")

    # Exécution du benchmark
    all_results: list[dict[str, Any]] = []

    for i, question in enumerate(questions, 1):
        logger.info(f"[{i}/{len(questions)}] {question[:60]}...")
        print(f"  [{i:02d}/{len(questions):02d}] {_trunc(question, 60)}", end="", flush=True)

        runs = benchmark_question(
            question=question,
            chain=chain,
            callback=callback,
            vectorstore=vectorstore,
            n_runs=args.n,
        )

        agg = aggregate_runs(runs)
        result_row: dict[str, Any] = {"question": question, **agg}
        all_results.append(result_row)

        print(
            f"  → total={agg['total_ms_mean']:.0f}ms"
            f"  llm={agg['llm_ms_mean']:.0f}ms"
            f"  retriever={agg['retriever_ms_mean']:.0f}ms"
        )

        # Pause entre les questions pour ne pas saturer l'API Mistral
        if i < len(questions):
            time.sleep(1)

    # Affichage des résultats
    print_results(all_results, n_runs=args.n)

    # Sauvegarde CSV
    if not args.no_csv:
        save_csv(all_results, args.output)


if __name__ == "__main__":
    main()
