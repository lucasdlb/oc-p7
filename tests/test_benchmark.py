"""Tests unitaires pour scripts/benchmark_rag.py.

Couvre les fonctions pures sans appel réseau :
  - RAGTimingCallback (reset, on_retriever_start/end, on_llm_start/end)
  - aggregate_runs
  - _trunc
  - load_default_questions
  - save_csv
"""

import json
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from scripts.benchmark_rag import (
    RAGTimingCallback,
    _trunc,
    aggregate_runs,
    load_default_questions,
    save_csv,
)

# ---------------------------------------------------------------------------
# RAGTimingCallback
# ---------------------------------------------------------------------------


class TestRAGTimingCallback:
    def test_initial_state(self):
        cb = RAGTimingCallback()
        assert cb.retriever_ms == 0.0
        assert cb.llm_ms == 0.0
        assert cb._retriever_start == 0.0
        assert cb._llm_start == 0.0

    def test_reset_clears_all_fields(self):
        cb = RAGTimingCallback()
        cb.retriever_ms = 123.0
        cb.llm_ms = 456.0
        cb._retriever_start = 1.0
        cb._llm_start = 2.0
        cb.reset()
        assert cb.retriever_ms == 0.0
        assert cb.llm_ms == 0.0
        assert cb._retriever_start == 0.0
        assert cb._llm_start == 0.0

    def test_on_retriever_start_sets_timestamp(self):
        cb = RAGTimingCallback()
        cb.on_retriever_start({}, "query", run_id=uuid4())
        assert cb._retriever_start > 0.0

    def test_on_retriever_end_computes_elapsed(self):
        cb = RAGTimingCallback()
        cb._retriever_start = 1.0  # simulé dans le passé
        # on_retriever_end utilise perf_counter() — sera > _retriever_start
        cb.on_retriever_end([], run_id=uuid4())
        assert cb.retriever_ms > 0.0

    def test_on_retriever_end_skips_if_no_start(self):
        cb = RAGTimingCallback()
        # _retriever_start = 0.0 → branche if non prise
        cb.on_retriever_end([], run_id=uuid4())
        assert cb.retriever_ms == 0.0

    def test_on_llm_start_sets_timestamp(self):
        cb = RAGTimingCallback()
        cb.on_llm_start({}, ["prompt"], run_id=uuid4())
        assert cb._llm_start > 0.0

    def test_on_llm_end_computes_elapsed(self):
        cb = RAGTimingCallback()
        cb._llm_start = 1.0
        mock_response = MagicMock()
        cb.on_llm_end(mock_response, run_id=uuid4())
        assert cb.llm_ms > 0.0

    def test_on_llm_end_skips_if_no_start(self):
        cb = RAGTimingCallback()
        mock_response = MagicMock()
        cb.on_llm_end(mock_response, run_id=uuid4())
        assert cb.llm_ms == 0.0


# ---------------------------------------------------------------------------
# aggregate_runs
# ---------------------------------------------------------------------------


class TestAggregateRuns:
    def _make_run(self, embed, retriever, llm, total):
        return {
            "embed_ms": embed,
            "retriever_ms": retriever,
            "llm_ms": llm,
            "total_ms": total,
        }

    def test_single_run_mean_equals_value(self):
        runs = [self._make_run(100, 200, 300, 600)]
        agg = aggregate_runs(runs)
        assert agg["embed_ms_mean"] == pytest.approx(100.0)
        assert agg["retriever_ms_mean"] == pytest.approx(200.0)
        assert agg["llm_ms_mean"] == pytest.approx(300.0)
        assert agg["total_ms_mean"] == pytest.approx(600.0)

    def test_single_run_min_max_equal_value(self):
        runs = [self._make_run(100, 200, 300, 600)]
        agg = aggregate_runs(runs)
        assert agg["embed_ms_min"] == pytest.approx(100.0)
        assert agg["embed_ms_max"] == pytest.approx(100.0)

    def test_multiple_runs_mean(self):
        runs = [
            self._make_run(100, 200, 1000, 1300),
            self._make_run(200, 300, 2000, 2500),
            self._make_run(300, 400, 3000, 3700),
        ]
        agg = aggregate_runs(runs)
        assert agg["embed_ms_mean"] == pytest.approx(200.0)
        assert agg["llm_ms_mean"] == pytest.approx(2000.0)

    def test_multiple_runs_min_max(self):
        runs = [
            self._make_run(100, 200, 1000, 1300),
            self._make_run(200, 300, 2000, 2500),
            self._make_run(300, 400, 3000, 3700),
        ]
        agg = aggregate_runs(runs)
        assert agg["embed_ms_min"] == pytest.approx(100.0)
        assert agg["embed_ms_max"] == pytest.approx(300.0)

    def test_p95_single_run(self):
        runs = [self._make_run(500, 100, 2000, 2600)]
        agg = aggregate_runs(runs)
        # p95 sur 1 valeur = la valeur elle-même
        assert agg["embed_ms_p95"] == pytest.approx(500.0)

    def test_all_keys_present(self):
        runs = [self._make_run(1, 2, 3, 6)]
        agg = aggregate_runs(runs)
        for metric in ["embed_ms", "retriever_ms", "llm_ms", "total_ms"]:
            for stat in ["mean", "min", "max", "p95"]:
                assert f"{metric}_{stat}" in agg


# ---------------------------------------------------------------------------
# _trunc
# ---------------------------------------------------------------------------


class TestTrunc:
    def test_short_string_unchanged(self):
        assert _trunc("hello", 10) == "hello"

    def test_exact_length_unchanged(self):
        assert _trunc("hello", 5) == "hello"

    def test_long_string_truncated_with_ellipsis(self):
        result = _trunc("abcdefghij", 6)
        assert len(result) == 6
        assert result.endswith("…")

    def test_truncated_content_is_prefix(self):
        result = _trunc("abcdefghij", 6)
        assert result == "abcde…"

    def test_empty_string(self):
        assert _trunc("", 5) == ""


# ---------------------------------------------------------------------------
# load_default_questions
# ---------------------------------------------------------------------------


class TestLoadDefaultQuestions:
    def test_returns_questions_list(self, tmp_path):
        data = [
            {"question": "Q1?", "ground_truth": "A1"},
            {"question": "Q2?", "ground_truth": "A2"},
        ]
        path = tmp_path / "test_set.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        questions = load_default_questions(path)
        assert questions == ["Q1?", "Q2?"]

    def test_returns_only_question_field(self, tmp_path):
        data = [{"question": "Q?", "ground_truth": "A", "context_uids": [1, 2]}]
        path = tmp_path / "test_set.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        questions = load_default_questions(path)
        assert questions == ["Q?"]

    def test_empty_file_returns_empty_list(self, tmp_path):
        path = tmp_path / "test_set.json"
        path.write_text("[]", encoding="utf-8")
        assert load_default_questions(path) == []


# ---------------------------------------------------------------------------
# save_csv
# ---------------------------------------------------------------------------


class TestSaveCsv:
    def test_creates_csv_with_correct_headers(self, tmp_path):
        results = [
            {
                "question": "Q?",
                "embed_ms_mean": 100.0,
                "llm_ms_mean": 2000.0,
                "total_ms_mean": 2100.0,
            }
        ]
        output = tmp_path / "out.csv"
        save_csv(results, output)
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "question" in content
        assert "embed_ms_mean" in content
        assert "Q?" in content

    def test_creates_parent_dirs(self, tmp_path):
        results = [{"question": "Q?", "total_ms_mean": 1000.0}]
        output = tmp_path / "subdir" / "results.csv"
        save_csv(results, output)
        assert output.exists()

    def test_empty_results_no_file_written(self, tmp_path):
        output = tmp_path / "out.csv"
        save_csv([], output)
        assert not output.exists()

    def test_multiple_rows(self, tmp_path):
        results = [
            {"question": "Q1?", "total_ms_mean": 1000.0},
            {"question": "Q2?", "total_ms_mean": 2000.0},
        ]
        output = tmp_path / "out.csv"
        save_csv(results, output)
        lines = output.read_text(encoding="utf-8").strip().splitlines()
        # header + 2 data rows
        assert len(lines) == 3
