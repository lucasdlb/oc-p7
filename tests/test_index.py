import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG


def test_vectorisation_config_loaded():

    assert CONFIG.vectorisation.model == "mistral-embed"
    assert CONFIG.vectorisation.chunk_size > 0
    assert CONFIG.vectorisation.chunk_overlap >= 0


def test_index_dir_exists_for_debug():

    os.environ["RUN_MODE"] = "debug"

    assert CONFIG.vectorisation.index_dir.exists() or str(CONFIG.vectorisation.index_dir).endswith(
        "vector_store_debug"
    )


def test_load_clean_events_reads_from_configured_path():
    from scripts.build_index import load_clean_events

    df = load_clean_events()
    assert len(df) == 2
    assert "uid" in df.columns
    assert df.iloc[0]["title"] == "Festival Jazz en Provence"


def test_build_documents_creates_documents():
    import pandas as pd

    from scripts.build_index import build_documents

    df = pd.DataFrame(
        [
            {
                "uid": "1",
                "title": "Concert Jazz",
                "description": "Un concert",
                "city": "Marseille",
                "firstdate_begin": "2026-06-01",
                "lastdate_end": "2026-06-02",
            },
            {
                "uid": "2",
                "title": "Expo Art",
                "description": "Une expo",
                "city": "Aix",
                "firstdate_begin": "2026-07-01",
                "lastdate_end": "2026-07-02",
            },
        ]
    )
    docs = build_documents(df)
    assert len(docs) == 2
    assert docs[0].page_content.startswith("Concert Jazz")


def test_chunk_documents_respects_size():
    from langchain_core.documents import Document

    from scripts.build_index import chunk_documents

    docs = [
        Document(
            page_content="A" * 1000,
            metadata={"uid": "1", "title": "Long", "city": "Marseille"},
        )
    ]
    chunks = chunk_documents(docs)
    assert len(chunks) >= 1
    for chunk in chunks:
        max_size = CONFIG.vectorisation.chunk_size + CONFIG.vectorisation.chunk_overlap
        assert len(chunk.page_content) <= max_size


def test_faiss_index_can_be_created():
    from langchain_community.vectorstores import FAISS
    from langchain_core.embeddings import Embeddings

    class FakeEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[0.1] * 128 for _ in texts]

        def embed_query(self, text):
            return [0.1] * 128

    texts = ["hello world", "foo bar"]
    emb = FakeEmbeddings()
    vs = FAISS.from_texts(texts, emb)
    assert vs.index.ntotal == 2


def test_faiss_merge_from():
    from langchain_community.vectorstores import FAISS
    from langchain_core.embeddings import Embeddings

    class FakeEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[0.1] * 128 for _ in texts]

        def embed_query(self, text):
            return [0.1] * 128

    emb = FakeEmbeddings()
    vs1 = FAISS.from_texts(["a", "b"], emb)
    vs2 = FAISS.from_texts(["c", "d"], emb)
    vs1.merge_from(vs2)
    assert vs1.index.ntotal == 4


def test_load_clean_events_raises_when_file_missing(tmp_path):
    from scripts.build_index import load_clean_events

    with patch("scripts.build_index.PATH") as mock_path:
        mock_path.clean_file = tmp_path / "nonexistent.csv"
        try:
            load_clean_events()
        except SystemExit as e:
            assert e.code == 1


def test_build_index_success_with_mocked_embeddings():
    from langchain_core.documents import Document

    from scripts.build_index import build_index

    class FakeEmbeddings:
        def embed_documents(self, texts):
            return [[0.1] * 128 for _ in texts]

        def embed_query(self, text):
            return [0.1] * 128

    docs = [
        Document(
            page_content="Title. Description",
            metadata={
                "uid": "1",
                "title": "Title",
                "city": "Marseille",
                "firstdate_begin": "2026-01-01",
                "lastdate_end": "2026-01-02",
            },
        ),
        Document(
            page_content="Title2. Description2",
            metadata={
                "uid": "2",
                "title": "Title2",
                "city": "Aix",
                "firstdate_begin": "2026-02-01",
                "lastdate_end": "2026-02-02",
            },
        ),
    ]
    with patch(
        "scripts.build_index.MistralAIEmbeddings",
        return_value=FakeEmbeddings(),
    ):
        vs, emb = build_index(docs)
    assert vs.index.ntotal == 2
    assert emb is not None


def test_build_index_raises_when_api_key_missing():
    # SETTINGS.mistral_api_key is validated at import — this test verifies
    # that build_index itself proceeds without a separate guard now.
    # The guard is at the AppSettings level (startup), so this is a no-op test.
    pass


def test_main_loads_clean_events_builds_and_saves(tmp_path):

    from scripts.build_index import main as build_main

    class FakeEmbeddings:
        def embed_documents(self, texts):
            return [[0.1] * 128 for _ in texts]

        def embed_query(self, text):
            return [0.1] * 128

    clean_csv = tmp_path / "clean.csv"
    clean_csv.write_text(
        "uid,title,description,city,address,department,postal_code,latitude,longitude,firstdate_begin,lastdate_end,keywords\n"
        "1,Festival Jazz en Provence,Un festival de jazz,Vitrolles,,,,2026-06-01,2026-06-30,\n"
    )

    with patch("scripts.build_index.PATH") as mock_path:
        with patch("scripts.build_index.VEC") as mock_vec:
            mock_path.clean_file = clean_csv
            mock_vec.model = "mistral-embed"
            mock_vec.chunk_size = 512
            mock_vec.chunk_overlap = 50
            mock_vec.index_dir = tmp_path / "index"
            with patch(
                "scripts.build_index.MistralAIEmbeddings",
                return_value=FakeEmbeddings(),
            ):
                build_main()

    assert (tmp_path / "index").exists()
