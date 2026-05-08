import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def mock_csv_path():
    data = "uid,title,description,city,department\n1,Test,Desc,Marseille,Bouches-du-Rhône\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(data)
        return Path(f.name)


@pytest.fixture
def mock_csv_with_invalid():
    data = (
        "uid,title,description,city,department\n"
        "1,Valid,Desc,Marseille,Bouches-du-Rhône\n"
        "2,,Desc,Marseille,Bouches-du-Rhône\n"
        "3,Valid,,Marseille,Bouches-du-Rhône\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(data)
        return Path(f.name)
