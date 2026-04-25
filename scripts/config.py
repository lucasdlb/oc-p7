"""Configuration partagée pour les scripts de fetch et clean."""

from pathlib import Path

OPENDATASOFT_BASE = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
PAGE_SIZE = 100
DEPARTMENT_FILTER = "Bouches-du-Rhône"


def find_project_root() -> Path:
    """Find the project root by searching for pyproject.toml upward."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("pyproject.toml not found — is this a uv project?")


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_FILE = DATA_DIR / "events_raw.json"
CLEAN_FILE = DATA_DIR / "events_clean.csv"
SKIPPED_FILE = DATA_DIR / "events_skipped.json"
