"""Configuration du projet chargée depuis config.toml (ou debug.toml).

Les valeurs sont lues depuis config.toml (production) ou debug.toml (debug),
sélectionné par la variable d'environnement RUN_MODE. Les chemins sont résolus
relativement à la racine du projet (répertoire contenant pyproject.toml).
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


class PathsConfig(BaseModel):
    """Chemins des fichiers de données.

    Attributes:
        data_dir: Répertoire contenant les données (data/).
        raw_file: Fichier JSON brut issu de fetch_events.py.
        clean_file: Fichier CSV nettoyé utilisé pour l'indexation.
        skipped_file: Fichier JSON loguant les événements ignorés.
    """

    data_dir: Path
    raw_file: Path
    clean_file: Path
    skipped_file: Path


class OpendataConfig(BaseModel):
    """Paramètres de l'API OpenDataSoft.

    Attributes:
        base_url: URL de base de l'API OpenDataSoft.
        page_size: Nombre d'événements par page (1-100).
        dept_filter: Filtre géographique (ex: "Bouches-du-Rhône").
    """

    base_url: str
    page_size: int = Field(default=100, ge=1, le=100)
    dept_filter: str


class VectorisationConfig(BaseModel):
    """Paramètres de vectorisation Mistral.

    Attributes:
        model: Nom du modèle d'embedding (ex: "mistral-embed").
        chunk_size: Taille maximale d'un chunk en caractères.
        chunk_overlap: Chevauchement entre chunks adjacents.
        index_dir: Répertoire où sauvegarder le index FAISS.
    """

    model: str
    chunk_size: int = Field(default=512, ge=1)
    chunk_overlap: int = Field(default=50, ge=0)
    index_dir: Path


class Config(BaseModel):
    """Configuration globale du projet."""

    paths: PathsConfig
    opendata: OpendataConfig
    vectorisation: VectorisationConfig


def find_project_root() -> Path:
    """Trouve la racine du projet en cherchant pyproject.toml upward.

    Returns:
        Chemin absolu du répertoire contenant pyproject.toml.

    Raises:
        FileNotFoundError: Si pyproject.toml n'est pas trouvé.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("pyproject.toml not found — is this a uv project?")


def _resolve_paths(raw: dict, root: Path) -> dict:
    """Résout les chemins relatifs par rapport à la racine du projet.

    Args:
        raw: Dict brut issu du fichier TOML (non encore résolu).
        root: Racine du projet (répertoire contenant pyproject.toml).

    Returns:
        Dict avec tous les chemins résolus en absolus.
    """
    data_dir = Path(raw["paths"]["data_dir"])
    if not data_dir.is_absolute():
        raw["paths"]["data_dir"] = root / data_dir

    for key in ("raw_file", "clean_file", "skipped_file"):
        p = Path(raw["paths"][key])
        if not p.is_absolute():
            raw["paths"][key] = root / p

    index_dir = Path(raw["vectorisation"]["index_dir"])
    if not index_dir.is_absolute():
        raw["vectorisation"]["index_dir"] = root / index_dir

    return raw


def _load() -> Config:
    """Charge et retourne la configuration active.

    Lit RUN_MODE depuis l'environnement (défaut: "production").
    Charge debug.toml si RUN_MODE="debug", sinon config.toml.

    Returns:
        Instance Config avec tous les paramètres validés par Pydantic.
    """
    load_dotenv()

    root = find_project_root()
    run_mode = os.getenv("RUN_MODE", "production")
    config_file = "debug.toml" if run_mode == "debug" else "config.toml"
    logger.info(f"RUN_MODE={run_mode} — loading {config_file}")

    toml_path = root / config_file
    raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    raw = _resolve_paths(raw, root)

    return Config(**raw)


CONFIG = _load()
OPENDATA = CONFIG.opendata
PATH = CONFIG.paths
VEC = CONFIG.vectorisation
OPENDATASOFT_BASE = OPENDATA.base_url
