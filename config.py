"""Configuration du projet chargée depuis config.toml (ou debug.toml).

Les valeurs statiques (chemins, API params, vectorisation) sont lues depuis
config.toml (production) ou debug.toml (debug), sélectionné par RUN_MODE.

Les secrets et variables d'environnement (MISTRAL_API_KEY, RUN_MODE, DEBUG)
sont validés au démarrage via AppSettings (pydantic-settings).
"""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Project root — resolved once at import time, relative to this file
# ---------------------------------------------------------------------------


def _find_project_root() -> Path:
    """Trouve la racine du projet en cherchant pyproject.toml vers le haut.

    Démarre depuis ce fichier (config.py) et remonte jusqu'à trouver
    un répertoire contenant pyproject.toml. Fonctionne quel que soit le
    répertoire de travail courant.

    Returns:
        Chemin absolu de la racine du projet.
    """
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent


PROJECT_ROOT = _find_project_root()


# ---------------------------------------------------------------------------
# Environment / secrets — validated at import time via pydantic-settings
# ---------------------------------------------------------------------------


class AppSettings(BaseSettings):
    """Variables d'environnement et secrets de l'application.

    Lus depuis .env (ou l'environnement). Échec rapide au démarrage si
    une variable requise est absente.

    Attributes:
        mistral_api_key: Clé API Mistral (requise pour embeddings et LLM).
        run_mode: Mode d'exécution — "debug" ou "production" (défaut).
        debug: Active les logs verbeux en texte brut (défaut: False).
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mistral_api_key: SecretStr
    run_mode: str = "production"
    debug: bool = False
    hf_token: str | None = None


# ---------------------------------------------------------------------------
# Static config — loaded from config.toml or debug.toml
# ---------------------------------------------------------------------------


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
    """Charge et retourne la configuration TOML active.

    Lit RUN_MODE depuis SETTINGS (défaut: "production").
    Charge debug.toml si RUN_MODE="debug", sinon config.toml.

    Returns:
        Instance Config avec tous les paramètres validés par Pydantic.
    """
    config_file = "debug.toml" if SETTINGS.run_mode == "debug" else "config.toml"
    logger.info(f"RUN_MODE={SETTINGS.run_mode} — loading {config_file}")

    toml_path = PROJECT_ROOT / config_file
    raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    raw = _resolve_paths(raw, PROJECT_ROOT)

    return Config(**raw)


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")  # injecte le .env dans os.environ (HF_TOKEN, etc.)
SETTINGS = AppSettings()  # type: ignore[call-arg]  # pydantic-settings reads env/dotenv

CONFIG = _load()
OPENDATA = CONFIG.opendata
PATH = CONFIG.paths
VEC = CONFIG.vectorisation
OPENDATASOFT_BASE = OPENDATA.base_url
