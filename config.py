"""Configuration du projet chargée depuis config.toml (ou debug.toml)."""

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
    data_dir: Path
    raw_file: Path
    clean_file: Path
    skipped_file: Path


class OpendataConfig(BaseModel):
    base_url: str
    page_size: int = Field(default=100, ge=1, le=100)
    dept_filter: str


class VectorisationConfig(BaseModel):
    model: str
    chunk_size: int = Field(default=512, ge=1)
    chunk_overlap: int = Field(default=50, ge=0)
    index_dir: Path


class Config(BaseModel):
    paths: PathsConfig
    opendata: OpendataConfig
    vectorisation: VectorisationConfig


def find_project_root() -> Path:
    """Find the project root by searching for pyproject.toml upward."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("pyproject.toml not found — is this a uv project?")


def _resolve_paths(raw: dict, root: Path) -> dict:
    data_dir = Path(raw["paths"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    raw["paths"]["data_dir"] = data_dir

    for key in ("raw_file", "clean_file", "skipped_file"):
        p = Path(raw["paths"][key])
        if not p.is_absolute():
            raw["paths"][key] = root / p

    index_dir = Path(raw["vectorisation"]["index_dir"])
    if not index_dir.is_absolute():
        raw["vectorisation"]["index_dir"] = root / index_dir

    return raw


def _load() -> Config:
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
