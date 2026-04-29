"""Configuration du projet chargée depuis config.toml."""

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from pydantic import BaseModel, Field


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


def _load() -> Config:
    root = find_project_root()
    toml_path = root / "config.toml"
    raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))

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

    return Config(**raw)


CONFIG = _load()
OPENDATA = CONFIG.opendata
PATH = CONFIG.paths
VEC = CONFIG.vectorisation
OPENDATASOFT_BASE = OPENDATA.base_url
