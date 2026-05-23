"""
Nettoyage des événements bruts OpenDataSoft.
Supprime les événements incomplets et nettoie le HTML des descriptions.

Usage:
    uv run python scripts/clean_events.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()))
)

import pandas as pd
from bs4 import BeautifulSoup

from config import PATH
from logging_config import setup_logging

logger = setup_logging(__name__)


def clean_html(text: str) -> str:
    """Nettoie le HTML d'une chaîne de caractères.

    Retire toutes les balises HTML et retourne le texte brut,
    avec les espaces normalisés.

    Args:
        text: Chaîne contenant du HTML (peut être vide ou None).

    Returns:
        Texte nettoyé sans balises, strips des espaces.
        Retourne "" si text est vide ou None.
    """
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)


def is_valid_event(record: dict) -> tuple[bool, str]:
    """Vérifie qu'un enregistrement OpenDataSoft est complet.

    Un événement valide doit avoir :
    - title_fr (titre en français)
    - description_fr (description en français)
    - timings (créneaux horaires — même vide, doit être présent)

    Args:
        record: Dict représentant un événement brut depuis l'API OpenDataSoft.

    Returns:
        Tuple (True, "") si l'événement est valide.
        Tuple (False, reason) sinon, où reason décrit le champ manquant.
    """
    if not record.get("title_fr"):
        return False, "missing title_fr"
    if not record.get("description_fr"):
        return False, "missing description_fr"
    if not record.get("timings"):
        return False, "missing timings"
    return True, ""


def clean_record(record: dict) -> dict:
    """Normalise et nettoie un enregistrement brut OpenDataSoft.

    Convertit les noms de champs OpenDataSoft en noms normalisés
    (snake_case) et nettoie le HTML des champs texte.

    Args:
        record: Dict brut issu de l'API OpenDataSoft avec les champs
            title_fr, description_fr, location_city, etc.

    Returns:
        Dict nettoyé avec les clés : uid, title, description, city,
        address, department, postal_code, latitude, longitude,
        firstdate_begin, lastdate_end, keywords.
    """
    return {
        "uid": record.get("uid"),
        "title": record["title_fr"],
        "description": clean_html(record["description_fr"]),
        "city": record.get("location_city", ""),
        "address": record.get("location_address", ""),
        "department": record.get("location_department", ""),
        "postal_code": record.get("location_postalcode", ""),
        "latitude": record.get("location_latitude"),
        "longitude": record.get("location_longitude"),
        "firstdate_begin": record.get("firstdate_begin", ""),
        "lastdate_end": record.get("lastdate_end", ""),
        "keywords": record.get("keywords_fr", ""),
    }


def main() -> None:
    """Point d'entrée : lit le JSON brut, filtre, nettoie et exporte en CSV."""
    if not PATH.raw_file.exists():
        logger.error(f"{PATH.raw_file} not found — run fetch_events.py first")
        return

    raw_records = json.loads(PATH.raw_file.read_text())
    logger.info(f"Loaded {len(raw_records)} raw records")

    valid_records: list[dict] = []
    skipped_records: list[dict] = []

    for record in raw_records:
        if is_valid_event(record)[0]:
            valid_records.append(clean_record(record))
        else:
            reason = is_valid_event(record)[1]
            skipped_records.append({"uid": record.get("uid"), "reason": reason})
            logger.debug(f"Skipped [{record.get('uid')}]: {reason}")

    logger.info(f"Valid records: {len(valid_records)}")
    logger.warning(f"Skipped records: {len(skipped_records)}")

    df = pd.DataFrame(valid_records)
    PATH.data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(PATH.clean_file, index=False)
    logger.info(f"Saved cleaned events to {PATH.clean_file}")

    if skipped_records:
        PATH.skipped_file.write_text(json.dumps(skipped_records, ensure_ascii=False, indent=2))
        logger.warning(f"Logged {len(skipped_records)} skipped events to {PATH.skipped_file}")


if __name__ == "__main__":
    main()
