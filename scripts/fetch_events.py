"""
Récupération des événements culturels des Bouches-du-Rhône
via l'API OpenDataSoft (dataset evenements-publics-openagenda).

Aucune clé API requise.

Usage:
    uv run python scripts/fetch_events.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(
    0, str(next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()))
)

import requests

from config import OPENDATA, OPENDATASOFT_BASE, PATH


def fetch_page(offset: int) -> dict:
    """Récupère une page d'événements depuis OpenDataSoft.

    Args:
        offset: Offset de pagination pour l'API OpenDataSoft.

    Returns:
        Dict contenant les clés 'results' (liste d'événements) et
        'total_count' (nombre total d'événements matching la requête).
    """
    dept_filter = (
        '(location_department="Bouches-du-Rhône" '
        'OR startswith(location_postalcode, "13")) '
        "AND lastdate_end > now()"
    )
    params = {
        "where": dept_filter,
        "limit": OPENDATA.page_size,
        "offset": offset,
    }
    response = requests.get(OPENDATASOFT_BASE, params=params)
    response.raise_for_status()
    return response.json()


def fetch_all_events() -> list[dict]:
    """Récupère tous les événements via pagination.

    Effectue des appels successifs à fetch_page() en augmentant l'offset
    jusqu'à récupération de tous les événements (filtre: Bouches-du-Rhône,
    date de fin > maintenant).

    Returns:
        Liste de dictionnaires représentant chaque événement.
    """
    all_records: list[dict] = []
    offset = 0

    print("Connexion à OpenDataSoft...")
    data = fetch_page(offset=0)
    total = data.get("total_count", 0)
    print(f"{total} événements trouvés dans les Bouches-du-Rhône\n")

    all_records.extend(data.get("results", []))

    while len(all_records) < total:
        offset += OPENDATA.page_size
        print(f"  Récupération {offset}/{total}...")
        data = fetch_page(offset=offset)
        records = data.get("results", [])
        if not records:
            break
        all_records.extend(records)
        time.sleep(0.3)

    print(f"\n✅ {len(all_records)} événements récupérés au total")
    return all_records


def is_future_event(timings_str: str) -> bool:
    """Vérifie qu'au moins un créneau horaire est dans le futur.

    Parse le JSON timings (liste de {begin, end}) et vérifie si au moins
    un slot a une date de fin postérieure à maintenant (UTC).

    Args:
        timings_str: Chaîne JSON contenant les créneaux de l'événement.
            Ex: '[{"begin":"2026-06-01T00:00:00Z","end":"2026-06-02T00:00:00Z"}]'

    Returns:
        True si au moins un créneau est dans le futur, False sinon.
        Retourne False si timings_str est vide ou le parsing échoue.
    """
    if not timings_str:
        return False
    try:
        timings = json.loads(timings_str)
        now = datetime.now(timezone.utc)
        return any(datetime.fromisoformat(slot["end"]) > now for slot in timings)
    except (json.JSONDecodeError, KeyError, ValueError):
        return False


def main() -> None:
    """Point d'entrée: récupère tous les événements et les sauvegarde."""
    records = fetch_all_events()
    PATH.data_dir.mkdir(parents=True, exist_ok=True)
    PATH.raw_file.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"Sauvé dans {PATH.raw_file}")


if __name__ == "__main__":
    main()
