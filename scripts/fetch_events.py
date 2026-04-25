"""
Récupération des événements culturels des Bouches-du-Rhône
via l'API OpenDataSoft (dataset evenements-publics-openagenda).

Aucune clé API requise.
"""

import json
import time
from datetime import datetime, timezone

import requests
from config import DATA_DIR, OPENDATASOFT_BASE, PAGE_SIZE, RAW_FILE


def fetch_page(offset: int) -> dict:
    """Récupère une page d'événements depuis OpenDataSoft."""
    dept_filter = (
        '(location_department="Bouches-du-Rhône" '
        'OR startswith(location_postalcode, "13")) '
        "AND lastdate_end > now()"
    )
    params = {
        "where": dept_filter,
        "limit": PAGE_SIZE,
        "offset": offset,
    }
    response = requests.get(OPENDATASOFT_BASE, params=params)
    response.raise_for_status()
    return response.json()


def fetch_all_events() -> list:
    """Récupère tous les événements via pagination."""
    all_records = []
    offset = 0

    print("Connexion à OpenDataSoft...")
    data = fetch_page(offset=0)
    total = data.get("total_count", 0)
    print(f"{total} événements trouvés dans les Bouches-du-Rhône\n")

    all_records.extend(data.get("results", []))

    while len(all_records) < total:
        offset += PAGE_SIZE
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
    """Vérifie qu'au moins un créneau est dans le futur."""
    if not timings_str:
        return False
    try:
        timings = json.loads(timings_str)
        now = datetime.now(timezone.utc)
        return any(datetime.fromisoformat(slot["end"]) > now for slot in timings)
    except (json.JSONDecodeError, KeyError, ValueError):
        return False


def main():
    records = fetch_all_events()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"Sauvé dans {RAW_FILE}")


if __name__ == "__main__":
    main()
