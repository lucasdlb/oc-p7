"""
Nettoyage des événements bruts OpenDataSoft.
Supprime les événements incomplets et nettoie le HTML des descriptions.
"""

import json
import logging

import pandas as pd
from bs4 import BeautifulSoup
from config import CLEAN_FILE, DATA_DIR, RAW_FILE, SKIPPED_FILE

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def clean_html(text: str) -> str:
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)


def is_valid_event(record: dict) -> tuple[bool, str]:
    if not record.get("title_fr"):
        return False, "missing title_fr"
    if not record.get("description_fr"):
        return False, "missing description_fr"
    if not record.get("timings"):
        return False, "missing timings"
    return True, ""


def clean_record(record: dict) -> dict:
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


def main():
    if not RAW_FILE.exists():
        logger.error(f"{RAW_FILE} not found — run fetch_events.py first")
        return

    raw_records = json.loads(RAW_FILE.read_text())
    logger.info(f"Loaded {len(raw_records)} raw records")

    valid_records = []
    skipped_records = []

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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_FILE, index=False)
    logger.info(f"Saved cleaned events to {CLEAN_FILE}")

    if skipped_records:
        SKIPPED_FILE.write_text(json.dumps(skipped_records, ensure_ascii=False, indent=2))
        logger.warning(f"Logged {len(skipped_records)} skipped events to {SKIPPED_FILE}")


if __name__ == "__main__":
    main()
