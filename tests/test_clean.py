import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_clean_html_strips_tags():
    from scripts.clean_events import clean_html

    result = clean_html("Hello <b>world</b>!")
    assert "<b>" not in result
    assert "world" in result


def test_clean_html_handles_nested_tags():
    from scripts.clean_events import clean_html

    result = clean_html("<p>Para with <em>emphasis</em></p>")
    assert "<p>" not in result
    assert "emphasis" in result


def test_is_valid_event_requires_timings_field():
    from scripts.clean_events import is_valid_event

    record = {
        "uid": "123",
        "title_fr": "Concert Jazz",
        "description_fr": "Un festival",
        "timings": '[{"begin":"2026-07-01T00:00:00Z","end":"2026-07-02T00:00:00Z"}]',
    }
    valid, reason = is_valid_event(record)
    assert valid is True


def test_is_valid_event_rejects_missing_uid():
    from scripts.clean_events import is_valid_event

    record = {"uid": "", "title_fr": "Concert", "description_fr": "Desc"}
    valid, reason = is_valid_event(record)
    assert valid is False


def test_is_valid_event_rejects_missing_title():
    from scripts.clean_events import is_valid_event

    record = {"uid": "123", "title_fr": "", "description_fr": "Desc"}
    valid, reason = is_valid_event(record)
    assert valid is False


def test_is_valid_event_rejects_missing_description():
    from scripts.clean_events import is_valid_event

    record = {"uid": "123", "title_fr": "Concert", "description_fr": ""}
    valid, reason = is_valid_event(record)
    assert valid is False


def test_is_valid_event_rejects_missing_timings():
    from scripts.clean_events import is_valid_event

    record = {"uid": "123", "title_fr": "Concert", "description_fr": "Desc"}
    valid, reason = is_valid_event(record)
    assert valid is False
    assert "timings" in reason


def test_main_writes_no_skipped_file_when_all_valid(tmp_path):
    import json

    from scripts.clean_events import main as clean_main

    raw_records = [
        {"uid": "1", "title_fr": "Valid", "description_fr": "Desc", "timings": "[]"},
        {"uid": "2", "title_fr": "Also Valid", "description_fr": "Desc2", "timings": "[]"},
    ]
    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(raw_records))

    with patch("scripts.clean_events.PATH") as mock_path:
        mock_path.raw_file = raw_file
        mock_path.data_dir = tmp_path
        mock_path.clean_file = tmp_path / "clean.csv"
        mock_path.skipped_file = tmp_path / "skipped.json"
        clean_main()

    assert not mock_path.skipped_file.exists()


def test_clean_record_with_all_fields():
    from scripts.clean_events import clean_record

    record = {
        "uid": "42",
        "title_fr": "My Title",
        "description_fr": "My <b>bold</b> description",
        "location_city": "Aix",
        "location_address": "1 Rue Centrale",
        "location_department": "Bouches-du-Rhône",
        "location_postalcode": "13100",
        "location_latitude": 43.53,
        "location_longitude": 5.45,
        "firstdate_begin": "2026-07-01",
        "lastdate_end": "2026-07-15",
        "keywords_fr": "jazz, festival",
    }
    cleaned = clean_record(record)
    assert cleaned["uid"] == "42"
    assert cleaned["city"] == "Aix"
    assert cleaned["postal_code"] == "13100"
    assert cleaned["latitude"] == 43.53
    assert cleaned["keywords"] == "jazz, festival"
    assert "<b>" not in cleaned["description"]


def test_main_skips_when_raw_file_missing(tmp_path):
    from scripts.clean_events import main as clean_main

    with patch("scripts.clean_events.PATH") as mock_path:
        mock_path.raw_file = tmp_path / "nonexistent.json"
        mock_path.data_dir = tmp_path
        clean_main()


def test_clean_record_uses_title_fr_field():
    from scripts.clean_events import clean_record

    record = {
        "uid": "123",
        "title_fr": "Concert Jazz",
        "description_fr": "Un festival",
        "city": "Marseille",
    }
    cleaned = clean_record(record)
    assert cleaned["title"] == "Concert Jazz"


def test_clean_record_cleans_html_in_description():
    from scripts.clean_events import clean_record

    record = {
        "uid": "123",
        "title_fr": "Concert Jazz",
        "description_fr": "Concert avec <b>HTML</b> tags",
        "location_city": "Marseille",
        "location_address": "",
        "location_department": "",
        "location_postalcode": "",
        "firstdate_begin": "",
        "lastdate_end": "",
        "keywords_fr": "",
    }
    cleaned = clean_record(record)
    assert "<b>" not in cleaned["description"]
    assert "HTML" in cleaned["description"]


def test_clean_html_returns_empty_for_empty_string():
    from scripts.clean_events import clean_html

    assert clean_html("") == ""


def test_main_valid_and_skipped_records(tmp_path):
    from scripts.clean_events import main as clean_main

    raw_records = [
        {"uid": "1", "title_fr": "Valid", "description_fr": "Desc", "timings": "[]"},
        {"uid": "2", "title_fr": "", "description_fr": "Desc", "timings": "[]"},
    ]
    import json

    raw_file = tmp_path / "raw.json"
    raw_file.write_text(json.dumps(raw_records))

    with patch("scripts.clean_events.PATH") as mock_path:
        mock_path.raw_file = raw_file
        mock_path.data_dir = tmp_path
        mock_path.clean_file = tmp_path / "clean.csv"
        mock_path.skipped_file = tmp_path / "skipped.json"
        clean_main()

    import pandas as pd

    df = pd.read_csv(mock_path.clean_file)
    assert len(df) == 1
    assert str(df.iloc[0]["uid"]) == "1"
    assert mock_path.skipped_file.exists()
