import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeResponse:
    def __init__(self, json_data):
        self._json = json_data

    def json(self):
        return self._json

    def get(self, key, default=None):
        return self._json.get(key, default)

    def raise_for_status(self):
        pass


def test_fetch_page_uses_department_filter():

    from scripts.fetch_events import fetch_page

    captured_params = {}

    def fake_get(url, params=None, **kwargs):
        captured_params.update(params or {})

        class Resp:
            def json(self):
                return {"results": [], "total_count": 0}

            def raise_for_status(self):
                pass

        return Resp()

    with patch("requests.get", side_effect=fake_get):
        fetch_page(offset=0)
        assert "where" in captured_params
        assert "Bouches-du-Rhône" in captured_params["where"]


def test_fetch_page_uses_offset_param():
    from scripts.fetch_events import fetch_page

    captured_params = {}

    def fake_get(url, params=None, **kwargs):
        captured_params.update(params or {})

        class Resp:
            def json(self):
                return {"results": [], "total_count": 0}

            def raise_for_status(self):
                pass

        return Resp()

    with patch("requests.get", side_effect=fake_get):
        fetch_page(offset=50)
        assert captured_params["offset"] == 50


def test_is_future_event_rejects_past_dates():
    from scripts.fetch_events import is_future_event

    timings = '[{"begin":"2020-01-01T00:00:00Z","end":"2020-01-02T00:00:00Z"}]'
    assert is_future_event(timings) is False


def test_is_future_event_accepts_future_dates():
    from scripts.fetch_events import is_future_event

    timings = '[{"begin":"2027-01-01T00:00:00+00:00","end":"2027-01-02T00:00:00+00:00"}]'
    assert is_future_event(timings) is True


def test_fetch_page_returns_results_key():
    from scripts.fetch_events import fetch_page

    with patch("requests.get") as mock_get:
        mock_get.return_value = FakeResponse(
            {"results": [{"uid": "123", "title_fr": "Test"}], "pagination": {}}
        )
        result = fetch_page(offset=0)
        assert "results" in result
        assert len(result["results"]) == 1


def test_fetch_page_handles_empty_results():
    from scripts.fetch_events import fetch_page

    with patch("requests.get") as mock_get:
        mock_get.return_value = FakeResponse({"results": [], "pagination": {"count": 0}})
        result = fetch_page(offset=0)
        assert result["results"] == []


def test_is_future_event_rejects_empty_string():
    from scripts.fetch_events import is_future_event

    assert is_future_event("") is False


def test_is_future_event_rejects_invalid_json():
    from scripts.fetch_events import is_future_event

    assert is_future_event("not valid json") is False


def test_is_future_event_rejects_malformed_date():
    from scripts.fetch_events import is_future_event

    timings = '[{"begin":"2020-01-01T00:00:00Z","end":"not-a-date"}]'
    assert is_future_event(timings) is False


def test_fetch_all_events_handles_pagination():
    from scripts.fetch_events import fetch_all_events

    responses = [
        {"results": [{"uid": "1"}], "total_count": 3},
        {"results": [{"uid": "2"}], "total_count": 3},
        {"results": [{"uid": "3"}], "total_count": 3},
    ]

    with patch("scripts.fetch_events.fetch_page") as mock_page:
        mock_page.side_effect = [FakeResponse(r) for r in responses]
        with patch("scripts.fetch_events.time.sleep"):
            records = fetch_all_events()
    assert len(records) == 3
    assert mock_page.call_count >= 2


def test_main_writes_raw_file(tmp_path):
    import json

    from scripts.fetch_events import main as fetch_main

    mock_records = [{"uid": "1", "title_fr": "Test"}]
    with patch("scripts.fetch_events.fetch_all_events", return_value=mock_records):
        with patch("scripts.fetch_events.PATH") as mock_path:
            mock_path.data_dir = tmp_path
            mock_path.raw_file = tmp_path / "raw.json"
            fetch_main()
    assert mock_path.raw_file.exists()
    assert json.loads(mock_path.raw_file.read_text()) == mock_records
