import io
import json
from urllib.error import HTTPError, URLError

import pytest

from scripts import inventory_deepcardiosim as module


def response(payload=None):
    return io.BytesIO(json.dumps(payload or {"files": [{"key": "sample.pt"}]}).encode())


def test_transient_gateway_failure_retries_and_honors_bounded_retry_after(monkeypatch):
    calls, delays = [], []

    def fetch(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise HTTPError(module.RECORD_API, 504, "gateway timeout", {"Retry-After": "120"}, None)
        return response()

    monkeypatch.setattr(module, "urlopen", fetch)
    monkeypatch.setattr(module.time, "sleep", delays.append)
    assert module.fetch_record()["files"][0]["key"] == "sample.pt"
    assert len(calls) == 2
    assert delays == [30]


def test_permanent_http_error_does_not_retry(monkeypatch):
    calls = []

    def fetch(*args, **kwargs):
        calls.append(1)
        raise HTTPError(module.RECORD_API, 404, "missing", {}, None)

    monkeypatch.setattr(module, "urlopen", fetch)
    with pytest.raises(HTTPError):
        module.fetch_record()
    assert len(calls) == 1


def test_exhausted_network_failure_still_fails(monkeypatch):
    calls, delays = [], []

    def fetch(*args, **kwargs):
        calls.append(1)
        raise URLError("offline")

    monkeypatch.setattr(module, "urlopen", fetch)
    monkeypatch.setattr(module.time, "sleep", delays.append)
    with pytest.raises(URLError, match="offline"):
        module.fetch_record(attempts=3)
    assert len(calls) == 3
    assert delays == [1, 2]


@pytest.mark.parametrize("payload", [{}, {"files": []}, {"files": "invalid"}])
def test_bad_response_is_not_retried_or_replaced_with_snapshot(monkeypatch, payload):
    calls = []

    def fetch(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(module, "urlopen", fetch)
    with pytest.raises(ValueError, match="files collection"):
        module.fetch_record()
    assert len(calls) == 1
