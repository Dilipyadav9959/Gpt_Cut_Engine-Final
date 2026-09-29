"""HTTP boundary tests: lifecycle, validation, bounded processing, and dispatch."""

import inspect
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from editron_cut_engine import api


@pytest.fixture
def service(monkeypatch, tmp_path):
    workdirs = []

    def make_work(**kwargs):
        work = tmp_path / f"request-{len(workdirs)}"
        work.mkdir()
        workdirs.append(work)
        return str(work)

    plan = {"duration": 5, "cut_ranges": [], "repetition_cuts": [], "detected_language": "te"}
    engine = MagicMock()
    engine.analyze.return_value = plan

    def render(source, output, language):
        Path(output).write_bytes(b"rendered-output")
        return plan

    engine.process.side_effect = render
    monkeypatch.setattr(api, "CutEngine", MagicMock(return_value=engine))
    monkeypatch.setattr(api.tempfile, "mkdtemp", make_work)
    monkeypatch.setattr(api, "_worker", threading.BoundedSemaphore(1))
    with TestClient(api.app) as client:
        yield client, engine, workdirs


def test_preview_never_renders_and_cleans_up(service):
    client, engine, workdirs = service
    response = client.post("/v1/cut", files={"file": ("../escape.mp4", b"video")},
                           data={"preview": "true", "language": "te"})
    assert response.status_code == 200
    assert response.json()["plan"]["detected_language"] == "te"
    engine.process.assert_not_called()
    source = engine.analyze.call_args.args[0]
    assert source.name == "input.media"
    assert source.parent == workdirs[0]
    assert not workdirs[0].exists()


def test_render_returns_file_then_cleans_up(service):
    client, engine, workdirs = service
    response = client.post("/v1/cut", files={"file": ("clip.mp4", b"video")})
    assert response.status_code == 200
    assert response.content == b"rendered-output"
    assert response.headers["x-editron-cuts"] == "0"
    engine.analyze.assert_not_called()
    assert not workdirs[0].exists()


@pytest.mark.parametrize("field,value", [("mode", "pro"), ("silence_mode", "unsafe"),
                                         ("repetitions", "anything"), ("min_pause", "-1"),
                                         ("padding", "nan")])
def test_invalid_configuration_rejected_before_work(service, field, value):
    client, engine, workdirs = service
    response = client.post("/v1/cut", files={"file": ("clip.mp4", b"video")}, data={field: value})
    assert response.status_code == 422
    assert not workdirs
    engine.process.assert_not_called()


def test_busy_returns_503_without_creating_workspace(service):
    client, _, workdirs = service
    api._worker.acquire()
    try:
        response = client.post("/v1/cut", files={"file": ("clip.mp4", b"video")})
    finally:
        api._worker.release()
    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    assert not workdirs


@pytest.mark.parametrize("body,status", [(b"", 422), (b"too large", 413)])
def test_empty_and_oversized_uploads_rejected_and_cleaned(service, monkeypatch, body, status):
    client, engine, workdirs = service
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 4)
    response = client.post("/v1/cut", files={"file": ("clip.mp4", body)})
    assert response.status_code == status
    assert not workdirs[0].exists()
    engine.process.assert_not_called()


def test_oversized_request_rejected_before_multipart_spooling(service, monkeypatch):
    client, _, workdirs = service
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 4)
    response = client.post("/v1/cut", content=b"x" * (65536 + 5),
                           headers={"content-type": "application/octet-stream"})
    assert response.status_code == 413
    assert not workdirs


def test_streamed_request_limit_without_content_length(service, monkeypatch):
    client, _, workdirs = service
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 4)
    body = (b"--test-boundary\r\nContent-Disposition: form-data; name=\"file\"; "
            b"filename=\"input.mp4\"\r\nContent-Type: video/mp4\r\n\r\n"
            + b"x" * (65536 + 5) + b"\r\n--test-boundary--\r\n")
    response = client.post("/v1/cut", content=iter([body[:1000], body[1000:]]),
                           headers={"content-type": "multipart/form-data; boundary=test-boundary"})
    assert response.status_code == 413
    assert not workdirs


def test_failed_file_transmission_cleans_up(service, monkeypatch):
    client, _, workdirs = service

    async def fail_send(*args, **kwargs):
        raise RuntimeError("Disconnected while sending")

    monkeypatch.setattr(api.FileResponse, "__call__", fail_send)
    with pytest.raises(RuntimeError, match="Disconnected"):
        client.post("/v1/cut", files={"file": ("clip.mp4", b"video")})
    assert not workdirs[0].exists()


def test_processing_failure_cleans_up_and_releases_worker(service):
    client, engine, workdirs = service
    engine.analyze.side_effect = RuntimeError("specific internal path")
    response = client.post("/v1/cut", files={"file": ("clip.mp4", b"video")}, data={"preview": "true"})
    assert response.status_code == 500
    assert "specific internal path" not in response.text
    assert not workdirs[0].exists()
    assert api._worker.acquire(blocking=False)
    api._worker.release()


def test_processing_route_uses_fastapi_threadpool():
    assert not inspect.iscoroutinefunction(api.cut)


def test_health(service):
    client, _, _ = service
    assert client.get("/health").json()["status"] == "ok"
