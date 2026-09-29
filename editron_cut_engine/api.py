"""Local HTTP integration with bounded work and request-scoped temporary files."""

import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from . import __version__
from .engine import CutEngine


logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = int(os.environ.get("EDITRON_MAX_UPLOAD_BYTES", str(512 * 1024 * 1024)))
if MAX_UPLOAD_BYTES <= 0:
    raise ValueError("EDITRON_MAX_UPLOAD_BYTES must be positive")
_worker = threading.BoundedSemaphore(1)
app = FastAPI(title="EDITRON local editing engine", version=__version__)


class UploadBodyLimitMiddleware:
    """Bound multipart spooling before FastAPI resolves the UploadFile parameter."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/v1/cut":
            return await self.app(scope, receive, send)
        limit = MAX_UPLOAD_BYTES + 64 * 1024  # Multipart boundary and form-field overhead.
        headers = dict(scope.get("headers", []))
        try:
            declared_length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            declared_length = 0
        if declared_length > limit:
            response = JSONResponse({"detail": "Request body exceeds the upload limit"}, status_code=413)
            return await response(scope, receive, send)
        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > limit:
                raise HTTPException(status_code=413, detail="Request body exceeds the upload limit")
            return message

        return await self.app(scope, limited_receive, send)


app.add_middleware(UploadBodyLimitMiddleware)


def _cleanup(work):
    try:
        shutil.rmtree(work)
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("Could not remove temporary request directory %s", work, exc_info=True)


class TemporaryFileResponse(FileResponse):
    """Release temporary media even when response transmission fails or disconnects."""

    def __init__(self, path, *, work, **kwargs):
        super().__init__(path, background=BackgroundTask(_cleanup, work), **kwargs)
        self.work = work

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            _cleanup(self.work)


def _copy_upload(upload, destination):
    total = 0
    with destination.open("wb") as handle:
        while chunk := upload.file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Upload exceeds EDITRON_MAX_UPLOAD_BYTES")
            handle.write(chunk)
    if total == 0:
        raise HTTPException(status_code=422, detail="Uploaded video is empty")


@app.get("/health")
def health():
    return {"status": "ok", "engine": "editron", "version": __version__}


@app.post("/v1/cut")
def cut(
    file: UploadFile = File(...),
    language: str = Form("auto", min_length=1, max_length=32),
    repetitions: Literal["off", "conservative", "balanced", "aggressive"] = Form("conservative"),
    silence_mode: Literal["auto", "auto-editor", "vad", "rms", "off"] = Form("auto"),
    mode: Literal["fast", "smart"] = Form("smart"),
    preview: bool = Form(False),
    min_pause: float = Form(0.8, ge=0, le=60),
    padding: float = Form(0.08, ge=0, le=5),
):
    # FastAPI runs synchronous routes in its threadpool, keeping health checks responsive.
    if not _worker.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="Editing worker is busy", headers={"Retry-After": "5"})
    work = None
    response_owns_directory = False
    try:
        work = Path(tempfile.mkdtemp(prefix="editron_api_"))
        # Never join a user-supplied upload filename to a filesystem directory.
        source = work / "input.media"
        output = work / "edited.mp4"
        _copy_upload(file, source)
        engine = CutEngine(mode=mode, silence_mode=silence_mode, repetitions=repetitions,
                           min_pause=min_pause, padding=padding)
        if preview:
            return {"plan": engine.analyze(source, language)}
        plan = engine.process(source, output, language)
        if not output.is_file():
            raise RuntimeError("Renderer did not produce an output file")
        response = TemporaryFileResponse(
            output, work=work, media_type="video/mp4", filename="edited.mp4",
            headers={"X-Editron-Cuts": str(len(plan.get("cut_ranges", []))),
                     "X-Editron-Repetition-Cuts": str(len(plan.get("repetition_cuts", [])))},
        )
        response_owns_directory = True
        return response
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OSError, RuntimeError) as exc:
        logger.exception("Editron request failed")
        raise HTTPException(status_code=500, detail="Media processing failed; inspect the server log") from exc
    finally:
        if work is not None and not response_owns_directory:
            _cleanup(work)
        _worker.release()
