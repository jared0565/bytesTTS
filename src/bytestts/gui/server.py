# SPDX-License-Identifier: MPL-2.0
"""The control panel's loopback server (spec §6.3): the page, and the API it drives.

Every request passes the Guard (protection.py) first. Every error body is {"error": "<one line>"}.
"""

from __future__ import annotations

import os
from importlib import resources
from typing import Any

import onnxruntime as ort
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException
from starlette.requests import Request

from bytestts import __version__
from bytestts.catalog import catalog
from bytestts.catalog.catalog import VoiceInfo
from bytestts.errors import FrontendUnavailable, VoiceNotFound
from bytestts.gui import jobs as jobs_module
from bytestts.gui.downloads import Downloads, DownloadStatus
from bytestts.gui.jobs import Loader, SpeechJobs
from bytestts.gui.protection import Guard
from bytestts.gui.session import Busy, CpuSlot, Session, one_line
from bytestts.text.phonemizers import espeak

TEXT_LIMIT = 5000
FORMATS = ("wav",)  # stage 1; stage 2 adds what soundfile can write
QUALITIES = ("standard", "high")
_STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def _voice_json(v: VoiceInfo, state: str) -> dict[str, Any]:
    return {
        "id": v.id,
        "language": v.language,
        "speakers": [*v.featured, *(s for s in v.speakers if s not in v.featured)],
        "licence": v.licence,
        "data_licence": v.data_licence,
        "attribution": v.attribution,
        "sample_rate": v.sample_rate,
        "size_bytes": v.size_bytes,
        "state": state,
    }


def _download_json(s: DownloadStatus) -> dict[str, Any]:
    return {"voice": s.voice, "state": s.state, "done": s.done, "total": s.total, "error": s.error}


def create_app(*, port: int, secret: str, max_threads: int, session: Session, loader: Loader | None = None) -> FastAPI:
    """The panel's app. `loader=None` means jobs.load_voice, looked up at call time (tests patch it)."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(Guard, port=port, secret=secret, raw_body_paths=frozenset({"/api/export"}))
    slot = CpuSlot()
    downloads = Downloads()

    def default_loader(voice: str, device: jobs_module.Device, threads: int) -> Any:
        return jobs_module.load_voice(voice, device, threads)

    speech = SpeechJobs(slot, on_damaged=downloads.mark_damaged, loader=loader or default_loader)
    app.state.slot, app.state.downloads, app.state.jobs, app.state.session = slot, downloads, speech, session

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return _error(exc.status_code, one_line(Exception(exc.detail)))

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {"loc": (), "msg": "bad request"}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return _error(400, one_line(Exception(f"{where}: {first.get('msg')}" if where else first.get("msg"))))

    @app.exception_handler(Exception)
    async def failure(request: Request, exc: Exception) -> JSONResponse:
        return _error(500, one_line(exc))

    for path, (name, media_type) in _STATIC.items():

        def page(name: str = name, media_type: str = media_type) -> Response:
            body = resources.files("bytestts.gui").joinpath("static", name).read_bytes()
            return Response(body, media_type=media_type)

        app.add_api_route(path, page, methods=["GET"], include_in_schema=False)

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        try:
            espeak.find_espeak()
            found: dict[str, Any] = {"found": True, "hint": None}
        except FrontendUnavailable:
            found = {"found": False, "hint": espeak.INSTALL_HINT}
        return {
            "version": __version__,
            "cuda_available": "CUDAExecutionProvider" in ort.get_available_providers(),
            "espeak": found,
            "cpu_count": os.cpu_count() or 1,
            "max_threads": max_threads,
            "text_limit": TEXT_LIMIT,
            "formats": list(FORMATS),
            "busy": slot.busy,
        }

    @app.get("/api/voices")
    def voices() -> list[dict[str, Any]]:
        return [_voice_json(v, downloads.state(v.id)) for v in catalog.list_voices()]

    @app.post("/api/voices/{voice_id}/download", status_code=202)
    def start_download(voice_id: str) -> Any:
        try:
            downloads.start(voice_id)
        except VoiceNotFound as e:
            return _error(404, one_line(e))
        except Busy as e:
            return _error(409, one_line(e))
        return {"state": "running"}

    @app.get("/api/voices/{voice_id}/download")
    def download_status(voice_id: str) -> Any:
        try:
            return _download_json(downloads.status(voice_id))
        except VoiceNotFound as e:
            return _error(404, one_line(e))

    @app.post("/api/voices/{voice_id}/download/cancel")
    def cancel_download(voice_id: str) -> Any:
        try:
            downloads.cancel(voice_id)
            return {"state": downloads.status(voice_id).state}
        except VoiceNotFound as e:
            return _error(404, one_line(e))

    return app
