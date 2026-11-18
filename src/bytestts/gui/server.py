# SPDX-License-Identifier: MPL-2.0
"""The control panel's loopback server (spec §6.3): the page, and the API it drives.

Every request passes the Guard (protection.py) first. Every error body is {"error": "<one line>"}.
"""

from __future__ import annotations

import os
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import onnxruntime as ort
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict
from starlette.background import BackgroundTask
from starlette.exceptions import HTTPException
from starlette.requests import Request

from bytestts import __version__
from bytestts.catalog import catalog
from bytestts.catalog.catalog import VoiceInfo
from bytestts.errors import FrontendUnavailable, VoiceNotFound
from bytestts.gui import jobs as jobs_module
from bytestts.gui import saving
from bytestts.gui.downloads import Downloads, DownloadStatus
from bytestts.gui.jobs import Loader, SpeakRequest, SpeechJobs
from bytestts.gui.protection import Guard
from bytestts.gui.saving import TicketError
from bytestts.gui.session import Busy, CpuSlot, Session, one_line
from bytestts.text.phonemizers import espeak

TEXT_LIMIT = 5000  # code points
MIN_SPEED, MAX_SPEED = 0.5, 2.0
FORMATS = ("wav",)  # stage 1; stage 2 adds what soundfile can write
QUALITIES = ("standard", "high")
_STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}


class JobBody(BaseModel):
    """A Speak request from the page. Strict: "1" is not a number, and an unknown field is refused."""

    model_config = ConfigDict(extra="forbid", strict=True)

    text: str
    voice: str
    speaker: str | None = None
    speed: float
    device: Literal["cpu", "cuda"]
    threads: int


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
        where = ".".join(str(p) for p in first.get("loc", ())[1:])  # drop "body", "query" or "path"
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

    @app.post("/api/jobs", status_code=201)
    def start_job(job: JobBody) -> Any:
        if not 1 <= len(job.text) <= TEXT_LIMIT:
            return _error(400, f"text must be 1 to {TEXT_LIMIT} characters")
        if not MIN_SPEED <= job.speed <= MAX_SPEED:  # NaN fails this too
            return _error(400, f"speed must be between {MIN_SPEED} and {MAX_SPEED}")
        if not 1 <= job.threads <= max_threads:
            return _error(400, f"threads must be 1 to {max_threads}")
        manifest = catalog.load_catalog().get(job.voice)
        if manifest is None:
            return _error(400, f"unknown voice {job.voice}")
        state = downloads.state(job.voice)
        if state == "missing":
            return _error(400, f"voice {job.voice} is not downloaded")
        if state == "needs-repair":
            return _error(400, f"voice {job.voice} needs repair")
        if job.speaker is not None:
            if not manifest.speakers:
                return _error(400, f"voice {job.voice} has no speakers")
            if job.speaker not in manifest.speakers:
                return _error(400, f"speaker {job.speaker} is not in voice {job.voice}")
        request = SpeakRequest(job.text, job.voice, job.speaker, job.speed, job.device, job.threads)
        try:
            return {"id": speech.start(request)}
        except Busy as e:
            return _error(409, one_line(e))

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: int) -> Any:
        try:
            s = speech.status(job_id)
        except KeyError:
            return _error(404, f"unknown job {job_id}")
        return {
            "id": s.id,
            "state": s.state,
            "sentences_done": s.sentences_done,
            "error": s.error,
            "warnings": list(s.warnings),
        }

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: int) -> Any:
        try:
            speech.stop(job_id)
            return {"state": speech.status(job_id).state}
        except KeyError:
            return _error(404, f"unknown job {job_id}")

    @app.get("/api/jobs/{job_id}/audio")
    def job_audio(job_id: int) -> Response:
        try:
            return Response(speech.audio(job_id), media_type="audio/wav")
        except KeyError:
            return _error(404, f"no audio for job {job_id}")

    @app.post("/api/export")
    async def export(request: Request, format: str, quality: str = "standard", ticket: str | None = None) -> Any:
        if format not in FORMATS:
            return _error(400, f"format {format} is not available")
        if quality not in QUALITIES:
            return _error(400, f"quality must be one of {', '.join(QUALITIES)}")
        target: Path | None = None
        if ticket is not None:
            try:
                target = session.tickets.redeem(ticket)
            except TicketError as e:
                return _error(400, one_line(e))
        cap = saving.BYTE_CAP
        length = request.headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > cap):
            return _error(413, f"the audio is larger than {cap} bytes")
        try:
            slot.acquire("export")
        except Busy as e:
            return _error(409, one_line(e))
        temporary: list[Path] = []
        try:
            fd, name = tempfile.mkstemp(prefix="bytestts-", suffix=".wav")
            temporary.append(Path(name))
            with os.fdopen(fd, "wb") as out:
                try:
                    await saving.receive_body(request.stream(), out, cap=cap)
                except saving.BodyTooLarge as e:
                    return _error(413, one_line(e))
            try:
                saving.check_wav(temporary[0])
            except saving.BadAudio as e:
                return _error(400, one_line(e))
            where = target.parent if target is not None else None
            fd, name = tempfile.mkstemp(prefix="bytestts-", suffix=".part" if target else ".wav", dir=where)
            os.close(fd)
            result = Path(name)
            temporary.append(result)
            saving.copy_wav(temporary[0], result)
            if target is not None:
                saving.replace_into_place(result, target)
                return {"saved": target.name}
            temporary.remove(result)  # the response deletes it once sent
            return FileResponse(
                result, media_type="audio/wav", background=BackgroundTask(result.unlink, missing_ok=True)
            )
        finally:
            for path in temporary:
                path.unlink(missing_ok=True)
            slot.release()

    @app.post("/api/quit")
    def quit_panel() -> Response:
        return JSONResponse({"ok": True}, background=BackgroundTask(session.request_quit))

    return app
