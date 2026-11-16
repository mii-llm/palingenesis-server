"""HTTP API for a trusted operator of a single training host."""
import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .monitor import progress
from .store import Store
from .worker import DISTRIBUTED, Dispatcher

Operation = Literal[
    "init", "train", "distill", "distill-cache", "distill-score", "rl", "harbor", "harness",
    "merge", "surgery", "internals", "autopilot", "prepare", "prepare-multi", "s0-tune",
    "diagnose", "inspect", "validate", "profile", "monitor", "loss", "gradients", "version",
]
Argument = Annotated[str, StringConstraints(max_length=8192, pattern=r"^[^\x00]*$")]


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Operation
    args: list[Argument] = Field(default_factory=list, max_length=256)
    processes: int = Field(default=1, ge=1, le=1024)
    expected_steps: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def distributed_command(self):
        if self.processes > 1 and self.operation not in DISTRIBUTED:
            raise ValueError("processes > 1 is supported for train, rl and distill-cache")
        return self


class Job(BaseModel):
    id: str
    operation: Operation
    state: Literal["queued", "running", "succeeded", "failed", "interrupted"]
    created_at: str
    started_at: str | None
    finished_at: str | None
    exit_code: int | None
    error: str | None
    status_url: str
    logs_url: str


class JobStatus(Job):
    progress: dict | None


def describe(row):
    url = f"/v1/jobs/{row['id']}"
    return {**dict(row), "status_url": url, "logs_url": url + "/logs"}


def create_app():
    store = Store(Path(os.environ.get("PGS_SERVER_STATE", ".palingenesis-server")))
    workdir = Path(os.environ.get("PGS_SERVER_WORKDIR", ".")).resolve()
    token = os.environ.get("PGS_SERVER_TOKEN")
    if not token or not token.isascii():
        raise ValueError("Set PGS_SERVER_TOKEN to a random ASCII bearer token")
    if not workdir.is_dir():
        raise ValueError(f"PGS_SERVER_WORKDIR is missing: {workdir}")

    @asynccontextmanager
    async def lifespan(app):
        store.initialize()
        dispatcher = Dispatcher(store)
        dispatcher.start()
        try:
            yield
        finally:
            dispatcher.stopping.set()
            dispatcher.join()

    auth = HTTPBearer(auto_error=False)

    def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(auth)):
        if credentials is None or not hmac.compare_digest(credentials.credentials.encode(), token.encode()):
            raise HTTPException(401, "Invalid bearer token", headers={"WWW-Authenticate": "Bearer"})

    app = FastAPI(title="Palingenesis Server", version="0.1.0", lifespan=lifespan,
                  dependencies=[Depends(authorize)])

    def find(job_id):
        job = store.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        return job

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/jobs", status_code=202, response_model=Job)
    def submit(request: JobRequest, response: Response):
        """Queue a Palingenesis CLI operation and return its polling URL."""
        job = describe(find(store.enqueue(request, workdir)))
        response.headers["Location"] = job["status_url"]
        return job

    @app.get("/v1/jobs", response_model=list[Job])
    def jobs(limit: Annotated[int, Query(ge=1, le=100)] = 20,
             offset: Annotated[int, Query(ge=0, le=2**63-1)] = 0):
        return [describe(row) for row in store.recent(limit, offset)]

    @app.get("/v1/jobs/{job_id}", response_model=JobStatus)
    def status(job_id: str):
        row = find(job_id)
        return {**describe(row), "progress": progress(store, row)}

    @app.get("/v1/jobs/{job_id}/logs", response_class=Response,
             responses={200: {"content": {"text/plain": {}}, "description": "Bounded stdout and stderr"}})
    def logs(job_id: str, offset: Annotated[int | None, Query(ge=0, le=2**63-1)] = None,
             limit: Annotated[int, Query(ge=1, le=65536)] = 8192):
        """Tail by default, or read forward from a byte offset. Resume at X-Next-Offset."""
        find(job_id)
        start, data, size = store.read_log(job_id, offset, limit)
        return Response(data, media_type="text/plain", headers={
            "X-Log-Offset": str(start), "X-Next-Offset": str(start + len(data)),
            "X-Log-Size": str(size), "Cache-Control": "no-store",
        })

    return app
