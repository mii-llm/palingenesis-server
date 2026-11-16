"""Launch one job in a detached worker, with a lock shared by its SDK process."""
import fcntl
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading

from .store import Store

DISTRIBUTED = {
    "train": "palingenesis.train",
    "rl": "palingenesis.rl.trainer",
    "distill-cache": "palingenesis.distill.build",
}


def command(job):
    args = json.loads(job["args"])
    if job["processes"] > 1:
        return [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
                f"--nproc-per-node={job['processes']}", "-m", DISTRIBUTED[job["operation"]], *args]
    return [sys.executable, "-u", "-m", "palingenesis.cli", job["operation"], *args]


def run(store, lock_fd):
    job = store.claim()
    if job is None:
        return
    code = None
    error = None
    try:
        with store.log_path(job["id"]).open("ab", buffering=0) as log:
            result = subprocess.run(command(job), cwd=job["workdir"], stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock_fd,),
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
            code = result.returncode
            if code:
                error = f"Palingenesis exited with code {code}"
    except (OSError, ValueError) as exc:
        error = str(exc)
    store.finish(job["id"], code, error)


class Dispatcher(threading.Thread):
    def __init__(self, store):
        super().__init__(name="palingenesis-dispatch", daemon=True)
        self.store = store
        self.stopping = threading.Event()

    def run(self):
        child = None
        while not self.stopping.is_set():
            try:
                if (child is None or child.poll() is not None) and self.store.has_work():
                    child = self.start_worker()
            except Exception:
                logging.exception("Could not dispatch a Palingenesis job")
            self.stopping.wait(0.5)
        if child is not None:
            # An embedded API can restart its lifespan without exiting Python.
            # Reap our worker when it finishes, while letting shutdown return now.
            threading.Thread(target=child.wait, daemon=True, name="palingenesis-reap").start()

    def start_worker(self):
        with (self.store.root / "worker.lock").open("ab") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return None
            # Closing our copy leaves the lock with the detached worker and SDK.
            env = os.environ.copy()
            env.pop("PGS_SERVER_TOKEN", None)
            return subprocess.Popen(
                [sys.executable, "-m", "palingenesis_server.worker", str(self.store.root), str(lock.fileno())],
                pass_fds=(lock.fileno(),), start_new_session=True, stdin=subprocess.DEVNULL, env=env,
                stdout=subprocess.DEVNULL, stderr=None,
            )


if __name__ == "__main__":
    run(Store(Path(sys.argv[1])), int(sys.argv[2]))
