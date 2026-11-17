import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from palingenesis_server.api import create_app
from palingenesis_server.store import Store


class JobsTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.state = self.root / "state"
        env = patch.dict(os.environ, PGS_SERVER_STATE=str(self.state), PGS_SERVER_WORKDIR=str(self.work),
                         PGS_SERVER_TOKEN="test-operator-token")
        env.start()
        self.addCleanup(env.stop)

    def client(self):
        return TestClient(create_app(), headers={"Authorization": "Bearer test-operator-token"})

    def submit(self, api, operation, args):
        response = api.post("/v1/jobs", json={"operation": operation, "args": args})
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.headers["Location"], response.json()["status_url"])
        return response.json()

    def wait(self, api, job, wanted):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            response = api.get(job["status_url"])
            self.assertEqual(response.status_code, 200, response.text)
            state = response.json()["state"]
            if state == wanted:
                return response.json()
            if state in {"succeeded", "failed", "interrupted"}:
                self.fail(f"Unexpected {state}: {api.get(job['logs_url']).text}")
            time.sleep(0.05)
        self.fail(f"Timed out waiting for {wanted}")

    def test_real_sdk_success_failure_and_cursor(self):
        with self.client() as api:
            (self.work / "steps.log").write_text("step=1 loss=0.25 lr=0.1 tok/s=10 dt=2s\n")
            job = self.submit(api, "monitor", ["--log_file", "steps.log", "--brief"])
            self.assertEqual(self.wait(api, job, "succeeded")["exit_code"], 0)
            log = api.get(job["logs_url"]).content
            self.assertIn(b"[HEALTHY] step=1", log)
            first = api.get(job["logs_url"], params={"offset": 0, "limit": 5})
            rest = api.get(job["logs_url"], params={"offset": first.headers["X-Next-Offset"]})
            self.assertEqual(first.content + rest.content, log)
            self.assertEqual(api.get(job["logs_url"], params={"limit": 5}).content, log[-5:])
            self.assertEqual(api.get(job["logs_url"], params={"offset": len(log)}).content, b"")
            bad = self.submit(api, "monitor", [])
            self.assertEqual(self.wait(api, bad, "failed")["exit_code"], 2)
            self.assertIn("--log_file", api.get(bad["logs_url"]).text)
            self.assertEqual([x["id"] for x in api.get("/v1/jobs").json()], [bad["id"], job["id"]])

    def test_auth_and_validation(self):
        with self.client() as api:
            self.assertEqual(api.get("/v1/jobs", headers={"Authorization": ""}).status_code, 401)
            for payload in ({"operation": "sh"}, {"operation": "init", "processes": 2},
                            {"operation": "train", "args": ["bad\x00arg"]}):
                self.assertEqual(api.post("/v1/jobs", json=payload).status_code, 422)
            self.assertEqual(api.get("/v1/jobs/missing/logs").status_code, 404)

    def test_live_sdk_monitor_and_api_restart(self):
        # A controlled training process lets the test restart the HTTP service mid-step.
        package = self.work / "palingenesis"
        package.mkdir()
        (package / "__init__.py").touch()
        (package / "cli.py").write_text(
            "from pathlib import Path\nimport time\n"
            "print('step=1 loss=0.25 lr=0.01 tok/s=10 dt=2s', flush=True)\n"
            "for _ in range(300):\n"
            "    if Path('finish').exists(): break\n"
            "    time.sleep(0.05)\n"
            "else: raise SystemExit(3)\n"
        )
        with self.client() as api:
            job = self.submit(api, "train", ["--config", "example.yaml"])
            try:
                self.wait(api, job, "running")
                deadline = time.monotonic() + 5
                while api.get(job["status_url"]).json()["progress"]["status"] == "NO_DATA":
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.05)
                self.assertIn("step=1 loss=0.25", api.get(job["logs_url"]).text)
                second = self.submit(api, "version", [])
            except BaseException:
                (self.work / "finish").touch()
                raise
        with self.client() as api:
            try:
                status = api.get(job["status_url"]).json()
                self.assertEqual(status["state"], "running")
                self.assertEqual(status["progress"]["current_step"], 1)
                self.assertEqual(status["progress"]["current_loss"], 0.25)
                self.assertEqual(api.get(second["status_url"]).json()["state"], "queued")
            finally:
                (self.work / "finish").touch()
            self.wait(api, job, "succeeded")
            self.wait(api, second, "succeeded")
            self.assertEqual(api.get(job["logs_url"]).text.count("step=1"), 1)

    def test_interrupted_worker_and_nonfinite_metrics(self):
        store = Store(self.state)
        store.initialize()
        from palingenesis_server.api import JobRequest

        job_id = store.enqueue(JobRequest(operation="train"), self.work)
        store.claim()
        store.log_path(job_id).write_text("step=1 loss=nan lr=0.1\n")
        with self.client() as api:
            result = self.wait(api, {"status_url": f"/v1/jobs/{job_id}"}, "interrupted")
            self.assertIsNone(result["exit_code"])
            self.assertEqual(result["progress"]["status"], "CRASHED")
            self.assertIsNone(result["progress"]["current_loss"])
