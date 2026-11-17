"""Run with Python inside the running release container, including on a CPU runner."""
import json
import os
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def request(path, body=None):
    headers = {"Authorization": "Bearer " + os.environ["PGS_SERVER_TOKEN"]}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    return urlopen(Request("http://127.0.0.1:8000" + path, data=data, headers=headers), timeout=5)


def main():
    deadline = time.monotonic() + 90
    while True:
        try:
            with request("/health") as response:
                assert json.load(response)["status"] == "ok"
            break
        except URLError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)

    with request("/v1/jobs", {"operation": "init", "args": ["--list"]}) as response:
        assert response.status == 202
        job = json.load(response)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        with request(job["status_url"]) as response:
            status = json.load(response)
        if status["state"] in {"queued", "running"}:
            time.sleep(1)
            continue
        with request(job["logs_url"]) as response:
            log = response.read()
        assert status["state"] == "succeeded" and status["exit_code"] == 0, (status, log)
        assert b"pgs train" in log, log
        print("Container health, authenticated submission, polling and SDK log verified.")
        return
    raise TimeoutError("SDK job did not finish within 90 seconds")


if __name__ == "__main__":
    main()
