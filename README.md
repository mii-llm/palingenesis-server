# Palingenesis Server

Submit Palingenesis operations over HTTP, poll job state and training metrics, and read console output while the job runs. The server runs on a single Linux or macOS host with Python 3.11+. A SQLite queue schedules one job at a time. A distributed training job can use several GPUs on that host.

## Run with Docker

Use v0.1.0 on a Linux x86-64 NVIDIA training host. The image installs the SDK's locked CUDA 12.9 training environment. The host needs an NVIDIA driver compatible with CUDA 12.9 and the NVIDIA Container Toolkit.

```sh
docker pull ghcr.io/mii-llm/palingenesis-server:0.1.0
export PGS_SERVER_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
docker run -d --name palingenesis-server --init --gpus all --shm-size=2g \
  -p 127.0.0.1:8000:8000 -e PGS_SERVER_TOKEN \
  -v pgs-work:/workspace -v pgs-state:/state -v pgs-cache:/cache \
  ghcr.io/mii-llm/palingenesis-server:0.1.0
curl -fsS -H "Authorization: Bearer $PGS_SERVER_TOKEN" http://127.0.0.1:8000/health
```

For a private package, authenticate Docker to `ghcr.io` with an account that can read the package. To build the same tag from this checkout, run `docker build --platform linux/amd64 -t ghcr.io/mii-llm/palingenesis-server:0.1.0 .`.

The container runs as UID/GID 10001. The work volume holds configs, datasets and outputs. The state volume holds the SQLite queue and logs, and the cache volume holds downloaded models and compiled kernels. To use a host project, replace `-v pgs-work:/workspace` with a bind mount of your project directory and give UID 10001 write access to its output directories. The `init` operation can also generate a config directly in the work volume.

An API process restart on the host can leave its detached worker running. Stopping the entire container stops its training processes too. On the next start with the same state volume, the server marks unfinished running jobs as interrupted and dispatches queued jobs.

## Run from source

From this repository, install both sibling projects in the same Python environment. On a training host, use the Python environment with the PyTorch build and Palingenesis extras for your hardware and workload.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ../palingenesis -e .
export PGS_SERVER_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export PGS_SERVER_WORKDIR="$(cd ../palingenesis && pwd)"
export PGS_SERVER_STATE="$PWD/.palingenesis-server"
palingenesis-server --host 127.0.0.1 --port 8000
```

Open [interactive API docs](http://127.0.0.1:8000/docs) and use **Authorize** with the token, or use the examples below from a shell with the same token. All job, log and health endpoints require bearer authentication. The token grants operator access to Palingenesis commands, including their file, plugin and model-loading options. Give it only to callers trusted with the service account. Use TLS at your reverse proxy for remote clients.

Relative paths in CLI arguments and configs resolve as they would when running the command from `PGS_SERVER_WORKDIR` (default is the server's startup directory). Place your configs and datasets on the training host. Job state and captured logs live under `PGS_SERVER_STATE` (default `.palingenesis-server` in the startup directory). Use the same state directory after a restart.

## Submit, poll and read logs

This example lists the SDK's bundled configs and runs on CPU.

```sh
base=http://127.0.0.1:8000
job=$(curl -fsS "$base/v1/jobs" \
  -H "Authorization: Bearer $PGS_SERVER_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"operation":"init","args":["--list"]}' |
  python -c 'import json,sys; print(json.load(sys.stdin)["status_url"])')

curl -fsS -H "Authorization: Bearer $PGS_SERVER_TOKEN" "$base$job"
curl -fsi -H "Authorization: Bearer $PGS_SERVER_TOKEN" "$base$job/logs"
```

`POST /v1/jobs` returns HTTP 202 and a `Location` header. Poll the returned `status_url` every few seconds. Jobs move from `queued` to `running`, then to `succeeded` or `failed` based on the SDK exit code. Status includes timestamps, the exit code and an error description when available. For diagnostics, a nonzero exit code can mean the SDK found an issue. The diagnostic report is in the log.

The API process can restart while the worker finishes its current job. Queued jobs resume dispatch when the API is running. If the worker itself dies before it records the outcome, recovery marks the job `interrupted` once the SDK process releases its lock. The operator can inspect logs and checkpoints before submitting a new job.

| Endpoint | Result |
| --- | --- |
| `GET /health` | API liveness |
| `POST /v1/jobs` | Accept a job with an SDK operation and an array of individual CLI arguments |
| `GET /v1/jobs?limit=20&offset=0` | Newest jobs first, up to 100 per page |
| `GET /v1/jobs/{id}` | Job state and a current progress snapshot for training operations |
| `GET /v1/jobs/{id}/logs?limit=8192` | Most recent bytes of combined stdout and stderr |
| `GET /v1/jobs/{id}/logs?offset=0&limit=8192` | Read forward from a byte offset, up to 65,536 bytes at a time |

To follow a log without gaps, start at offset zero and use the response's `X-Next-Offset` in the next request. Empty output at the current end is normal while a job is running. After observing a terminal state, drain the log until `X-Next-Offset` equals `X-Log-Size`. Chunks are bytes, so a streaming client should use an incremental UTF-8 decoder. Logs and job history are retained on disk.

## Training and diagnostics

Before launching SFT, submit the SDK's pre-training diagnostic for the intended config and GPU memory capacity. Poll the diagnostic to completion and inspect its report.

```json
{"operation":"diagnose","args":["--config","configs/llama3_8b.yaml","--mode","pre","--gpu_memory_gb","80","--json"]}
```

Then submit the training job. The optional `expected_steps` supplies the total for the SDK's ETA calculation. Set it to match the training config.

```json
{"operation":"train","args":["--config","configs/llama3_8b.yaml","--train.max_steps","5000"],"expected_steps":5000}
```

For SFT, DPO and off-policy distillation, `progress` uses the SDK's monitor on the most recent 256 KiB of captured log. It reports the latest step, loss, throughput, trend, health and ETA when available. For RL and on-policy distillation it exposes the latest logged step and its numeric metrics. Before a step appears in the window the snapshot reports `NO_DATA`. The process exit code determines the terminal job state. Other operations expose lifecycle and log output. To analyze an existing training log, submit `monitor` or `loss` with the SDK's `--log_file` argument.

The operation field accepts the Palingenesis CLI commands for training, RL, distillation, preparation, model inspection, diagnostics, merging and the other commands enumerated in the API docs. Each element of `args` is one argument passed directly to the selected command. Set `processes` for single-node distributed `train`, `rl` or `distill-cache` jobs to launch them with PyTorch's distributed runner. Training artifacts are written to the paths selected by the SDK configuration.

## Test

```sh
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
```

The tests run CPU-only SDK commands and use a controlled long-running process to exercise live logs, monitor metrics, queue serialization and API restart.
