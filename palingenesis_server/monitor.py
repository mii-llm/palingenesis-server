"""Read a bounded snapshot without loading model weights into the API process."""
import math
import re

from palingenesis.agent_tooling.monitor_run import analyze_run, parse_training_log

WINDOW = 256 * 1024
ROLLOUT_STEP = re.compile(r"(?:^|\s)step (\d+) \| (.+)$")


def progress(store, job):
    operation = job["operation"]
    if operation not in {"train", "rl", "distill", "autopilot"}:
        return None
    start, data, _ = store.read_log(job["id"], None, WINDOW)
    if start:
        data = data.partition(b"\n")[2]
    # The last line may still be in the middle of a write.
    if job["state"] == "running" and not data.endswith(b"\n"):
        data = data.rpartition(b"\n")[0]
    text = data.decode("utf-8", errors="replace")
    if operation in {"rl", "distill"}:
        for line in reversed(text.splitlines()):
            match = ROLLOUT_STEP.search(line)
            if match:
                metrics = {}
                for item in match[2].split():
                    key, sep, value = item.partition("=")
                    if sep:
                        try:
                            number = float(value)
                        except ValueError:
                            continue
                        metrics[key] = number if math.isfinite(number) else None
                return {"current_step": int(match[1]), "metrics": metrics}
        return {"status": "NO_DATA"}
    report = analyze_run(parse_training_log(text), job["expected_steps"])
    # The monitor reports diverged runs with NaN/Inf. HTTP JSON requires finite numbers.
    return {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in report.items()}
