"""Show how the TimeGAN jobs on Rangpur are doing: queue state, phase, step, scores and time left.

Run on the login node, from the project folder:
  python3 rangpur_status.py
or from the laptop:
  ssh s4980031@rangpur.compute.eait.uq.edu.au "cd comp3710/TimeGAN_LOB_s49800311 && python3 rangpur_status.py"
Pass a log path to read that log instead of the newest slurm-<job>.out.
Standard library only, so it runs without the conda env.
"""
import ast
import glob
import os
import re
import subprocess
import sys

AFTER_TRAINING_MIN = 10   # test evaluation, CPU audit of five tickers and the GIFs
REFERENCE_SCORE = 0.665   # best validation score of the full arm with 10k joint steps (laptop)
PHASE_LINE = re.compile(r"Phase (\w+): steps (\d+) to (\d+)")
FINISHED_LINE = re.compile(r"Phase (\w+) finished: (\d+) steps in (\d+) s")
STEP_LINE = re.compile(r"\s+\[(\w+)\s+(\d+)\] (\{.*\})")


def run(cmd):
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True).stdout
    except FileNotFoundError:
        return ""


def to_seconds(elapsed):
    """Slurm times: M:SS, H:MM:SS or D-HH:MM:SS."""
    days = 0
    if "-" in elapsed:
        d, elapsed = elapsed.split("-")
        days = int(d)
    parts = [int(p) for p in elapsed.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours, minutes, secs = parts
    return days * 86400 + hours * 3600 + minutes * 60 + secs


def duration(seconds):
    seconds = max(0, int(seconds))
    return "{}h {:02d}m".format(seconds // 3600, seconds % 3600 // 60)


def queue():
    """My jobs in the queue, keyed by job id."""
    jobs = {}
    for line in run(["squeue", "--me", "-h", "-o", "%i|%j|%T|%M|%R"]).splitlines():
        job, name, state, elapsed, reason = line.split("|", 4)
        jobs[job] = {"name": name, "state": state, "elapsed": elapsed, "reason": reason}
    return jobs


def final_state(job):
    """State, exit code and run time of a job that has left the queue."""
    out = run(["sacct", "-j", job, "-n", "-X", "-o", "State,ExitCode,Elapsed"]).split()
    return " ".join(out) if out else "unknown (sacct gave nothing)"


def read_log(path):
    """Phases, finished phase times, the latest validation line and the best checkpoint."""
    log = {"phases": {}, "finished": {}, "last": None, "best": None, "evaluating": False, "audited": False,
           "done": False}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = PHASE_LINE.match(line)
            if m:
                log["phases"][m.group(1)] = int(m.group(3))
            m = FINISHED_LINE.match(line)
            if m:
                log["finished"][m.group(1)] = int(m.group(3))
            m = STEP_LINE.match(line)
            if m:
                metrics = ast.literal_eval(m.group(3))
                log["last"] = (m.group(1), int(m.group(2)), metrics)
                if metrics.get("new_best"):
                    log["best"] = (m.group(1), int(m.group(2)), metrics["score"])
            log["evaluating"] |= line.startswith("Test hour evaluation")
            log["audited"] |= line.startswith("Comparison written")
            log["done"] |= line.startswith("Finished")
    return log


def report_progress(log, elapsed):
    """Where training is and, once the joint phase runs, how long is left (elapsed in seconds or None)."""
    if log["done"]:
        print("  Job script finished: training, audit and GIFs are done.")
        return
    if log["audited"]:
        print("  Audit written; drawing the GIFs.")
        return
    if log["evaluating"]:
        print("  Training finished; evaluating the test hour, then the audit and GIFs (about {} min).".format(
            AFTER_TRAINING_MIN))
        return
    if log["last"] is None:
        print("  Loading data or in the first 500 steps; no validation line yet.")
        return
    phase, step, metrics = log["last"]
    total = log["phases"].get(phase, 0)
    print("  Phase {}: step {} of {} ({:.0f}%)".format(phase, step, total, 100 * step / max(total, 1)))
    if phase == "joint" and elapsed is not None:
        joint_seconds = elapsed - sum(log["finished"].values())
        rate = step / max(joint_seconds, 1)
        left = (total - step) / max(rate, 1e-9) + 60 * AFTER_TRAINING_MIN
        print("  Joint phase: {:.1f} steps/s, about {} left (training, audit and GIFs)".format(rate, duration(left)))
    elif phase != "joint":
        print("  Time left is estimated once the joint phase starts.")
    tickers = sorted(k[len("score_"):] for k in metrics if k.startswith("score_"))
    if "score" in metrics:
        per_ticker = "  ".join("{} {:.3f}".format(t, metrics["score_" + t]) for t in tickers)
        shown = "   (" + per_ticker + ")" if tickers else ""
        print("  Latest validation score {:.3f}{}".format(metrics["score"], shown))
    if log["best"]:
        print("  Best so far: {:.3f} at {} step {}   (10k-step run's best: {:.3f})".format(
            log["best"][2], log["best"][0], log["best"][1], REFERENCE_SCORE))


def show_errors(out_path):
    err_path = out_path[:-4] + ".err"
    if not os.path.exists(err_path):
        return
    with open(err_path, errors="replace") as fh:
        lines = [line.rstrip() for line in fh if line.strip()]
    if any("Error" in line or "Traceback" in line for line in lines):
        print("  Errors in {}:".format(os.path.basename(err_path)))
        for line in lines[-8:]:
            print("    " + line)


def job_id(path):
    m = re.search(r"(\d+)\.out$", path)
    return m.group(1) if m else ""


def main():
    jobs = queue()
    print("Queue:" if jobs else "Queue: none of your jobs are queued or running.")
    for job, j in sorted(jobs.items()):
        print("  {}  {:16s} {:8s} {:>10s}  {}".format(job, j["name"], j["state"], j["elapsed"], j["reason"]))

    logs = sys.argv[1:] or [p for p in [max(glob.glob("slurm-setup-*.out"), key=job_id, default=None),
                                        max(glob.glob("slurm-[0-9]*.out"), key=job_id, default=None)] if p]
    for path in logs:
        job = job_id(path)
        print("\n{} (job {}):".format(path, job or "?"))
        if job in jobs:
            print("  {} for {}".format(jobs[job]["state"].lower(), jobs[job]["elapsed"]))
        elif job:
            print("  Left the queue: " + final_state(job))
        if "setup" in os.path.basename(path):
            with open(path, errors="replace") as fh:
                for line in fh.read().splitlines()[-4:]:
                    print("  " + line)
        else:
            running = job in jobs and jobs[job]["state"] == "RUNNING"
            report_progress(read_log(path), to_seconds(jobs[job]["elapsed"]) if running else None)
        show_errors(path)


if __name__ == "__main__":
    main()
