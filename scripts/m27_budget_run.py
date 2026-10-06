#!/usr/bin/env python3
"""Run one M27 GPU stage with a budget reservation and lifecycle-based GPU-hour ledger."""
import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "outputs/m27/gpu_hour_ledger.csv"
LIFECYCLE = ROOT / "outputs/m27/raw/h20_formal_lifecycle.jsonl"
LIMIT = 168.0
FIELDS = ["stage", "start_time", "end_time", "gpu_count", "elapsed_hours", "gpu_hours", "cumulative_gpu_hours", "reason"]


def stamp(t):
    return dt.datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


def usage_between(start_offset, end, start):
    if not LIFECYCLE.exists():
        return 0., "no GPU lifecycle start"
    starts = {}
    usage = 0.
    with LIFECYCLE.open() as f:
        f.seek(start_offset)
        for line in f:
            event = json.loads(line)
            key = (event.get("exp"), event.get("arm"))
            if event.get("event") == "start":
                starts[key] = event
            elif event.get("event") == "stop" and key in starts:
                began = starts.pop(key)
                usage += max(0., event["unix_ts"] - began["unix_ts"]) * len(began["instances"]) / 3600
    for began in starts.values():
        usage += max(0., end - max(start, began["unix_ts"])) * len(began["instances"]) / 3600
    return usage, "lifecycle start/stop; incomplete sessions charged until stage end" if starts else "lifecycle start/stop"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--gpu-count", type=int, required=True)
    p.add_argument("--reserve-hours", type=float, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command and a.command[0] == "--" else a.command
    if not command or a.gpu_count < 1 or a.reserve_hours <= 0:
        p.error("GPU count, reservation and command required")
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    if not LEDGER.exists():
        with LEDGER.open("w", newline="") as f:
            csv.writer(f).writerow(FIELDS)
    with LEDGER.open(newline="") as f:
        rows = list(csv.DictReader(f))
    used = float(rows[-1]["cumulative_gpu_hours"]) if rows else 0.
    reserved = a.gpu_count * a.reserve_hours
    if used + reserved > LIMIT:
        raise SystemExit(f"HARD GATE: used {used:.3f} + next reserved {reserved:.3f} > {LIMIT} GPU-h")
    print(f"[budget] used={used:.3f} GPU-h; next reservation={reserved:.3f}; hard cap={LIMIT}", flush=True)
    offset = LIFECYCLE.stat().st_size if LIFECYCLE.exists() else 0
    start = time.time()
    try:
        rc = subprocess.call(command, cwd=ROOT, timeout=a.reserve_hours * 3600)
    except subprocess.TimeoutExpired:
        rc = 124
        print(f"[budget] reservation wall timeout after {a.reserve_hours:.3f} h; inspect M27 PIDs before retry", flush=True)
    finally:
        end = time.time()
        actual, method = usage_between(offset, end, start)
        with LEDGER.open("a", newline="") as f:
            csv.writer(f).writerow([a.stage, stamp(start), stamp(end), a.gpu_count,
                                    f"{(end - start)/3600:.6f}", f"{actual:.6f}",
                                    f"{used + actual:.6f}", f"rc={locals().get('rc', 'interrupted')}; {method}"])
        print(f"[budget] stage={a.stage} actual={actual:.3f} cumulative={used+actual:.3f} GPU-h", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
