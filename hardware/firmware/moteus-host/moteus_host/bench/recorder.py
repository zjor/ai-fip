"""Write immutable raw bench samples and run provenance."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .servo import STATE_FIELDS


TELEMETRY_FIELDS = (
    "sample",
    "host_time_s",
    "scheduled_time_s",
    "request_time_s",
    "response_time_s",
    "experiment_phase",
    "command_mode",
    "command_position_rev",
    "command_velocity_rev_s",
    "feedforward_torque_nm",
    "maximum_torque_nm",
    "velocity_limit_rev_s",
    "accel_limit_rev_s2",
    "watchdog_timeout_s",
    *STATE_FIELDS,
)
EVENT_FIELDS = ("host_time_s", "event", "value", "unit", "note")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_metadata(workdir: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=workdir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        return {
            "revision": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class RunRecorder:
    def __init__(
        self,
        *,
        output_root: Path,
        experiment: str,
        resolved_config: dict[str, Any],
    ) -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        base_run_id = f"{timestamp}_{experiment}"
        run_id = base_run_id
        suffix = 1
        while (output_root / run_id).exists():
            suffix += 1
            run_id = f"{base_run_id}_{suffix}"

        self.path = output_root / run_id
        self.path.mkdir(parents=True)
        self._start_utc = datetime.now(UTC)
        self._metadata: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "experiment": experiment,
            "status": "running",
            "start_time_utc": self._start_utc.isoformat(),
            "command": sys.argv,
            "resolved_config": resolved_config,
            "software": {
                "python": sys.version.split()[0],
                "moteus": _package_version("moteus"),
                "project": _package_version("fip-moteus-host"),
                "git": _git_metadata(Path(__file__).resolve().parents[2]),
            },
        }
        _write_json(self.path / "run.json", self._metadata)

        self._telemetry_file = (self.path / "telemetry.csv").open(
            "w", newline=""
        )
        self._telemetry = csv.DictWriter(
            self._telemetry_file, fieldnames=TELEMETRY_FIELDS, extrasaction="raise"
        )
        self._telemetry.writeheader()
        self._events_file = (self.path / "events.csv").open("w", newline="")
        self._events = csv.DictWriter(self._events_file, fieldnames=EVENT_FIELDS)
        self._events.writeheader()
        self._sample_count = 0

    def attach_controller(self, configuration: str, device: dict[str, Any]) -> None:
        (self.path / "controller-config.txt").write_text(configuration)
        self._metadata["controller"] = device
        _write_json(self.path / "run.json", self._metadata)

    def write_sample(self, sample: dict[str, Any]) -> None:
        row = {field: sample.get(field) for field in TELEMETRY_FIELDS}
        self._telemetry.writerow(row)
        self._sample_count += 1
        if self._sample_count % 50 == 0:
            self._telemetry_file.flush()

    def event(
        self,
        host_time_s: float,
        event: str,
        *,
        value: Any = None,
        unit: str | None = None,
        note: str | None = None,
    ) -> None:
        self._events.writerow(
            {
                "host_time_s": host_time_s,
                "event": event,
                "value": value,
                "unit": unit,
                "note": note,
            }
        )
        self._events_file.flush()

    def close(self, status: str, reason: str | None = None) -> None:
        if self._telemetry_file.closed:
            return
        self._telemetry_file.flush()
        self._events_file.flush()
        self._telemetry_file.close()
        self._events_file.close()
        self._metadata.update(
            {
                "status": status,
                "reason": reason,
                "end_time_utc": datetime.now(UTC).isoformat(),
                "sample_count": self._sample_count,
            }
        )
        _write_json(self.path / "run.json", self._metadata)
