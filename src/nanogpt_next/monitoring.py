from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch


class TrainingStatusStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "status": "initializing",
            "updated_at": time.time(),
        }

    def update(self, **fields: Any) -> None:
        with self._lock:
            self._state.update(fields)
            self._state["updated_at"] = time.time()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)


class TrainingStatusServer:
    def __init__(self, store: TrainingStatusStore, host: str, port: int) -> None:
        self.store = store
        self.host = host
        self.port = port
        self.error: str | None = None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        handler_cls = self._build_handler()
        try:
            self._server = ThreadingHTTPServer((self.host, self.port), handler_cls)
        except OSError as exc:
            self.error = str(exc)
            return False
        self._thread = threading.Thread(target=self._server.serve_forever, name="nanogpt-next-status-api", daemon=True)
        self._thread.start()
        return True

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join()

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        store = self.store

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path not in {"/", "/status", "/healthz"}:
                    self.send_response(404)
                    self.end_headers()
                    return
                payload = {
                    "ok": True,
                    "path": self.path,
                    "status": store.snapshot(),
                }
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return

        return Handler


class DeviceTemperatureMonitor:
    def __init__(
        self,
        device: torch.device,
        enabled: bool,
        max_celsius: float,
        resume_celsius: float,
        poll_interval: float,
        query_command: str = "",
    ) -> None:
        self.device = device
        self.enabled = enabled
        self.max_celsius = max_celsius
        self.resume_celsius = resume_celsius
        self.poll_interval = poll_interval
        self.query_command = query_command.strip()
        self.backend_name = "disabled"
        self.unavailable_reason: str | None = None
        self._command: list[str] | str | None = None
        self._use_shell = False
        if self.enabled:
            self._initialize_backend()

    def is_available(self) -> bool:
        return self.enabled and self._command is not None and self.unavailable_reason is None

    def should_pause(self, temperature_celsius: float | None) -> bool:
        return temperature_celsius is not None and temperature_celsius > self.max_celsius

    def read_temperature(self) -> float | None:
        if not self.is_available() or self._command is None:
            return None
        try:
            completed = subprocess.run(
                self._command,
                capture_output=True,
                text=True,
                timeout=5,
                shell=self._use_shell,
                check=False,
            )
        except Exception as exc:
            self.unavailable_reason = f"temperature query failed: {exc}"
            return None
        if completed.returncode != 0:
            self.unavailable_reason = completed.stderr.strip() or completed.stdout.strip() or "temperature query failed"
            return None
        temperature = _parse_temperature(completed.stdout)
        if temperature is None:
            self.unavailable_reason = "temperature query output did not contain a numeric value"
        return temperature

    def throttle_if_needed(
        self,
        current_temperature: float | None,
        status_store: TrainingStatusStore | None = None,
        current_step: int | None = None,
        progress = None,
    ) -> tuple[float | None, float, bool]:
        if not self.should_pause(current_temperature):
            return current_temperature, 0.0, False

        paused_since = time.perf_counter()
        temperature = current_temperature
        if progress is not None:
            progress.write(
                f"thermal throttle: step={current_step} gpu_temp={temperature:.1f}C exceeds {self.max_celsius:.1f}C; pausing"
            )
        if status_store is not None:
            status_store.update(
                thermal_paused=True,
                temperature_celsius=temperature,
                thermal_threshold_celsius=self.max_celsius,
                thermal_resume_celsius=self.resume_celsius,
                thermal_pause_started_at=time.time(),
            )

        while temperature is not None and temperature > self.resume_celsius:
            time.sleep(self.poll_interval)
            temperature = self.read_temperature()
            if status_store is not None:
                status_store.update(
                    thermal_paused=True,
                    temperature_celsius=temperature,
                    thermal_threshold_celsius=self.max_celsius,
                    thermal_resume_celsius=self.resume_celsius,
                    thermal_pause_seconds=time.perf_counter() - paused_since,
                )

        paused_seconds = time.perf_counter() - paused_since
        if progress is not None:
            if temperature is None:
                progress.write("thermal throttle: temperature query became unavailable; resuming training")
            else:
                progress.write(
                    f"thermal throttle: resumed at gpu_temp={temperature:.1f}C after {paused_seconds:.1f}s"
                )
        if status_store is not None:
            status_store.update(
                thermal_paused=False,
                temperature_celsius=temperature,
                thermal_pause_seconds=paused_seconds,
                thermal_pause_started_at=None,
            )
        return temperature, paused_seconds, True

    def _initialize_backend(self) -> None:
        if self.query_command:
            self.backend_name = "custom"
            self._command = self.query_command
            self._use_shell = True
            return
        if self.device.type == "cuda":
            binary = shutil.which("nvidia-smi") or shutil.which("nvidia-smi.exe")
            if binary is None:
                self.backend_name = "nvidia-smi"
                self.unavailable_reason = "nvidia-smi was not found in PATH"
                return
            device_index = 0 if self.device.index is None else int(self.device.index)
            self.backend_name = "nvidia-smi"
            self._command = [
                binary,
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
                "-i",
                str(device_index),
            ]
            return
        self.backend_name = "unsupported"
        self.unavailable_reason = (
            f"automatic temperature monitoring is not implemented for device type '{self.device.type}'; "
            "set monitoring.thermal_query_command to provide a custom shell command"
        )


def _parse_temperature(raw_output: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", raw_output)
    if match is None:
        return None
    return float(match.group(0))
