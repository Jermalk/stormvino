#!/home/jerzy/ov_env/bin/python3
"""
monitor_sidecar.py — GPU metrics + server health proxy for SVP monitor.
Runs independently of ov_server; survives server restarts/model loads.
Serves GET /metrics on :11436 (JSON, CORS open).
"""

import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.error import URLError
from urllib.request import urlopen

import psutil

SERVER_URL    = "http://localhost:11435"
SIDECAR_PORT  = 11436
HWMON_XE      = "/sys/class/hwmon/hwmon5"
VRAM_TOTAL_MiB = 24480          # B60 — confirmed by probe
ENGINE_KEYS   = ["rcs", "ccs", "bcs", "vcs", "vecs"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _read(path: str, default: str = "") -> str:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


def _read_int(path: str) -> int | None:
    v = _read(path).strip()
    return int(v) if v.isdigit() else None


# ── VRAM via /proc/fdinfo (xe driver, no sudo) ────────────────────────────────

def _vram_by_proc() -> dict[str, int]:
    """Returns {comm_name: vram_mib} for all xe DRM clients."""
    procs: dict[str, int] = {}
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            fd_dir = f"/proc/{pid}/fdinfo"
            try:
                total = 0
                for fd in os.listdir(fd_dir):
                    try:
                        content = _read(f"{fd_dir}/{fd}")
                        if "drm-driver:\txe" not in content:
                            continue
                        for line in content.splitlines():
                            m = re.match(r"drm-total-vram0:\s+(\d+)\s+KiB", line)
                            if m:
                                total = max(total, int(m.group(1)) // 1024)
                    except OSError:
                        pass
                if total > 0:
                    name = _read(f"/proc/{pid}/comm", pid).strip()
                    procs[name] = procs.get(name, 0) + total
            except OSError:
                pass
    except OSError:
        pass
    return procs


# ── Engine utilisation via fdinfo cycle-counter delta ─────────────────────────

def _fdinfo_cycles(pid: str) -> dict[str, int]:
    """Sum drm-cycles-* and drm-total-cycles-* across all xe fds for one pid."""
    result: dict[str, int] = {}
    try:
        for fd in os.listdir(f"/proc/{pid}/fdinfo"):
            content = _read(f"/proc/{pid}/fdinfo/{fd}")
            if "drm-driver:\txe" not in content:
                continue
            for line in content.splitlines():
                m = re.match(r"(drm-(?:cycles|total-cycles)-\w+):\s*(\d+)", line)
                if m:
                    result[m.group(1)] = result.get(m.group(1), 0) + int(m.group(2))
    except OSError:
        pass
    return result


def _find_server_pid() -> str | None:
    try:
        for pid in os.listdir("/proc"):
            if pid.isdigit() and _read(f"/proc/{pid}/comm").strip() == "ov_server":
                return pid
    except OSError:
        pass
    return None


# ── Shared poller state ───────────────────────────────────────────────────────

_state: dict = {
    "server_up":    False,
    "server_health": None,
    "vram_live": {
        "total_gb": round(VRAM_TOTAL_MiB / 1024, 2),
        "used_gb":  0.0,
        "by_proc":  {},
    },
    "system": None,
}
_lock = threading.Lock()


# ── GPU thread (500 ms) ───────────────────────────────────────────────────────

_prev_cycles: dict[str, int] = {}
_prev_energy: int = 0
_prev_time:   float = time.time()


def _poll_gpu() -> None:
    global _prev_cycles, _prev_energy, _prev_time
    while True:
        try:
            # --- VRAM ---
            by_proc = _vram_by_proc()
            used_mib = sum(by_proc.values())
            vram = {
                "total_gb": round(VRAM_TOTAL_MiB / 1024, 2),
                "used_gb":  round(used_mib / 1024, 2),
                "by_proc":  {k: round(v / 1024, 2) for k, v in by_proc.items()},
            }

            # --- temperatures ---
            t2 = _read_int(f"{HWMON_XE}/temp2_input")
            t3 = _read_int(f"{HWMON_XE}/temp3_input")
            temp_gt  = round(t2 / 1000, 1) if t2 else None
            temp_mem = round(t3 / 1000, 1) if t3 else None
            fan      = _read_int(f"{HWMON_XE}/fan1_input") or None
            p_cap    = _read_int(f"{HWMON_XE}/power1_cap")
            power_cap = round(p_cap / 1_000_000, 1) if p_cap else None
            e1       = _read_int(f"{HWMON_XE}/energy1_input") or 0

            # --- instantaneous power from energy delta ---
            now = time.time()
            dt  = now - _prev_time
            if _prev_energy and dt > 0 and e1 >= _prev_energy:
                power_w = round((e1 - _prev_energy) / 1_000_000 / dt, 1)
            else:
                power_w = None
            _prev_energy = e1

            # --- engine utilisation ---
            pid = _find_server_pid()
            cur_cycles = _fdinfo_cycles(pid) if pid else {}
            eng_pct: dict[str, float] = {}
            if _prev_cycles and dt > 0:
                for eng in ENGINE_KEYS:
                    used_key  = f"drm-cycles-{eng}"
                    total_key = f"drm-total-cycles-{eng}"
                    du = cur_cycles.get(used_key,  0) - _prev_cycles.get(used_key,  0)
                    dt2 = cur_cycles.get(total_key, 0) - _prev_cycles.get(total_key, 0)
                    eng_pct[eng] = round(min(100.0, max(0.0, du / dt2 * 100)), 1) if dt2 > 0 else 0.0
            _prev_cycles = cur_cycles
            _prev_time   = now

            gpu = {
                "temp_gt_c":   temp_gt,
                "temp_mem_c":  temp_mem,
                "fan_rpm":     fan,
                "power_w":     power_w,
                "power_cap_w": power_cap,
                "engine_pct":  eng_pct,
                "vram_used_mib":  used_mib,
                "vram_total_mib": VRAM_TOTAL_MiB,
            }

            with _lock:
                _state["vram_live"] = vram
                if _state["system"]:
                    _state["system"]["gpu"] = gpu
                else:
                    _state["system"] = {"gpu": gpu, "cpu": None, "memory": None}

        except Exception:
            pass
        time.sleep(0.5)


# ── Health proxy thread (1 s) ─────────────────────────────────────────────────

def _poll_health() -> None:
    while True:
        try:
            with urlopen(f"{SERVER_URL}/health", timeout=2) as r:
                health = json.loads(r.read())
            with _lock:
                _state["server_up"]     = True
                _state["server_health"] = health
        except Exception:
            with _lock:
                _state["server_up"] = False
        time.sleep(1.0)


# ── System (CPU + memory) thread (2 s) ───────────────────────────────────────

def _poll_system() -> None:
    while True:
        try:
            per_core = psutil.cpu_percent(interval=None, percpu=True)
            freq     = psutil.cpu_freq()
            load     = list(psutil.getloadavg())
            mem      = psutil.virtual_memory()
            swap     = psutil.swap_memory()
            temps: dict[str, float] = {}
            try:
                for sensor, readings in psutil.sensors_temperatures().items():
                    for r in readings:
                        temps[r.label or sensor] = r.current
            except Exception:
                pass

            cpu_data = {
                "percent":      round(sum(per_core) / len(per_core), 1) if per_core else 0.0,
                "per_core":     [round(p, 1) for p in per_core],
                "freq_ghz":     round(freq.current / 1000, 2) if freq else None,
                "freq_max_ghz": round(freq.max     / 1000, 2) if freq else None,
                "load_avg":     [round(x, 3) for x in load],
                "temps":        temps,
            }
            mem_data = {
                "ram_pct":      mem.percent,
                "ram_used_gb":  round(mem.used      / 1024**3, 1),
                "ram_total_gb": round(mem.total     / 1024**3, 1),
                "ram_avail_gb": round(mem.available / 1024**3, 1),
                "swap_pct":     swap.percent,
                "swap_used_gb": round(swap.used     / 1024**3, 1),
                "swap_total_gb": round(swap.total   / 1024**3, 1),
            }
            with _lock:
                if _state["system"]:
                    _state["system"]["cpu"]    = cpu_data
                    _state["system"]["memory"] = mem_data
                else:
                    _state["system"] = {"gpu": None, "cpu": cpu_data, "memory": mem_data}
        except Exception:
            pass
        time.sleep(2.0)


# ── HTTP server ───────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass   # silence access log

    def do_GET(self) -> None:
        if self.path not in ("/metrics", "/metrics/"):
            self.send_error(404)
            return
        with _lock:
            payload = json.dumps(_state).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()


if __name__ == "__main__":
    threading.Thread(target=_poll_gpu,    daemon=True).start()
    threading.Thread(target=_poll_health, daemon=True).start()
    threading.Thread(target=_poll_system, daemon=True).start()
    time.sleep(0.8)   # first polls settle
    srv = HTTPServer(("", SIDECAR_PORT), _Handler)
    srv.socket.setsockopt(6, 1, 1)  # SO_REUSEADDR via SOL_TCP
    print(f"ov-monitor-sidecar listening on :{SIDECAR_PORT}", flush=True)
    srv.serve_forever()
