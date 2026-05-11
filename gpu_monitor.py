"""
GPU metrics background poller for /monitor/api/system.

Ported from ov_monitor.py (terminal monitor). Reads sysfs/debugfs on the
xe driver (Arc B60) — no sudo needed. Runs a daemon thread at 1.5s interval.

Call start() once at server startup. get_data() is safe to call from any thread.
"""
import logging
import os
import re
import threading
import time

log = logging.getLogger("ov_server")

DRI_B60  = "/sys/kernel/debug/dri/0000:03:00.0"
HWMON_XE = "/sys/class/hwmon/hwmon5"
ENGINE_CYCLE_KEYS = ["rcs", "ccs", "bcs", "vcs", "vecs"]

_poller: "GpuPoller | None" = None


def start() -> None:
    global _poller
    if _poller is None:
        _poller = GpuPoller()


def get_data() -> dict:
    return _poller.get() if _poller else {}


def _read(path: str, default: str | None = None) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return default


def _read_int(path: str, default: int = 0) -> int:
    v = _read(path)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default


def _read_vram_mm() -> tuple[int, int]:
    """Returns (used_mib, total_mib) from vram0_mm debugfs."""
    text = _read(f"{DRI_B60}/vram0_mm", "") or ""
    used = total = 0
    for line in text.splitlines():
        m = re.match(r'\s*usage:\s*(\d+)', line)
        if m:
            used = int(m.group(1)) // (1024 * 1024)
        m = re.match(r'\s*size:\s*(\d+)', line)
        if m:
            total = int(m.group(1)) // (1024 * 1024)
    return used, total or 24480


def _read_fdinfo_cycles(pid: int) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        fdinfo_dir = f"/proc/{pid}/fdinfo"
        for fd in os.listdir(fdinfo_dir):
            content = _read(f"{fdinfo_dir}/{fd}", "") or ""
            if "drm-driver:\txe" not in content:
                continue
            for line in content.splitlines():
                m = re.match(r'(drm-(?:cycles|total-cycles)-\w+):\s*(\d+)', line)
                if m:
                    k, v = m.group(1), int(m.group(2))
                    result[k] = result.get(k, 0) + v
    except Exception:
        pass
    return result


def _find_server_pid() -> int | None:
    try:
        for pid_str in os.listdir("/proc"):
            if not pid_str.isdigit():
                continue
            if (_read(f"/proc/{pid_str}/comm", "") or "").strip() != "python3":
                continue
            fdinfo_dir = f"/proc/{pid_str}/fdinfo"
            try:
                for fd in os.listdir(fdinfo_dir):
                    if "drm-driver:\txe" in (_read(f"{fdinfo_dir}/{fd}", "") or ""):
                        return int(pid_str)
            except OSError:
                pass
    except Exception:
        pass
    return None


class GpuPoller:
    def __init__(self) -> None:
        self._data: dict = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._prev_cycles: dict[str, int] = {}
        self._prev_energy = 0
        self._prev_time = time.time()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="gpu-poller"
        )
        self._thread.start()
        log.info("GPU monitor poller started")

    def _poll(self) -> dict:
        d: dict = {}
        t2 = _read_int(f"{HWMON_XE}/temp2_input")
        t3 = _read_int(f"{HWMON_XE}/temp3_input")
        d["temp_gt_c"]  = round(t2 / 1000, 1) if t2 else None
        d["temp_mem_c"] = round(t3 / 1000, 1) if t3 else None

        fan = _read_int(f"{HWMON_XE}/fan1_input")
        d["fan_rpm"] = fan if fan else None

        p1 = _read_int(f"{HWMON_XE}/power1_cap")
        d["power_cap_w"] = round(p1 / 1_000_000, 1) if p1 else None
        d["_energy1_uj"] = _read_int(f"{HWMON_XE}/energy1_input")

        used_mib, total_mib = _read_vram_mm()
        d["vram_used_mib"]  = used_mib
        d["vram_total_mib"] = total_mib

        pid = _find_server_pid()
        d["_cycles"] = _read_fdinfo_cycles(pid) if pid else {}
        return d

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                d = self._poll()
                now = time.time()
                dt = now - self._prev_time

                e1 = d.pop("_energy1_uj", 0)
                if self._prev_energy and dt > 0 and e1 >= self._prev_energy:
                    d["power_w"] = round((e1 - self._prev_energy) / 1_000_000 / dt, 1)
                else:
                    d["power_w"] = None
                self._prev_energy = e1

                cur = d.pop("_cycles", {})
                eng_pct: dict[str, float] = {}
                if self._prev_cycles and dt > 0:
                    for eng in ENGINE_CYCLE_KEYS:
                        used_key  = f"drm-cycles-{eng}"
                        total_key = f"drm-total-cycles-{eng}"
                        delta_used  = cur.get(used_key,  0) - self._prev_cycles.get(used_key,  0)
                        delta_total = cur.get(total_key, 0) - self._prev_cycles.get(total_key, 0)
                        if delta_total > 0:
                            eng_pct[eng] = round(
                                min(100.0, max(0.0, delta_used / delta_total * 100)), 1
                            )
                self._prev_cycles = cur
                d["engine_pct"] = eng_pct
                self._prev_time = now
                with self._lock:
                    self._data = d
            except Exception as exc:
                log.debug(f"gpu_monitor poll: {exc}")
            time.sleep(1.5)

    def get(self) -> dict:
        with self._lock:
            return dict(self._data)

    def stop(self) -> None:
        self._stop.set()
