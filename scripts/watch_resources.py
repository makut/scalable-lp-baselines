"""External resource watcher for experiments and baseline runners.

The watcher can either wrap a command:

    python scripts/watch_resources.py -- torchrun --standalone --nproc_per_node=8 -m node2vec.train --config ...

or attach to an already running root PID:

    python scripts/watch_resources.py --pid 12345

It samples the whole process tree/process group, which is the important bit
for torchrun, multiprocessing workers, and external baseline launchers. CPU
and RAM are collected with psutil. NVIDIA GPU utilization and global device
memory usage are collected through nvidia-smi when it is available.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
import os
import signal
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import psutil
except ImportError as exc:  # pragma: no cover - exercised only on minimal envs.
    raise SystemExit(
        "scripts/watch_resources.py requires psutil. Install it with `pip install psutil` "
        "in the environment where you run experiments."
    ) from exc


LOGGER = logging.getLogger("watch_resources")
ProcessKey = tuple[int, float]


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def now_local_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def utc_slug() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def mib(num_bytes: Optional[float]) -> Optional[float]:
    if num_bytes is None:
        return None
    return float(num_bytes) / (1024.0 * 1024.0)


def rounded(value: Optional[float], digits: int = 3) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), digits)


def process_key(proc: psutil.Process) -> Optional[ProcessKey]:
    try:
        return int(proc.pid), float(proc.create_time())
    except (psutil.Error, OSError):
        return None


def same_process(key: ProcessKey, proc: psutil.Process) -> bool:
    current = process_key(proc)
    if current is None:
        return False
    return key[0] == current[0] and abs(key[1] - current[1]) < 0.001


def is_live_process(proc: psutil.Process) -> bool:
    try:
        if not proc.is_running():
            return False
        return proc.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, OSError):
        return False


def safe_getpgid(pid: int) -> Optional[int]:
    if not hasattr(os, "getpgid"):
        return None
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def safe_getsid(pid: int) -> Optional[int]:
    if not hasattr(os, "getsid"):
        return None
    try:
        return os.getsid(pid)
    except OSError:
        return None


class ProcessTracker:
    """Tracks roots, descendants, and optionally POSIX process groups/sessions."""

    def __init__(
        self,
        root_pids: Iterable[int],
        *,
        track_mode: str,
        include_children: bool = True,
    ) -> None:
        self.track_mode = track_mode
        self.include_children = include_children
        self.root_keys: set[ProcessKey] = set()
        self.processes: dict[ProcessKey, psutil.Process] = {}
        self.pgids: set[int] = set()
        self.sids: set[int] = set()

        for pid in root_pids:
            proc = psutil.Process(int(pid))
            key = process_key(proc)
            if key is None:
                raise ValueError(f"cannot inspect PID {pid}")
            self.root_keys.add(key)
            pgid = safe_getpgid(proc.pid)
            sid = safe_getsid(proc.pid)
            if pgid is not None:
                self.pgids.add(pgid)
            if sid is not None:
                self.sids.add(sid)
            self.processes[key] = proc

    @property
    def root_pids(self) -> list[int]:
        return sorted(pid for pid, _ in self.root_keys)

    def _remember(self, proc: psutil.Process, out: dict[ProcessKey, psutil.Process]) -> None:
        if not is_live_process(proc):
            return
        key = process_key(proc)
        if key is None:
            return
        cached = self.processes.get(key)
        out[key] = cached if cached is not None else proc

    def _alive_root_processes(self) -> list[psutil.Process]:
        roots: list[psutil.Process] = []
        for key in self.root_keys:
            pid, _ = key
            try:
                proc = psutil.Process(pid)
            except psutil.NoSuchProcess:
                continue
            except (psutil.Error, OSError):
                continue
            if same_process(key, proc) and is_live_process(proc):
                roots.append(proc)
        return roots

    def refresh(self) -> list[psutil.Process]:
        current: dict[ProcessKey, psutil.Process] = {}

        for key, proc in list(self.processes.items()):
            try:
                if is_live_process(proc) and same_process(key, proc):
                    current[key] = proc
            except (psutil.Error, OSError):
                continue

        roots = self._alive_root_processes()
        for root in roots:
            self._remember(root, current)
            if self.include_children:
                try:
                    for child in root.children(recursive=True):
                        self._remember(child, current)
                except (psutil.Error, OSError):
                    LOGGER.debug("Cannot list children for PID %s", root.pid, exc_info=True)

        if self.track_mode in {"process-group", "session"}:
            self._refresh_by_posix_scope(current)

        self.processes = current
        return [proc for _, proc in sorted(current.items(), key=lambda item: item[0][0])]

    def _refresh_by_posix_scope(self, current: dict[ProcessKey, psutil.Process]) -> None:
        if self.track_mode == "process-group" and not self.pgids:
            return
        if self.track_mode == "session" and not self.sids:
            return

        try:
            iterator = psutil.process_iter(["pid", "create_time"])
            for proc in iterator:
                try:
                    if self.track_mode == "process-group":
                        pgid = safe_getpgid(proc.pid)
                        if pgid is None or pgid not in self.pgids:
                            continue
                    else:
                        sid = safe_getsid(proc.pid)
                        if sid is None or sid not in self.sids:
                            continue
                    self._remember(proc, current)
                except (psutil.Error, OSError):
                    continue
        except (psutil.Error, OSError):
            LOGGER.debug("Cannot enumerate processes for %s tracking", self.track_mode, exc_info=True)


@dataclass
class ProcessMetrics:
    pid_count: int
    pids: list[int]
    cpu_percent: float
    rss_bytes: int
    vms_bytes: int
    uss_bytes: Optional[int]
    thread_count: int
    top_processes: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "pid_count": self.pid_count,
            "pids": self.pids,
            "cpu_percent": rounded(self.cpu_percent),
            "rss_mib": rounded(mib(self.rss_bytes)),
            "vms_mib": rounded(mib(self.vms_bytes)),
            "uss_mib": rounded(mib(self.uss_bytes)),
            "thread_count": self.thread_count,
            "top_processes": self.top_processes,
        }


def collect_process_metrics(procs: list[psutil.Process], *, top_n: int, collect_uss: bool) -> ProcessMetrics:
    cpu_total = 0.0
    rss_total = 0
    vms_total = 0
    uss_total = 0
    uss_available = False
    thread_total = 0
    details: list[dict[str, Any]] = []
    pids: list[int] = []

    for proc in procs:
        try:
            with proc.oneshot():
                pid = int(proc.pid)
                pids.append(pid)
                cpu = float(proc.cpu_percent(interval=None))
                cpu_total += cpu
                mem = proc.memory_info()
                rss_total += int(mem.rss)
                vms_total += int(mem.vms)
                if collect_uss:
                    try:
                        full_mem = proc.memory_full_info()
                        uss = int(getattr(full_mem, "uss"))
                        uss_total += uss
                        uss_available = True
                    except (AttributeError, psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                        uss = None
                    except (psutil.Error, OSError):
                        uss = None
                else:
                    uss = None
                try:
                    threads = int(proc.num_threads())
                    thread_total += threads
                except (psutil.Error, OSError):
                    threads = None
                try:
                    name = proc.name()
                except (psutil.Error, OSError):
                    name = ""
                try:
                    cmdline = " ".join(proc.cmdline())
                except (psutil.Error, OSError):
                    cmdline = ""
                details.append(
                    {
                        "pid": pid,
                        "name": name,
                        "cpu_percent": rounded(cpu),
                        "rss_mib": rounded(mib(mem.rss)),
                        "uss_mib": rounded(mib(uss)),
                        "threads": threads,
                        "cmdline": cmdline[:300],
                    }
                )
        except (psutil.NoSuchProcess, psutil.ZombieProcess, OSError):
            continue
        except (psutil.AccessDenied, PermissionError):
            pids.append(int(proc.pid))
            continue
    top_processes = sorted(
        details,
        key=lambda item: (float(item.get("rss_mib") or 0.0), float(item.get("cpu_percent") or 0.0)),
        reverse=True,
    )[: max(0, top_n)]
    return ProcessMetrics(
        pid_count=len(pids),
        pids=sorted(set(pids)),
        cpu_percent=cpu_total,
        rss_bytes=rss_total,
        vms_bytes=vms_total,
        uss_bytes=uss_total if uss_available else None,
        thread_count=thread_total,
        top_processes=top_processes,
    )


def parse_optional_float(value: str) -> Optional[float]:
    text = value.strip()
    if not text or text in {"N/A", "[N/A]", "Not Supported", "[Not Supported]"}:
        return None
    for suffix in ("MiB", "W", "%", "C"):
        text = text.replace(suffix, "")
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        return None


def csv_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in csv.reader(text.splitlines()):
        clean = [cell.strip() for cell in row]
        if clean and any(cell for cell in clean):
            rows.append(clean)
    return rows


@dataclass
class GpuDeviceInfo:
    index: str
    uuid: str
    name: str
    memory_total_mib: Optional[float]

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "uuid": self.uuid,
            "name": self.name,
            "memory_total_mib": rounded(self.memory_total_mib),
        }


@dataclass
class GpuDeviceSample:
    index: str
    uuid: str
    name: str
    memory_used_mib: Optional[float]
    memory_total_mib: Optional[float]
    utilization_gpu_percent: Optional[float]
    utilization_memory_percent: Optional[float]
    power_draw_w: Optional[float]
    temperature_c: Optional[float]

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "uuid": self.uuid,
            "name": self.name,
            "memory_used_mib": rounded(self.memory_used_mib),
            "memory_total_mib": rounded(self.memory_total_mib),
            "utilization_gpu_percent": rounded(self.utilization_gpu_percent),
            "utilization_memory_percent": rounded(self.utilization_memory_percent),
            "power_draw_w": rounded(self.power_draw_w),
            "temperature_c": rounded(self.temperature_c),
        }


@dataclass
class GpuMetrics:
    available: bool
    devices: list[GpuDeviceSample]
    error: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "error": self.error,
            "devices": [device.to_json() for device in self.devices],
        }


class NvidiaSmiSampler:
    def __init__(self, *, enabled: bool, timeout_sec: float) -> None:
        self.enabled = enabled
        self.timeout_sec = timeout_sec
        self.available = False
        self.error: Optional[str] = None
        self.inventory: list[GpuDeviceInfo] = []
        if enabled:
            self.inventory = self._query_inventory()
            self.available = bool(self.inventory)

    def _run(self, args: list[str]) -> tuple[Optional[str], Optional[str]]:
        try:
            completed = subprocess.run(
                ["nvidia-smi", *args],
                capture_output=True,
                check=False,
                text=True,
                timeout=self.timeout_sec,
            )
        except FileNotFoundError:
            return None, "nvidia-smi not found"
        except PermissionError:
            return None, "nvidia-smi exists but is not executable"
        except subprocess.TimeoutExpired:
            return None, f"nvidia-smi timed out after {self.timeout_sec:.1f}s"
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "").strip()
            return None, message or f"nvidia-smi exited with {completed.returncode}"
        return completed.stdout, None

    def _query_inventory(self) -> list[GpuDeviceInfo]:
        out, err = self._run(
            [
                "--query-gpu=index,uuid,name,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        if err is not None:
            self.error = err
            LOGGER.info("GPU metrics disabled: %s", err)
            return []
        devices: list[GpuDeviceInfo] = []
        for row in csv_rows(out or ""):
            if len(row) < 4:
                continue
            devices.append(
                GpuDeviceInfo(
                    index=row[0],
                    uuid=row[1],
                    name=row[2],
                    memory_total_mib=parse_optional_float(row[3]),
                )
            )
        return devices

    def _query_devices(self) -> tuple[list[GpuDeviceSample], Optional[str]]:
        out, err = self._run(
            [
                "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        if err is not None:
            return [], err
        devices: list[GpuDeviceSample] = []
        for row in csv_rows(out or ""):
            if len(row) < 9:
                continue
            devices.append(
                GpuDeviceSample(
                    index=row[0],
                    uuid=row[1],
                    name=row[2],
                    memory_used_mib=parse_optional_float(row[3]),
                    memory_total_mib=parse_optional_float(row[4]),
                    utilization_gpu_percent=parse_optional_float(row[5]),
                    utilization_memory_percent=parse_optional_float(row[6]),
                    power_draw_w=parse_optional_float(row[7]),
                    temperature_c=parse_optional_float(row[8]),
                )
            )
        return devices, None

    def sample(self) -> GpuMetrics:
        if not self.enabled:
            return GpuMetrics(available=False, devices=[], error="disabled")
        if not self.available:
            return GpuMetrics(
                available=False,
                devices=[],
                error=self.error or "no NVIDIA GPUs found",
            )

        devices, device_err = self._query_devices()
        if device_err is not None:
            return GpuMetrics(
                available=False,
                devices=[],
                error=device_err,
            )

        return GpuMetrics(
            available=True,
            devices=devices,
        )


@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    memory_available_mib: float

    def to_json(self) -> dict[str, Any]:
        return {
            "cpu_percent": rounded(self.cpu_percent),
            "memory_percent": rounded(self.memory_percent),
            "memory_available_mib": rounded(self.memory_available_mib),
        }


def collect_system_metrics() -> SystemMetrics:
    mem_info = psutil.virtual_memory()
    return SystemMetrics(
        cpu_percent=float(psutil.cpu_percent(interval=None)),
        memory_percent=float(mem_info.percent),
        memory_available_mib=float(mem_info.available) / (1024.0 * 1024.0),
    )


@dataclass
class WatchSample:
    timestamp: str
    elapsed_sec: float
    process: ProcessMetrics
    system: SystemMetrics
    gpu: GpuMetrics

    def to_json(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "elapsed_sec": rounded(self.elapsed_sec),
            "process": self.process.to_json(),
            "system": self.system.to_json(),
            "gpu": self.gpu.to_json(),
        }


@dataclass
class OnlineStats:
    count: int = 0
    total: float = 0.0
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def add(self, value: Any) -> None:
        if value is None or value == "":
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        if math.isnan(numeric) or math.isinf(numeric):
            return
        self.count += 1
        self.total += numeric
        self.minimum = numeric if self.minimum is None else min(self.minimum, numeric)
        self.maximum = numeric if self.maximum is None else max(self.maximum, numeric)

    def to_json(self) -> dict[str, Any]:
        mean = None if self.count == 0 else self.total / self.count
        return {
            "count": self.count,
            "min": rounded(self.minimum),
            "max": rounded(self.maximum),
            "mean": rounded(mean),
        }


class SampleWriter:
    BASE_COLUMNS = [
        "timestamp",
        "elapsed_sec",
        "tracked_pid_count",
        "tracked_thread_count",
        "process_cpu_percent",
        "process_rss_mib",
        "process_uss_mib",
        "process_vms_mib",
        "system_cpu_percent",
        "system_memory_percent",
        "system_memory_available_mib",
        "gpu_memory_used_mib",
        "gpu_max_utilization_percent",
        "gpu_mean_utilization_percent",
    ]

    def __init__(self, out_dir: Path, gpu_inventory: list[GpuDeviceInfo]) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_indexes = [device.index for device in gpu_inventory]
        self.columns = list(self.BASE_COLUMNS)
        for index in self.gpu_indexes:
            prefix = self._gpu_prefix(index)
            self.columns.extend(
                [
                    f"{prefix}_utilization_percent",
                    f"{prefix}_memory_used_mib",
                    f"{prefix}_memory_total_mib",
                    f"{prefix}_memory_utilization_percent",
                    f"{prefix}_power_draw_w",
                    f"{prefix}_temperature_c",
                ]
            )
        self.csv_path = self.out_dir / "samples.csv"
        self.jsonl_path = self.out_dir / "samples.jsonl"
        self.csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self.jsonl_file = self.jsonl_path.open("w", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.columns)
        self.csv_writer.writeheader()
        self.stats: dict[str, OnlineStats] = {column: OnlineStats() for column in self.columns if column != "timestamp"}
        self.sample_count = 0

    @staticmethod
    def _gpu_prefix(index: str) -> str:
        safe = "".join(char if char.isalnum() else "_" for char in str(index))
        return f"gpu{safe}"

    def write(self, sample: WatchSample) -> None:
        row = self.flatten(sample)
        self.csv_writer.writerow(row)
        self.jsonl_file.write(json.dumps(sample.to_json(), ensure_ascii=False) + "\n")
        self.csv_file.flush()
        self.jsonl_file.flush()
        self.sample_count += 1
        for column, stats in self.stats.items():
            stats.add(row.get(column))

    def flatten(self, sample: WatchSample) -> dict[str, Any]:
        devices_by_index = {device.index: device for device in sample.gpu.devices}
        used_values = [device.memory_used_mib for device in sample.gpu.devices if device.memory_used_mib is not None]
        util_values = [
            device.utilization_gpu_percent
            for device in sample.gpu.devices
            if device.utilization_gpu_percent is not None
        ]
        row: dict[str, Any] = {
            "timestamp": sample.timestamp,
            "elapsed_sec": rounded(sample.elapsed_sec),
            "tracked_pid_count": sample.process.pid_count,
            "tracked_thread_count": sample.process.thread_count,
            "process_cpu_percent": rounded(sample.process.cpu_percent),
            "process_rss_mib": rounded(mib(sample.process.rss_bytes)),
            "process_uss_mib": rounded(mib(sample.process.uss_bytes)),
            "process_vms_mib": rounded(mib(sample.process.vms_bytes)),
            "system_cpu_percent": rounded(sample.system.cpu_percent),
            "system_memory_percent": rounded(sample.system.memory_percent),
            "system_memory_available_mib": rounded(sample.system.memory_available_mib),
            "gpu_memory_used_mib": rounded(sum(used_values)) if used_values else "",
            "gpu_max_utilization_percent": rounded(max(util_values)) if util_values else "",
            "gpu_mean_utilization_percent": rounded(sum(util_values) / len(util_values)) if util_values else "",
        }
        for index in self.gpu_indexes:
            prefix = self._gpu_prefix(index)
            device = devices_by_index.get(index)
            row[f"{prefix}_utilization_percent"] = "" if device is None else rounded(device.utilization_gpu_percent)
            row[f"{prefix}_memory_used_mib"] = "" if device is None else rounded(device.memory_used_mib)
            row[f"{prefix}_memory_total_mib"] = "" if device is None else rounded(device.memory_total_mib)
            row[f"{prefix}_memory_utilization_percent"] = "" if device is None else rounded(device.utilization_memory_percent)
            row[f"{prefix}_power_draw_w"] = "" if device is None else rounded(device.power_draw_w)
            row[f"{prefix}_temperature_c"] = "" if device is None else rounded(device.temperature_c)
        return row

    def close(self) -> None:
        self.csv_file.close()
        self.jsonl_file.close()

    def stats_json(self) -> dict[str, Any]:
        return {column: stats.to_json() for column, stats in self.stats.items() if stats.count > 0}


def prime_cpu_counters(tracker: ProcessTracker) -> None:
    for proc in tracker.refresh():
        try:
            proc.cpu_percent(interval=None)
        except psutil.Error:
            continue
    psutil.cpu_percent(interval=None)


def write_run_metadata(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    command: Optional[list[str]],
    root_pids: list[int],
    track_mode: str,
    gpu_inventory: list[GpuDeviceInfo],
) -> None:
    metadata = {
        "started_at": now_local_iso(),
        "command": command,
        "attached_pids": args.pid,
        "root_pids": root_pids,
        "track_mode": track_mode,
        "include_children": not args.no_children,
        "interval_sec": args.interval,
        "duration_sec": args.duration,
        "collect_uss": not args.no_uss,
        "gpu_memory_scope": "global_device",
        "gpu_devices": [device.to_json() for device in gpu_inventory],
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if command:
        (out_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")


def command_finished(process: Optional[subprocess.Popen[Any]]) -> bool:
    return process is not None and process.poll() is not None


def signal_process_group(process: subprocess.Popen[Any], sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        try:
            process.send_signal(sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def normalize_exit_code(code: Optional[int]) -> int:
    if code is None:
        return 0
    if code < 0:
        return 128 + abs(code)
    return code


def live_summary(sample: WatchSample) -> str:
    gpu_bits = []
    for device in sample.gpu.devices:
        util = device.utilization_gpu_percent
        used_mem = device.memory_used_mib
        if util is None and used_mem is None:
            continue
        gpu_bits.append(f"gpu{device.index}: util={rounded(util, 1)}% used_mem={rounded(used_mem, 1)}MiB")
    gpu_text = " | ".join(gpu_bits) if gpu_bits else "gpu=n/a"
    return (
        f"t={sample.elapsed_sec:.1f}s pids={sample.process.pid_count} "
        f"cpu={sample.process.cpu_percent:.1f}% rss={mib(sample.process.rss_bytes) or 0:.1f}MiB "
        f"uss={mib(sample.process.uss_bytes) if sample.process.uss_bytes is not None else 'n/a'}MiB "
        f"{gpu_text}"
    )


def run(args: argparse.Namespace) -> int:
    command = list(args.command or [])
    if command and command[0] == "--":
        command = command[1:]
    pids = list(args.pid or [])
    if bool(command) == bool(pids):
        raise SystemExit("Use exactly one mode: either pass --pid PID or put the command after `--`.")

    process: Optional[subprocess.Popen[Any]] = None
    if command:
        LOGGER.info("Starting command: %s", shlex.join(command))
        process = subprocess.Popen(command, start_new_session=True)
        pids = [int(process.pid)]
        requested_track_mode = "process-group" if args.track == "auto" else args.track
    else:
        requested_track_mode = "tree" if args.track == "auto" else args.track

    tracker = ProcessTracker(
        pids,
        track_mode=requested_track_mode,
        include_children=not args.no_children,
    )
    sampler = NvidiaSmiSampler(enabled=not args.no_gpu, timeout_sec=args.nvidia_smi_timeout)
    out_dir = args.out_dir or Path(f"resource_watch_{utc_slug()}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(
        out_dir,
        args=args,
        command=command or None,
        root_pids=tracker.root_pids,
        track_mode=requested_track_mode,
        gpu_inventory=sampler.inventory,
    )
    writer = SampleWriter(out_dir, sampler.inventory)

    LOGGER.info("Writing samples to %s and %s", writer.csv_path, writer.jsonl_path)
    prime_cpu_counters(tracker)
    start_monotonic = time.monotonic()
    started_at = now_local_iso()
    next_print = start_monotonic + args.print_every if args.print_every > 0 else math.inf
    interrupted = False

    try:
        while True:
            elapsed_before_sleep = time.monotonic() - start_monotonic
            if args.duration is not None and elapsed_before_sleep >= args.duration:
                break

            sleep_for = args.interval
            if args.duration is not None:
                sleep_for = min(sleep_for, max(0.0, args.duration - elapsed_before_sleep))
            if sleep_for > 0:
                time.sleep(sleep_for)

            procs = tracker.refresh()
            if command:
                if command_finished(process) and not procs:
                    break
            elif not procs:
                LOGGER.info("No tracked processes remain; stopping watcher")
                break

            process_metrics = collect_process_metrics(procs, top_n=args.top_processes, collect_uss=not args.no_uss)
            sample = WatchSample(
                timestamp=now_local_iso(),
                elapsed_sec=time.monotonic() - start_monotonic,
                process=process_metrics,
                system=collect_system_metrics(),
                gpu=sampler.sample(),
            )
            writer.write(sample)

            if time.monotonic() >= next_print:
                LOGGER.info(live_summary(sample))
                next_print = time.monotonic() + args.print_every
    except KeyboardInterrupt:
        interrupted = True
        LOGGER.warning("Interrupted")
        if process is not None and process.poll() is None:
            LOGGER.warning("Forwarding SIGINT to command process group")
            signal_process_group(process, signal.SIGINT)
            try:
                process.wait(timeout=args.shutdown_grace)
            except subprocess.TimeoutExpired:
                LOGGER.warning("Command did not stop after SIGINT; sending SIGTERM")
                signal_process_group(process, signal.SIGTERM)
                try:
                    process.wait(timeout=max(1.0, args.shutdown_grace / 2.0))
                except subprocess.TimeoutExpired:
                    LOGGER.warning("Command did not stop after SIGTERM; sending SIGKILL")
                    signal_process_group(process, signal.SIGKILL)
                    process.wait(timeout=max(1.0, args.shutdown_grace / 2.0))
    finally:
        writer.close()

    finished_at = now_local_iso()
    duration = time.monotonic() - start_monotonic
    exit_code = normalize_exit_code(None if process is None else process.poll())
    if interrupted and process is None:
        exit_code = 130

    summary = {
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": rounded(duration),
        "sample_count": writer.sample_count,
        "exit_code": exit_code,
        "interrupted": interrupted,
        "command": command or None,
        "root_pids": tracker.root_pids,
        "track_mode": requested_track_mode,
        "metrics": writer.stats_json(),
        "output_files": {
            "csv": str(writer.csv_path),
            "jsonl": str(writer.jsonl_path),
            "summary": str(out_dir / "summary.json"),
            "metadata": str(out_dir / "run_metadata.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOGGER.info("Summary written to %s", out_dir / "summary.json")
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample CPU/RAM/NVIDIA GPU metrics for an experiment command or an existing PID.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/watch_resources.py -- python -m seal.seal_link_pred --config seal/configs/link_prediction_default.yaml\n"
            "  python scripts/watch_resources.py -- torchrun --standalone --nproc_per_node=8 -m node2vec.train --config node2vec/configs/node2vec_default.yaml\n"
            "  python scripts/watch_resources.py --pid 12345 --out-dir runs/watch_node2vec_8gpu\n"
        ),
    )
    parser.add_argument("--pid", action="append", type=int, help="Root PID to attach to. Can be passed multiple times.")
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds.")
    parser.add_argument("--duration", type=float, default=None, help="Optional maximum watcher duration in seconds.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for samples.csv/jsonl/summary.json.")
    parser.add_argument(
        "--track",
        choices=["auto", "tree", "process-group", "session"],
        default="auto",
        help=(
            "Process scope. auto=process-group for wrapped commands and tree for attached PIDs. "
            "Use process-group/session for launchers that keep workers outside the normal child tree."
        ),
    )
    parser.add_argument("--no-children", action="store_true", help="Track only the root PID(s), not recursive children.")
    parser.add_argument("--no-gpu", action="store_true", help="Disable nvidia-smi sampling.")
    parser.add_argument("--no-uss", action="store_true", help="Skip USS memory collection; RSS/VMS are still sampled.")
    parser.add_argument("--top-processes", type=int, default=5, help="Store this many largest processes in each JSONL sample.")
    parser.add_argument("--print-every", type=float, default=10.0, help="Print live one-line summaries every N seconds; 0 disables.")
    parser.add_argument("--nvidia-smi-timeout", type=float, default=2.0, help="Timeout for each nvidia-smi query in seconds.")
    parser.add_argument("--shutdown-grace", type=float, default=30.0, help="Grace period when Ctrl-C stops a wrapped command.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run, placed after `--`.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")
    if args.print_every < 0:
        parser.error("--print-every must be non-negative")
    configure_logging(args.verbose)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
