"""
advisor/system_probe.py — снимок состояния системы

API:
    probe_system() -> SystemSnapshot

Snapshot содержит CPU, RAM, GPU (VRAM), диск, torch-версию.
psutil и pynvml опциональны — если нет, поля возвращают None.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class GpuInfo:
    name: str
    vram_total_gb:  float
    vram_used_gb:   float
    vram_free_gb:   float
    utilization_pct: Optional[float]   # None если pynvml нет
    temp_c:          Optional[float]


@dataclass
class SystemSnapshot:
    cpu_count:        int
    cpu_freq_mhz:     Optional[float]
    cpu_load_pct:     Optional[float]     # среднее за 1 с
    ram_total_gb:     float
    ram_used_gb:      float
    ram_free_gb:      float
    disk_free_gb:     float
    torch_version:    str
    cuda_available:   bool
    gpus:             list[GpuInfo] = field(default_factory=list)

    @property
    def primary_gpu(self) -> Optional[GpuInfo]:
        return self.gpus[0] if self.gpus else None


def probe_system(data_root: str = ".") -> SystemSnapshot:
    import os
    cpu_count = os.cpu_count() or 1

    # psutil (optional)
    try:
        import psutil
        vm    = psutil.virtual_memory()
        ram_t = vm.total    / 1024**3
        ram_u = vm.used     / 1024**3
        ram_f = vm.available/ 1024**3
        cpu_f = psutil.cpu_freq()
        cpu_freq  = cpu_f.current if cpu_f else None
        cpu_load  = psutil.cpu_percent(interval=0.2)
        disk_free = psutil.disk_usage(data_root).free / 1024**3
    except ImportError:
        ram_t, ram_u, ram_f = 0.0, 0.0, 0.0
        cpu_freq, cpu_load  = None, None
        disk_free           = 0.0

    snap = SystemSnapshot(
        cpu_count     = cpu_count,
        cpu_freq_mhz  = cpu_freq,
        cpu_load_pct  = cpu_load,
        ram_total_gb  = ram_t,
        ram_used_gb   = ram_u,
        ram_free_gb   = ram_f,
        disk_free_gb  = disk_free,
        torch_version = torch.__version__,
        cuda_available= torch.cuda.is_available(),
    )

    # CUDA (через torch, без pynvml)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / 1024**3
            used  = torch.cuda.memory_allocated(i) / 1024**3
            free  = total - used

            # pynvml для utilization/temp (optional)
            util, temp = None, None
            try:
                import pynvml
                pynvml.nvmlInit()
                h    = pynvml.nvmlDeviceGetHandleByIndex(i)
                ut   = pynvml.nvmlDeviceGetUtilizationRates(h)
                util = float(ut.gpu)
                temp = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass

            snap.gpus.append(GpuInfo(
                name             = props.name,
                vram_total_gb    = total,
                vram_used_gb     = used,
                vram_free_gb     = free,
                utilization_pct  = util,
                temp_c           = temp,
            ))

    return snap
