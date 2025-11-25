import platform
import sys
import psutil
import cpuinfo
import torch
from typing import Dict, Any

def get_hardware_info() -> Dict[str, Any]:
    print("[INFO] Collecting hardware info...")

    info = {
        "platform": platform.platform(),
        "python_version": sys.version.split(' ')[0],
        "cpu": {},
        "ram": {},
        "gpu": {}
    }

    # === 2. POPRAWKA: Użyj cpuinfo zamiast platform ===
    info["cpu"]["name"] = cpuinfo.get_cpu_info()['brand_raw']

    info["cpu"]["physical_cores"] = psutil.cpu_count(logical=False)
    info["cpu"]["logical_cores"] = psutil.cpu_count(logical=True)
    freq = psutil.cpu_freq()
    info["cpu"]["max_frequency_mhz"] = round(freq.max, 2)

    ram_stats = psutil.virtual_memory()
    info["ram"]["total_gb"] = round(ram_stats.total / (1024 ** 3), 2)


    if torch.cuda.is_available():
        info["gpu"]["cuda_available"] = True
        gpu_id = 0
        info["gpu"]["name"] = torch.cuda.get_device_name(gpu_id)
        vram_total_bytes = torch.cuda.mem_get_info(gpu_id)[1]
        info["gpu"]["vram_total_gb"] = round(vram_total_bytes / (1024 ** 3), 2)

    else:
        info["gpu"]["cuda_available"] = False
        info["gpu"]["name"] = "N/A (CUDA not available)"
        info["gpu"]["vram_total_gb"] = 0.0

    return info