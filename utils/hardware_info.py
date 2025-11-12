import platform
import sys
import psutil
import cpuinfo

def get_hardware_info() -> None:
    info = {
        "system": platform.platform(),
        "python_version": sys.version,
        "cpu": {},
        "ram": {},
        "gpu": {},
    }

    info["cpu"]["name"] = cpuinfo.get_cpu_info()["brand_raw"]
    info["cpu"]["physical_cores"] = psutil.cpu_count(logical=False)
    info["cpu"]["logical_cores"] = psutil.cpu_count(logical=True)
    info["cpu"]["max_frequency_ghz"] = psutil.cpu_freq().max/1000


    for x,y in info.items():
        print(x, y)
    print(psutil.cpu_freq())

get_hardware_info()