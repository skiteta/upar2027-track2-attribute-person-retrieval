"""Small stage diagnostics; no environment mutation or remote communication."""
from pathlib import Path
import resource

import torch


def runtime_snapshot():
    result = {"peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    try:
        rows = Path("/proc/self/status").read_text().splitlines()
        result["rss_kib"] = next(int(line.split()[1]) for line in rows if line.startswith("VmRSS:"))
    except (OSError, StopIteration, ValueError):
        result["rss_kib"] = None
    for name in ("memory.current", "memory.max", "memory.events"):
        try:
            result["cgroup_"+name.replace(".", "_")] = (Path("/sys/fs/cgroup") / name).read_text().strip()[:512]
        except OSError:
            pass
    if torch.cuda.is_initialized():
        free, total = torch.cuda.mem_get_info()
        result.update(cuda_free_bytes=free, cuda_total_bytes=total,
                      cuda_allocated_bytes=torch.cuda.memory_allocated(),
                      cuda_reserved_bytes=torch.cuda.memory_reserved())
    return result
