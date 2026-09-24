#!/bin/bash
# Free GPUs per node in the `gpus` queue (qlist contains gpuHost). Run before every GPU qsub.
# Usage: bash scripts/gpu_avail.sh          -> table + totals
#        bash scripts/gpu_avail.sh <node>   -> also nvidia-smi on that node (model / memory)
pbsnodes -a 2>/dev/null | python3 -c '
import sys
nodes, cur = {}, None
for l in sys.stdin:
    if l and not l[0].isspace(): cur = l.strip(); nodes[cur] = {}
    elif cur and " = " in l:
        k, v = l.strip().split(" = ", 1); nodes[cur][k] = v
free = tot = 0
print("%-9s %-14s %10s %5s %7s  queues" % ("node", "state", "free/ngpus", "cpus", "mem GB"))
for n, d in sorted(nodes.items()):
    g = int(d.get("resources_available.ngpus", 0) or 0)
    if not g or "gpuHost" not in d.get("resources_available.qlist", ""): continue
    u = int(d.get("resources_assigned.ngpus", 0) or 0); s = d["state"]
    if "down" in s or "offline" in s: continue
    ncpu = int(d.get("resources_available.ncpus", 0)) - int(d.get("resources_assigned.ncpus", 0) or 0)
    mem = int(d.get("resources_available.mem", "0mb")[:-2]) // 1024
    print("%-9s %-14s %4d/%-5d %5d %7d  %s" % (n, s, g-u, g, ncpu, mem, d.get("resources_available.qlist", "")))
    free += g - u; tot += g
print(f"--- {free}/{tot} GPUs free in the gpus queue")
'
if [ -n "$1" ]; then ssh -o BatchMode=yes -o ConnectTimeout=5 "$1" nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv; fi
