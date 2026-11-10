"""
Run a long job (training) under a resource guard, so it cannot take the machine down.

On the 16 GB dev Mac an offline training run once swapped until the SSD filled and macOS crashed
(TRAINING.md §3). This samples the job's memory, the system's available memory, swap and free disk
every 0.5 s, logs a trend line every --trend-sec, keeps the Mac from idle-sleeping while the job runs
(caffeinate), and kills the job's whole process tree when a limit is crossed. Training resumes from
its last epoch checkpoint with the same command.

    python tools/train_guard.py --log out/logs/train_offline_guard.log -- \\
        verus --train-offline --cpu --init-from out/models/offline/legacy_2025-10/verus_baseline_offline.keras

Available memory is the system-wide figure: on Apple Silicon the GPU (tensorflow-metal) allocates
from the same unified memory, and those allocations do not show up in the job's RSS.
"""
import argparse, re, shutil, subprocess, sys, time

import psutil


def swap_used_gb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) / 1024 if m else 0.0


def process_tree(p):
    try:
        return [p] + p.children(recursive=True)
    except psutil.Error:
        return [p]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-avail-gb", type=float, default=1.5, help="kill below this much available memory")
    ap.add_argument("--max-swap-growth-gb", type=float, default=1.5, help="kill when swap grows by more than this")
    ap.add_argument("--min-disk-gb", type=float, default=6.0, help="kill below this much free disk (swap lives on it)")
    ap.add_argument("--disk", default="/System/Volumes/Data" if sys.platform == "darwin" else "/")
    ap.add_argument("--trend-sec", type=float, default=300.0, help="trend line interval (0 = off)")
    ap.add_argument("--log", help="also append the guard's lines to this file")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command to run")
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command given (put it after --)")

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [guard] {msg}"
        print(line, flush=True)
        if a.log:
            with open(a.log, "a") as f:
                f.write(line + "\n")

    swap0 = swap_used_gb()
    proc = subprocess.Popen(cmd)
    ps = psutil.Process(proc.pid)
    if shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(proc.pid)])
    log(f"started pid {proc.pid}: {' '.join(cmd)}")
    log(f"limits: available >= {a.min_avail_gb} GB, swap growth <= {a.max_swap_growth_gb} GB "
        f"(from {swap0:.2f} GB), free disk >= {a.min_disk_gb} GB")

    peak_rss, min_avail, peak_swap, killed, t_trend = 0.0, float("inf"), swap0, None, time.time()
    while proc.poll() is None:
        rss = 0.0
        for c in process_tree(ps):
            try:
                rss += c.memory_info().rss / 2**30
            except psutil.Error:
                pass
        avail = psutil.virtual_memory().available / 2**30
        swap = swap_used_gb()
        disk = shutil.disk_usage(a.disk).free / 2**30
        peak_rss, min_avail, peak_swap = max(peak_rss, rss), min(min_avail, avail), max(peak_swap, swap)
        if a.trend_sec and time.time() - t_trend >= a.trend_sec:
            t_trend = time.time()
            log(f"rss {rss:.2f} GB, available {avail:.2f} GB, swap {swap:.2f} GB, free disk {disk:.1f} GB")
        if avail < a.min_avail_gb:
            killed = f"available memory {avail:.2f} GB"
        elif swap - swap0 > a.max_swap_growth_gb:
            killed = f"swap {swap:.2f} GB"
        elif disk < a.min_disk_gb:
            killed = f"free disk {disk:.1f} GB"
        if killed:
            for c in reversed(process_tree(ps)):
                try:
                    c.kill()
                except psutil.Error:
                    pass
            break
        time.sleep(0.5)
    proc.wait()
    log(f"exit {proc.returncode}; peak rss {peak_rss:.2f} GB, min available {min_avail:.2f} GB, "
        f"swap {swap0:.2f} -> {peak_swap:.2f} GB" + (f"; KILLED ({killed})" if killed else ""))
    sys.exit(proc.returncode if proc.returncode is not None else 1)


if __name__ == "__main__":
    main()
