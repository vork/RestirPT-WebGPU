#!/usr/bin/env python3
"""Background GPU load per process (perf2 V-PERF; docs/decisions/perf2-api.md §4), from IOKit AGXDeviceUserClient
AppUsage.accumulatedGPUTime (ns) deltas.

usage: gpu-bg.py [secs]                    one interval (default 5 s): top processes by GPU ms/s + device utilisation
       gpu-bg.py --loop <interval_s> <log>  append one line per interval (GPU lock holder, Chrome vs other ms/s) until killed
"""
import collections, os, plistlib, re, subprocess, sys, time

LOCK = '/tmp/restirpt-gpu.lock'


def usage_from_registry(entries):
    """Pair usage with its owning registry object, independent of property order."""
    acc = collections.Counter()
    pending = list(entries)
    while pending:
        entry = pending.pop()
        pending.extend(entry.get('IORegistryEntryChildren', []))
        match = re.fullmatch(r'pid (\d+), (.*)', entry.get('IOUserClientCreator', ''))
        if not match:
            continue
        key = f'{match.group(2).strip()} ({match.group(1)})'
        acc[key] += sum(item.get('accumulatedGPUTime', 0) for item in entry.get('AppUsage', []))
    return acc


def snap():
    # Text ioreg output can list AppUsage BEFORE IOUserClientCreator. Carrying the last creator across lines
    # attributes that usage to another process. The archive form preserves each client's property dictionary.
    raw = subprocess.check_output(['ioreg', '-a', '-r', '-c', 'AGXDeviceUserClient'])
    return usage_from_registry(plistlib.loads(raw))


def util():
    s = subprocess.run(['ioreg', '-r', '-d', '1', '-c', 'IOAccelerator', '-w0'], capture_output=True, text=True).stdout
    m = re.search(r'"Device Utilization %"=(\d+)', s); return int(m.group(1)) if m else -1


def rates(a, b, dt):
    return sorted((((b[k] - a.get(k, 0)) / 1e6 / dt), k) for k in b if b[k] - a.get(k, 0) > 0)[::-1]


def once(secs):
    a = snap(); t0 = time.time(); us = []
    while time.time() - t0 < secs: us.append(util()); time.sleep(0.5)
    b = snap(); dt = time.time() - t0
    d = rates(a, b, dt)
    print(f'interval {dt:.1f} s; device util samples mean {sum(us)/len(us):.0f}% (min {min(us)}, max {max(us)})')
    print(f'total GPU ms/s over all clients: {sum(x for x, _ in d):.0f}')
    for v, k in d[:12]: print(f'  {v:7.1f} ms/s  {k}')


def loop(iv, path):
    log = open(path, 'a', buffering=1)
    a = snap(); ta = time.time()
    while True:
        time.sleep(iv)
        b = snap(); tb = time.time()
        d = rates(a, b, tb - ta)
        try: holder = ','.join(os.listdir(LOCK))
        except FileNotFoundError: holder = '-'
        chrome = sum(v for v, k in d if 'Google Chrome' in k); other = sum(v for v, k in d if 'Google Chrome' not in k)
        log.write(f"{time.strftime('%H:%M:%S')} lock={holder} chrome={chrome:.0f} other={other:.0f} ms/s | " + ' '.join(f'{k}={v:.0f}' for v, k in d[:6]) + '\n')
        a, ta = b, tb


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--loop': loop(float(sys.argv[2]), sys.argv[3])
    else: once(float(sys.argv[1]) if len(sys.argv) > 1 else 5)
