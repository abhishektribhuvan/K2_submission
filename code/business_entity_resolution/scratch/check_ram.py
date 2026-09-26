import psutil

vm = psutil.virtual_memory()
print(f"Total RAM:     {vm.total / (1024**3):.2f} GB")
print(f"Available RAM: {vm.available / (1024**3):.2f} GB")
print(f"Used RAM:      {vm.used / (1024**3):.2f} GB ({vm.percent}%)")

print("\nTop 7 Processes by RAM:")
procs = []
for p in psutil.process_iter(['name', 'pid', 'memory_info']):
    try:
        mem = p.info['memory_info'].rss if p.info['memory_info'] else 0
        procs.append((p.info['name'], p.info['pid'], mem))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

procs.sort(key=lambda x: x[2], reverse=True)
for name, pid, mem in procs[:7]:
    print(f"  {name:<25} (PID: {pid:<6}): {mem / (1024**2):.1f} MB")
