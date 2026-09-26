import psutil

print("All Python processes:")
for p in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
    try:
        if 'python' in p.info['name'].lower():
            rss = p.info['memory_info'].rss / (1024**2) if p.info.get('memory_info') else 0
            cmd = ' '.join(p.info.get('cmdline') or [])[:80]
            print(f"  PID {p.info['pid']}: {rss:.1f} MB | {cmd}")
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
