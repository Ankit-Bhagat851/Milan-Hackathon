import subprocess
import psutil
import platform

def kill_apps(process_names):
    process_names = [p.lower() for p in process_names]
    for pname in process_names:
        print(f"🛑 Trying to close: {pname}")
        if platform.system() == "Windows":
            try:
                subprocess.run(["taskkill", "/f", "/im", pname], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"✅ Force-closed {pname} using taskkill")
                continue
            except subprocess.CalledProcessError:
                print(f"⚠️ Taskkill failed for {pname}, trying psutil fallback...")

        for proc in psutil.process_iter(["pid", "name"]):
            try:
                if proc.info["name"].lower() == pname:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except psutil.TimeoutExpired:
                        proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

def launch_apps(app_paths):
    for app in app_paths:
        try:
            print(f"🚀 Launching: {app}")
            subprocess.Popen([app], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"❌ Failed to launch {app}: {e}")