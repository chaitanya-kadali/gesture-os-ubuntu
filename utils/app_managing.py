import os
import signal
from collections import deque, defaultdict

import subprocess
import time


class AppProcessManager:
    def __init__(self):
        # app_name -> queue of Popen objects
        self.process_map = defaultdict(deque)

    def open_app(self, app_name, command):
        """
        Launch app and store its process
        """
        proc = subprocess.Popen(
            command,
            preexec_fn=os.setsid  # isolate process group
        )
        self.process_map[app_name].append(proc)
        print(f"[OPENED] {app_name} PID={proc.pid}")

    def close_app(self, app_name, mode="fifo"):
        """
        Close one instance of an app
        mode = 'fifo' | 'lifo'
        """
        if not self.process_map[app_name]:
            print(f"[INFO] No running instances of {app_name}")
            return

        if mode == "fifo":
            proc = self.process_map[app_name].popleft()
        else:
            proc = self.process_map[app_name].pop()

        try:
            os.killpg(proc.pid, signal.SIGTERM)
            print(f"[CLOSED] {app_name} PID={proc.pid}")
        except ProcessLookupError:
            print(f"[WARN] PID {proc.pid} already terminated")

    # def list_apps(self):
    #     for app, q in self.process_map.items():
    #         pids = [p.pid for p in q]
    #         print(f"{app}: {pids}")
