import os
import platform
import subprocess
import time

import pd


def py4pdtimer(message):
    if message == "start":
        pd.set_obj_var("py4pd_timer", time.time())
    elif message == "end":
        timer = time.time() - pd.get_obj_var("py4pd_timer")
        pd.print("It spent " + str(int(timer * 1000)) + " ms.")
    else:
        pd.error("Bad args to py4pdtimer")


def getMemoryUse():
    if platform.system() == "Linux":
        try:
            pid = os.getpid()
            if isinstance(pid, list):
                pd.error("More than one pid found for " + "pd")
                return 0
            memoryUse = int(
                subprocess.check_output(["ps", "-o", "rss=", str(pid)]).strip()
            )
            memoryUse = int(memoryUse / 1024)
            return memoryUse
        except subprocess.CalledProcessError as e:
            pd.error(f"Error retrieving memory usage: {e}")
            return 0
    elif platform.system() == "Darwin":
        try:
            command = "ps -o rss -p " + str(os.getpid())
            result = subprocess.check_output(command, shell=True, text=True)
            memory = int(int(result.split("\n")[1]) / 1024)
            return memory
        except subprocess.CalledProcessError:
            return None

    elif platform.system() == "Windows":
        try:
            command = "tasklist | findstr pd"
            result = subprocess.check_output(command, shell=True, text=True)
            try:
                kbytes = float(result.split()[4]) * 1000
            except:
                kbytes = float(result.split()[4].replace(",", ".")) * 1000
            mb = kbytes / 1024
            return mb
        except subprocess.CalledProcessError:
            return None

    else:
        return 1
