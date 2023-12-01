import pd
import time
import subprocess
import platform

def py4pdtimer(message):
    if (message == "start"):
        pd.set_obj_var("py4pd_timer", time.time())
    elif (message == "end"):
        timer = time.time() - pd.get_obj_var("py4pd_timer")
        pd.print("It spent " + str(int(timer * 1000)) + " ms.")
    else:
        pd.error("Bad args to py4pdtimer")

def getMemoryUse(programName):
    if platform.system() == 'Linux':
        try:
            pid = subprocess.check_output(["pidof", programName])
            pid = pid.decode("utf-8").strip()
            if isinstance(pid, list):
                pd.error("More than one pid found for " + programName)
                return 0    
            memoryUse = int(subprocess.check_output(["ps", "-o", "rss=", pid]).strip())
            memoryUse = int(memoryUse / 1024)
            return memoryUse
        except subprocess.CalledProcessError as e:
            pd.error(f"Error retrieving memory usage: {e}")
            return 0
    elif platform.system() == 'Darwin':
        try:
            command = 'top -l 1 -stats pid,command,cpu,mem | grep Pd'
            result = subprocess.check_output(command, shell=True, text=True)
            result = result.split()[3]
            result = result[:-1]           
            return result
        except subprocess.CalledProcessError:
            return None
    else:
        return 1


