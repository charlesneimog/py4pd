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
    # Check if the platform is Linux
    if platform.system() == 'Linux':
        try:
            # Execute pidof to get the pid of the program using programName
            pid = subprocess.check_output(["pidof", programName])
            # Convert the byte string to a string and remove the newline character
            pid = pid.decode("utf-8").strip()
            if isinstance(pid, list):
                pd.error("More than one pid found for " + programName)
                return 0    
            memoryUse = int(subprocess.check_output(["ps", "-o", "rss=", pid]).strip())
            memoryUse = int(memoryUse / 1024)
            # Return the memory usage
            return memoryUse
        except subprocess.CalledProcessError as e:
            pd.error(f"Error retrieving memory usage: {e}")
            return 0
    else:
        # Return 1 for non-Linux platforms (Mac and Windows)
        return 1


