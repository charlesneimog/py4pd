import pd
import time
import subprocess


def py4pdtimer(message):
    if (message == "start"):
        pd.set_global_var("py4pd_timer", time.time())
    elif (message == "end"):
        timer = time.time() - pd.get_global_var("py4pd_timer")
        pd.print("It spent " + str(int(timer * 1000)) + " ms.")
    else:
        pd.error("Bad args to py4pdtimer")





def getMemoryUse(programName):
    # execute pidof to get the pid of the program using programName
    pid = subprocess.check_output(["pidof", programName])
    # convert the byte string to a string and remove the newline character
    pid = pid.decode("utf-8").strip()
    if isinstance(pid, list):
        pd.error("More than one pid found for " + programName)
        return 0    
    memoryUse = int(subprocess.check_output(["ps", "-o", "rss=", pid]).strip())
    memoryUse = int(memoryUse / 1024)
    # return the memory usage
    return memoryUse



