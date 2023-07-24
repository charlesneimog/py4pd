import pd
import time
import subprocess


def py4pdtimer(message):
    if (message == "start"):
        pd.setglobalvar("py4pd_timer", time.time())
    elif (message == "end"):
        timer = time.time() - pd.getglobalvar("py4pd_timer")
        pd.print("It spent " + str(int(timer * 1000)) + " ms.")
    else:
        pd.error("Bad args to py4pdtimer")





def getMemoryUse(programName):
    # execute pidof to get the pid of the program using programName
    pid = subprocess.check_output(["pidof", programName])
    # convert the byte string to a string and remove the newline character
    pid = pid.decode("utf-8").strip()
    memoryUse = int(subprocess.check_output(["ps", "-o", "rss=", pid]).strip())
    memoryUse = int(memoryUse / 1024)
    # return the memory usage
    return memoryUse



