import pd
import time


def py4pdtimer(message):
    if (message == "start"):
        pd.setglobalvar("py4pd_timer", time.time())
    elif (message == "end"):
        timer = time.time() - pd.getglobalvar("py4pd_timer")
        pd.print("It spent " + str(int(timer * 1000)) + " ms.")
    else:
        pd.error("Bad args to py4pdtimer")

