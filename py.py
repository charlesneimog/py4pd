from random import *

def sum(x, y):
    "It sums two numbers."
    x = int(x)
    y = int(y)
    return x + y

def arithm_ser(begin, end, step):
    "It calculates the arithmetic series."
    list = []
    for x in range(begin, end, step):
        list.append(x)
    return list

def fibonacci(n):
    "Calculate the nth fibonacci number."
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def thread_test():
    "It tests the threading module. Just return the hour after 5 seconds."
    import time
    import pd # import the py4pd module (embedded in the python interpreter)
    pd.message("Starting thread...")
    time.sleep(5)
    pd.message("Thread finished.")
    return time.strftime("%H:%M:%S")

def pd_output():
    "It sends some output to the py4pd output."
    import pd # import the py4pd module (embedded in the python interpreter)
    import time
    for x in range(10):
        pd.out(x)
        time.sleep(0.5)
    
def pd_message():
    "It sends a message to the py4pd message box."
    import pd # import the py4pd module (embedded in the python interpreter)
    pd.message("Hello from python!")
    return None

def pd_error():
    "It sends a message to the py4pd message box."
    import pd # import the py4pd module (embedded in the python interpreter)
    # NOT WORKING
    pd.error("Python error!")
    return None