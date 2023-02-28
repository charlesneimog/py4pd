from random import *
from neoscore.common import *
import numpy as np
import os
import pd

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
    pd.print("Starting thread...")
    time.sleep(5)
    pd.print("Thread finished.")
    return time.strftime("%H:%M:%S")

def pd_output():
    "It sends some output to the py4pd output."
    import pd # import the py4pd module (embedded in the python interpreter)
    for x in range(10):
        pd.out(x)
    
def pd_message():
    "It sends a message to the py4pd message box."
    import pd # import the py4pd module (embedded in the python interpreter)
    pd.print("Hello from python!")
    return None

def pd_error():
    "It sends a message to the py4pd message box."
    import pd # import the py4pd module (embedded in the python interpreter)
    # NOT WORKING
    pd.error("Python error!")
    return None

def pd_send():
    "It sends a message to the py4pd message box."
    pd.send()

def pd_tabwrite():
    "It sends a message to the py4pd message box."
    list = []
    tablen = randint(10, 200)
    for i in range(tablen):
        randomnumber = randint(-100, 100)
        list.append(randomnumber * 0.01)
    pd.tabwrite("test", list, resize=True)

def pd_audio(audio):
    "It sends a message to the py4pd message box."
    # get first 10 samples
    if type(audio) == np.ndarray:
        pd.out("numpy")
    else:
        pd.out("list")

def pd_audioout(audio):
    "It sends a message to the py4pd message box."
    # audio is a numpy array, multiply by 0.5
    audio = np.multiply(audio, 0.3) 
    return audio

def pd_audionoise(audio):
    "It sends a message to the py4pd message box."
    audiolen = len(audio) + 1
    noise = np.random.normal(0, 1, audiolen)
    return noise

def pd_tabread():
    "It sends a message to the py4pd message box."
    myarray = pd.tabread("test")
    return myarray

def whereFiles():
    "It returns the path of the files."
    import os
    return os.__file__

def noArgs():
    "It returns the path of the files."
    return "ok"

def neoscoreTest():
    if os.name == 'posix':
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'

    pitch = 'c'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    neoscore.setup()
    POSITION = (Mm(0), Mm(0))
    staff = Staff(POSITION, None, Mm(30))
    saxClef = 'treble'
    Clef(ZERO, staff, saxClef) # .ppm
    note = [(pitch, '', 4)]
    Chordrest(Mm(5), staff, note, (int(1), int(1)))
    scriptPath = os.path.dirname(os.path.abspath(__file__))
    if os.name == 'nt':
        filename = f'{script_dir}/neoscoretest.png'
    else:
        filename = f'{script_dir}/neoscoretest.png'
    neoscore.render_image(rect=None, dest=filename, dpi=150, wait=True)
    neoscore.shutdown()
    return 1

