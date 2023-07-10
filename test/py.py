import pd
from random import randint

try:
    import numpy as np
    numpyIsInstalled = True
except Exception as e:
    import platform
    py4pdObjectFolder = pd.py4pdfolder()
    if platform.system() != "Darwin":
        pd.error("=== Numpy not installed, to solve this... ===")
        pd.print("\n", show_prefix=False)
        pd.print("    1º Create new object with 'py4pd -lib py4pd'", show_prefix=False)
        pd.print("\n", show_prefix=False)
        pd.print("    2º Create new object 'py.pip'", show_prefix=False)
        pd.print("\n", show_prefix=False)
        pd.print("    3º Send the message 'global numpy' and wait for the installation", show_prefix=False)
        pd.print("\n", show_prefix=False)
        pd.error("==================================")
    else:
        pd.error("=== Numpy not installed, to solve this... ===")
        pd.print("    Open the terminal and run this line:", show_prefix=False)
        pd.print("\n")
        pd.print("cd '" + py4pdObjectFolder + "' && pip install numpy -t ./resources/py-modules", show_prefix=False)
        pd.print("\n", show_prefix=False)

    numpyIsInstalled = False

# ================================================
# ==============  Functions  =====================
# ================================================


def pdsum(x, y):
    "It sums two numbers."
    x = int(x)
    y = int(y)
    return x + y


def arithm_ser(begin, end, step):
    "It calculates the arithmetic series."
    begin = int(begin)
    end = int(end)
    step = int(step)
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
        return fibonacci(n - 1) + fibonacci(n - 2)


# ================================================
# ================== AUDIO  ======================
# ================================================

def pd_tempfolder():
    "It returns the temp folder path."
    tempfolder = pd.tempfolder()
    pd.print(str(tempfolder))


def pd_output():
    "It sends some output to the py4pd output."
    for x in range(10):
        pd.out(x)

def example_pdout():
    for x in range(2):
        pd.out(x, symbol="myloop")
    for x in range(2):
        pd.out(x, symbol="myloop2")
    return None

def pdout(x):
    pd.out(x, symbol="myloop")
    pd.out(x, symbol="myloop2")
    return None

def pd_print():
    "It sends a message to the py4pd message box."
    pd.print("Hello from python!")
    return None


def pd_error():
    "It sends a message to the py4pd message box."
    pd.error("pd error from Python, check the script.")
    return None


def pd_send():
    "It sends a message to the py4pdreceiver receive."	
    pd.send("py4pdreceiver", "hello from python!")
    pd.send("py4pdreceiver", 1)
    pd.send("py4pdreceiver", [1, 2, 3, 4, 5])
    return 0


def pd_tabwrite():
    "It sends a message to the py4pd message box."
    list = []
    tablen = randint(10, 200)
    i = 0
    while i < tablen:
        # gerar aleatoric number between -1 and 1
        list.append(randint(-100, 100) / 100)
        i += 1
    pd.tabwrite("test", list, resize=True)


def printall(x, y):
    "It sends a message to the py4pd message box."
    pd.print(str(x + y))

# ================================================
# ================ Audio =========================
# ================================================


def audioin(audio):
    length = len(audio)
    return length

def audioout(freq, amplitude):
    if freq is None: 
        return np.zeros(pd.vecsize())
    else:
        phase = pd.getglobalvar("PHASE")
        if phase is None:
            phase = 0.0 # set the initial value of phase in the first call of the function
        output = np.zeros(pd.vecsize())
        increment = (freq * 2.0 * np.pi) / pd.samplerate()
        for i in range(pd.vecsize()):
            output[i] = np.sin(phase) * amplitude
            phase += increment 
            if phase > 2 * np.pi:
                phase -= 2 * np.pi
        # ---------
        pd.setglobalvar("PHASE", phase) 
        # it saves the value of phase for the next call of the function
        # without this, we will not have a continuous sine wave.
        return output

def audio(audio, amplitude):
    if amplitude is None:
        amplitude = 0.2
    audio = np.multiply(audio, amplitude)
    return audio


def py4pdLoadObjects():
    global numpyIsInstalled
    if numpyIsInstalled:
        pd.addobject(audioin, "audioin", objtype="AUDIOIN")
        pd.addobject(audioout, "audioout", objtype="AUDIOOUT")
        pd.addobject(audio, "audio", objtype="AUDIO")
    
