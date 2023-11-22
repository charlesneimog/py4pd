import pd
import sys
from random import randint

try:
    import numpy as np
    numpyIsInstalled = True
except Exception as e:
    pd.pip_install("numpy")
    numpyIsInstalled = False
    pd.error("You must restart Pure Data to use numpy.")
    sys.exit()

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
def generate_sine_wave(frequency, amplitude, phase, num_samples, sampling_rate):
    angular_frequency = 2 * np.pi * frequency
    t = np.arange(num_samples) / sampling_rate
    sine_wave = amplitude * np.sin(angular_frequency * t + phase)
    last_phase = phase + angular_frequency * t[-1]
    return sine_wave, last_phase


def mksenoide(freqs, amps, phases, vectorsize, samplerate):
    n = len(freqs) 
    nframes = vectorsize
    out = np.zeros((n, nframes), dtype=np.float64)  # Modify the shape of the output array
    new_phases = np.zeros(n, dtype=np.float64)  # Array to store the new phases
    for i in range(n):
        out[i], new_phases[i] = generate_sine_wave(freqs[i], amps[i], phases[i], nframes, samplerate)
        if new_phases[i] > 2 * np.pi:
            new_phases[i] -= 2 * np.pi
    return out, new_phases


def sinusoids(freqs, amps):  
    vectorsize = pd.get_vec_size()
    samplerate = pd.get_sample_rate()
    if freqs is None or amps is None:
        return None
    if len(freqs) != len(amps):
        return None
    phases = pd.get_obj_var("PHASE", initial_value=np.zeros(len(freqs)))
    freqs = np.array(freqs, dtype=np.float64)
    amps = np.array(amps, dtype=np.float64)
    out, new_phases = mksenoide(freqs, amps, phases, vectorsize, samplerate)
    pd.set_obj_var("PHASE", new_phases)
    return out




def py4pdLoadObjects():
    pd.add_object(sinusoids, 'sinusoids~', objtype=pd.AUDIOOUT) 
    
