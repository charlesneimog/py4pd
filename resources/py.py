import pd
from random import randint
import os
import time
import math

try:
    from neoscore.common import *
except Exception as e:
    pd.error(str(e))
    pd.error(
        "Please, run 'pip install neoscore -t ./py-modules' in the terminal from current folder")

try:
    import numpy as np
except Exception as e:
    pd.error(str(e))
    pd.error(
        "Please, run 'pip install numpy -t ./py-modules' in the terminal from current folder")

try:
    import matplotlib
    # NOTE:  Always use 'agg' backend or other than Tkinter. Tkinket will
    # conflict with Pd
    matplotlib.use('agg')
    from matplotlib import pyplot as plt
except Exception as e:
    pd.error(str(e))
    pd.error("Error importing matplotlib")
    pd.error(
        "Please, run 'pip install matplotlib -t ./py-modules' in the terminal from current folder")

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


def thread_test():
    "It tests the threading module. Just return the hour after 5 seconds."
    pd.print("Starting thread...")
    time.sleep(5)
    pd.print("Thread finished.")
    return time.strftime("%H:%M:%S")


# ================================================
# ==================  Pd  ========================
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
    pd.tabwrite("pd.tabwrite", list, resize=True)


def printall(x, y):
    "It sends a message to the py4pd message box."
    pd.print(str(x + y))

# ================================================
# ================ Audio =========================
# ================================================

def fft(audio):
    fft = np.fft.fft(audio)
    ifft = np.fft.ifft(fft)
    # get real part of ifft
    ifft = np.real(ifft)
    ifft = ifft.astype(np.float32) # for now, we must use float32
    # ifft to tuple
    key = pd.getkey("output")
    if key == "numpy" or key == None:
        return ifft
    elif key == "tuple":
        iff = tuple(ifft)
        return iff
    else:
        return ifft.tolist()


def pd_audio(audio):
    "It sends a message to the py4pd message box."
    # get first 10 samples
    if type(audio) == np.ndarray:
        pd.out("numpy")
    else:
        pd.out("list")

def audioin(audio):
    "It sends a message to the py4pd message box."
    # get first 10 samples
    if type(audio) == np.ndarray:
        pd.out("numpy")
    else:
        pd.out("list")

def pd_audioout(audio):
    "It sends a message to the py4pd message box."
    # audio is a numpy array, multiply by 0.5
    if type(audio) == np.ndarray:
        audio = np.multiply(audio, 0.2)
        audio = audio.tolist()
    else:
        audio = [x * 0.05 for x in audio]
    return audio


def senoide():
    sample_rate = pd.SAMPLERATE
    # duration of 64 samples
    duration = pd.VECSIZE
    freq = pd.getkey("freq")
    if freq == None:
        freq = 440
    # create a sine wave
    sine_wave = np.sin(2 * np.pi * np.arange(duration) * freq / sample_rate)
    # convert to np.float32
    sine_wave = sine_wave.astype(np.float32)
    return sine_wave.tolist()


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




def getpitchKey(pitch):
    note = {
        # natural
        'c': ['c', ''],
        'd': ['d', ''],
        'e': ['e', ''],
        'f': ['f', ''],
        'g': ['g', ''],
        'a': ['a', ''],
        'b': ['b', ''],
        # sharp
        'c#': ['c', 'accidentalSharp'],
        'd#': ['d', 'accidentalSharp'],
        'e#': ['e', 'accidentalSharp'],
        'f#': ['f', 'accidentalSharp'],
        'g#': ['g', 'accidentalSharp'],
        'a#': ['a', 'accidentalSharp'],
        'b#': ['b', 'accidentalSharp'],
        # flat
        'cb': ['c', 'accidentalFlat'],
        'db': ['d', 'accidentalFlat'],
        'eb': ['e', 'accidentalFlat'],
        'fb': ['f', 'accidentalFlat'],
        'gb': ['g', 'accidentalFlat'],
        'ab': ['a', 'accidentalFlat'],
        'bb': ['b', 'accidentalFlat'],
    }
    return note[pitch]


def note(pitches):
    try:
        neoscore.shutdown() # to avoid errors
    except BaseException:
        pass
    neoscore.setup()
    py4pdTMPfolder = pd.tempfolder()
    for file in py4pdTMPfolder:
        if file.endswith(".ppm"):
            try:
                os.remove(py4pdTMPfolder + "/" + file)
            except BaseException:
                pass
    staffSoprano = Staff((Mm(0), Mm(0)), None, Mm(30))
    trebleClef = 'treble'
    Clef(ZERO, staffSoprano, trebleClef)
    staffBaixo = Staff((ZERO, Mm(15)), None, Mm(30))
    bassClef = 'bass'
    Clef(ZERO, staffBaixo, bassClef)
    Path.rect((Mm(-10), Mm(-10)), None, Mm(42), Mm(42),
              Brush(Color(0, 0, 0, 0)), Pen(thickness=Mm(0.5)))
    for pitch in pitches:
        # in pitch remove not number
        pitchWithoutNumber = pitch.replace(pitch[-1], '')
        pitchOctave = int(pitch[-1])
        pitchClass, accidental = getpitchKey(pitchWithoutNumber)
        note = [(pitchClass, accidental, pitchOctave)]
        if pitchOctave < 4:
            Chordrest(Mm(5), staffBaixo, note, (int(1), int(1)))
        else:
            Chordrest(Mm(5), staffSoprano, note, (int(1), int(1)))
    randomNumber = randint(1, 100)
    notePathName = py4pdTMPfolder + "/" + pitch + f"{randomNumber}.ppm"
    neoscore.render_image(rect=None, dest=notePathName, dpi=150, wait=True)
    neoscore.shutdown()
    if os.name == 'nt':
        notePathName = notePathName.replace("\\", "/")
    pd.show(notePathName)
    return None


def neoscoreTest():
    if os.name == 'posix':
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    pitch = 'c'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    neoscore.setup()
    POSITION = (Mm(0), Mm(0))
    staff = Staff(POSITION, None, Mm(30))
    saxClef = 'treble'
    Clef(ZERO, staff, saxClef)  # .ppm
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



def dft(freq_hz):
    plt.clf()
    plt.switch_backend('agg')
    home_path = pd.tempfolder()
    NUMPY_DATA = pd.tabread('audioArray')
    NUMPY_DATA = np.array(NUMPY_DATA)
    for file in os.listdir(home_path):
        if file.endswith(".gif") or file.endswith(".png"):
            os.remove(home_path + "/" + file)
    round_index = freq_hz
    k = float(round_index / (pd.samplerate() /
              len(NUMPY_DATA) / len(NUMPY_DATA)))
    all_index = []
    for i in range(len(NUMPY_DATA)):
        formula = math.e ** (math.pi * 2 * 1j * k * i)
        all_index.append(formula)

    graph = []
    for i in range(len(all_index)):
        graph.append(all_index[i] * float(NUMPY_DATA[i]))
    imag = []
    real = []
    for i in range(len(graph)):
        imag.append(float(graph[i].imag))
        real.append(float(graph[i].real))

    imag = np.array(imag)
    real = np.array(real)

    plt.plot(imag, real, color='black', linewidth=0.4)
    plt.xlabel('Imaginary')
    plt.ylabel('Real')

    freq_hz = int(round(freq_hz, 0))
    random_number = randint(10, 99)
    plt.savefig(f'{home_path}/canvas{freq_hz}{random_number}.png', dpi=60)
    pd.show(f'{home_path}/canvas{freq_hz}{random_number}.png')
    return 0

