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
    pd.print(
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
    pd.print(
        "Please, run 'pip install matplotlib -t ./py-modules' in the terminal from current folder")

try:
    from PIL import Image
except Exception as e:
    pd.error(str(e))
    pd.print(
        "Please, run 'pip install PIL -t ./py-modules' in the terminal from current folder")

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
    if type(audio) == np.ndarray:
        audio = np.multiply(audio, 0.2)
    else:
        audio = [x * 0.05 for x in audio]
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
    pd.out(1)


def dft(freq_hz):
    plt.clf()
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

    # CenterOfMass = sum(graph) / len(graph)
    imag = np.array(imag)
    real = np.array(real)
    plt.switch_backend('agg')  # Change backend to avoid error in PureData

    # define draw, circle and point to add to the graph
    plt.plot(imag, real, color='black', linewidth=0.4)
    # plt.plot(CenterOfMass.real, CenterOfMass.imag, 'ro')

    # plot names of axis
    plt.xlabel('Imaginary')
    plt.ylabel('Real')

    # plt.plot(imag_centroid, real_centroid, 'ro', color='blue')
    # add a new circle that are in the center of max and min of real and imaginary
    # center = (max(imag) + min(imag)) / 2, (max(real) + min(real)) / 2
    # radius = max(imag) - center[0] if (max(imag) > max(real)) else max(real) - center[1]
    # plt.gca().add_patch(plt.Circle(center, radius=radius, fill=False, color='blue', linewidth=2))

    # with circle draw a cross
    # plt.plot([center[0], center[0]], [center[1] - radius, center[1] + radius], color='blue', linewidth=2)
    # plt.plot([center[0] - radius, center[0] + radius], [center[1], center[1]], color='blue', linewidth=2)

    # add triangle retangle
    # plt.plot([center[0], CenterOfMass.real], [center[1], CenterOfMass.imag], color='green', linewidth=2)
    # plt.plot([CenterOfMass.real, CenterOfMass.imag], [CenterOfMass.imag, center[1]], color='green', linewidth=2)

    # Save image, convert to gif and remove png
    freq_hz = int(round(freq_hz, 0))
    random_number = randint(10, 99)
    # save the plt using mpimg
    plt.savefig(f'{home_path}/canvas{freq_hz}{random_number}.jpg', dpi=60)
    # convert to gif
    im = Image.open(f'{home_path}/canvas{freq_hz}{random_number}.jpg')
    im.save(f'{home_path}/canvas{freq_hz}{random_number}.gif')
    # remove png
    os.remove(f'{home_path}/canvas{freq_hz}{random_number}.jpg')
    # show the image
    output = f'{home_path}/canvas{freq_hz}{random_number}.gif'
    # in lisp, do namestring equivalent in python
    pd.show(output)
    return 0

