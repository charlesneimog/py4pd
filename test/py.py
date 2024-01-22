import os
import sys
from random import randint

import numpy as np

try:
    import pd
except:
    print("Pd not imported")

# ================================================
# ==============  Functions  =====================
# ================================================


def getTestId():
    pythonVersionMajor = sys.version_info[0]
    pythonVersionMinor = sys.version_info[1]

    platform = sys.platform
    return f"{platform}-{pythonVersionMajor}.{pythonVersionMinor}"


def sumlist(x, valuetimes):
    mylist = []
    for i in x:
        print(i)
        mylist.append(i * valuetimes)
    return mylist


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
# ================== LISTS  ======================
# ================================================


def randomNumpyArray():
    return np.random.rand(1, 64).tolist()[0]


# ================================================
# ================== TESTS  ======================
# ================================================
def importsvgpathtools():
    try:
        import svgpathtools

        return 1
    except Exception as e:
        print(e)
        return 0


def getPy4pdCfg():
    import os

    import svgpathtools

    print(os.getcwd())
    os.system("ls ")

    with open("./py4pd.cfg", "w") as config_file:
        value = svgpathtools.__file__.replace("svgpathtools\\__init__.py", "")
        config_file.write(f"packages = {value}")


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


def pdouttest(y):
    for x in range(y):
        pd.out(x, out_n=x)


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
    out = np.zeros((n, nframes))  # Array with 2 dimensions: channels by frames
    new_phases = np.zeros(n)  # Array to store the new phases
    for i in range(n):
        out[i], new_phases[i] = generate_sine_wave(
            freqs[i], amps[i], phases[i], nframes, samplerate
        )
    return out, new_phases


def sinusoids(freqs, amps):
    if freqs is None or amps is None:
        pd.error("You need to set the amplitudes")
        return np.zeros((1, 64))
    if len(freqs) != len(amps):
        pd.error("The length of freqs and amps are different")
        return None

    phases = pd.get_obj_var("PHASE", initial_value=np.zeros(len(freqs)))
    freqs = np.array(freqs, dtype=np.float64)
    amps = np.array(amps, dtype=np.float64)
    out, new_phases = mksenoide(freqs, amps, phases, 64, pd.get_sample_rate())
    pd.set_obj_var("PHASE", new_phases)
    return out


def audioin(audio):
    num_channels = audio.shape[0]
    fftresult = []
    for channel in range(num_channels):
        channel_data = audio[channel, :]
        fft_result = np.fft.fft(channel_data)
        first_real_number = np.real(fft_result[0])
        fftresult.append(first_real_number)
    return fftresult


def generate_audio_noise():
    noise_array = np.random.rand(5, 64)
    return noise_array


def audioInOut(audio):
    num_channels = audio.shape[0]
    transformed_audio = np.empty_like(audio, dtype=np.float64)

    for channel in range(num_channels):
        channel_data = audio[channel, :]
        fft_result = np.fft.fft(channel_data)
        ifft_result = np.fft.ifft(fft_result).real
        transformed_audio[channel, :] = ifft_result

    return transformed_audio


def testpdMethod(method):
    try:
        if method == "get_patch_dir":
            return pd.get_patch_dir()
        elif method == "get_vec_size":
            return pd.get_vec_size()
        elif method == "get_py4pd_dir":
            return pd.get_py4pd_dir()
        elif method == "tabwrite":
            return pd.tabwrite("pd-test", [10, 20, 30, 40, 50], resize=True)
        elif method == "tabread":
            return pd.tabread("pd-test", numpy=False) == [10, 20, 30, 40, 50]
        elif method == "print":
            return pd.print("Ok")
        elif method == "logpost":
            return pd.logpost(1, "logpost")
        elif method == "error":
            return pd.error("my test")
        elif method == "search-patches":
            return pd.get_pd_search_paths()
        elif method == "get_patch_zoom":
            return pd.get_patch_zoom()
        elif method == "get_outlet_count":
            return pd.get_outlet_count()
        elif method == "get_obj_args":
            return pd.get_obj_args()
        elif method == "objargs":
            pd.print(pd.get_obj_args())
        else:
            return -1

    except Exception as e:
        pd.print(e)
        return -1


# ================================================
# ================ SETUP =========================
# ================================================


def py4pdLoadObjects():
    pd.add_object(getTestId, "getTestId")

    # pd module test
    pd.add_object(pdouttest, "pd.out", num_aux_outlets=4)

    # Others
    pd.add_object(sinusoids, "sinusoids~", obj_type=pd.AUDIOOUT)
    pd.add_object(audioin, "audioin~", obj_type=pd.AUDIOIN)
    pd.add_object(audioInOut, "audio~", obj_type=pd.AUDIO)
    pd.add_object(generate_audio_noise, "pynoise~", obj_type=pd.AUDIOOUT)
    pd.add_object(randomNumpyArray, "random-array", py_out=True)
    pd.add_object(testpdMethod, "pd-mod")
