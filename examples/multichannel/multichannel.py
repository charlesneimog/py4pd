import pd
import numpy as np
from numba import jit 


@jit(nopython=True, cache=True, fastmath=True)
def generate_sine_wave(frequency, amplitude, phase, num_samples, sampling_rate):
    angular_frequency = 2 * np.pi * frequency
    t = np.arange(num_samples) / sampling_rate
    sine_wave = amplitude * np.sin(angular_frequency * t + phase)
    last_phase = phase + angular_frequency * t[-1]
    return sine_wave, last_phase


@jit(nopython=True, cache=True, fastmath=True)
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
    vectorsize = pd.vecsize()
    samplerate = pd.samplerate()
    if freqs is None or amps is None:
        return None
    if len(freqs) != len(amps):
        return None
    phases = pd.getglobalvar("PHASE", initial_value=np.zeros(len(freqs)))
    freqs = np.array(freqs, dtype=np.float64)
    amps = np.array(amps, dtype=np.float64)
    out, new_phases = mksenoide(freqs, amps, phases, vectorsize, samplerate)
    pd.setglobalvar("PHASE", new_phases)
    return out


def audioTest(audio):
    nchan, length = audio.shape
    return length / pd.nchs()


def timesaudio(audio, times):
    # it is a 2 dim array, multiply by some number
    return audio * times



# ============================================
# ==================== PD ====================
# ============================================

def py4pdLoadObjects():
    pd.addobject(sinusoids, 'sinusoids~', objtype=pd.AUDIOOUT) 
    pd.addobject(audioTest, 'nch~', objtype=pd.AUDIOIN)
    pd.addobject(timesaudio, 'times~', objtype=pd.AUDIO)
    

