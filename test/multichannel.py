import numpy as np
import pd


def randomNumpyArray():
    return np.random.rand(1, 64).tolist()[0]


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


# ============================================
# ==================== PD ====================
# ============================================


def py4pdLoadObjects():
    pd.add_object(sinusoids, "sinusoids~", objtype=pd.AUDIOOUT)
    pd.add_object(audioin, "audioin~", objtype=pd.AUDIOIN)
    pd.add_object(audioInOut, "audio~", objtype=pd.AUDIO)
    pd.add_object(generate_audio_noise, "pynoise~", objtype=pd.AUDIOOUT)
    pd.add_object(randomNumpyArray, "random-array", pyout=True)
