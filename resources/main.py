import pd
import numpy

def audioin(audio):
    # do one simple fft and output the real part
    fft = numpy.fft.fft(audio)
    fft = numpy.real(fft)
    return fft.tolist()


def audioout(freq, amplitude):
    if freq is None: 
        return numpy.zeros(pd.vecsize())
    else:
        phase = pd.getglobalvar("PHASE", initial_value=0.0)
        output = numpy.zeros(pd.vecsize())
        increment = (freq * 2.0 * numpy.pi) / pd.samplerate()
        for i in range(pd.vecsize()):
            output[i] = numpy.sin(phase) * amplitude
            phase += increment 
            if phase > 2 * numpy.pi:
                phase -= 2 * numpy.pi
        # ---------
        pd.setglobalvar("PHASE", phase) 
        # it saves the value of phase for the next call of the function
        # without this, we will not have a continuous sine wave.
        return output

def audio(audio, amplitude):
    if amplitude is None:
        amplitude = 0.2
    audio = numpy.multiply(audio, amplitude)
    return audio


def py4pdLoadObjects():
    pd.addobject(audioin, "audioin", objtype="AUDIOIN")
    pd.addobject(audioout, "audioout", objtype="AUDIOOUT")
    pd.addobject(audio, "audio", objtype="AUDIO")
