import pd
import numpy

def audioin(audio):
    length = len(audio)
    return length

def audioout(freq, amplitude):
    if freq is None: 
        return numpy.zeros(pd.vecsize())
    else:
        phase = pd.getglobalvar("PHASE")
        if phase is None:
            phase = 0.0 # set the initial value of phase in the first call of the function
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
