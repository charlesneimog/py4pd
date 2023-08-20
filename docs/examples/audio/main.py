import pd
import numpy

def audio(audio, amplitude):
    if amplitude is None:
        amplitude = 0.2
    audio = numpy.multiply(audio, amplitude)
    return audio


def py4pdLoadObjects():
    pd.add_object(audio, "audio", objtype=pd.AUDIO)
