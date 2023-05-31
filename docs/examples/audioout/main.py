import pd
import numpy

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

def py4pdLoadObjects():
    pd.addobject(audioout, "audioout", objtype=pd.AUDIOOUT)
