import pd
import numpy

def audioout(freq, amplitude):
    if freq is None: 
        return numpy.zeros(pd.get_vec_size())
    else:
        phase = pd.get_obj_var("PHASE", initial_value=0.0)
        output = numpy.zeros(pd.get_vec_size())
        increment = (freq * 2.0 * numpy.pi) / pd.get_sample_rate()
        for i in range(pd.get_vec_size()):
            output[i] = numpy.sin(phase) * amplitude
            phase += increment 
            if phase > 2 * numpy.pi:
                phase -= 2 * numpy.pi
        # ---------
        pd.set_obj_var("PHASE", phase) 
        # it saves the value of phase for the next call of the function
        # without this, we will not have a continuous sine wave.
        return output

def py4pdLoadObjects():
    pd.add_object(audioout, "audioout", objtype=pd.AUDIOOUT)
