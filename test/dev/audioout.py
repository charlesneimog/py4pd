import pd
import numpy

def audioout(freq, amplitude):
    if freq is None: 
        return numpy.zeros(pd.get_vec_size())
    else:
        phase = pd.get_obj_var("PHASE1", initial_value=0.0)
        output = numpy.zeros((2, pd.get_vec_size()))
        
        increment = (freq * 2.0 * numpy.pi) / pd.get_sample_rate()
        for i in range(pd.get_vec_size()):
            output[0][i] = numpy.sin(phase) * amplitude
            phase += increment 
            if phase > 2 * numpy.pi:
                phase -= 2 * numpy.pi
        pd.set_obj_var("PHASE1", phase) 

        phase2 = pd.get_obj_var("PHASE2", initial_value=0.0)

        increment = ((freq * 1.5) * 2.0 * numpy.pi) / pd.get_sample_rate()
        for i in range(pd.get_vec_size()):
            output[1][i] = numpy.sin(phase2) * amplitude
            phase2 += increment 
            if phase2 > 2 * numpy.pi:
                phase2 -= 2 * numpy.pi
        pd.set_obj_var("PHASE2", phase2) 
        
        # create 2d array for 2 channels
        return output

def py4pdLoadObjects():
    pd.add_object(audioout, "audioout", objtype=pd.AUDIOOUT)
