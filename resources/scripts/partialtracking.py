import pd
import sndtrck
import numpy as np


def lorisPartialTracking(tabname):
    audio = np.array(pd.tabread(tabname))
    spectrum = sndtrck.analyze_samples(audio)
    # spectrum.plot()
    # spectrum.show()
    


