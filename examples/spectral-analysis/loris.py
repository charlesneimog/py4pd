import pd
import loristrck as lt
import numpy as np


def resynthSound(tableName):
    samples = pd.tabread(tableName, numpy=True)
    samples = np.array(samples, dtype=np.float64)
    partials = lt.analyze(samples, pd.samplerate(), resolution=100)
    partials, _ = lt.util.select(partials,  maxfreq=10000)
    matplotaxis = lt.util.plot_partials(partials)
    # save figure
    fig = matplotaxis.get_figure()

    # remove margins
    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.07, top=0.95)

    imgNumber = pd.getglobalvar("imgNumber", initial_value=0)
    fig.savefig(pd.tempfolder() + f"/loris{imgNumber}.png")
    pd.setglobalvar("imgNumber", imgNumber + 1)
    return pd.tempfolder() + "/loris" + str(imgNumber) + ".png"









def py4pdLoadObjects():
    pd.addobject(resynthSound, "loris")

    
    # My License, Name and University, others information
    pd.print("", show_prefix=False)
    pd.print("GPL3 | by Charles K. Neimog", show_prefix=False)
    pd.print("University of São Paulo", show_prefix=False)
    pd.print("", show_prefix=False)
