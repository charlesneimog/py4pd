import pd
import loristrck as lt
import numpy as np


def resynthSound(tableName):
    samples = pd.tabread(tableName, numpy=True)
    samples = np.array(samples, dtype=np.float64)
    partials = lt.analyze(samples, pd.get_sample_rate(), resolution=100)
    partials, _ = lt.util.select(partials,  maxfreq=10000)
    matplotaxis = lt.util.plot_partials(partials)
    # save figure
    fig = matplotaxis.get_figure()

    # remove margins
    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.07, top=0.95)

    imgNumber = pd.get_obj_var("imgNumber", initial_value=0)
    fig.savefig(pd.get_temp_dir() + f"/loris{imgNumber}.png")
    pd.set_obj_var("imgNumber", imgNumber + 1)
    return pd.get_temp_dir() + "/loris" + str(imgNumber) + ".png"


def py4pdLoadObjects():
    pd.add_object(resynthSound, "loris")


