import loristrck as lt
import os
import numpy as np

def lorisAnalisys(audio, resolution=60):
    print("Analysing audio... Wait...")
    # if audio start with ./ add the path
    if audio[0:2] == "./":
        script_dir = os.path.dirname(__file__)
        audio = os.path.join(script_dir, audio[2:])
    samples, sr = lt.util.sndreadmono(audio)
    partials = lt.analyze(samples, sr, resolution=resolution)
    selected, noise = lt.util.select(partials, mindur=0.02, maxfreq=12000, minamp=-35, minbps=2)
    # translate numpy.ndarray to list
    for partial in selected:
        print(partial.tolist())
        break
    




