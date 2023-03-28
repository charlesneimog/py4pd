import pd
from random import randint
import os
import time
import math
import sys

try:
    from neoscore.common import *
except Exception as e:
    pd.error(str(e))
    pd.error("Please, run 'pip install neoscore -t ./py-modules' in the terminal from current folder")
    sys.exit(1)

try:
    import numpy as np
except Exception as e:
    pd.error(str(e))
    pd.print("Please, run 'pip install numpy -t ./py-modules' in the terminal from current folder")
    sys.exit(1)
	
try:
    import matplotlib
    # NOTE:  Always use 'agg' backend or other than Tkinter. Tkinket will
    # conflict with Pd
    matplotlib.use('agg')
    from matplotlib import pyplot as plt
except Exception as e:
    pd.error(str(e))
    pd.error("Error importing matplotlib")
    pd.print(
        "Please, run 'pip install matplotlib -t ./py-modules' in the terminal from current folder")
    sys.exit(1)


def getpitchKey(pitch):
    note = {
        # natural
        'c': ['c', ''],
        'd': ['d', ''],
        'e': ['e', ''],
        'f': ['f', ''],
        'g': ['g', ''],
        'a': ['a', ''],
        'b': ['b', ''],
        # sharp
        'c#': ['c', 'accidentalSharp'],
        'd#': ['d', 'accidentalSharp'],
        'e#': ['e', 'accidentalSharp'],
        'f#': ['f', 'accidentalSharp'],
        'g#': ['g', 'accidentalSharp'],
        'a#': ['a', 'accidentalSharp'],
        'b#': ['b', 'accidentalSharp'],
        # flat
        'cb': ['c', 'accidentalFlat'],
        'db': ['d', 'accidentalFlat'],
        'eb': ['e', 'accidentalFlat'],
        'fb': ['f', 'accidentalFlat'],
        'gb': ['g', 'accidentalFlat'],
        'ab': ['a', 'accidentalFlat'],
        'bb': ['b', 'accidentalFlat'],
    }
    return note[pitch]


def note(pitches):
    try:
        neoscore.shutdown() # to avoid errors
    except BaseException:
        pass
    neoscore.setup()
    py4pdTMPfolder = pd.tempfolder()
    for file in py4pdTMPfolder:
        if file.endswith(".ppm"):
            try:
                os.remove(py4pdTMPfolder + "/" + file)
            except BaseException:
                pass
    staffSoprano = Staff((Mm(0), Mm(0)), None, Mm(30))
    trebleClef = 'treble'
    Clef(ZERO, staffSoprano, trebleClef)
    staffBaixo = Staff((ZERO, Mm(15)), None, Mm(30))
    bassClef = 'bass'
    Clef(ZERO, staffBaixo, bassClef)
    Path.rect((Mm(-10), Mm(-10)), None, Mm(42), Mm(42),
              Brush(Color(0, 0, 0, 0)), Pen(thickness=Mm(-0.5)))
    for pitch in pitches:
        # in pitch remove not number
        pitchWithoutNumber = pitch.replace(pitch[-1], '')
        pitchOctave = int(pitch[-1])
        pitchClass, accidental = getpitchKey(pitchWithoutNumber)
        note = [(pitchClass, accidental, pitchOctave)]
        if pitchOctave < 4:
            Chordrest(Mm(5), staffBaixo, note, (int(1), int(1)))
        else:
            Chordrest(Mm(5), staffSoprano, note, (int(1), int(1)))
    randomNumber = randint(1, 100)
    notePathName = py4pdTMPfolder + "/" + pitch + f"{randomNumber}.ppm"
    neoscore.render_image(rect=None, dest=notePathName, dpi=150, wait=True)
    neoscore.shutdown()
    if os.name == 'nt':
        notePathName = notePathName.replace("\\", "/")
    #pd.print(str(notePathName))
    pd.show(notePathName)
    return None

def stringTest(a):
	pd.print(str(a))
	return "Ok"

def py4pdLoadObjects():
    pd.addobject(note, "score", objtype="VIS")
    pd.addobject(stringTest, "stringTest", objtype="VIS")

