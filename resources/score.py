import pd
import os
import random

try:
    from neoscore.common import *
except:
    pd.error("Seems that neoscore is not installed.")
    pd.print("Please install neoscore running: pip install neoscore -t ./py-modules")
    pd.print("And restart the patch.")
    pd.print("Remember that you need to have python3.11 installed.")

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
        neoscore.shutdown()
    except:
        pass
    neoscore.setup()
    scriptPath = os.path.dirname(os.path.abspath(__file__))
    allFiles = os.listdir(scriptPath + "/__pycache__")
    for file in allFiles:
        if file.endswith(".ppm"):
            try:
                os.remove(scriptPath + "/__pycache__/" + file)
            except:
                pass
    staffSoprano = Staff((Mm(0), Mm(0)), None, Mm(30))
    trebleClef = 'treble'
    Clef(ZERO, staffSoprano, trebleClef)
    staffBaixo = Staff((ZERO, Mm(15)), None, Mm(30))
    bassClef = 'bass'
    Clef(ZERO, staffBaixo, bassClef)   
    Path.rect((Mm(-10), Mm(-10)), None, Mm(42), Mm(42), Brush(Color(0, 0, 0, 0)), Pen(thickness=Mm(0.5)),
)
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
    randomNumber = random.randint(1, 100) 
    notePathName = scriptPath + "/__pycache__/note_" + pitch + f"{randomNumber}.ppm"
    neoscore.render_image(rect=None, dest=notePathName, dpi=150, wait=True)
    neoscore.shutdown()
    pd.show(notePathName)
    return None


