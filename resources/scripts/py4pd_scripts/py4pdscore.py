import pd
from random import randint
import os
os.environ['QT_QPA_PLATFORM'] = "xcb" 
if os.name == 'nt':
    os.environ['QT_QPA_PLATFORM'] = "windows"
    
try:
    from neoscore.common import *
except Exception as e:
    pd.error(str(e))
    pd.error("To fix this, send the message 'global nescore' to the object py.pip and restart Pd.")
    # TODO: CREATE pd.install('neoscore') to install the module


def getpitchKey(pitch, cents=0):
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


def chord(pitches):
    try:
        neoscore.shutdown()
    except BaseException:
        pass
    neoscore.setup()

    if isinstance(pitches, str):
        pitches = [pitches]

    # make all string with small letters
    pitches = [x.lower() for x in pitches]

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
              Brush(Color(255, 255, 255, 0)), Pen(thickness=Mm(0)))
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
    pd.show(notePathName)
    return None



def note(pitch):
    try:
        neoscore.shutdown()
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
              Brush(Color(255, 255, 255, 0)), Pen(thickness=Mm(0)))
    # get cents
    # the note can be represent as c#5 cb3 
    # with cents we have  c#5+50 cb3-50
    # get cents and remove from pitch
    cents = 0
    if '+' in pitch:
        cents = int(pitch.split('+')[1])
        pitch = pitch.split('+')[0]
    if '-' in pitch:
        cents = int(pitch.split('-')[1]) * -1
        pitch = pitch.split('-')[0]
    

    # in pitch remove not number
    pitchWithoutNumber = pitch.replace(pitch[-1], '')
    pitchOctave = int(pitch[-1])
    pitchClass, accidental = getpitchKey(pitchWithoutNumber,  cents)
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
    pd.print(str(notePathName))
    pd.show(notePathName)
    return None

