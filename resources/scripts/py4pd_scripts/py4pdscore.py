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



def neoscore_midicent2note(midicent):

    """
    Converts a midicent note number to a note name string.
    """
    if isinstance(midicent, list):
        return [neoscore_midicent2note(x) for x in midicent]
    sharpOrFlat = pd.getkey("accidentals")
    if sharpOrFlat is None or sharpOrFlat == "sharp":
        sharpOrFlat = "sharp"
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'C']
    elif sharpOrFlat == "flat":
        note_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C']
    else:
        raise ValueError("Invalid value for sharpOrFlat: " + str(sharpOrFlat))

    # multiply by 0.01, then round to nearest integer, then multiply by 100

    newmidicent = round(float(midicent) * 0.01) * 100
    desviation = midicent - newmidicent
    desviation = round(desviation, 1)
    octave = int(midicent // 1200) - 1
    note = int(newmidicent / 100) % 12
    if desviation > 40 and desviation < 60:
        return f"{note_names[note]}+{octave}"
    elif desviation < -40 and desviation > -60:
        return f"{note_names[note]}-{octave}"
    else:
        return f"{note_names[note]}{octave}"




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

        # quarter-tone sharp
        'c+': ['c', 'accidentalQuarterToneSharpStein'],
        'd+': ['d', 'accidentalQuarterToneSharpStein'],
        'e+': ['e', 'accidentalQuarterToneSharpStein'],
        'f+': ['f', 'accidentalQuarterToneSharpStein'],
        'g+': ['g', 'accidentalQuarterToneSharpStein'],
        'a+': ['a', 'accidentalQuarterToneSharpStein'],
        'b+': ['b', 'accidentalQuarterToneSharpStein'],

        # quarter-tone flat
        'c-': ['c', 'accidentalQuarterToneFlatStein'],
        'd-': ['d', 'accidentalQuarterToneFlatStein'],
        'e-': ['e', 'accidentalQuarterToneFlatStein'],
        'f-': ['f', 'accidentalQuarterToneFlatStein'],
        'g-': ['g', 'accidentalQuarterToneFlatStein'],
        'a-': ['a', 'accidentalQuarterToneFlatStein'],
        'b-': ['b', 'accidentalQuarterToneFlatStein'],

        # three-quarter-tone sharp
        'c#+': ['c', 'accidentalThreeQuarterTonesSharpStein'],
        'd#+': ['d', 'accidentalThreeQuarterTonesSharpStein'],
        'e#+': ['e', 'accidentalThreeQuarterTonesSharpStein'],
        'f#+': ['f', 'accidentalThreeQuarterTonesSharpStein'],
        'g#+': ['g', 'accidentalThreeQuarterTonesSharpStein'],
        'a#+': ['a', 'accidentalThreeQuarterTonesSharpStein'],
        'b#+': ['b', 'accidentalThreeQuarterTonesSharpStein'],

        # three-quarter-tone flat
        'cb-': ['c', 'accidentalThreeQuarterTonesFlatZimmermann'],
        'db-': ['d', 'accidentalThreeQuarterTonesFlatZimmermann'],
        'eb-': ['e', 'accidentalThreeQuarterTonesFlatZimmermann'],
        'fb-': ['f', 'accidentalThreeQuarterTonesFlatZimmermann'],
        'gb-': ['g', 'accidentalThreeQuarterTonesFlatZimmermann'],
        'ab-': ['a', 'accidentalThreeQuarterTonesFlatZimmermann'],
        'bb-': ['b', 'accidentalThreeQuarterTonesFlatZimmermann'],


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
    elif isinstance(pitches, int):
        try:
            pitches = [neoscore_midicent2note(pitches)]
        except BaseException:
            pd.error("The integer must be a midicent")
            return
    elif isinstance(pitches, float):
        try:
            pitches = [neoscore_midicent2note(int(pitches))]
        except BaseException:
            pd.error("The float must be a midicent")
            return

    if isinstance(pitches, list):
        newPitches = []
        for pitch in pitches:
            if isinstance(pitch, str):
                newPitches.append(pitch)
            elif isinstance(pitch, int):
                newPitches.append(neoscore_midicent2note(pitch))
            elif isinstance(pitch, float):
                newPitches.append(neoscore_midicent2note(int(pitch)))

            else:
                pd.error("The list must contain only strings (c4, c#4, c+4, etc) or integers (midicents)")
                return
        pitches = newPitches
    pitches = [x.lower() for x in pitches]
    py4pdTMPfolder = pd.tempfolder()
    staffSoprano = Staff((Mm(0), Mm(0)), None, Mm(30))
    trebleClef = 'treble'
    Clef(ZERO, staffSoprano, trebleClef)
    staffBaixo = Staff((ZERO, Mm(15)), None, Mm(30))
    bassClef = 'bass'
    Clef(ZERO, staffBaixo, bassClef)
    Path.rect((Mm(-10), Mm(-10)), None, Mm(42), Mm(42),
              Brush(Color(255, 255, 255, 0)), Pen(thickness=Mm(0)))

    for pitch in pitches:
        pitchWithoutNumber = pitch.replace(pitch[-1], '')
        pitchOctave = int(pitch[-1])
        pitchClass, accidental = getpitchKey(pitchWithoutNumber)
        note = [(pitchClass, accidental, pitchOctave)]
        try:
            if pitchOctave < 4:
                Chordrest(Mm(5), staffBaixo, note, (int(1), int(1)))
            else:
                Chordrest(Mm(5), staffSoprano, note, (int(1), int(1)))
        except Exception as e:
            pd.error(e)
            return

    scoreNumber = pd.getglobalvar('scoreNumber')
    if scoreNumber is None:
        scoreNumber = 0
    notePathName = py4pdTMPfolder + "/" + pd.getobjpointer() + "_" + str(scoreNumber) + ".ppm"
    pd.setglobalvar('scoreNumber', scoreNumber + 1)
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

