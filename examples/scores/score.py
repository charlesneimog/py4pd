from neoscore.common import *
import os
import random
import pd


def note(pitch):
    scriptPath = os.path.dirname(os.path.abspath(__file__))
    for file in os.listdir(scriptPath):
        if file.endswith(".ppm"):
            try:
                os.remove(os.path.join(scriptPath, file))
            except:
                pass
                
    neoscore.setup()
    try:
        randomNumber = random.randint(1, 100)
        POSITION = (Mm(0), Mm(0))
        staff = Staff(POSITION, None, Mm(30))
        saxClef = 'treble'
        Clef(ZERO, staff, saxClef) # .ppm
        note = [(pitch, '', 4)]
        Chordrest(Mm(5), staff, note, (int(1), int(1)))
        scriptPath = os.path.dirname(os.path.abspath(__file__))
        notePathName = scriptPath + "/note_" + pitch + f"{randomNumber}.ppm"
        neoscore.render_image(rect=None, dest=notePathName, dpi=150, wait=True)
        neoscore.shutdown()
        return f"open note_{pitch}{randomNumber}.ppm"
    except:
        pd.print("Error generating score")
        neoscore.shutdown()
    
    
    




