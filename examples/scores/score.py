from neoscore.common import *
import os
import random
import pd


def note(pitch):
    scriptPath = os.path.dirname(os.path.abspath(__file__))
    # make this in 
    for file in os.listdir(scriptPath + "/__pycache__"):
        if file.endswith(".ppm"):
            try:
                os.remove(os.path.join(scriptPath + "/__pycache__", file))
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
        notePathName = scriptPath + "/__pycache__/note_" + pitch + f"{randomNumber}.ppm"
        neoscore.show(display_page_geometry=False)
        neoscore.shutdown()
        return f"open  ./__pycache__/note_{pitch}{randomNumber}.ppm"
    except:
        pd.print("Error generating score")
        neoscore.shutdown()
    
    
    




