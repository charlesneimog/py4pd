from neoscore.common import *
import os
import random


def note(*args):
    pitch = args[0]
    neoscore.setup()
    randomNumber = random.randint(1, 100)
    POSITION = (Mm(0), Mm(0))
    staff = Staff(POSITION, None, Mm(30))
    saxClef = 'treble'
    Clef(ZERO, staff, saxClef) # .ppm
    note = [(args, '', 4)]
    Chordrest(Mm(5), staff, note, (int(1), int(1)))
    scriptPath = os.path.dirname(os.path.abspath(__file__))
    notePathName = scriptPath + "/note_" + pitch + f"{randomNumber}.ppm"
    neoscore.render_image(rect=None, dest=notePathName, dpi=150, wait=True)
    neoscore.shutdown()
    return f"open note_{pitch}{randomNumber}.ppm"

# inspect the note function
import inspect

# discover if function argument is *args or *kwargs
def is_args(func):
    return inspect.signature(func).parameters.get('args', None) is not None

print(is_args(note))
    


