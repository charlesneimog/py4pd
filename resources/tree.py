import pd
import math
import random
import os


IMAGE_NUMBER = 0

try:
    from neoscore.common import *
except ImportError:
    pd.error("neoscore is not installed")

def om_tree(tree):
    """This build om-tree from a tree"""
    global IMAGE_NUMBER
    try:
        neoscore.shutdown()
    except BaseException:
        pass
    neoscore.setup()

    staffSoprano = Staff((Mm(0), Mm(0)), None, Mm(30))
    trebleClef = 'treble'
    Clef(ZERO, staffSoprano, trebleClef)

    Path.rect((Mm(-5), Mm(-5)), None, Mm(22), Mm(22),
          Brush(Color(255, 255, 255, 0)), Pen(thickness=Mm(-1)))

    for measure in tree:
        if (len(measure)) == 2:
            timeSignature = measure[0]
            rhythmTree = measure[1]
            if any(isinstance(elem, list) for elem in rhythmTree):
                pd.print("There is tuplets inside measure")
            else:
                totalOfAttacks = sum(rhythmTree)
                isTuplet = totalOfAttacks / timeSignature[0]
                for key in dir(staffSoprano):
                    pd.print("Staff: " + str(key))
                staffSoprano.length = Mm(3)
                if math.log2(isTuplet).is_integer():
                    timeUnit = timeSignature[0] * isTuplet
                    space = Mm(3)
                    for value in rhythmTree:
                        Chordrest(space, staffSoprano, ["c"], (int(value), int(timeUnit)))
                        space += Mm(5)
                        
                else:
                    pd.print("Is tuplet")

        elif (len(measure)) == 1:
            pd.print("Measure: " + str(measure[0]))
        else:
            pd.error("Invalid measure: " + str(measure))
            return
    notePathName = pd.home() + "/" + f"{IMAGE_NUMBER}.ppm"
    IMAGE_NUMBER += 1
    neoscore.render_image(rect=None, dest=notePathName, dpi=150, wait=True)
    neoscore.shutdown()
    if os.name == 'nt':
        notePathName = notePathName.replace("\\", "/")
    return notePathName


def py4pdLoadObjects():
    pd.addobject(om_tree, "omtree")
