import pd

try:
    from src.pip import *
except Exception as e:
    pd.error("Error to add py.pip: " + str(e))

try:
    from src.list import *
    from src.math import *
    from src.convertion import *
    from src.loop import *
    from src.info import *
    from src.convertion import *
    from src.operators import *
    from src.tree import *
    from src.show import *
    from src.musicconvertions import *

except Exception as e:
    pd.error("Error loading py4pd objects: " + str(e))
    pd.addobject(pipinstall, "py.pip")


def py4pdLoadObjects():
    # Pip install
    pd.addobject(pipinstall, "py.pip")

   # Logic Functions
    pd.addobject(pyand, "py.and")
    pd.addobject(pyor, "py.or")

    # info
    pd.addobject(pdprint, "py.print", no_outlet=True)
    
    # Convertion Objects
    pd.addobject(py2pd, "py2pd")
    pd.addobject(pd2py, "pd2py", pyout=True)
    pd.addobject(pdlist2pylist, "py.mklist", pyout=True)

    # List Functions
    pd.addobject(pylen, "py.len")
    pd.addobject(nth, "py.nth", pyout=True)
    pd.addobject(omlist, "py.list", pyout=True)
    pd.addobject(pymax, "py.max")
    pd.addobject(pymin, "py.min")
    pd.addobject(pyreduce, "py.reduce", pyout=True)
    pd.addobject(mat_trans, "py.mattrans", pyout=True)
    pd.addobject(rotate, "py.rotate", pyout=True)
    pd.addobject(flat, "py.flat")

    # Loop Functions
    # these are special objects.
    pd.addobject(pyiterate, "py.iterate", pyout=True, num_aux_outlets=1, ignore_none_return=True) # these are special objects, they don't have a pyout argument but output py data types
    pd.addobject(pycollect, "py.collect", pyout=True) # these are special objects, they don't have a pyout argument but output py data types

    # Math Functions
    pd.addobject(omsum, "py.sum")
    pd.addobject(omminus, "py.minus")
    pd.addobject(omtimes, "py.times")
    pd.addobject(omdiv, "py.div")
    pd.addobject(omabs, "py.abs")

    # Rhythm Tree
    pd.addobject(extract_numbers, "py.rhythm_tree")
    
    # img 
    pd.addobject(py4pdshow, "py.show", objtype=pd.VIS)


    # music convertions
    pd.addobject(freq2midicent, "f2mc")
    pd.addobject(midicent2freq, "mc2f")
    pd.addobject(midicent2note, "mc2n")

    
