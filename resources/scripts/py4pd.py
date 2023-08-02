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
    from src.test import *
    from src.utils import *

except Exception as e:
    pd.error("Error loading py4pd objects: " + str(e))
    pd.addobject(pipinstall, "py.pip")


def mysumarg(a=3, b=2, c=5, d=4):
    pd.print(f"mysumarg: {a} + {b} + {c} + {d}")
    return a + b + c + d 


def py4pdLoadObjects():
    # Pip install
    pd.addobject(pipinstall, "py.pip")
    
    pd.addobject(mysumarg, "py.mysumarg")

    # Utils
    pd.addobject(getObjectArgs, "py.getargs") 

   # Logic Functions
    pd.addobject(pyand, "py.and", pyout=True)
    pd.addobject(pyand, "py&&", pyout=True)
    pd.addobject(pyor, "py||", pyout=True)
    pd.addobject(pyequal, "py==", pyout=True)
    pd.addobject(pygreater, "py>", pyout=True)
    pd.addobject(pylower, "py<", pyout=True)
    pd.addobject(py4pdif, "py.if", pyout=True)
    pd.addobject(ommod, "py//", pyout=True)
    pd.addobject(pyisin, "pyisin") 
    

    # info
    pd.addobject(pdprint, "py.print", no_outlet=True)
    
    # Convertion Objects
    pd.addobject(py2pd, "py2pd", ignore_none_return=True)
    pd.addobject(pd2py, "pd2py")
    pd.addobject(pdlist2pylist, "py.mklist", pyout=True)

    # List Functions
    pd.addobject(pylen, "py.len")
    pd.addobject(nth, "py.nth", pyout=True)
    pd.addobject(omappend, "py.append", pyout=True)
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
    pd.addobject(pycollect, "py.collect", pyout=True, ignore_none_return=True) # these are special objects, they don't have a pyout argument but output py data types
    pd.addobject(pyrecursive, "py.recursive", pyout=True, ignore_none_return=True)
    pd.addobject(pytrigger, "py.trigger", pyout=True, ignore_none_return=True, require_outlet_n=True)
    pd.addobject(pygate, "py.gate", pyout=True, require_outlet_n=True, ignore_none_return=True)

    # Math Functions
    pd.addobject(omsum, "py+")
    pd.addobject(omminus, "py-")
    pd.addobject(omtimes, "py*")
    pd.addobject(omdiv, "py/")
    pd.addobject(omabs, "py.abs")
    pd.addobject(py4pdListComprehension, "py.listcomp", pyout=True)

    # Rhythm Tree
    pd.addobject(extract_numbers, "py.rhythm_tree")
    
    # img 
    pd.addobject(py4pdshow, "py.show", objtype=pd.VIS)

    # music convertions
    pd.addobject(freq2midicent, "f2mc")
    pd.addobject(midicent2freq, "mc2f")
    pd.addobject(midicent2note, "mc2n")

    # test
    pd.addobject(py4pdtimer, "py.timer", no_outlet=True)
    pd.addobject(getMemoryUse, "py.memuse")

    
