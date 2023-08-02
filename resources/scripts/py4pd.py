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
    pd.add_object(pipinstall, "py.pip")


def mysumarg(a=3, b=2, c=5, d=4):
    pd.print(f"mysumarg: {a} + {b} + {c} + {d}")
    return a + b + c + d 


def py4pdLoadObjects():
    # Pip install
    pd.add_object(pipinstall, "py.pip")
    
    pd.add_object(mysumarg, "py.mysumarg")

    # Utils
    pd.add_object(getObjectArgs, "py.getargs") 

   # Logic Functions
    pd.add_object(pyand, "py.and", pyout=True)
    pd.add_object(pyand, "py&&", pyout=True)
    pd.add_object(pyor, "py||", pyout=True)
    pd.add_object(pyequal, "py==", pyout=True)
    pd.add_object(pygreater, "py>", pyout=True)
    pd.add_object(pylower, "py<", pyout=True)
    pd.add_object(py4pdif, "py.if", pyout=True)
    pd.add_object(ommod, "py//", pyout=True)
    pd.add_object(pyisin, "pyisin") 
    

    # info
    pd.add_object(pdprint, "py.print", no_outlet=True)
    
    # Convertion Objects
    pd.add_object(py2pd, "py2pd", ignore_none_return=True)
    pd.add_object(pd2py, "pd2py")
    pd.add_object(pdlist2pylist, "py.mklist", pyout=True)

    # List Functions
    pd.add_object(pylen, "py.len")
    pd.add_object(nth, "py.nth", pyout=True)
    pd.add_object(omappend, "py.append", pyout=True)
    pd.add_object(omlist, "py.list", pyout=True)
    pd.add_object(pymax, "py.max")
    pd.add_object(pymin, "py.min")
    pd.add_object(pyreduce, "py.reduce", pyout=True)
    pd.add_object(mat_trans, "py.mattrans", pyout=True)
    pd.add_object(rotate, "py.rotate", pyout=True)
    pd.add_object(flat, "py.flat")

    # Loop Functions
    # these are special objects.
    pd.add_object(pyiterate, "py.iterate", pyout=True, num_aux_outlets=1, ignore_none_return=True) # these are special objects, they don't have a pyout argument but output py data types
    pd.add_object(pycollect, "py.collect", pyout=True, ignore_none_return=True) # these are special objects, they don't have a pyout argument but output py data types
    pd.add_object(pyrecursive, "py.recursive", pyout=True, ignore_none_return=True)
    pd.add_object(pytrigger, "py.trigger", pyout=True, ignore_none_return=True, require_outlet_n=True)
    pd.add_object(pygate, "py.gate", pyout=True, require_outlet_n=True, ignore_none_return=True)

    # Math Functions
    pd.add_object(omsum, "py+")
    pd.add_object(omminus, "py-")
    pd.add_object(omtimes, "py*")
    pd.add_object(omdiv, "py/")
    pd.add_object(omabs, "py.abs")
    pd.add_object(py4pdListComprehension, "py.listcomp", pyout=True)

    # Rhythm Tree
    pd.add_object(extract_numbers, "py.rhythm_tree")
    
    # img 
    pd.add_object(py4pdshow, "py.show", objtype=pd.VIS)

    # music convertions
    pd.add_object(freq2midicent, "f2mc")
    pd.add_object(midicent2freq, "mc2f")
    pd.add_object(midicent2note, "mc2n")

    # test
    pd.add_object(py4pdtimer, "py.timer", no_outlet=True)
    pd.add_object(getMemoryUse, "py.memuse")

    
