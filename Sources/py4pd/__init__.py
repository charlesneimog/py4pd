import os

import pd
from py4pd.convertion import pd2py, pdlist2pylist, py2pd
from py4pd.info import *
from py4pd.libs import *
from py4pd.list import *
from py4pd.loop import *
from py4pd.musicconvertions import *
from py4pd.openmusic import *
from py4pd.operators import *
from py4pd.pip import *
from py4pd.pymath import *
from py4pd.show import *
from py4pd.test import *
from py4pd.tree import *
from py4pd.utils import *

# try to import all files inside src.


def numpyIsInstalled():
    try:
        import numpy

        return True
    except:
        return False


def mysumarg(a=3, b=2, c=5, d=4):
    pd.print(f"mysumarg: {a} + {b} + {c} + {d}")
    return a + b + c + d


def py4pd_setup():
    numpyInstalled = numpyIsInstalled()

    # Utils
    pd.add_object(getObjectArgs, "py.getargs")

    # Logic Functions

    pd.add_object(pyisnone, "py.isnone", py_out=True)
    pd.add_object(pyand, "py.and", py_out=True, help_patch="py.logic-help.pd")
    pd.add_object(pyand, "py&&", py_out=True, help_patch="py.logic-help.pd")
    pd.add_object(pyor, "py.or", py_out=True, help_patch="py.logic-help.pd")
    pd.add_object(pyor, "py||", py_out=True, help_patch="py.logic-help.pd")
    pd.add_object(pyequal, "py.equal", py_out=True, help_patch="py.logic-help.pd")
    pd.add_object(pyequal, "py==", py_out=True, help_patch="py.logic-help.pd")
    pd.add_object(pygreater, "py>", py_out=True, help_patch="py.logic-help.pd")
    pd.add_object(pylower, "py<", py_out=True, help_patch="py.logic-help.pd")
    pd.add_object(py4pdif, "py.if", py_out=True, help_patch="py.logic-help.pd")
    pd.add_object(pyisin, "pyisin", py_out=True, help_patch="py.logic-help.pd")

    # info
    pd.add_object(pdprint, "py.print", no_outlet=True, help_path="py.print-help.pd")
    pd.add_object(pdraise, "py.raise", no_outlet=True)

    # Convertion Objects
    pd.add_object(py2pd, "py2pd", ignore_none=True)
    pd.add_object(pd2py, "pd2py", py_out=True)
    pd.add_object(pdlist2pylist, "py.mklist", py_out=True)

    # List Functions
    pd.add_object(pylen, "py.len")
    pd.add_object(nth, "py.nth", py_out=True)
    pd.add_object(omappend, "py.append", py_out=True)
    pd.add_object(omlist, "py.list", py_out=True)
    pd.add_object(pysplit, "py.split", py_out=True, require_outlet_n=True)
    pd.add_object(pymax, "py.max")
    pd.add_object(pymin, "py.min")
    pd.add_object(pyreduce, "py.reduce", py_out=True)
    pd.add_object(mat_trans, "py.mattrans", py_out=True)
    pd.add_object(rotate, "py.rotate", py_out=True)
    pd.add_object(flat, "py.flat")

    pd.add_object(getObjectType, "py.type")

    if numpyInstalled:
        pd.add_object(np2list, "py.np2list", py_out=True)
        pd.add_object(list2np, "py.list2np", py_out=True)

    # Loop Functions
    pd.add_object(pyrange, "py.range", py_out=True, ignore_none=True)
    pd.add_object(
        pyiterate, "py.iterate", py_out=True, n_outlets=2, ignore_none=True
    )  # these are special objects, they don't have a pyout argument but output py data types
    pd.add_object(
        pycollect, "py.collect", py_out=True, ignore_none=True
    )  # these are special objects, they don't have a pyout argument but output py data types
    pd.add_object(pyrecursive, "py.recursive", py_out=True, ignore_none=True)
    pd.add_object(
        pytrigger,
        "py.trigger",
        py_out=True,
        ignore_none=True,
        require_outlet_n=True,
    )
    pd.add_object(
        pygate, "py.gate", py_out=True, require_outlet_n=True, ignore_none=True
    )

    # Math Functions
    pd.add_object(omsum, "py+")
    pd.add_object(omminus, "py-")
    pd.add_object(omtimes, "py*")
    pd.add_object(omdiv, "py/")
    pd.add_object(omabs, "py.abs")
    pd.add_object(ommod, "py//", py_out=True)
    pd.add_object(pdround, "py.round")

    pd.add_object(py4pdListComprehension, "py.listcomp", py_out=True)
    pd.add_object(py4pdListComprehension, "py.expr", py_out=True)

    # Rhythm Tree
    pd.add_object(omtree, "py.omtree", playable=True, n_outlets=2, ignore_none=True)

    # OpenMusic
    pd.add_object(arithm_ser, "py.arithm-ser")

    # img
    pd.add_object(py4pdshow, "py.show", obj_type=pd.VIS)

    # music convertions
    pd.add_object(freq2midicent, "py.f2mc")
    pd.add_object(midicent2freq, "py.mc2f")
    pd.add_object(midicent2note, "py.mc2n")

    # test
    pd.add_object(py4pdtimer, "py.timer", no_outlet=True)
    pd.add_object(getMemoryUse, "py.memuse")
