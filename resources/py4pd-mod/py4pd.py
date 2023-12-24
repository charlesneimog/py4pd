import os

import pd

# list all folders inside src
thisFile = os.path.abspath(__file__)

# try to import all files inside src. (must use this file)
for root, dirs, files in os.walk(os.path.join(thisFile, "src")):
    for file in files:
        if file.endswith(".py") and file != "__init__.py":
            try:
                exec(f"from src.{file[:-3]} import *")
            except Exception as e:
                pd.error(f"Error importing {file}: {e}")


from src.convertion import *
from src.info import *
from src.libs import *
from src.list import *
from src.loop import *
from src.math import *
from src.musicconvertions import *
from src.openmusic import *
from src.operators import *
from src.pip import *
from src.show import *
from src.test import *
from src.tree import *
from src.utils import *

# try to import all files inside src.


def mysumarg(a=3, b=2, c=5, d=4):
    pd.print(f"mysumarg: {a} + {b} + {c} + {d}")
    return a + b + c + d


def py4pdLoadObjects():
    # Utils
    pd.add_object(getObjectArgs, "py.getargs")

    # Logic Functions
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

    # Convertion Objects
    pd.add_object(py2pd, "py2pd", ignore_none_return=True)
    pd.add_object(pd2py, "pd2py", py_out=True)
    pd.add_object(pdlist2pylist, "py.mklist", py_out=True)

    # List Functions
    pd.add_object(pylen, "py.len")
    pd.add_object(nth, "py.nth", py_out=True)
    pd.add_object(omappend, "py.append", py_out=True)
    pd.add_object(omlist, "py.list", py_out=True)
    pd.add_object(pymax, "py.max")
    pd.add_object(pymin, "py.min")
    pd.add_object(pyreduce, "py.reduce", py_out=True)
    pd.add_object(mat_trans, "py.mattrans", py_out=True)
    pd.add_object(rotate, "py.rotate", py_out=True)
    pd.add_object(flat, "py.flat")

    # Loop Functions
    # these are special objects.
    pd.add_object(
        pyiterate, "py.iterate", py_out=True, num_aux_outlets=1, ignore_none_return=True
    )  # these are special objects, they don't have a pyout argument but output py data types
    pd.add_object(
        pycollect, "py.collect", py_out=True, ignore_none_return=True
    )  # these are special objects, they don't have a pyout argument but output py data types
    pd.add_object(pyrecursive, "py.recursive", py_out=True, ignore_none_return=True)
    pd.add_object(
        pytrigger,
        "py.trigger",
        py_out=True,
        ignore_none_return=True,
        require_outlet_n=True,
    )
    pd.add_object(
        pygate, "py.gate", py_out=True, require_outlet_n=True, ignore_none_return=True
    )

    # Math Functions
    pd.add_object(omsum, "py+")
    pd.add_object(omminus, "py-")
    pd.add_object(omtimes, "py*")
    pd.add_object(omdiv, "py/")
    pd.add_object(omabs, "py.abs")
    pd.add_object(ommod, "py//", py_out=True)

    pd.add_object(py4pdListComprehension, "py.listcomp", py_out=True)

    # Rhythm Tree
    pd.add_object(extract_numbers, "py.rhythm_tree")

    # OpenMusic
    pd.add_object(arithm_ser, "py.arithm-ser")

    # img
    pd.add_object(py4pdshow, "py.show", obj_type=pd.VIS)

    # music convertions
    pd.add_object(freq2midicent, "f2mc")
    pd.add_object(midicent2freq, "mc2f")
    pd.add_object(midicent2note, "mc2n")

    # test
    pd.add_object(py4pdtimer, "py.timer", no_outlet=True)
    pd.add_object(getMemoryUse, "py.memuse")

    # py4pd libs
    py4pd_libs = pd.new_object("py4pd.libs")
    py4pd_libs.addmethod("install", downloadPy4pdLibraries)
    py4pd_libs.addmethod("libraries", listPy4pdLibraries)
    py4pd_libs.help_patch = "py4pd.libs-help.pd"
    py4pd_libs.add_object()
