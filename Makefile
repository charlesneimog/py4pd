# library name
lib.name = py4pd

ldlibs = -L ${py_lib}  -l $(version)

# input source file (class name == source file basename)
class.sources = src/py4pd.c

# all extra files to be included in binary distribution of the library
datafiles = py4pd-help.pd 

# include Makefile.pdlibbuilder
# (for real-world projects see the "Project Management" section
# in tips-tricks.md)
PDLIBBUILDER_DIR=./pd-lib-builder/
include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder