# library name
lib.name = py4pd

# python libs

cflags = -I $(PYTHON_INCLUDE)

# if running in Windows, define ldlibs = -L ${PYTHON_LIB} -l $(PYTHON_VERSION)
# if running in  Linux, define ldlibs =  -l python$(PYTHON_VERSION)
# if running in  Mac, define ldlibs = -l python$(PYTHON_VERSION)

uname := $(shell uname -s)

ifeq (MINGW,$(findstring MINGW,$(uname)))
  ldlibs = -L ${PYTHON_LIB} -l $(PYTHON_VERSION)
endif

ifeq (Linux,$(findstring Linux,$(uname)))
  ldlibs = -l $(PYTHON_VERSION) 
endif

ifeq (Darwin,$(findstring Darwin,$(uname)))
  ldlibs = -l $(PYTHON_VERSION)
endif

# input source file (class name == source file basename)
class.sources = src/py4pd.c

# all extra files to be included in binary distribution of the library
datafiles = py4pd-help.pd 

# include Makefile.pdlibbuilder
# (for real-world projects see the "Project Management" section
# in tips-tricks.md)
PDLIBBUILDER_DIR=./pd-lib-builder/
include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder