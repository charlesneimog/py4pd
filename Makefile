# library name
lib.name = py4pd

# python libs

uname := $(shell uname -s)

ifeq (MINGW,$(findstring MINGW,$(uname)))
  # remove -Wcast-function-type for mingw 
  cflags = -I $(PYTHON_INCLUDE) -I $(NUMPY_INCLUDE) -Wno-cast-function-type  
  ldlibs =  $(PYTHON_DLL)  
  pythondll_name = $(shell basename $(PYTHON_DLL))
  $(shell cp $(PYTHON_DLL) $(pythondll_name))

endif

ifeq (Linux,$(findstring Linux,$(uname)))
  cflags = -I $(PYTHON_INCLUDE) -Wno-cast-function-type
  ldlibs = -l $(PYTHON_VERSION) 
endif

ifeq (Darwin,$(findstring Darwin,$(uname)))
  cflags = -I $(PYTHON_INCLUDE) -Wno-cast-function-type
  ldlibs = -l $(PYTHON_VERSION)
endif

# input source file (class name == source file basename)
py4pd.class.sources = src/py4pd.c 
py4pd~.class.sources = src/py4pd_tilde.c

# all extra files to be included in binary distribution of the library
datafiles = \
$(wildcard Help-files/*.pd) \
$(wildcard scripts/*.py) \
$(wildcard py4pd-help.pd) \
$(PYTHON_DLL)

# include Makefile.pdlibbuilder
# (for real-world projects see the "Project Management" section
# in tips-tricks.md)
PDLIBBUILDER_DIR=./pd-lib-builder/
include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder

