# library name
lib.name = py4pd

# python libs

uname := $(shell uname -s)

# remove -Wunused-variable warnings

ifeq (MINGW,$(findstring MINGW,$(uname)))
  cflags = -I $(PYTHON_INCLUDE) -Wno-cast-function-type -Wno-unused-variable 
  ldlibs =  $(PYTHON_DLL) -lwinpthread
  pythondll_name = $(shell basename $(PYTHON_DLL))
  $(shell cp $(PYTHON_DLL) $(pythondll_name))

else ifeq (Linux,$(findstring Linux,$(uname)))
  cflags = -I $(PYTHON_INCLUDE) -Wno-cast-function-type -Wno-unused-variable
  ldlibs = -l $(PYTHON_VERSION) 

else ifeq (Darwin,$(findstring Darwin,$(uname)))
  cflags = -I $(PYTHON_INCLUDE) -Wno-cast-function-type -Wno-unused-variable -mmacosx-version-min=10.9
  ldlibs = -L "/Library/Frameworks/Python.framework/Versions/3.11/lib/" -l python3.11

else
  $(error "Unknown system type: $(uname)")
  $(shell exit 1)

endif

# =================================== Sources ===================================

py4pd.class.sources = src/py4pd.c src/module.c 
# py4pd~.class.sources = src/py4pd_tilde.c

# =================================== Data ======================================
datafiles = \
$(wildcard Help-files/*.pd) \
$(wildcard scripts/*.py) \
$(wildcard py.py) \
$(wildcard py4pd-help.pd) \
$(wildcard python311._pth) \
$(PYTHON_DLL)

# =================================== Pd Lib Builder =============================

PDLIBBUILDER_DIR=./pd-lib-builder/
include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder

