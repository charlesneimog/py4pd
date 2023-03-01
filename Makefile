lib.name = py4pd

uname := $(shell uname -s)

ifeq (MINGW,$(findstring MINGW,$(uname)))
  PYTHON_INCLUDE := $(shell cat pythonincludes.txt)
  PYTHON_PATH := $(shell cat pythonpath.txt)
  NUMPY_INCLUDE := $(shell cat numpyincludes.txt)
  PYTHON_DLL := $(PYTHON_PATH)/python311.dll
  cflags = -I $(PYTHON_INCLUDE) -I $(NUMPY_INCLUDE) -Wno-cast-function-type -Wno-unused-variable 
  ldlibs =  $(PYTHON_DLL) -lwinpthread 
else ifeq (Linux,$(findstring Linux,$(uname)))
  PYTHON_INCLUDE := $(shell python3.11 -c 'import sysconfig;print(sysconfig.get_config_var("INCLUDEPY"))')
  NUMPY_INCLUDE := $(shell python3.11 -c 'import numpy.distutils.misc_util as np_utils; print(np_utils.get_numpy_include_dirs()[0])')
  cflags = -I $(PYTHON_INCLUDE) -I $(NUMPY_INCLUDE) -Wno-cast-function-type -Wl,-export-dynamic 
  ldlibs = -l $(PYTHON_VERSION) -Wl,-export-dynamic
else ifeq (Darwin,$(findstring Darwin,$(uname)))
  PYTHON_INCLUDE := $(shell python3.11 -c 'import sysconfig;print(sysconfig.get_config_var("INCLUDEPY"))')
  NUMPY_INCLUDE := $(shell python3.11 -c 'import numpy.distutils.misc_util as np_utils; print(np_utils.get_numpy_include_dirs()[0])')
  cflags = -I $(PYTHON_INCLUDE) -I $(NUMPY_INCLUDE) -Wno-cast-function-type -mmacosx-version-min=10.9
  PYTHON_LIB := $(shell python3.11 -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))')
  ldlibs = -L $(PYTHON_LIB) -l $(PYTHON_VERSION) 
else
  $(error "Unknown system type: $(uname)")
  $(shell exit 1)

endif

# =================================== Sources ===================================

py4pd.class.sources = src/py4pd.c src/py4pd_utils.c src/pd_module.c src/py4pd_pic.c

# =================================== Data ======================================
datafiles = \
$(wildcard Help-files/*.pd) \
$(wildcard scripts/*.py) \
$(wildcard py.py) \
$(wildcard py4pd-help.pd) \
$(wildcard python311._pth) \
$(PYTHON_DLL)

# =================================== Pd Lib Builder =============================

PDLIBBUILDER_DIR=./resources/pd-lib-builder/
include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder

