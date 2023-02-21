# library name
lib.name = py4pd

uname := $(shell uname -s)

ifeq (MINGW,$(findstring MINGW,$(uname)))
  PYTHON_INCLUDE := $(shell cat pythonincludes.txt)
  PYTHON_PATH := $(shell cat pythonpath.txt)
  PYTHON_DLL := $(PYTHON_PATH)/python311.dll
  # PYTHON_DLL = $PYTHON_PATH/python311.dll
  EXTRA_INCLUDES = -I $(PYTHON_INCLUDE) 
  cflags = -I $(PYTHON_INCLUDE) -Wno-cast-function-type -Wno-unused-variable 
  ldlibs =  $(PYTHON_DLL) -lwinpthread -lregex
  # pythondll_name = $(shell basename $(PYTHON_DLL))
  # $(shell cp $(PYTHON_DLL) $(pythondll_name))

else ifeq (Linux,$(findstring Linux,$(uname)))
  PYTHON_INCLUDE := $(shell python3 -c 'import sysconfig;print(sysconfig.get_config_var("INCLUDEPY"))')
  cflags = -I $(PYTHON_INCLUDE) -Wno-cast-function-type -Wno-unused-variable
  ldlibs = -l $(PYTHON_VERSION) 

else ifeq (Darwin,$(findstring Darwin,$(uname)))
  PYTHON_INCLUDE := $(shell python3 -c 'import sysconfig;print(sysconfig.get_config_var("INCLUDEPY"))')
  cflags = -I $(PYTHON_INCLUDE) -Wno-cast-function-type -Wno-unused-variable -mmacosx-version-min=10.9
  PYTHON_LIB := $(shell python3 -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))')
  ldlibs = -L $(PYTHON_LIB) -l $(PYTHON_VERSION) 

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

