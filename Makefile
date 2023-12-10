all: py4pd

lib.name = py4pd
uname := $(shell uname -s)

# =================================== Windows ===================================
ifeq (MINGW,$(findstring MINGW,$(uname)))
	PYTHON_INCLUDE := $(shell cat pythonincludes.txt)
	PYTHON_PATH := $(shell cat pythonpath.txt)
	NUMPY_INCLUDE := $(shell cat numpyincludes.txt)
	PYTHON_DLL := $(PYTHON_PATH)/python311.dll
	cflags = -I '$(PYTHON_INCLUDE)' -I '$(NUMPY_INCLUDE)' -Wno-cast-function-type -Wno-unused-variable 
	ldlibs =  '$(PYTHON_DLL)' -lwinpthread 

# =================================== Linux =====================================
else ifeq (Linux,$(findstring Linux,$(uname)))
	# $(shell rm -f src/*.o)
  	PYTHON_INCLUDE := $(shell $(PYTHON_VERSION) -c 'import sysconfig;print(sysconfig.get_config_var("INCLUDEPY"))')
	NUMPY_INCLUDE := $(shell $(PYTHON_VERSION) -c 'import numpy; print(numpy.get_include())')
	cflags = -I $(PYTHON_INCLUDE) -I $(NUMPY_INCLUDE) -Wno-cast-function-type
  	ldlibs = -l dl -l $(PYTHON_VERSION) 

# =================================== MacOS =====================================
else ifeq (Darwin,$(findstring Darwin,$(uname)))
  	PYTHON_INCLUDE := $(shell $(PYTHON_VERSION) -c 'import sysconfig;print(sysconfig.get_config_var("INCLUDEPY"))')
	NUMPY_INCLUDE := $(shell $(PYTHON_VERSION) -c 'import numpy; print(numpy.get_include())')
	ifeq ($(extension),d_arm64)
		cflags = -I $(PYTHON_INCLUDE) -I $(NUMPY_INCLUDE) -Wno-bad-function-cast -mmacosx-version-min=12 
	else
  		cflags = -I $(PYTHON_INCLUDE) -I $(NUMPY_INCLUDE) -Wno-bad-function-cast -mmacosx-version-min=10.9 
	endif
  	PYTHON_LIB := $(shell $(PYTHON_VERSION) -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))')
  	ldlibs = -l dl -L $(PYTHON_LIB) -l $(PYTHON_VERSION) -Wno-null-pointer-subtraction
	
else
  $(error "Unknown system type: $(uname)")
  $(shell exit 1)
endif

# =================================== Sources ===================================

py4pd.class.sources = src/py4pd.c src/utils.c src/module.c src/pic.c src/ext-libraries.c src/ext-class.c src/player.c

# =================================== Data ======================================
datafiles = \
$(wildcard Help-files/*.pd) \
$(wildcard scripts/*.py) \
$(wildcard py.py) \
$(wildcard py4pd-help.pd) \
$(PYTHON_DLL)

# =================================== Pd Lib Builder =============================

PDLIBBUILDER_DIR=./resources/pd-lib-builder/

ifeq ($(extension),d_arm64)
  override arch := arm64
endif

include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder


