all: py4pd

lib.name = py4pd
uname := $(shell uname -s)

# =================================== Windows ===================================
ifeq (MINGW,$(findstring MINGW,$(uname)))
	PYTHON_INCLUDE := $(shell cat pythonincludes.txt)
	PYTHON_PATH := $(shell cat pythonpath.txt)
	NUMPY_INCLUDE := $(shell cat numpyincludes.txt)
	PYTHON_DLL := $(PYTHON_PATH)/python310.dll
	cflags = -l dl -I $(PYTHON_INCLUDE) -I $(NUMPY_INCLUDE) -Wno-cast-function-type -Wno-unused-variable -DPY4PD_EDITOR=\"nvim\"
	ldlibs =  $(PYTHON_DLL) -l dl -lwinpthread -Xlinker --export-all-symbols

# =================================== Linux =====================================
else ifeq (Linux,$(findstring Linux,$(uname)))
  	PYTHON_INCLUDE := $(shell $(PYTHON_VERSION) -c 'import sysconfig;print(sysconfig.get_config_var("INCLUDEPY"))')
	NUMPY_INCLUDE := $(shell $(PYTHON_VERSION) -c 'import numpy; print(numpy.get_include())')
	cflags = -I $(PYTHON_INCLUDE) -I $(NUMPY_INCLUDE) -g -Wno-cast-function-type -Wl,-export-dynamic -DPY4PD_EDITOR=\"nvim\"
  	ldlibs = -g -l dl -l $(PYTHON_VERSION) -Xlinker -export-dynamic 

# =================================== MacOS =====================================
else ifeq (Darwin,$(findstring Darwin,$(uname)))
  	PYTHON_INCLUDE := $(shell $(PYTHON_VERSION) -c 'import sysconfig;print(sysconfig.get_config_var("INCLUDEPY"))')
	NUMPY_INCLUDE := $(shell $(PYTHON_VERSION) -c 'import numpy; print(numpy.get_include())')
	ifeq ($(extension),d_arm64)
		cflags = -I $(PYTHON_INCLUDE) -I $(NUMPY_INCLUDE) -Wno-cast-function-type -mmacosx-version-min=12 -DPY4PD_EDITOR=\"nvim\"
	else
  		cflags = -I $(PYTHON_INCLUDE) -I $(NUMPY_INCLUDE) -Wno-cast-function-type -mmacosx-version-min=10.9 -DPY4PD_EDITOR=\"nvim\"
	endif
  	PYTHON_LIB := $(shell $(PYTHON_VERSION) -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))')
  	ldlibs = -l dl -L $(PYTHON_LIB) -l $(PYTHON_VERSION) -Wno-null-pointer-subtraction
else
  $(error "Unknown system type: $(uname)")
  $(shell exit 1)
endif

# =====================================================
# ==================== py4pd ==========================
# =====================================================

# ifeq (MINGW,$(findstring MINGW,$(uname)))
#     CC := gcc
#     py4pd.exe: src/py4pd_exe.c
# 	    $(CC) -o py4pd.exe src/py4pd_exe.c -I $(PYTHON_INCLUDE) -l dl -l $(PYTHON_VERSION) -Wno-cast-function-type -Wl,-export-dynamic
# else ifeq (Linux,$(findstring Linux,$(uname)))
#     CC := gcc
#     py4pd: src/py4pd_exe.c
# 	    $(CC) -o py4pd src/py4pd_exe.c -I $(PYTHON_INCLUDE) -l dl -l $(PYTHON_VERSION) -Wno-cast-function-type -Wl,-export-dynamic
# else ifeq (Darwin,$(findstring Darwin,$(uname)))
#     PYTHON_INCLUDE := $(shell $(PYTHON_VERSION) -c 'import sysconfig;print(sysconfig.get_config_var("INCLUDEPY"))')
#     PYTHON_LIB := $(shell $(PYTHON_VERSION) -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))')
#     CC := clang
#     py4pd: src/py4pd_exe.c
# 	    $(CC) -o py4pd src/py4pd_exe.c -I $(PYTHON_INCLUDE) -l dl -L $(PYTHON_LIB) -l $(PYTHON_VERSION) -Wno-cast-function-type
# else
#     $(error "Unknown system type: $(uname)")
#     $(shell exit 1)
# endif
#

# =================================== Sources ===================================

py4pd.class.sources = src/py4pd.c src/utils.c src/module.c src/pic.c src/ext-libraries.c

# =================================== Data ======================================
datafiles = \
$(wildcard Help-files/*.pd) \
$(wildcard scripts/*.py) \
$(wildcard py.py) \
$(wildcard py4pd-help.pd) \
# $(wildcard python311._pth) \
$(PYTHON_DLL)

# =================================== Pd Lib Builder =============================

PDLIBBUILDER_DIR=./resources/pd-lib-builder/

ifeq ($(extension),d_arm64)
  override arch := arm64
endif

include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder


# py4pd_exe: src/py4pd_exe.c
# 	$(info ) 
# 	$(info ========== py4pd StandAlone =========)
# 	$(info)
#
# ifeq (MINGW,$(findstring MINGW,$(uname)))
# 	$(CC) -o py4pd.exe src/py4pd_exe.c -I $(PYTHON_INCLUDE) -l dl -l $(PYTHON_VERSION) -Wno-cast-function-type -Wl,-export-dynamic
# else ifeq (Linux,$(findstring Linux,$(uname)))
# 	$(CC) -o py4pd src/py4pd_exe.c -I $(PYTHON_INCLUDE) -l dl -l $(PYTHON_VERSION) -Wno-cast-function-type -Wl,-export-dynamic
# else ifeq (Darwin,$(findstring Darwin,$(uname)))
# 	PYTHON_INCLUDE := $(shell $(PYTHON_VERSION) -c 'import sysconfig;print(sysconfig.get_config_var("INCLUDEPY"))')
# 	PYTHON_LIB := $(shell $(PYTHON_VERSION) -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))')
# 	$(info $(PYTHON_INCLUDE))
# 	$(info $(PYTHON_LIB))
# 	$(CC) -o py4pd src/py4pd_exe.c -I $(PYTHON_INCLUDE) -l dl -L $(PYTHON_LIB) -l $(PYTHON_VERSION) -Wno-cast-function-type -Wl,-export-dynamic
# else
# 	$(error "Unknown system type: $(uname)")
# 	$(shell exit 1)
# endif
#
# # in normal mode, compile py4pd and the py4pd_exe
# all: py4pd py4pd_exe

