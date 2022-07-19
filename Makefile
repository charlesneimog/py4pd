# Makefile to build class '_template_' for Pure Data.
# Needs Makefile.pdlibbuilder as helper makefile for platform-dependent build
# settings and rules.

# library name
lib.name = py4pd

# How to add include paths to the compiler:
# - for Windows: add '-Ipath' to the compiler command line
# - for MacOSX: add '-Ipath' to the compiler command line

cflags = I"C:/Users/Neimog/AppData/Local/Programs/Python/Python310/include/" L"C:/Users/Neimog/AppData/Local/Programs/Python/Python310/libs/" -lpython310 

ldlibs = L"C:/Users/Neimog/AppData/Local/Programs/Python/Python310/libs/" -lpython310

# input source file (class name == source file basename)
class.sources = src/py4pd.c

# all extra files to be included in binary distribution of the library
datafiles = py4pd-help.pd 

# include Makefile.pdlibbuilder
# (for real-world projects see the "Project Management" section
# in tips-tricks.md)
PDLIBBUILDER_DIR=./pd-lib-builder/
include $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder

