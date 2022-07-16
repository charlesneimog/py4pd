lib.name = py4pd

cflags = -I"C:/Users/Neimog/AppData/Local/Programs/Python/Python310/include/" -L"C:/Users/Neimog/AppData/Local/Programs/Python/Python310/libs/" -lpython310

class.sources = py4pd.c

PDLIBBUILDER_DIR=pd-lib-builder/
include $(firstword $(wildcard $(PDLIBBUILDER_DIR)/Makefile.pdlibbuilder Makefile.pdlibbuilder))

# 