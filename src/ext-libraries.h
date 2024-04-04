// clang-format off
#ifndef PY4PD_LIB_H
#define PY4PD_LIB_H

#include "py4pd.h"

void *Py4pdLib_NewObj(t_symbol *s, int argc, t_atom *argv);
PyObject *Py4pdLib_AddObj(PyObject *self, PyObject *args, PyObject *keywords); 




#endif


