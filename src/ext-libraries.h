#ifndef PYLIBRARY_MODULE_H
#define PYLIBRARY_MODULE_H

#include "py4pd.h"

extern void *py_newObject(t_symbol *s, int argc, t_atom *argv);
extern void *py_freeObject(t_py *x);
extern PyObject *pdAddPyObject(PyObject *self, PyObject *args, PyObject *keywords);

#endif
