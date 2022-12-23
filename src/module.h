// if py4pd.h not include then include it

#ifndef Py4PD_MODULE_H
#define Py4PD_MODULE_H

#include "py4pd.h"

PyObject *pd_output(PyObject *self, PyObject *args);
PyObject *pdprint(PyObject *self, PyObject *args);
PyObject *pderror(PyObject *self, PyObject *args);

PyMODINIT_FUNC PyInit_pd(void);

#endif

