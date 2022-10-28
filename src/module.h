// if py4pd.h not include then include it
#include "py4pd.h"

// declare function in module.c

PyObject *pd_output(PyObject *self, PyObject *args);
PyObject *pdmessage(PyObject *self, PyObject *args);
PyObject *pderror(PyObject *self, PyObject *args);

PyMODINIT_FUNC PyInit_pd(void);
