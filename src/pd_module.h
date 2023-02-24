// if py4pd.h not include then include it

#ifndef PY4PD_MODULE_H
#define PY4PD_MODULE_H

#include "py4pd.h"

extern PyObject *pdout(PyObject *self, PyObject *args);
extern PyObject *pdprint(PyObject *self, PyObject *args);
extern PyObject *pderror(PyObject *self, PyObject *args);

extern PyMODINIT_FUNC PyInit_pd(void);




#endif

