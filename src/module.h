// clang-format off
#ifndef PY4PD_MODULE_H
#define PY4PD_MODULE_H

#include "py4pd.h"

extern PyMethodDef PdMethods[];
PyMODINIT_FUNC PyInit_pd(void);
void Py4pdMod_FreePdcollectHash(pdcollectHash* hash_table);

extern PyObject *Py4pdLib_AddObj(PyObject *self, PyObject *args, PyObject *keywords);

#endif


