// if py4pd.h not include then include it

#ifndef PY4PD_MODULE_H
#define PY4PD_MODULE_H

#include "py4pd.h"

extern PyMethodDef PdMethods[];

PyMODINIT_FUNC PyInit_pd(void);


#endif

