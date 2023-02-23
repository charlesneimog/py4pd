// if py4pd.h not include then include it

#ifndef PY4PD_MODULE_H
#define PY4PD_MODULE_H

#include "py4pd.h"

PyObject *pdout(PyObject *self, PyObject *args); 
PyObject *pdprint(PyObject *self, PyObject *args);
PyObject *pderror(PyObject *self, PyObject *args);
static PyObject *pdmoduleError;



// =================================
PyMethodDef PdMethods[] = {                                                          // here we define the function spam_system
    {"out", pdout, METH_VARARGS, "Output in out0 from PureData"},                           // one function for now
    {"print", pdprint, METH_VARARGS, "Print informations in PureData Console"},             // one function for now
    {"error", pderror, METH_VARARGS, "Print error in PureData"},                            // one function for now
    {NULL, NULL, 0, NULL}
};

// =================================

struct PyModuleDef pdmodule = {
    PyModuleDef_HEAD_INIT,
    "pd", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module,
             or -1 if the module keeps state in global variables. */
    PdMethods,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit_pd(void){
    PyObject *m;
    m = PyModule_Create(&pdmodule);
    if (m == NULL)
        return NULL;
    pdmoduleError = PyErr_NewException("spam.error", NULL, NULL);
    Py_XINCREF(pdmoduleError);
    if (PyModule_AddObject(m, "error", pdmoduleError) < 0){
        Py_XDECREF(pdmoduleError);
        Py_CLEAR(pdmoduleError);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}

#endif

