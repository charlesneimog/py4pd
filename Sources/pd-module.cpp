#include <m_pd.h>

#include "pd-module.hpp"
#include "py4pd.hpp"

#define PY_ARRAY_UNIQUE_SYMBOL PY4PD_NUMPYARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_25_API_VERSION
#include <numpy/arrayobject.h>

// ─────────────────────────────────────
static PyObject *Print(PyObject *Self, PyObject *Args, PyObject *Keys) {
    const char *Message;
    if (PyTuple_Size(Args) > 0) {
        if (true) { //
            startpost("[py4pd]: ");
        }
        for (int i = 0; i < PyTuple_Size(Args); i++) {
            PyObject *arg = PyTuple_GetItem(Args, i);
            PyObject *str = PyObject_Str(arg);
            startpost(PyUnicode_AsUTF8(str));
            startpost(" ");
        }
        startpost("\n");
    }
    Py_RETURN_TRUE;
}

// ─────────────────────────────────────
PYBIND11_EMBEDDED_MODULE(pd, m) {
    printf("pd-module.cpp\n");
    m.def("print", &Print);
}
