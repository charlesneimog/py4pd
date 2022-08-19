#include <m_pd.h>
#include <g_canvas.h>

#include <py4pd.h>

// If windows 64bits include 
#ifdef _WIN64
#include <windows.h>
#else 
#include <pthread.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

// Python include
#include <Python.h>

#include "py4pd.c"

// function C to use in Python code;
static PyObject *Pd_FloatOut(PyObject *self, PyObject *args){
    t_float f;
    if (!PyArg_ParseTuple(args, "f", &f))
        return NULL;
    outlet_float(((t_py *)self)->out_A, f);
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *Pd_SymbolOut(PyObject *self, PyObject *args){
    char *s;
    if (!PyArg_ParseTuple(args, "s", &s))
        return NULL;
    outlet_symbol(((t_py *)self)->out_A, gensym(s));
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *Pd_BangOut(PyObject *self, PyObject *args){
    outlet_bang(((t_py *)self)->out_A);
    Py_INCREF(Py_None);
    return Py_None;
}