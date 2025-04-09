#include "py4pd.h"

// ─────────────────────────────────────
static PyObject *pdpy_clock_create(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    t_pdpy_clock *self;
    self = (t_pdpy_clock *)type->tp_alloc(type, 0);
    if (self == NULL) {
        PyErr_SetString(PyExc_TypeError, "ClockType not ready");
        return NULL;
    }
    return (PyObject *)self;
}

// ─────────────────────────────────────
static void pdpy_clock_destruct(t_pdpy_clock *self) {
    Py_XDECREF(self->function);
    clock_unset(self->clock);
    clock_free(self->clock);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// ─────────────────────────────────────
static PyObject *pdpy_clock_delay(t_pdpy_clock *self, PyObject *args) {
    float f;
    if (PyArg_ParseTuple(args, "f", &f)) {
        clock_delay(self->clock, f);
        Py_RETURN_TRUE;
    }
    PyErr_SetString(PyExc_TypeError, "expected float");
    return NULL;
}

// ─────────────────────────────────────
static PyObject *pdpy_clock_unset(t_pdpy_clock *self, PyObject *args) {
    if (self->clock != NULL) {
        clock_unset(self->clock);
        Py_RETURN_TRUE;
    }
    Py_RETURN_TRUE;
}

// ─────────────────────────────────────
static PyObject *pdpy_clock_set(t_pdpy_clock *self, PyObject *args) {
    float time;
    if (!PyArg_ParseTuple(args, "f", &time)) {
        PyErr_SetString(PyExc_TypeError, "expected float");
        return NULL;
    }

    if (self->clock != NULL) {
        clock_set(self->clock, time);
        Py_RETURN_TRUE;
    }
    Py_RETURN_TRUE;
}

// ─────────────────────────────────────
static PyMethodDef Clock_methods[] = {
    {"delay", (PyCFunction)pdpy_clock_delay, METH_VARARGS,
     "Sets the clock so that it will go off (and call the clock method) after time milliseconds."},
    {"unset", (PyCFunction)pdpy_clock_unset, METH_NOARGS,
     "Unsets the clock, canceling any timeout that has been set previously"},
    {"set", (PyCFunction)pdpy_clock_set, METH_NOARGS,
     "Sets the clock so that it will go off at the specified absolute systime"},
    {NULL, NULL, 0, NULL} // Sentinel
};

// ─────────────────────────────────────
static PyTypeObject ClockType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "puredata.NewObject.clock",
    .tp_basicsize = sizeof(t_pdpy_clock),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = pdpy_clock_create,
    .tp_dealloc = (destructor)pdpy_clock_destruct,
    .tp_methods = Clock_methods,
};

// ─────────────────────────────────────
PyObject *pdpy_newclock(PyObject *self, PyObject *args) {
    if (PyType_Ready(&ClockType) < 0) {
        PyErr_SetString(PyExc_TypeError, "new_clock ClockType not ready");
        return NULL;
    }
    // get pyfunction to be executed
    PyObject *func;
    if (!PyArg_ParseTuple(args, "O", &func)) {
        PyErr_SetString(PyExc_TypeError, "new_clock require a function as argument");
        return NULL;
    }

    if (!PyCallable_Check(func)) {
        PyErr_SetString(PyExc_TypeError, "new_clock function is not callable");
        return NULL;
    }

    PyObject *funcname = PyObject_GetAttrString(func, "__name__");
    if (!funcname) {
        PyErr_SetString(PyExc_TypeError, "new_clock function has no __name__ attribute");
        return NULL;
    }
    const char *funcnamestr = PyUnicode_AsUTF8(funcname);

    t_pdpy_clock *clock = (t_pdpy_clock *)PyObject_CallObject((PyObject *)&ClockType, NULL);
    if (!clock) {
        return NULL;
    }

    PyObject *pdmod = PyImport_ImportModule("puredata");
    if (pdmod == NULL) {
        PyErr_Print();
        pdpy_printerror(NULL);
        PyErr_SetString(PyExc_ImportError, "Failed to import 'puredata' module");
        return NULL;
    }
    PyObject *capsule = PyObject_GetAttrString(pdmod, "_currentobj");
    t_pdpy_pdobj *ptr;
    t_pdpy_pyclass *cls = (t_pdpy_pyclass *)self;

    if (!capsule) {
        PyErr_SetString(PyExc_AttributeError, "No '_currentobj' attribute found");
        return NULL;
    } else if (!Py_IsNone(capsule)) {
        if (!PyCapsule_CheckExact(capsule)) {
            PyErr_SetString(PyExc_TypeError, "Expected a PyCapsule object");
            Py_DECREF(capsule);
            return NULL;
        }

        ptr = PyCapsule_GetPointer(capsule, "_currentobj");
        if (!ptr) {
            PyErr_SetString(PyExc_TypeError, "Pointer in _currentobj is NULL");
            Py_DECREF(capsule);
            return NULL;
        }
    } else if (cls->pdobj != NULL) {
        ptr = cls->pdobj;
    } else {
        PyErr_SetString(PyExc_TypeError, "Impossible to get PureData objct pointer");
        return NULL;
    }

    Py_INCREF(func);
    clock->pd = pdpy_proxyclock_class;
    clock->owner = ptr;
    clock->function = func;
    clock->functionname = funcnamestr;
    clock->clock = clock_new(clock, (t_method)pdpy_clock_execute);

    ptr->clocks =
        (t_pdpy_clock **)resizebytes(ptr->clocks, ptr->clocks_size * sizeof(t_pdpy_clock *),
                                     (ptr->clocks_size + 1) * sizeof(t_pdpy_clock *));

    ptr->clocks[ptr->clocks_size] = clock;
    ptr->clocks_size++;

    return (PyObject *)clock;
}

// ─────────────────────────────────────
void pdpy_clock_execute(t_pdpy_clock *x) {
    if (x->function == NULL) {
        pd_error(x->owner, "Clock function is NULL");
        return;
    }

    if (!PyCallable_Check(x->function)) {
        pd_error(x->owner, "Clock function is not callable");
        return;
    }

    PyObject_CallNoArgs(x->function);
    return;
}
