#include <string.h>

#include <m_pd.h>

#include <g_canvas.h>
#include <m_imp.h>
#include <s_stuff.h>

#include <Python.h>

typedef struct _pdpy_pyclass t_pdpy_pyclass;
typedef struct _pdpy_clock t_pdpy_clock;
static t_class *pdpy_proxyinlet_class = NULL;
static t_class *pdpy_pyobjectout_class = NULL;
static t_class *pdpy_proxyclock_class = NULL;

#define PYOBJECT -1997
#define PY4PDSIGTOTAL(s) ((t_int)((s)->s_length * (s)->s_nchans))

// ╭─────────────────────────────────────╮
// │          Object Base Class          │
// ╰─────────────────────────────────────╯
typedef struct _pdpy_objptr {
    t_pd x_pd;
    t_symbol *id;
    PyObject *pValue;
} t_pdpy_objptr;

// ─────────────────────────────────────
typedef struct _pdpy_pdobj {
    t_object obj;
    t_sample sample;
    t_canvas *canvas;

    // PyClass
    const char *script_filename;
    PyObject *pyclass;
    char id[MAXPDSTRING];

    // dsp
    PyObject *dspfunction;
    unsigned nchs;
    unsigned vecsize;
    unsigned siginlets;
    unsigned sigoutlets;

    // clock
    t_pdpy_clock **clocks;
    int clocks_size;

    // properties
    t_symbol *current_frame;
    t_symbol *properties_receiver;
    int checkbox_count;
    int current_row;
    int current_col;

    // in and outs
    t_outlet **outs;
    t_inlet **ins;
    int outletsize;
    int inletsize;
    struct pdpy_proxyinlet *proxy_in;
    t_pdpy_objptr *outobjptr; // PyObject <type> <pointer>
} t_pdpy_pdobj;

// ─────────────────────────────────────
typedef struct _pdpy_pyclass {
    PyObject_HEAD const char *name;
    t_pdpy_pdobj *pdobj;
    PyObject *outlets;
    PyObject *inlets;
    PyObject *pyargs;
} t_pdpy_pyclass;

// ─────────────────────────────────────
typedef struct pdpy_proxyinlet {
    t_pd pd;
    t_pdpy_pdobj *owner;
    unsigned int id;
} t_pdpy_proxyinlet;

// ─────────────────────────────────────
typedef struct _pdpy_clock {
    PyObject_HEAD PyObject *function;
    t_pd pd;
    t_clock *clock;
    t_pdpy_pdobj *owner;
    const char *functionname;
    float delay_time;
} t_pdpy_clock;

// ╭─────────────────────────────────────╮
// │            Declarations             │
// ╰─────────────────────────────────────╯
static void pdpy_execute(t_pdpy_pdobj *x, char *methodname, t_symbol *s, int argc, t_atom *argv);
static void pdpy_proxyinlet_init(t_pdpy_proxyinlet *p, t_pdpy_pdobj *owner, unsigned int id);
static void pdpy_proxy_anything(t_pdpy_proxyinlet *proxy, t_symbol *s, int argc, t_atom *argv);
static void pdpy_clock_execute(t_pdpy_clock *x);
static void pdpy_printerror(t_pdpy_pdobj *x);
static void pdpy_pyobject(t_pdpy_pdobj *x, t_symbol *s, t_symbol *id);
// static PyObject *pdpy_newpyclass(t_symbol *s, t_pdpy_pdobj *x, PyObject *pdmod, PyObject
// *pyargs);

// ╭─────────────────────────────────────╮
// │                Clock                │
// ╰─────────────────────────────────────╯
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
static PyObject *pdpy_newclock(PyObject *self, PyObject *args) {
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
static void pdpy_clock_execute(t_pdpy_clock *x) {
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

// ─────────────────────────────────────
static PyObject *py4pdobj_converttopy(int argc, t_atom *argv) {
    PyObject *pValue = PyList_New(argc);
    for (int i = 0; i < argc; i++) {
        if (argv[i].a_type == A_FLOAT) {
            int isInt = atom_getintarg(i, argc, argv) == atom_getfloatarg(i, argc, argv);
            if (isInt) {
                t_int number = atom_getintarg(i, argc, argv);
                PyList_SetItem(pValue, i, PyLong_FromLong(number));
            } else {
                t_float number = atom_getfloatarg(i, argc, argv);
                PyList_SetItem(pValue, i, PyFloat_FromDouble(number));
            }
        } else if (argv[i].a_type == A_SYMBOL) {
            t_symbol *k = atom_getsymbolarg(i, argc, argv);
            PyList_SetItem(pValue, i, PyUnicode_FromString(k->s_name));
        }
    }

    return pValue;
}

// ─────────────────────────────────────
static void pdpy_proxyinlet_fwd(t_pdpy_proxyinlet *p, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    if (!argc) {
        return;
    }

    //
    char methodname[MAXPDSTRING];
    pd_snprintf(methodname, MAXPDSTRING, "in_%d_%s", p->id, atom_getsymbol(argv)->s_name);
    pdpy_execute(p->owner, methodname, atom_getsymbol(argv), argc - 1, argv + 1);
}

// ─────────────────────────────────────
void pdpy_proxyinlet_setup(void) {
    pdpy_proxyinlet_class =
        class_new(gensym("py4pd proxy inlet"), 0, 0, sizeof(t_pdpy_proxyinlet), 0, 0);
    if (pdpy_proxyinlet_class) {
        class_addanything(pdpy_proxyinlet_class, pdpy_proxy_anything);
        class_addmethod(pdpy_proxyinlet_class, (t_method)pdpy_proxyinlet_fwd, gensym("fwd"),
                        A_GIMME, 0);
    }
}

// ╭─────────────────────────────────────╮
// │        Python Object Pointer        │
// ╰─────────────────────────────────────╯
void pdpy_pyobjectoutput_setup(void) {
    pdpy_pyobjectout_class =
        class_new(gensym("_py4pd-ptr-class"), 0, 0, sizeof(t_pdpy_objptr), CLASS_PD, A_NULL);
}

// ─────────────────────────────────────
static t_pdpy_objptr *pdpy_createoutputptr(void) {
    t_pdpy_objptr *x = (t_pdpy_objptr *)getbytes(sizeof(t_pdpy_objptr));
    x->x_pd = pdpy_pyobjectout_class;

    char buf[64];
    int ret = pd_snprintf(buf, sizeof(buf), "<%p>", (void *)x);
    if (ret < 0 || ret >= sizeof(buf)) {
        buf[sizeof(buf) - 1] = '\0';
    }
    x->id = gensym(buf);
    pd_bind((t_pd *)x, x->id);
    return x;
}

// ─────────────────────────────────────
static PyObject *pdpy_getoutptr(t_symbol *s) {
    t_pdpy_objptr *x = (t_pdpy_objptr *)pd_findbyclass(s, pdpy_pyobjectout_class);
    return x ? x->pValue : NULL;
}

// ─────────────────────────────────────
t_class *pdpyobj_get_pdclass(const char *classname) {
    PyObject *pdmod = PyImport_ImportModule("puredata");
    if (!pdmod) {
        pd_error(NULL, "Failed to import puredata module in get_pdclass");
        pdpy_printerror(NULL);
        return NULL;
    }

    PyObject *obj_dict = PyObject_GetAttrString(pdmod, "_objects");
    if (!obj_dict || !PyDict_Check(obj_dict)) {
        pdpy_printerror(NULL);
        Py_XDECREF(obj_dict);
        Py_DECREF(pdmod);
        return NULL;
    }

    // Get the class dictionary
    PyObject *objclasses = PyDict_GetItemString(obj_dict, classname);
    if (!objclasses || !PyDict_Check(objclasses)) {
        pdpy_printerror(NULL);
        Py_DECREF(pdmod);
        return NULL; // Don't decref obj_dict since it's borrowed
    }

    // Get the capsule from the dictionary
    PyObject *pdclass = PyDict_GetItemString(objclasses, "pd_class");
    if (!pdclass || !PyCapsule_CheckExact(pdclass)) {
        pd_error(NULL, "Class '%s' not found or invalid in _objects", classname);
        Py_DECREF(pdmod);
        return NULL;
    }

    // Extract the t_class pointer from the capsule
    t_class *newobj = (t_class *)PyCapsule_GetPointer(pdclass, NULL);
    if (!newobj) {
        pd_error(NULL, "Invalid capsule for pdclass '%s'", classname);
        Py_DECREF(pdmod);
        return NULL;
    }

    // Cleanup references (do NOT decrement borrowed references)
    Py_DECREF(pdmod);

    return newobj;
}

// ─────────────────────────────────────
PyObject *pdpyobj_get_pyclass(t_pdpy_pdobj *x, const char *classname, PyObject *args) {
    PyObject *pdmod = PyImport_ImportModule("puredata");
    if (!pdmod) {
        pd_error(x, "Failed to import puredata module in get_pyclass");
        pdpy_printerror(NULL);
        return NULL;
    }

    PyObject *obj_dict = PyObject_GetAttrString(pdmod, "_objects");
    if (!obj_dict || !PyDict_Check(obj_dict)) {
        pdpy_printerror(NULL);
        Py_XDECREF(obj_dict);
        Py_DECREF(pdmod);
        return NULL;
    }

    // Get the class dictionary
    PyObject *objclasses = PyDict_GetItemString(obj_dict, classname);
    if (!objclasses) {
        pdpy_printerror(NULL);
        Py_DECREF(pdmod);
        return NULL; // Don't decref obj_dict since it's borrowed
    }

    PyObject *pyclass = PyDict_GetItemString(objclasses, "py_class");
    if (!pyclass) {
        pd_error(NULL, "Class '%s' not found or invalid in _objects", classname);
        Py_DECREF(pdmod);
        return NULL;
    }

    PyObject *currobj = PyCapsule_New((void *)x, "_currentobj", NULL);
    PyObject_SetAttrString(pdmod, "_currentobj", currobj);
    PyObject *newobj = PyObject_CallOneArg(pyclass, args);
    PyObject_SetAttrString(pdmod, "_currentobj", Py_None);

    // ╭─────────────────────────────────────╮
    // │   TODO: NEED MEMORY CHECK FOR ALL   │
    // │          NEW WAYS TO DO IT          │
    // ╰─────────────────────────────────────╯

    if (newobj == NULL) {
        pdpy_printerror(NULL);
        pd_error(NULL, "Failed to create python class");
    }
    PyObject *filename = PyDict_GetItemString(objclasses, "script_file");
    x->script_filename = PyUnicode_AsUTF8(filename);

    return newobj;
}

// ─────────────────────────────────────
static void pdpy_inlets(t_pdpy_pdobj *x) {
    PyObject *inlets = PyObject_GetAttrString((PyObject *)x->pyclass, "inlets");
    if (inlets == NULL) {
        return;
    }

    if (PyNumber_Check(inlets)) {
        int count = PyLong_AsLong(inlets);
        x->ins = (t_inlet **)malloc((count) * sizeof(t_inlet *));
        x->proxy_in = (t_pdpy_proxyinlet *)malloc((count) * sizeof(t_pdpy_proxyinlet));
        for (int i = 0; i < count; i++) {
            pdpy_proxyinlet_init(&x->proxy_in[i], x, i + 1);
            x->ins[i] = inlet_new(&x->obj, &x->proxy_in[i].pd, 0, 0);
        }
        return;
    } else if (PyTuple_Check(inlets)) {
        int count = PyTuple_Size(inlets);
        x->ins = (t_inlet **)malloc(count * sizeof(t_inlet *));
        x->proxy_in = (t_pdpy_proxyinlet *)malloc((count) * sizeof(t_pdpy_proxyinlet));

        // NOTE:I think that on signal objects, the first inlets are always signals
        x->siginlets = 0;
        for (int i = 0; i < count; i++) {
            PyObject *config = PyTuple_GetItem(inlets, i);
            if (config == NULL) {
                PyErr_SetString(PyExc_TypeError, "Invalid outlet type");
                continue;
            }
            const char *intype = PyUnicode_AsUTF8(config);
            t_symbol *sym = strcmp(intype, "anything") ? &s_signal : 0;
            pdpy_proxyinlet_init(&x->proxy_in[i], x, i + 1);
            x->ins[i] = inlet_new(&x->obj, &x->proxy_in[i].pd, sym, sym);
            if (strcmp(intype, "signal") == 0) {
                x->siginlets++;
            }
        }
        return;
    } else if (PyUnicode_Check(inlets)) {
        const char *outtype = PyUnicode_AsUTF8(inlets);
        x->ins = (t_inlet **)malloc(sizeof(t_inlet *));
        x->proxy_in = (t_pdpy_proxyinlet *)malloc(sizeof(t_pdpy_proxyinlet));

        t_symbol *sym = strcmp(outtype, "anything") ? &s_signal : 0;
        x->ins[0] = inlet_new(&x->obj, &x->proxy_in[0].pd, sym, sym);
        pdpy_proxyinlet_init(&x->proxy_in[0], x, 1);
        if (strcmp(outtype, "signal") == 0) {
            x->siginlets = 1;
        }
        return;
    }
    PyErr_SetString(PyExc_TypeError, "Invalid inlets configuration");
}

// ─────────────────────────────────────
static void pdpy_outlets(t_pdpy_pdobj *x) {
    PyObject *outlets = PyObject_GetAttrString(x->pyclass, "outlets");
    if (outlets == NULL) {
        return;
    }

    if (PyLong_Check(outlets)) {
        int count = PyLong_AsLong(outlets);
        x->outs = (t_outlet **)malloc(count * sizeof(t_outlet *));
        for (int i = 0; i < count; i++) {
            x->outs[i] = outlet_new(&x->obj, &s_anything);
        }
        x->outletsize = count;
        return;
    } else if (PyUnicode_Check(outlets)) {
        const char *sym = PyUnicode_AsUTF8(outlets);
        x->outs = (t_outlet **)malloc(1 * sizeof(t_outlet *));
        if (strcmp(sym, "signal") == 0) {
            x->outs[0] = outlet_new(&x->obj, &s_signal);
            x->sigoutlets++;
        } else {
            x->outs[0] = outlet_new(&x->obj, &s_anything);
        }
        x->outletsize = 1;
        return;
    } else if (PyTuple_Check(outlets)) {
        int count = PyTuple_Size(outlets);
        x->outs = (t_outlet **)malloc(count * sizeof(t_outlet *));
        x->sigoutlets = 0;
        for (int i = 0; i < count; i++) {
            PyObject *config = PyTuple_GetItem(outlets, i);
            if (config == NULL) {
                PyErr_SetString(PyExc_TypeError, "Invalid outlet type");
                continue;
            }
            const char *outtype = PyUnicode_AsUTF8(config);
            if (strcmp(outtype, "signal") == 0) {
                x->outs[i] = outlet_new(&x->obj, &s_signal);
                x->sigoutlets++;
            } else {
                x->outs[i] = outlet_new(&x->obj, &s_anything);
            }
        }
        x->outletsize = count;
        return;
    }

    PyErr_SetString(PyExc_TypeError, "Invalid outlets configuration");
    return;
}

// ─────────────────────────────────────
static void *pdpy_new(t_symbol *s, int argc, t_atom *argv) {
    t_class *pdobj = pdpyobj_get_pdclass(s->s_name);
    if (pdobj == NULL) {
        pd_error(NULL, "[%s] t_class for %s is unvalid, please report!", s->s_name, s->s_name);
        return NULL;
    }

    t_pdpy_pdobj *x = (t_pdpy_pdobj *)pd_new(pdobj);
    x->clocks_size = 0;
    PyObject *pdmod = PyImport_ImportModule("puredata");
    if (pdmod == NULL) {
        PyErr_SetString(PyExc_ImportError, "Failed to import 'puredata' module");
        return NULL;
    }

    PyObject *pyargs = PyList_New(argc);
    for (int i = 0; i < argc; i++) {
        if (argv[i].a_type == A_FLOAT) {
            PyList_SetItem(pyargs, i, PyFloat_FromDouble(argv[i].a_w.w_float));
        } else if (argv[i].a_type == A_SYMBOL) {
            PyList_SetItem(pyargs, i, PyUnicode_FromString(argv[i].a_w.w_symbol->s_name));
        } else {
            logpost(x, 2, "Unknown argument type: %s", atom_getsymbol(argv + i)->s_name);
        }
    }

    PyObject *pyclass = pdpyobj_get_pyclass(x, s->s_name, pyargs);
    if (pyclass == NULL) {
        pdpy_printerror(x);
        return NULL;
    }

    t_pdpy_pyclass *self = (t_pdpy_pyclass *)pyclass;
    x->pyclass = pyclass;

    pd_snprintf(x->id, MAXPDSTRING, "%p", x->pyclass);
    self->name = s->s_name;
    self->pdobj = x;
    self->pyargs = pyargs;
    pdpy_inlets(x);
    pdpy_outlets(x);

    x->outobjptr = pdpy_createoutputptr();
    return (void *)x;
}

// ─────────────────────────────────────
static void py4pdobj_free(t_pdpy_pdobj *x) {
    if (x->outs) {
        free(x->outs);
    }
    if (x->ins) {
        free(x->ins);
    }
    if (x->proxy_in) {
        free(x->proxy_in);
    }
    Py_DECREF(x->pyclass);
    for (int i = 0; i < x->clocks_size; i++) {
        t_pdpy_clock *c = x->clocks[i];
        clock_unset(c->clock);
        clock_free(c->clock);
    }

    pd_unbind((t_pd *)x->outobjptr, x->outobjptr->id);
    freebytes(x->outobjptr, sizeof(t_pdpy_objptr));
}

// ─────────────────────────────────────
void posterror(t_pdpy_pdobj *x) {
    PyErr_Print();
    PyErr_Clear();
}

// ─────────────────────────────────────
void pdpy_printerror(t_pdpy_pdobj *x) {
    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);

    if (pvalue == NULL) {
        pd_error(x, "[py4pd] Call failed, unknown error");
    } else {
        if (ptraceback != NULL) {
            PyObject *tracebackModule = PyImport_ImportModule("traceback");
            if (tracebackModule != NULL) {
                PyObject *formatException =
                    PyObject_GetAttrString(tracebackModule, "format_exception");
                if (formatException != NULL) {
                    PyObject *formattedException = PyObject_CallFunctionObjArgs(
                        formatException, ptype, pvalue, ptraceback, NULL);
                    if (formattedException != NULL) {
                        for (int i = 0; i < PyList_Size(formattedException); i++) {
                            pd_error(x, "\n%s",
                                     PyUnicode_AsUTF8(PyList_GetItem(formattedException, i)));
                            printf("\n%s", PyUnicode_AsUTF8(PyList_GetItem(formattedException, i)));
                        }
                        Py_DECREF(formattedException);
                    }
                    Py_DECREF(formatException);
                }
                Py_DECREF(tracebackModule);
            }
        } else {
            PyObject *pstr = PyObject_Str(pvalue);
            pd_error(x, "[py4pd] %s", PyUnicode_AsUTF8(pstr));
            printf("\n%s", PyUnicode_AsUTF8(pstr));
            Py_DECREF(pstr);
        }
    }
    Py_XDECREF(ptype);
    Py_XDECREF(pvalue);
    Py_XDECREF(ptraceback);
    PyErr_Clear();
}

// ─────────────────────────────────────
void pdpy_execute(t_pdpy_pdobj *x, char *methodname, t_symbol *s, int argc, t_atom *argv) {
    if (x->pyclass == NULL) {
        post("pyclass is NULL");
        return;
    }

    PyObject *method = PyObject_GetAttrString(x->pyclass, methodname);
    if (!method || !PyCallable_Check(method)) {
        PyErr_Clear();
        Py_XDECREF(method);
        pd_error(x, "[%s] method '%s' not found or not callable", x->obj.te_g.g_pd->c_name->s_name,
                 methodname);
        return;
    }

    PyObject *pValue = NULL;
    PyObject *pArg = NULL;
    if (strcmp(s->s_name, "bang") == 0) {
        pValue = PyObject_CallNoArgs(method);
    } else if (strcmp(s->s_name, "float") == 0) {
        t_float f = atom_getfloatarg(0, argc, argv);
        pArg = PyFloat_FromDouble(f);
        if (!pArg) {
            Py_DECREF(method);
            return;
        }
        pValue = PyObject_CallOneArg(method, pArg);
    } else if (strcmp(s->s_name, "symbol") == 0) {
        t_symbol *sym = atom_getsymbolarg(0, argc, argv); // Rename to avoid shadowing
        pArg = PyUnicode_FromString(sym->s_name);
        if (!pArg) {
            Py_DECREF(method);
            return;
        }
        pValue = PyObject_CallOneArg(method, pArg);
    } else if (strcmp(s->s_name, "list") == 0) {
        pArg = py4pdobj_converttopy(argc, argv);
        if (!pArg) {
            Py_DECREF(method);
            return;
        }
        pValue = PyObject_CallOneArg(method, pArg);
    } else {
        PyObject *pTuple = PyTuple_New(1);
        pArg = py4pdobj_converttopy(argc, argv);
        if (!pArg) {
            Py_DECREF(method);
            return;
        }
        PyTuple_SetItem(pTuple, 0, pArg);
        pValue = PyObject_CallObject(method, pTuple);
    }

    // Check for Python call errors
    if (pValue == NULL) {
        pdpy_printerror(x);
        Py_DECREF(method);
        return;
    }
    if (!Py_IsNone(pValue)) {
        pd_error(x,
                 "[%s] '%s' not return None, py4pd method functions must return None, use "
                 "'self.out' to output data to "
                 "outputs.",
                 x->obj.te_g.g_pd->c_name->s_name, methodname);
    }

    Py_XDECREF(pArg);
    Py_XDECREF(pValue);
}

// ─────────────────────────────────────
void pdpy_proxy_anything(t_pdpy_proxyinlet *proxy, t_symbol *s, int argc, t_atom *argv) {
    t_pdpy_pdobj *o = proxy->owner;
    char methodname[MAXPDSTRING];

    if (strcmp(s->s_name, "PyObject") == 0) {
        char *str = strdup(atom_getsymbol(argv)->s_name);
        char *p = str;
        while ((p = strchr(p, '.')) != NULL) {
            *p = '_';
        }
        pdpy_pyobject(proxy->owner, gensym(str), atom_getsymbol(argv + 1));
        free(str);
    } else {
        pd_snprintf(methodname, MAXPDSTRING, "in_%d_%s", proxy->id, s->s_name);
        pdpy_execute(o, methodname, s, argc, argv);
    }

    return;
}

// ─────────────────────────────────────
static void pdpy_anything(t_pdpy_pdobj *o, t_symbol *s, int argc, t_atom *argv) {
    char methodname[MAXPDSTRING];
    pd_snprintf(methodname, MAXPDSTRING, "in_1_%s", s->s_name);
    pdpy_execute(o, methodname, s, argc, argv);
}

// ─────────────────────────────────────
static void pdpy_pyobject(t_pdpy_pdobj *x, t_symbol *s, t_symbol *id) {
    if (x->pyclass == NULL) {
        post("pyclass is NULL");
        return;
    }

    char methodname[MAXPDSTRING];
    pd_snprintf(methodname, MAXPDSTRING, "in_1_pyobj_%s", s->s_name);

    PyObject *pyclass = (PyObject *)x->pyclass;
    PyObject *method = PyObject_GetAttrString(pyclass, methodname);
    if (!method || !PyCallable_Check(method)) {
        PyErr_Clear();
        pd_error(x, "[%s] method '%s' not found or not callable", x->obj.te_g.g_pd->c_name->s_name,
                 methodname);
        return;
    }

    PyObject *pArg = pdpy_getoutptr(id);
    PyObject *pValue = PyObject_CallOneArg(method, pArg);
    if (pValue == NULL) {
        pdpy_printerror(x);
        return;
    }

    if (!Py_IsNone(pValue)) {
        pd_error(x,
                 "[%s] '%s' not return None, py4pd method functions must return None, use "
                 "'self.out' to output data to "
                 "outputs.",
                 x->obj.te_g.g_pd->c_name->s_name, methodname);
    }

    Py_DECREF(method);
    Py_XDECREF(pArg);
    Py_XDECREF(pValue);

    return;
}

// ─────────────────────────────────────
static t_int *pdpy_perform(t_int *w) {
    t_pdpy_pdobj *x = (t_pdpy_pdobj *)(w[1]);
    int n = (int)w[2];

    if (x->dspfunction == NULL) {
        return w + x->siginlets + x->sigoutlets + 3;
    }

    PyObject *py_in = PyTuple_New(x->siginlets);
    t_sample *in;

    for (int i = 0; i < x->siginlets; i++) {
        in = (t_sample *)w[3 + i];
        PyObject *py_list = PyList_New(n);
        for (int j = 0; j < n; j++) {
            PyList_SetItem(py_list, j, PyFloat_FromDouble(in[j]));
        }
        PyTuple_SetItem(py_in, i, py_list);
    }

    PyObject *pValue = PyObject_CallOneArg(x->dspfunction, py_in);
    if (PyTuple_Check(pValue)) {
        int size = PyTuple_Size(pValue);
        if (size == x->sigoutlets) {
            for (int i = 0; i < x->sigoutlets; i++) {
                t_sample *out = (t_sample *)w[3 + x->siginlets + i];
                PyObject *pyout = PyTuple_GetItem(pValue, i);
                int arraysize = PyTuple_Size(pyout);
                for (int j = 0; j < arraysize; j++) {
                    PyObject *pyf = PyTuple_GetItem(pyout, j);
                    out[j] = PyFloat_AsDouble(pyf);
                }
            }
        } else if (size == n) {
            PyObject *pyf = PyTuple_GetItem(pValue, 0);
            if (PyFloat_Check(pyf)) {
                t_sample *out = (t_sample *)w[3 + x->siginlets];
                for (int j = 0; j < n; j++) {
                    PyObject *pyf = PyTuple_GetItem(pValue, j);
                    out[j] = PyFloat_AsDouble(pyf);
                }
            } else {
                PyErr_SetString(PyExc_RuntimeError, "Returned value inside Tuple is unknown!");
                pdpy_printerror(x);
            }
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Unknown Tuple size or way to process it");
            pdpy_printerror(x);
        }
    } else {
        PyErr_SetString(PyExc_RuntimeError, "Returned value is not a tuple");
        pdpy_printerror(x);
    }
    Py_DECREF(pValue);
    Py_DECREF(py_in);

    return w + x->siginlets + x->sigoutlets + 3;
}

// ─────────────────────────────────────
static void pdpy_dsp(t_pdpy_pdobj *x, t_signal **sp) {
    int sum = x->siginlets + x->sigoutlets;
    int sigvecsize = sum + 2; // +1 for x, +1 for blocksize
    t_int *sigvec = getbytes(sigvecsize * sizeof(t_int));
    for (int i = x->siginlets; i < sum; i++) {
        signal_setmultiout(&sp[i], 1);
    }

    sigvec[0] = (t_int)x;
    sigvec[1] = (t_int)sp[0]->s_n;
    for (int i = 0; i < sum; i++) {
        sigvec[i + 2] = (t_int)sp[i]->s_vec;
    }

    PyObject *pyclass = (PyObject *)x->pyclass;
    PyObject *pysr = PyLong_FromDouble(sp[0]->s_sr);
    if (PyObject_SetAttrString(pyclass, "samplerate", pysr) < 0) {
        pd_error(x, "Failed to set samplerate");
        PyErr_Clear();
    }

    PyObject *pyvec = PyLong_FromDouble(sp[0]->s_n);
    if (PyObject_SetAttrString(pyclass, "blocksize", pyvec) < 0) {
        pd_error(x, "Failed to set samplerate");
        PyErr_Clear();
    }

    PyObject *method = PyObject_GetAttrString(pyclass, "perform");
    if (!method || !PyCallable_Check(method)) {
        PyErr_Clear();
        pd_error(x, "[%s] No perform method defined or not callable",
                 x->obj.te_g.g_pd->c_name->s_name);
        for (int i = 0; i < sum; i++) {
            t_sample *out = (t_sample *)sp[i]->s_vec;
            dsp_add_zero(out, sp[i]->s_n);
        }
        freebytes(sigvec, sigvecsize * sizeof(t_int));
        return;
    }

    PyObject *dsp_method = PyObject_GetAttrString(pyclass, "dsp");
    if (dsp_method && PyCallable_Check(dsp_method)) {
        PyObject *py_in = PyTuple_New(3);
        PyTuple_SetItem(py_in, 0, PyFloat_FromDouble(sp[0]->s_sr));
        PyTuple_SetItem(py_in, 1, PyLong_FromDouble(sp[0]->s_n));
        PyTuple_SetItem(py_in, 2, PyLong_FromLong(x->siginlets));
        PyObject *r = PyObject_CallObject(dsp_method, py_in);
        Py_DECREF(py_in);
        if (r == NULL) {
            pdpy_printerror(x);
            for (int i = 0; i < sum; i++) {
                t_sample *out = (t_sample *)sp[i]->s_vec;
                dsp_add_zero(out, sp[i]->s_n);
            }
            freebytes(sigvec, sigvecsize * sizeof(t_int));
            return;
        }
        if (!PyObject_IsTrue(r)) {
            for (int i = 0; i < sum; i++) {
                t_sample *out = (t_sample *)sp[i]->s_vec;
                dsp_add_zero(out, sp[i]->s_n);
            }
            freebytes(sigvec, sigvecsize * sizeof(t_int));
            return;
        }
        Py_DECREF(r);
    } else {
        PyErr_Clear();
        pd_error(x, "[%s] Object class has no callable dsp method",
                 x->obj.te_g.g_pd->c_name->s_name);
        for (int i = 0; i < sum; i++) {
            t_sample *out = (t_sample *)sp[i]->s_vec;
            dsp_add_zero(out, sp[i]->s_n);
        }
        freebytes(sigvec, sigvecsize * sizeof(t_int));
        return;
    }

    PyErr_Clear();
    x->dspfunction = method;
    dsp_addv(pdpy_perform, sigvecsize, sigvec);
    freebytes(sigvec, sigvecsize * sizeof(t_int));
}

// ─────────────────────────────────────
static void pdpy_menu_open(t_pdpy_pdobj *o) {
    char name[MAXPDSTRING];
    pd_snprintf(name, MAXPDSTRING, "%s.pd_py", o->obj.te_g.g_pd->c_externdir->s_name);
    pdgui_vmess("::pd_menucommands::menu_openfile", "s", name);
    return;
}

// ─────────────────────────────────────
static void pdpy_proxyinlet_init(t_pdpy_proxyinlet *p, t_pdpy_pdobj *owner, unsigned int id) {
    p->pd = pdpy_proxyinlet_class;
    p->owner = owner;
    p->id = id;
}

// ╭─────────────────────────────────────╮
// │         PROPERTIES METHODS          │
// ╰─────────────────────────────────────╯
static PyObject *pyproperties_addcheckbox(PyObject *self, PyObject *args) {
    // t_pdpy_pdobj *o;
    // self:addcheckbox("Check Box 1", "updatecheckbox1", self.checkbox1)
    PyObject *name;
    PyObject *method;
    PyObject *init_value;
    if (PyArg_ParseTuple(args, "ssb", name, method, init_value)) {
    }

    // const char *s;
    // const char *text;
    // int init_value;
    //
    // char pdsend[MAXPDSTRING];
    // char checkid[MAXPDSTRING];
    // char checkvariable[MAXPDSTRING];
    // char sanitized_frame[MAXPDSTRING];
    // checkbox_count++;
    //
    // // Sanitize frame name (replace '.' with '_')
    // pd_snprintf(sanitized_frame, MAXPDSTRING, "%s", current_frame);
    // for (char *p = sanitized_frame; *p != '\0'; p++) {
    //     if (*p == '.') {
    //         *p = '_';
    //     }
    // }
    //
    // // Generate unique variable name
    // pd_snprintf(checkvariable, MAXPDSTRING, "::checkbox%d_%s_state", checkbox_count,
    //             sanitized_frame);
    //
    // // Initialize the Tcl variable to 0 (unchecked)
    // pdgui_vmess(0, "ssi", "set", checkvariable, init_value);
    //
    // // Build the pdsend command
    // pd_snprintf(pdsend, MAXPDSTRING, "eval pdsend [concat %s _properties checkbox %s $%s]",
    //             properties_receiver, method, checkvariable);
    //
    // // Create the checkbox
    // pd_snprintf(checkid, MAXPDSTRING, "%s.check%d", current_frame, checkbox_count);
    // pdgui_vmess(0, "ssssssss", "checkbutton", checkid, "-text", text, "-variable", checkvariable,
    //             "-command", pdsend);
    //
    // pdgui_vmess(0, "sssisi", "grid", checkid, "-row", current_row, "-column", current_col,
    //             "-sticky", "we");
    // pdlua_properties_updaterow(o);

    Py_RETURN_NONE;
}

// ─────────────────────────────────────
static PyObject *pyproperties_addtextinput(PyObject *self, PyObject *args) {

    //
    Py_RETURN_NONE;
}

// ─────────────────────────────────────
static PyMethodDef pyproperties_methods[] = {
    {"addcheckbox", pyproperties_addcheckbox, METH_VARARGS, "Add new checkbox on value"},
    {"addtextinput", pyproperties_addtextinput, METH_VARARGS, "Add new text input"},
    {NULL, NULL, 0, NULL}};

// ─────────────────────────────────────
static PyTypeObject pyproperties_type = {PyVarObject_HEAD_INIT(NULL, 0).tp_name = "pd_properties",
                                         .tp_basicsize = sizeof(PyObject),
                                         .tp_flags = Py_TPFLAGS_DEFAULT,
                                         .tp_doc = "MyClass objects",
                                         .tp_methods = pyproperties_methods,
                                         .tp_new = PyType_GenericNew};

// ─────────────────────────────────────
static PyObject *pdpy_properties_addcheckbox(PyObject *self, PyObject *args) {
    t_pdpy_pdobj *o;
    const char *s;

    const char *text;
    const char *method;
    int init_value;

    char pdsend[MAXPDSTRING];
    char checkid[MAXPDSTRING];
    char checkvariable[MAXPDSTRING];
    char sanitized_frame[MAXPDSTRING];
    o->checkbox_count++;

    // Sanitize frame name (replace '.' with '_')
    pd_snprintf(sanitized_frame, MAXPDSTRING, "%s", o->current_frame->s_name);
    for (char *p = sanitized_frame; *p != '\0'; p++) {
        if (*p == '.') {
            *p = '_';
        }
    }

    // Generate unique variable name
    pd_snprintf(checkvariable, MAXPDSTRING, "::checkbox%d_%s_state", o->checkbox_count,
                sanitized_frame);

    // Initialize the Tcl variable to 0 (unchecked)
    pdgui_vmess(0, "ssi", "set", checkvariable, init_value);

    // Build the pdsend command
    pd_snprintf(pdsend, MAXPDSTRING, "eval pdsend [concat %s _properties checkbox %s $%s]",
                o->properties_receiver->s_name, method, checkvariable);

    // Create the checkbox
    pd_snprintf(checkid, MAXPDSTRING, "%s.check%d", o->current_frame->s_name, o->checkbox_count);
    pdgui_vmess(0, "ssssssss", "checkbutton", checkid, "-text", text, "-variable", checkvariable,
                "-command", pdsend);

    pdgui_vmess(0, "sssisi", "grid", checkid, "-row", o->current_row, "-column", o->current_col,
                "-sticky", "we");
    // pdlua_properties_updaterow(o);

    Py_RETURN_NONE;
}

// ─────────────────────────────────────
static void pdpy_properties_receiver(t_pdpy_pdobj *o, t_symbol *s, int argc, t_atom *argv) {
    if (argc < 2) {
        return;
    }
}

// ─────────────────────────────────────
static void pdpy_properties_createdialog(t_pdpy_pdobj *o) {
    pdgui_vmess(0, "ssss", "toplevel", o->properties_receiver->s_name, "-class", "DialogWindow");
    pdgui_vmess(0, "ssss", "wm", "title", o->properties_receiver->s_name,
                "{[mydialog] Properties}");
    pdgui_vmess(0, "sss", "wm", "group", o->properties_receiver->s_name, ".");
    pdgui_vmess(0, "sssii", "wm", "resizable", o->properties_receiver->s_name, 0, 0);

    pdgui_vmess(0, "sss", "wm", "transient", o->properties_receiver->s_name, "$::focused_window");
    pdgui_vmess(0, "ssss", o->properties_receiver->s_name, "configure", "-menu",
                "$::dialog_menubar");
    pdgui_vmess(0, "sssfsf", o->properties_receiver->s_name, "configure", "-padx", 0.0f, "-pady",
                0.0f);
}

// ─────────────────────────────────────
static void pdpy_properties_setupbuttons(t_pdpy_pdobj *o) {
    char buttonsId[MAXPDSTRING];
    pd_snprintf(buttonsId, MAXPDSTRING, ".%p.buttons", (void *)o);

    char buttonCancelId[MAXPDSTRING];
    char buttonApplyId[MAXPDSTRING];
    char buttonOkId[MAXPDSTRING];
    pd_snprintf(buttonCancelId, MAXPDSTRING, ".%p.buttons.cancel", (void *)o);
    pd_snprintf(buttonApplyId, MAXPDSTRING, ".%p.buttons.apply", (void *)o);
    pd_snprintf(buttonOkId, MAXPDSTRING, ".%p.buttons.ok", (void *)o);

    char destroyCommand[MAXPDSTRING];
    pd_snprintf(destroyCommand, MAXPDSTRING, "destroy .%p", (void *)o);

    // Criando o frame dos botões
    pdgui_vmess(0, "sssf", "frame", buttonsId, "-pady", 5.0f);
    pdgui_vmess(0, "ssss", "pack", buttonsId, "-fill", "x");

    // Cancel (Close window)
    pdgui_vmess(0, "ssssss", "button", buttonCancelId, "-text", "Cancel", "-command",
                destroyCommand);
    pdgui_vmess(0, "sssssisisi", "pack", buttonCancelId, "-side", "left", "-expand", 1, "-padx", 10,
                "-ipadx", 10);

    // Apply (send all data to pd and lua obj) for this must be necessary to save all the variables
    // used in the object in a char [128][MAXPDSTRING], I don't think that this is good, or there is
    // better solution?
    // TODO: Need to dev the apply command
    pdgui_vmess(0, "ssss", "button", buttonApplyId, "-text", "Apply");
    pdgui_vmess(0, "sssssisisi", "pack", buttonApplyId, "-side", "left", "-expand", 1, "-padx", 10,
                "-ipadx", 10);

    // Ok
    pdgui_vmess(0, "ssssss", "button", buttonOkId, "-text", "OK", "-command", destroyCommand);
    pdgui_vmess(0, "sssssisisi", "pack", buttonOkId, "-side", "left", "-expand", 1, "-padx", 10,
                "-ipadx", 10);
}

// ─────────────────────────────────────
static void pdpy_properties(t_gobj *z, t_glist *owner) {
    t_pdpy_pdobj *o = (t_pdpy_pdobj *)z;
    PyObject *pyclass = (PyObject *)o->pyclass;
    PyObject *method = PyObject_GetAttrString(pyclass, "properties");
    if (!method || !PyCallable_Check(method)) {
        pd_error(o, "[%s] no properties method defined or not callable",
                 o->obj.te_g.g_pd->c_name->s_name);
        return;
    }

    char receiver[MAXPDSTRING];
    pd_snprintf(receiver, MAXPDSTRING, ".%p", o);
    o->properties_receiver = gensym(receiver);
    o->current_frame = NULL;
    pd_bind(&o->obj.ob_pd, o->properties_receiver); // new to unbind

    pdpy_properties_createdialog(o); // <-- create hidden window

    char frameId[MAXPDSTRING];
    snprintf(frameId, MAXPDSTRING, ".%p.main", (void *)o);
    pdgui_vmess(0, "sss", "wm", "deiconify", o->properties_receiver->s_name);
    pdgui_vmess(0, "sssf", "frame", frameId, "-padx", 15.0f, "-pady", 15.0f);
    pdgui_vmess(0, "sssssf", "pack", frameId, "-fill", "both", "-expand", 4.0f);
    pdgui_vmess(0, "sssfsf", "pack", frameId, "-pady", 10.f, "-padx", 10.f);

    // Create properties class
    if (PyType_Ready(&pyproperties_type) < 0) {
        PyErr_Print();
        return;
    }
    // TODO: Add receiver on class to unbind it

    PyObject *p = PyObject_CallObject((PyObject *)&pyproperties_type, NULL);
    if (!p) {
        pdpy_printerror(o);
        return;
    }

    // TODO: Create an internal module
    PyObject *r = PyObject_CallOneArg(method, p);
    if (r == NULL) {
        pdpy_printerror(o);
        pdgui_vmess(0, "ss", "destroy", o->properties_receiver->s_name);
        return;
    }
    pdpy_properties_setupbuttons(o);

    return;
}

// ─────────────────────────────────────
static t_class *pdpy_classnew(const char *n, bool dsp, bool prop) {
    t_class *c = class_new(gensym(n), (t_newmethod)pdpy_new, (t_method)py4pdobj_free,
                           sizeof(t_pdpy_pdobj), CLASS_NOINLET | CLASS_MULTICHANNEL, A_GIMME, 0);

    class_addmethod(c, (t_method)pdpy_menu_open, gensym("menu-open"), A_NULL);
    if (dsp) {
        class_addmethod(c, (t_method)pdpy_dsp, gensym("dsp"), A_CANT, 0);
    } else {
        class_setpropertiesfn(c, (t_propertiesfn)pdpy_properties);
    }

    return c;
}

// ─────────────────────────────────────
static int pdpy_create_newpyobj(PyObject *subclass, const char *name, const char *filename) {
    // TODO: Warning
    bool havedsp = false;
    PyObject *dsp = PyObject_GetAttrString(subclass, "perform");
    if (dsp) {
        havedsp = PyObject_IsTrue(dsp);
    } else {
        havedsp = false;
        PyErr_Clear(); // Ignore the error and clear the exception
    }
    Py_XDECREF(dsp);

    bool haveprop = false;
    PyObject *properties = PyObject_GetAttrString(subclass, "pd_properties");
    if (properties) {
        haveprop = PyObject_IsTrue(properties);
    } else {
        haveprop = false;
        PyErr_Clear(); // Ignore the error and clear the exception
    }
    Py_XDECREF(properties);

    // save object classes (pd and python)
    PyObject *pd_module = PyImport_ImportModule("puredata");
    if (pd_module == NULL) {
        pdpy_printerror(NULL);
        return 0;
    }

    // get all objects dict
    PyObject *pyexternals = PyObject_GetAttrString(pd_module, "_objects");
    if (pyexternals == NULL) {
        pdpy_printerror(NULL);
        return 0;
    }

    // create new dict for this object
    PyObject *externaldict = PyDict_New();

    // pyclass
    PyDict_SetItemString(externaldict, "py_class", subclass);
    PyDict_SetItemString(externaldict, "script_file", PyUnicode_FromString(filename));

    // pdclass
    t_class *pdclass = pdpy_classnew(name, havedsp, haveprop);
    PyObject *capsule = PyCapsule_New((void *)pdclass, NULL, NULL);
    if (!capsule) {
        pdpy_printerror(NULL);
        logpost(NULL, 0, "Failed to create capsule for pdclass");
        return 0;
    }

    if (PyDict_SetItemString(externaldict, "pd_class", capsule) < 0) {
        Py_DECREF(capsule); // Free capsule reference to prevent memory leak
        pdpy_printerror(NULL);
        logpost(NULL, 0, "Failed to add capsule to dictionary");
        return 0;
    }

    Py_DECREF(capsule); // Free capsule reference after adding to dictionary

    // save both classes in external dict
    if (PyDict_SetItemString(pyexternals, name, externaldict) < 0) {
        pdpy_printerror(NULL);
        logpost(NULL, 0, "Failed to create capsule for pdclass");
        return 0;
    }

    return 1;
}

// ─────────────────────────────────────
static int pdpy_validate_pd_subclasses(PyObject *module, const char *filename) {
    // Get pd.NewObject type
    PyObject *pd_module = PyImport_ImportModule("puredata");
    if (!pd_module) {
        PyErr_Print();
        return 0;
    }

    PyObject *new_object_type = PyObject_GetAttrString(pd_module, "NewObject");
    Py_DECREF(pd_module);
    if (!new_object_type) {
        PyErr_Print();
        return 0;
    }

    // Get module dictionary
    PyObject *module_dict = PyModule_GetDict(module);
    PyObject *items = PyDict_Items(module_dict);
    Py_ssize_t num_items = PyList_Size(items);

    int objects = 0;
    for (Py_ssize_t i = 0; i < num_items; i++) {
        PyObject *item = PyList_GetItem(items, i);
        PyObject *obj = PyTuple_GetItem(item, 1);
        if (PyType_Check(obj)) {
            if (PyObject_IsSubclass(obj, new_object_type) && (obj != new_object_type)) {
                PyObject *objname = PyObject_GetAttrString(obj, "name");
                if (!objname) {
                    pd_error(NULL,
                             "[py4pd] not possible to read %s inside %s, 'name' class attribute is "
                             "missing",
                             "test", "test");
                    continue;
                }
                int ok = pdpy_create_newpyobj(obj, PyUnicode_AsUTF8(objname), filename);
                if (!ok) {
                    pdpy_printerror(NULL);
                } else {
                    objects++;
                }
            }
        }
    }

    logpost(NULL, 3, "[py4pd] %d objects found inside %s", objects, filename);
    Py_DECREF(items);
    Py_DECREF(new_object_type);
    return 1;
}

// ─────────────────────────────────────
int pd4pd_loader_wrappath(int fd, const char *name, const char *dirbuf) {
    char filename[1024];
    pd_snprintf(filename, sizeof(filename), "%s/%s.pd_py", dirbuf, name);

    char modname[256];
    pd_snprintf(modname, sizeof(modname), "%s", name);

    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        PyErr_SetString(PyExc_IOError, "Failed to read script file");
        return 0;
    }
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *source = (char *)malloc(size + 1);
    if (!source) {
        fclose(fp);
        pdpy_printerror(NULL);
        return 0;
    }
    if (fread(source, 1, size, fp) != (size_t)size) {
        fclose(fp);
        free(source);
        pdpy_printerror(NULL);
        return 0;
    }
    source[size] = '\0';
    fclose(fp);

    // Create a new module with the qualified name
    PyObject *module = PyModule_New(modname);
    if (!module) {
        free(source);
        pdpy_printerror(NULL);
        return 0;
    }

    // Compile the source code (using the file name for error messages)
    PyObject *code_obj = Py_CompileString(source, filename, Py_file_input);
    free(source);
    if (!code_obj) {
        pdpy_printerror(NULL);
        Py_DECREF(module);
        return 0;
    }

    // Get the module’s dictionary in which the code will be executed
    PyObject *mod_dict = PyModule_GetDict(module);
    if (!mod_dict) {
        pdpy_printerror(NULL);
        Py_DECREF(code_obj);
        Py_DECREF(module);
        return 0;
    }

    if (PyDict_GetItemString(mod_dict, "__builtins__") == NULL) {
        PyObject *builtins = PyEval_GetBuiltins();
        if (PyDict_SetItemString(mod_dict, "__builtins__", builtins) < 0) {
            pdpy_printerror(NULL);
            Py_DECREF(module);
            return 0;
        }
    }

    // Evaluate the compiled code in the module’s namespace
    PyObject *result = PyEval_EvalCode(code_obj, mod_dict, mod_dict);
    Py_DECREF(code_obj);
    if (!result) {
        pdpy_printerror(NULL);
        Py_DECREF(module);
        return 0;
    }
    Py_DECREF(result);
    if (!pdpy_validate_pd_subclasses(module, filename)) {
        pdpy_printerror(NULL);
        Py_DECREF(module);
        return 0;
    }

    return 1;
}

// ─────────────────────────────────────
static int pdpyobj_init(t_pdpy_pyclass *self, PyObject *args) {
    char *objname;

    if (!PyArg_ParseTuple(args, "s", &objname)) {
        return -1;
    }

    PyObject *pdmod = PyImport_ImportModule("puredata");
    if (!pdmod) {
        PyErr_SetString(PyExc_ImportError, "Could not import module 'puredata'");
        return -1;
    }

    // Get or create the '_objects' dictionary
    PyObject *obj_dict = PyObject_GetAttrString(pdmod, "_objects");
    if (!obj_dict || !PyDict_Check(obj_dict)) {
        Py_XDECREF(obj_dict);
        obj_dict = PyDict_New();
        if (obj_dict == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Failed to create _objects dictionary");
            Py_DECREF(pdmod);
            return -1;
        }

        // Set _objects in puredata module
        if (PyObject_SetAttrString(pdmod, "_objects", obj_dict) < 0) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to set _objects in puredata module");
            Py_DECREF(obj_dict);
            Py_DECREF(pdmod);
            return -1;
        }
    }

    // Create a capsule to hold the pdclass pointer
    Py_DECREF(obj_dict);
    Py_DECREF(pdmod);

    return 0;
}

// ─────────────────────────────────────
static PyObject *pdpy_error(t_pdpy_pyclass *self, PyObject *args) {
    char msg[MAXPDSTRING] = "";
    size_t msg_len = 0;

    Py_ssize_t num_args = PyTuple_Size(args);
    if (num_args > 0) {
        for (Py_ssize_t i = 0; i < num_args; i++) {
            PyObject *arg = PyTuple_GetItem(args, i);
            PyObject *str_obj = PyObject_Str(arg); // Convert to string
            if (!str_obj) {
                continue;
            }

            const char *str = PyUnicode_AsUTF8(str_obj);
            if (str) {
                size_t str_len = strlen(str);
                if (msg_len + str_len + 4 >= MAXPDSTRING) {
                    Py_DECREF(str_obj);
                    pd_snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), "...");
                    break;
                }
                pd_snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), "%s", str);
                if (i < num_args - 1 && msg_len + 1 < MAXPDSTRING) {
                    pd_snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), " ");
                    msg_len++;
                }
            }

            Py_DECREF(str_obj);
        }
    }
    pd_error(self->pdobj, "[%s]: %s", self->name, msg);
    Py_RETURN_TRUE;
}

// ─────────────────────────────────────
static PyObject *pdpy_logpost(t_pdpy_pyclass *self, PyObject *args) {
    char msg[MAXPDSTRING] = "";
    size_t msg_len = 0;

    PyObject *pyloglevel = PyTuple_GetItem(args, 0);
    int loglevel = PyLong_AsLong(pyloglevel);

    Py_ssize_t num_args = PyTuple_Size(args);
    if (num_args > 0) {
        for (Py_ssize_t i = 1; i < num_args; i++) {
            PyObject *arg = PyTuple_GetItem(args, i);
            PyObject *str_obj = PyObject_Str(arg); // Convert to string
            if (!str_obj) {
                continue;
            }

            const char *str = PyUnicode_AsUTF8(str_obj);
            Py_DECREF(str_obj);
            if (str) {
                size_t str_len = strlen(str);
                if (msg_len + str_len + 4 >= MAXPDSTRING) {
                    pd_snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), "...");
                    break;
                }
                pd_snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), "%s", str);
                if (i < num_args - 1 && msg_len + 1 < MAXPDSTRING) {
                    pd_snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), " ");
                    msg_len++;
                }
            }
        }
    }
    logpost(self->pdobj, loglevel, "[%s]: %s", self->name, msg);
    Py_RETURN_TRUE;
}

// ─────────────────────────────────────
static PyObject *pdpy_out(t_pdpy_pyclass *self, PyObject *args, PyObject *keywords) {
    int outlet;
    int type;
    PyObject *pValue;

    if (!PyArg_ParseTuple(args, "iiO", &outlet, &type, &pValue)) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.out: wrong arguments");
        return NULL;
    }

    t_atomtype pdtype = type;
    t_pdpy_pdobj *x = self->pdobj;
    if (outlet > self->pdobj->outletsize - 1) {
        PyErr_SetString(PyExc_IndexError, "Index out of range for outlet");
        return NULL;
    }

    t_outlet *out = x->outs[outlet];
    if (pdtype == PYOBJECT) {
        x->outobjptr->pValue = pValue;
        t_atom args[2];
        SETSYMBOL(&args[0], gensym(Py_TYPE(pValue)->tp_name));
        SETSYMBOL(&args[1], x->outobjptr->id);
        outlet_anything(out, gensym("PyObject"), 2, args);
        Py_RETURN_TRUE;
    } else if (pdtype == A_FLOAT) {
        t_float v = PyFloat_AsDouble(pValue);
        outlet_float(out, v);
    } else if (pdtype == A_SYMBOL) {
        const char *s = PyUnicode_AsUTF8(pValue);
        outlet_symbol(out, gensym(s));
    } else if (pdtype == A_GIMME) {
        if (PyList_Check(pValue)) {
            int size = PyList_Size(pValue);
            PyObject *pValue_i;
            t_atom list_array[size];
            for (int i = 0; i < size; ++i) {
                pValue_i = PyList_GetItem(pValue, i); // borrowed reference
                if (PyLong_Check(pValue_i)) {
                    float result = (float)PyLong_AsLong(pValue_i);
                    SETFLOAT(&list_array[i], result);
                } else if (PyFloat_Check(pValue_i)) {
                    float result = PyFloat_AsDouble(pValue_i);
                    SETFLOAT(&list_array[i], result);
                } else if (PyUnicode_Check(pValue_i)) { // If the function return a
                    const char *result = PyUnicode_AsUTF8(pValue_i);
                    SETSYMBOL(&list_array[i], gensym(result));
                } else if (Py_IsNone(pValue_i)) {
                    PyErr_SetString(PyExc_TypeError, "NoneType not allowed");
                    return NULL;
                } else {
                    char msgerror[MAXPDSTRING];
                    pd_snprintf(msgerror, MAXPDSTRING, "Type not allowed %s",
                                Py_TYPE(pValue_i)->tp_name);
                    PyErr_SetString(PyExc_TypeError, msgerror);
                    return NULL;
                }
            }
            outlet_list(out, &s_list, size, list_array);
        } else {
            PyErr_SetString(PyExc_TypeError, "Output with pd.LIST require a list output");
        }
    }
    Py_RETURN_TRUE;
}

// ─────────────────────────────────────
static PyObject *pdpy_tabread(t_pdpy_pyclass *self, PyObject *args) {
    char *tabname;
    t_garray *pdarray;
    int vecsize;
    t_word *vec;

    if (!PyArg_ParseTuple(args, "s", &tabname)) {
        PyErr_SetString(PyExc_TypeError, "Wrong arguments, expected string tabname");
        return NULL;
    }

    t_symbol *pd_symbol = gensym(tabname);
    if (!(pdarray = (t_garray *)pd_findbyclass(pd_symbol, garray_class))) {
        PyErr_SetString(PyExc_TypeError, "self.tabread: array not found");
        return NULL;
    } else {
        int i;
        garray_getfloatwords(pdarray, &vecsize, &vec);
        PyObject *pAudio = PyTuple_New(vecsize);
        for (i = 0; i < vecsize; i++) {
            // PyTuple_SetItem steal the ref
            PyTuple_SetItem(pAudio, i, PyFloat_FromDouble(vec[i].w_float));
        }
        return pAudio;
    }
}

// ─────────────────────────────────────
static PyObject *pdpy_tabwrite(t_pdpy_pyclass *self, PyObject *args, PyObject *kwargs) {
    char *tabname;
    t_garray *pdarray;
    PyObject *array;
    t_word *vec;
    int vecsize;
    bool resize = false;

    if (!PyArg_ParseTuple(args, "sO", &tabname, &array)) {
        PyErr_SetString(PyExc_TypeError, "Wrong arguments, expected string tabname");
        return NULL;
    }

    if (kwargs != NULL && PyDict_Check(kwargs)) {
        PyObject *pString = PyUnicode_FromString(tabname);
        if (PyDict_Contains(kwargs, pString)) {
            resize = PyDict_GetItem(kwargs, pString);
        }
        Py_DECREF(pString);
    }

    t_symbol *pd_symbol = gensym(tabname);
    if (!(pdarray = (t_garray *)pd_findbyclass(pd_symbol, garray_class))) {
        PyErr_SetString(PyExc_TypeError, "self.tabwrite: array not found");
        return NULL;
    } else if (!garray_getfloatwords(pdarray, &vecsize, &vec)) {
        PyErr_SetString(PyExc_TypeError, "Bad template for tabwrite");
        return NULL;
    }

    if (PyList_Check(array)) {
        vecsize = PyList_Size(array);
        if (resize == 1) {
            garray_resize_long(pdarray, PyList_Size(array));
            garray_getfloatwords(pdarray, &vecsize, &vec);
        }
        for (int i = 0; i < vecsize; i++) {
            PyObject *PyFloat = PyList_GetItem(array, i);
            float result_float = PyFloat_AsDouble(PyFloat);
            vec[i].w_float = result_float;
        }
        garray_redraw(pdarray);
        Py_RETURN_TRUE;
    } else if (PyTuple_Check(array)) {
        vecsize = PyTuple_Size(array);
        if (resize == 1) {
            garray_resize_long(pdarray, PyTuple_Size(array));
            garray_getfloatwords(pdarray, &vecsize, &vec);
        }
        for (int i = 0; i < vecsize; i++) {
            PyObject *PyFloat = PyTuple_GetItem(array, i);
            float result_float = PyFloat_AsDouble(PyFloat);
            vec[i].w_float = result_float;
        }
        garray_redraw(pdarray);
        Py_RETURN_TRUE;
    } else {
        PyErr_SetString(PyExc_TypeError, "Input must be either a list or a tuple");
        return NULL;
    }
}

// ─────────────────────────────────────
static PyObject *pdpy_reload(t_pdpy_pyclass *self, PyObject *args) {

    const char *filename = self->pdobj->script_filename;
    const char *name = self->name;

    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        PyErr_SetString(PyExc_IOError, "Failed to open file");
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *source = (char *)malloc(size + 1);
    if (source == NULL) {
        fclose(fp);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for source\n");
        return 0;
    }
    if (fread(source, 1, size, fp) != (size_t)size) {
        fclose(fp);
        free(source);
        PyErr_SetString(PyExc_MemoryError, "Failed to read script file");
        return 0;
    }
    source[size] = '\0';
    fclose(fp);

    // Create a new module with the qualified name
    PyObject *module = PyModule_New(name);
    if (!module) {
        free(source);
        PyErr_Print();
        return 0;
    }

    // Compile the source code (using the file name for error messages)
    PyObject *code_obj = Py_CompileString(source, filename, Py_file_input);
    free(source);
    if (!code_obj) {
        PyErr_Print();
        Py_DECREF(module);
        return 0;
    }

    // Get the module’s dictionary in which the code will be executed
    PyObject *mod_dict = PyModule_GetDict(module);
    if (!mod_dict) {
        PyErr_Print();
        Py_DECREF(code_obj);
        Py_DECREF(module);
        return 0;
    }

    if (PyDict_GetItemString(mod_dict, "__builtins__") == NULL) {
        PyObject *builtins = PyEval_GetBuiltins();
        if (PyDict_SetItemString(mod_dict, "__builtins__", builtins) < 0) {
            PyErr_Print();
            Py_DECREF(module);
            return 0;
        }
    }

    // Evaluate the compiled code in the module’s namespace
    PyObject *result = PyEval_EvalCode(code_obj, mod_dict, mod_dict);
    Py_DECREF(code_obj);
    if (!result) {
        PyErr_Print();
        Py_DECREF(module);
        return 0;
    }

    Py_DECREF(result);
    if (!pdpy_validate_pd_subclasses(module, filename)) {
        pdpy_printerror(NULL);
        Py_DECREF(module);
        return NULL;
    }

    PyObject *newclass = pdpyobj_get_pyclass(self->pdobj, name, self->pyargs);
    if (newclass == NULL) {
        pdpy_printerror(NULL);
        Py_DECREF(module);
        return NULL;
    }

    t_pdpy_pyclass *newself = (t_pdpy_pyclass *)newclass;
    self->pdobj->pyclass = newclass;
    newself->name = self->name;
    newself->pdobj = self->pdobj;
    newself->pyargs = self->pyargs;
    PyObject *pyclass = (PyObject *)newself;

    // stops old clocks
    for (int i = 0; i < self->pdobj->clocks_size; i++) {
        t_pdpy_clock *clock = self->pdobj->clocks[i];
        PyObject *func = PyObject_GetAttrString(pyclass, clock->functionname);
        if (func != NULL && PyCallable_Check(func)) {
            Py_DECREF(clock->function);
            clock->function = func;
            Py_INCREF(func);
        } else {
            pd_error(self->pdobj, "Function %s not found or invalid, disabling clock",
                     clock->functionname);
            clock_unset(clock->clock);
        }
    }

    // dsp update
    if (self->pdobj->dspfunction != NULL) {
        PyObject *dspmethod = PyObject_GetAttrString(pyclass, "dsp_perform");
        if (!dspmethod || !PyCallable_Check(dspmethod)) {
            PyErr_Clear();
            self->pdobj->dspfunction = NULL;
        }
    }
    canvas_update_dsp();

    Py_DECREF(module);
    Py_DECREF((PyObject *)self);

    Py_RETURN_TRUE;
}

// ─────────────────────────────────────
static PyMethodDef pdpy_methods[] = {
    {"logpost", (PyCFunction)pdpy_logpost, METH_VARARGS, "Post things on PureData console"},
    {"error", (PyCFunction)pdpy_error, METH_VARARGS,
     "Print error, same as logpost with error level"},

    {"out", (PyCFunction)pdpy_out, METH_VARARGS | METH_KEYWORDS, "Post things on PureData console"},

    {"new_clock", (PyCFunction)pdpy_newclock, METH_VARARGS, "Return a clock object"},

    // arrays
    {"tabwrite", (PyCFunction)pdpy_tabwrite, METH_VARARGS | METH_KEYWORDS,
     "This read the content of a Pd Array"},
    {"tabread", (PyCFunction)pdpy_tabread, METH_VARARGS | METH_KEYWORDS,
     "This read the content of a Pd Array"},

    // reload
    {"reload", (PyCFunction)pdpy_reload, METH_NOARGS, "Reload the current script and object"},

    {NULL, NULL, 0, NULL} // Sentinel
};

// ─────────────────────────────────────
PyTypeObject pdpy_type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "NewObject",
    .tp_doc = "It creates new PureData objects",
    .tp_basicsize = sizeof(t_pdpy_pyclass),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // Allow inheritance
    .tp_methods = pdpy_methods,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)pdpyobj_init,
};

// ─────────────────────────────────────
static PyObject *pdpy_post(PyObject *self, PyObject *args) {
    (void)self;
    char *string;

    if (PyTuple_Size(args) > 0) {
        startpost("[Python]: ");
        for (int i = 0; i < PyTuple_Size(args); i++) {
            PyObject *arg = PyTuple_GetItem(args, i);
            PyObject *str = PyObject_Str(arg);
            startpost(PyUnicode_AsUTF8(str));
            startpost(" ");
            Py_DECREF(str);
        }
        startpost("\n");
    }
    Py_RETURN_TRUE;
}

// ─────────────────────────────────────
static PyObject *pdpy_hasgui(PyObject *self, PyObject *args) {
    //
    return PyLong_FromLong(sys_havegui());
}

// ╭─────────────────────────────────────╮
// │            MODULE LOADER            │
// ╰─────────────────────────────────────╯
static int ObjectLoader_init(PyObject *self, PyObject *args, PyObject *kwargs) {
    const char *fullname = NULL;
    const char *path = NULL;
    static char *kwlist[] = {"fullname", "path", NULL}; // Keyword argument names

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "ss", kwlist, &fullname, &path)) {
        return -1;
    }

    PyObject *py_fullname = PyUnicode_FromString(fullname);
    if (!py_fullname)
        return -1;
    if (PyObject_SetAttrString(self, "fullname", py_fullname) < 0) {
        Py_DECREF(py_fullname);
        return -1;
    }
    Py_DECREF(py_fullname);

    PyObject *py_path = PyUnicode_FromString(path);
    if (!py_path)
        return -1;
    if (PyObject_SetAttrString(self, "path", py_path) < 0) {
        Py_DECREF(py_path);
        return -1;
    }
    Py_DECREF(py_path);

    // Create 'spec' using importlib.util.spec_from_loader
    PyObject *util_module = PyImport_ImportModule("importlib.util");
    if (!util_module)
        return -1;

    PyObject *spec_func = PyObject_GetAttrString(util_module, "spec_from_loader");
    Py_DECREF(util_module);
    if (!spec_func)
        return -1;

    PyObject *spec_args = Py_BuildValue("(sO)", fullname, self); // (fullname, loader)
    if (!spec_args) {
        Py_DECREF(spec_func);
        return -1;
    }

    PyObject *spec = PyObject_CallObject(spec_func, spec_args);
    Py_DECREF(spec_func);
    Py_DECREF(spec_args);
    if (!spec)
        return -1;

    if (PyObject_SetAttrString(self, "spec", spec) < 0) {
        Py_DECREF(spec);
        return -1;
    }
    Py_DECREF(spec);

    return 0; // Success
}

// ─────────────────────────────────────
static PyObject *ObjectLoader_get_filename(PyObject *self, PyObject *args) {
    PyObject *fullname;
    if (!PyArg_ParseTuple(args, "O", &fullname))
        return NULL;
    return PyObject_GetAttrString(self, "path"); // Borrowed reference
}

// ─────────────────────────────────────
static PyObject *ObjectLoader_get_data(PyObject *self, PyObject *args) {
    const char *path;
    if (!PyArg_ParseTuple(args, "s", &path))
        return NULL;

    FILE *file = fopen(path, "rb");
    if (!file) {
        PyErr_SetFromErrnoWithFilename(PyExc_IOError, path);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);

    char *buffer = (char *)malloc(size);
    if (!buffer) {
        fclose(file);
        PyErr_NoMemory();
        return NULL;
    }

    size_t read = fread(buffer, 1, size, file);
    fclose(file);

    if (read != (size_t)size) {
        free(buffer);
        PyErr_SetString(PyExc_IOError, "Failed to read the entire file");
        return NULL;
    }

    PyObject *data = PyBytes_FromStringAndSize(buffer, size);
    free(buffer);
    return data;
}

// ─────────────────────────────────────
static PyObject *ObjectLoader_is_package(PyObject *self, PyObject *args) {
    PyObject *fullname;
    if (!PyArg_ParseTuple(args, "O", &fullname))
        return NULL;
    Py_RETURN_FALSE;
}

// ─────────────────────────────────────
static PyMethodDef ObjectLoader_methods[] = {
    {"get_filename", ObjectLoader_get_filename, METH_VARARGS, "Return the path to the source file"},
    {"get_data", ObjectLoader_get_data, METH_VARARGS, "Read the file's contents as bytes"},
    {"is_package", ObjectLoader_is_package, METH_VARARGS, "Check if the module is a package"},
    {NULL, NULL, 0, NULL} // Sentinel
};

// ─────────────────────────────────────
static PyType_Slot ObjectLoaderSlots[] = {
    {Py_tp_init, (void *)ObjectLoader_init}, {Py_tp_methods, ObjectLoader_methods}, {0, NULL}};

// ─────────────────────────────────────
static PyType_Spec ObjectLoaderSpec = {
    .name = "puredata._ObjectLoader",
    .basicsize = sizeof(PyObject),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE,
    .slots = ObjectLoaderSlots,
};

// ╭─────────────────────────────────────╮
// │             MODULE INIT             │
// ╰─────────────────────────────────────╯
PyMethodDef pdpy_modulemethods[] = {
    {"post", pdpy_post, METH_VARARGS, "Print informations in PureData Console"},
    {"hasgui", pdpy_hasgui, METH_NOARGS, "Return False is pd is running in console"},
    {NULL, NULL, 0, NULL} //
};

// ─────────────────────────────────────
static PyObject *pd4pdmodule_init(PyObject *self) {
    if (PyType_Ready(&pdpy_type) < 0) {
        return NULL;
    }

    // PyModule_AddObject steal ref
    // outlets output type
    PyModule_AddObject(self, "FLOAT", PyLong_FromLong(A_FLOAT));
    PyModule_AddObject(self, "SYMBOL", PyLong_FromLong(A_SYMBOL));
    PyModule_AddObject(self, "LIST", PyLong_FromLong(A_GIMME));
    PyModule_AddObject(self, "PYOBJECT", PyLong_FromLong(PYOBJECT));

    // inlets type
    PyModule_AddObject(self, "SIGNAL", PyUnicode_FromString(s_signal.s_name));
    PyModule_AddObject(self, "DATA", PyUnicode_FromString(s_anything.s_name));

    // NewObject class
    Py_INCREF(&pdpy_type);
    int r = PyModule_AddObject(self, "NewObject", (PyObject *)&pdpy_type);
    if (r != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to add NewObject to module");
        return NULL;
    }

    // Loader

    PyObject *abc_module = PyImport_ImportModule("importlib.abc");
    if (!abc_module)
        return NULL;
    PyObject *source_loader = PyObject_GetAttrString(abc_module, "SourceLoader");
    Py_DECREF(abc_module);
    if (!source_loader)
        return NULL;

    // Create bases tuple (inherit from SourceLoader)
    PyObject *bases = PyTuple_Pack(1, source_loader);
    Py_DECREF(source_loader);
    if (!bases)
        return NULL;

    // Create _ObjectLoader type dynamically
    PyObject *object_loader_type = PyType_FromSpecWithBases(&ObjectLoaderSpec, bases);
    Py_DECREF(bases);
    if (!object_loader_type)
        return NULL;

    // Add _ObjectLoader to the module
    if (PyModule_AddObject(self, "_ObjectLoader", object_loader_type) < 0) {
        Py_DECREF(object_loader_type);
        return NULL;
    }

    // objects
    PyObject *new_dict = PyDict_New();
    r = PyModule_AddObject(self, "_objects", new_dict);
    if (r != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to add NewObject to module");
        return NULL;
    }

    return 0;
}

// ─────────────────────────────────────
static PyModuleDef_Slot pdpy_moduleslots[] = { //
    {Py_mod_exec, pd4pdmodule_init},
    {0, NULL}};

// ─────────────────────────────────────
static struct PyModuleDef pdpy_module_def = {
    PyModuleDef_HEAD_INIT,
    .m_name = "puredata",
    .m_doc = "pd module provide function to interact with PureData, see the "
             "docs in www.charlesneimog.github.io/py4pd",
    .m_size = 0,
    .m_methods = pdpy_modulemethods,
    .m_slots = pdpy_moduleslots,
};

// ─────────────────────────────────────
PyMODINIT_FUNC pdpy_initpuredatamodule() {
    PyObject *m;
    m = PyModuleDef_Init(&pdpy_module_def);
    if (m == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create module");
        return NULL;
    }
    return m;
}
