#include <string.h>

#include <m_pd.h>

#include <g_canvas.h>
#include <m_imp.h>
#include <s_stuff.h>

#include <Python.h>

#define PY4PD_MAJOR_VERSION 1
#define PY4PD_MINOR_VERSION 0
#define PY4PD_MICRO_VERSION 0

// ╭─────────────────────────────────────╮
// │             Definitions             │
// ╰─────────────────────────────────────╯
//
extern PyMODINIT_FUNC pdpy_initpuredatamodule();
extern void pdpy_proxyinlet_setup(void);
extern void pdpy_pyobjectoutput_setup(void);
extern int pd4pd_loader_wrappath(int fd, const char *name, const char *dirbuf);

t_class *py4pd_class;
int objCount = 0;

// ─────────────────────────────────────
typedef struct _py {
    t_object obj;
} t_py;

typedef struct _pdpy_pyclass t_pdpy_pyclass;
typedef struct _pdpy_clock t_pdpy_clock;
typedef struct _pdpy_receiver t_pdpy_receiver;
static t_class *pdpy_proxyinlet_class = NULL;
static t_class *pdpy_pyobjectout_class = NULL;
static t_class *pdpy_proxyclock_class = NULL;

#define PYOBJECT -1997
#define PY4PDSIGTOTAL(s) ((t_int)((s)->s_length * (s)->s_nchans))

// ─────────────────────────────────────
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

// ─────────────────────────────────────
typedef struct _pdpy_receiver {
    PyObject_HEAD PyObject *function;
    t_pd pd;
    t_pdpy_pdobj *owner;
    const char *functionname;
    t_symbol *r;
} t_pdpy_receiver;

// ╭─────────────────────────────────────╮
// │            Declarations             │
// ╰─────────────────────────────────────╯
static void pdpy_execute(t_pdpy_pdobj *x, char *methodname, t_symbol *s, int argc, t_atom *argv);
static void pdpy_proxyinlet_init(t_pdpy_proxyinlet *p, t_pdpy_pdobj *owner, unsigned int id);
static void pdpy_proxy_anything(t_pdpy_proxyinlet *proxy, t_symbol *s, int argc, t_atom *argv);
static void pdpy_clock_execute(t_pdpy_clock *x);
static void pdpy_printerror(t_pdpy_pdobj *x);
// static void pdpy_pyobject(t_pdpy_pdobj *x, t_symbol *s, t_symbol *id);
static void pdpy_dealloc(t_pdpy_pyclass *self);

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
    if (self->clock) {
        clock_unset(self->clock);
        clock_free(self->clock);
        self->clock = NULL;
    }
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
        Py_DECREF(funcname);
        return NULL;
    }

    PyObject *pdmod = PyImport_ImportModule("puredata");
    if (pdmod == NULL) {
        PyErr_Print();
        pdpy_printerror(NULL);
        PyErr_SetString(PyExc_ImportError, "Failed to import 'puredata' module");
        Py_DECREF(funcname);
        Py_DECREF(clock);
        return NULL;
    }
    PyObject *capsule = PyObject_GetAttrString(pdmod, "_currentobj");
    t_pdpy_pdobj *ptr;
    t_pdpy_pyclass *cls = (t_pdpy_pyclass *)self;

    if (!capsule) {
        PyErr_SetString(PyExc_AttributeError, "No '_currentobj' attribute found");
        Py_DECREF(funcname);
        Py_DECREF(pdmod);
        Py_DECREF(clock);
        return NULL;
    } else if (!Py_IsNone(capsule)) {
        if (!PyCapsule_CheckExact(capsule)) {
            PyErr_SetString(PyExc_TypeError, "Expected a PyCapsule object");
            Py_DECREF(funcname);
            Py_DECREF(capsule);
            Py_DECREF(clock);
            Py_DECREF(pdmod);
            return NULL;
        }

        ptr = (t_pdpy_pdobj *)PyCapsule_GetPointer(capsule, "_currentobj");
        if (!ptr) {
            PyErr_SetString(PyExc_TypeError, "Pointer in _currentobj is NULL");
            Py_DECREF(funcname);
            Py_DECREF(capsule);
            Py_DECREF(clock);
            Py_DECREF(pdmod);
            return NULL;
        }
    } else if (cls->pdobj != NULL) {
        ptr = cls->pdobj;
    } else {
        PyErr_SetString(PyExc_TypeError, "Impossible to get PureData object pointer");
        Py_DECREF(funcname);
        Py_DECREF(pdmod);
        Py_DECREF(clock);
        return NULL;
    }

    Py_INCREF(func);
    clock->pd = pdpy_proxyclock_class;
    clock->owner = ptr;
    clock->function = func;
    clock->functionname = funcnamestr;
    clock->clock = clock_new(clock, (t_method)pdpy_clock_execute);

    ptr->clocks =
        (t_pdpy_clock **)resizebytes(ptr->clocks, ptr->clocks_size * (int)sizeof(t_pdpy_clock *),
                                     (ptr->clocks_size + 1) * (int)sizeof(t_pdpy_clock *));
    ptr->clocks[ptr->clocks_size] = clock;
    ptr->clocks_size++;

    Py_DECREF(funcname);
    Py_DECREF(pdmod);
    Py_DECREF(capsule);
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

    PyObject *r = PyObject_CallNoArgs(x->function);
    if (!r) {
        pdpy_printerror(x->owner);
    } else {
        Py_DECREF(r);
    }
    return;
}

// ╭─────────────────────────────────────╮
// │              Receiver               │
// ╰─────────────────────────────────────╯
// TODO:
// static PyObject *pdpy_newreceiver(PyObject *self, PyObject *args) {
//     PyObject *func;
//     char receiver[MAXPDSTRING];
//     if (!PyArg_ParseTuple(args, "sO", &receiver, &func)) {
//         PyErr_SetString(PyExc_TypeError, "new_receiver require a function as argument");
//         return NULL;
//     }
//
//     if (!PyCallable_Check(func)) {
//         PyErr_SetString(PyExc_TypeError, "new_receiver function is not callable");
//         return NULL;
//     }
//
//
// }
//
// ╭─────────────────────────────────────╮
// │            Py4pd objects            │
// ╰─────────────────────────────────────╯
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
    if (ret < 0 || ret >= (int)sizeof(buf)) {
        buf[sizeof(buf) - 1] = '\0';
    }
    x->pValue = NULL;
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

    PyObject *objclasses = PyDict_GetItemString(obj_dict, classname);
    if (!objclasses || !PyDict_Check(objclasses)) {
        pdpy_printerror(NULL);
        Py_DECREF(pdmod);
        return NULL;
    }

    PyObject *pdclass_capsule = PyDict_GetItemString(objclasses, "pd_class");
    if (!pdclass_capsule || !PyCapsule_CheckExact(pdclass_capsule)) {
        pd_error(NULL, "Class '%s' not found or invalid in _objects", classname);
        Py_DECREF(pdmod);
        return NULL;
    }

    t_class *newobj = (t_class *)PyCapsule_GetPointer(pdclass_capsule, NULL);
    if (!newobj) {
        pd_error(NULL, "Invalid capsule for pdclass '%s'", classname);
        Py_DECREF(pdmod);
        return NULL;
    }

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

    PyObject *objclasses = PyDict_GetItemString(obj_dict, classname);
    if (!objclasses) {
        pdpy_printerror(NULL);
        Py_DECREF(pdmod);
        return NULL;
    }

    PyObject *pyclass = PyDict_GetItemString(objclasses, "py_class");
    if (!pyclass) {
        pd_error(NULL, "Class '%s' not found or invalid in _objects", classname);
        Py_DECREF(pdmod);
        return NULL;
    }

    PyObject *currobj = PyCapsule_New((void *)x, "_currentobj", NULL);
    PyObject_SetAttrString(pdmod, "_currentobj", currobj);
    Py_DECREF(currobj);

    PyObject *newobj = PyObject_CallOneArg(pyclass, args);
    PyObject_SetAttrString(pdmod, "_currentobj", Py_None);

    if (newobj == NULL) {
        pdpy_printerror(NULL);
        pd_error(NULL, "Failed to create python class");
    }
    PyObject *filename = PyDict_GetItemString(objclasses, "script_file");
    if (filename)
        x->script_filename = PyUnicode_AsUTF8(filename);

    Py_DECREF(pdmod);
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
        x->ins = (t_inlet **)malloc((size_t)count * sizeof(t_inlet *));
        x->proxy_in = (t_pdpy_proxyinlet *)malloc((size_t)count * sizeof(t_pdpy_proxyinlet));
        for (int i = 0; i < count; i++) {
            pdpy_proxyinlet_init(&x->proxy_in[i], x, i + 1);
            x->ins[i] = inlet_new(&x->obj, &x->proxy_in[i].pd, 0, 0);
        }
        Py_DECREF(inlets);
        return;
    } else if (PyTuple_Check(inlets)) {
        int count = (int)PyTuple_Size(inlets);
        x->ins = (t_inlet **)malloc((size_t)count * sizeof(t_inlet *));
        x->proxy_in = (t_pdpy_proxyinlet *)malloc((size_t)count * sizeof(t_pdpy_proxyinlet));
        x->siginlets = 0;
        for (int i = 0; i < count; i++) {
            PyObject *config = PyTuple_GetItem(inlets, i);
            if (config == NULL) {
                PyErr_SetString(PyExc_TypeError, "Invalid inlet type");
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
        Py_DECREF(inlets);
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
        Py_DECREF(inlets);
        return;
    }
    Py_DECREF(inlets);
    PyErr_SetString(PyExc_TypeError, "Invalid inlets configuration");
}

// ─────────────────────────────────────
static void pdpy_outlets(t_pdpy_pdobj *x) {
    PyObject *outlets = PyObject_GetAttrString(x->pyclass, "outlets");
    if (outlets == NULL) {
        return;
    }

    if (PyLong_Check(outlets)) {
        int count = (int)PyLong_AsLong(outlets);
        x->outs = (t_outlet **)malloc((size_t)count * sizeof(t_outlet *));
        for (int i = 0; i < count; i++) {
            x->outs[i] = outlet_new(&x->obj, &s_anything);
        }
        x->outletsize = count;
        Py_DECREF(outlets);
        return;
    } else if (PyUnicode_Check(outlets)) {
        const char *sym = PyUnicode_AsUTF8(outlets);
        x->outs = (t_outlet **)malloc(sizeof(t_outlet *));
        if (strcmp(sym, "signal") == 0) {
            x->outs[0] = outlet_new(&x->obj, &s_signal);
            x->sigoutlets++;
        } else {
            x->outs[0] = outlet_new(&x->obj, &s_anything);
        }
        x->outletsize = 1;
        Py_DECREF(outlets);
        return;
    } else if (PyTuple_Check(outlets)) {
        int count = (int)PyTuple_Size(outlets);
        x->outs = (t_outlet **)malloc((size_t)count * sizeof(t_outlet *));
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
        Py_DECREF(outlets);
        return;
    }

    Py_DECREF(outlets);
    PyErr_SetString(PyExc_TypeError, "Invalid outlets configuration");
    return;
}

// ─────────────────────────────────────
static void *pdpy_new(t_symbol *s, int argc, t_atom *argv) {
    t_class *pdobj = pdpyobj_get_pdclass(s->s_name);
    if (pdobj == NULL) {
        pd_error(NULL, "[%s] t_class for %s is invalid, please report!", s->s_name, s->s_name);
        return NULL;
    }

    t_pdpy_pdobj *x = (t_pdpy_pdobj *)pd_new(pdobj);
    x->clocks_size = 0;
    x->clocks = NULL;
    x->dspfunction = NULL;
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
            logpost(x, 2, "Unknown argument type");
            PyList_SetItem(pyargs, i, PyUnicode_FromString("")); // fallback
        }
    }

    PyObject *pyclass = pdpyobj_get_pyclass(x, s->s_name, pyargs);
    if (pyclass == NULL) {
        pdpy_printerror(x);
        Py_DECREF(pyargs);
        Py_DECREF(pdmod);
        return NULL;
    }

    t_pdpy_pyclass *self = (t_pdpy_pyclass *)pyclass;
    x->pyclass = pyclass;

    pd_snprintf(x->id, MAXPDSTRING, "%p", x->pyclass);
    self->name = s->s_name;
    self->pdobj = x;
    // store args (INCREF because we keep it; will DECREF in dealloc)
    self->pyargs = pyargs;

    pdpy_inlets(x);
    pdpy_outlets(x);

    x->outobjptr = pdpy_createoutputptr();
    x->canvas = canvas_getcurrent();

    Py_DECREF(pdmod);
    return (void *)x;
}

// ─────────────────────────────────────
static void py4pdobj_free(t_pdpy_pdobj *x) {
    if (x->outs) {
        free(x->outs);
        x->outs = NULL;
    }
    if (x->ins) {
        free(x->ins);
        x->ins = NULL;
    }
    if (x->proxy_in) {
        free(x->proxy_in);
        x->proxy_in = NULL;
    }

    if (x->dspfunction) {
        Py_DECREF(x->dspfunction);
        x->dspfunction = NULL;
    }

    if (x->clocks_size > 0 && x->clocks) {
        for (int i = 0; i < x->clocks_size; i++) {
            if (x->clocks[i]) {
                Py_DECREF((PyObject *)x->clocks[i]); // triggers pdpy_clock_destruct
            }
        }
        freebytes(x->clocks, x->clocks_size * (int)sizeof(t_pdpy_clock *));
        x->clocks = NULL;
        x->clocks_size = 0;
    }

    if (x->pyclass) {
        Py_DECREF(x->pyclass);
        x->pyclass = NULL;
    }

    if (x->outobjptr) {
        if (x->outobjptr->pValue) {
            Py_DECREF(x->outobjptr->pValue);
            x->outobjptr->pValue = NULL;
        }
        pd_unbind((t_pd *)x->outobjptr, x->outobjptr->id);
        freebytes(x->outobjptr, sizeof(t_pdpy_objptr));
        x->outobjptr = NULL;
    }
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
            if (pstr) {
                pd_error(x, "[py4pd] %s", PyUnicode_AsUTF8(pstr));
                printf("\n%s", PyUnicode_AsUTF8(pstr));
                Py_DECREF(pstr);
            }
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
        t_symbol *sym = atom_getsymbolarg(0, argc, argv);
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
            Py_DECREF(pTuple);
            return;
        }
        PyTuple_SetItem(pTuple, 0, pArg); // steals pArg
        pArg = NULL;                      // prevent double DECREF
        pValue = PyObject_CallObject(method, pTuple);
        Py_DECREF(pTuple);
    }

    if (pValue == NULL) {
        pdpy_printerror(x);
        Py_DECREF(method);
        Py_XDECREF(pArg);
        return;
    }
    if (!Py_IsNone(pValue)) {
        pd_error(x,
                 "[%s] '%s' did not return None. py4pd methods must return None; use "
                 "'self.out' to output data.",
                 x->obj.te_g.g_pd->c_name->s_name, methodname);
    }

    Py_XDECREF(pArg);
    Py_XDECREF(pValue);
    Py_DECREF(method);
}

// ─────────────────────────────────────
void pdpy_proxy_anything(t_pdpy_proxyinlet *proxy, t_symbol *s, int argc, t_atom *argv) {
    t_pdpy_pdobj *o = proxy->owner;
    char methodname[MAXPDSTRING];

    if (strcmp(s->s_name, "PyObject") == 0) {
        pd_snprintf(methodname, MAXPDSTRING, "in_%d_pyobj", proxy->id);
        PyObject *method = PyObject_GetAttrString(o->pyclass, methodname);
        if (method == NULL) {
            pd_error(o, "[%s] method '%s' not found", o->obj.te_g.g_pd->c_name->s_name, methodname);
            return;
        }

        PyObject *pArg = pdpy_getoutptr(atom_getsymbol(argv + 1)); // borrowed
        PyObject *pValue = PyObject_CallOneArg(method, pArg);
        if (pValue == NULL) {
            return;
        }

        if (pValue == NULL) {
            pdpy_printerror(o);
            Py_DECREF(method);
            Py_XDECREF(pArg);
            return;
        }
        Py_DECREF(method);

    } else {
        pd_snprintf(methodname, MAXPDSTRING, "in_%d_%s", proxy->id, s->s_name);
        pdpy_execute(o, methodname, s, argc, argv);
    }
}

// ─────────────────────────────────────
static void pdpy_anything(t_pdpy_pdobj *o, t_symbol *s, int argc, t_atom *argv) {
    char methodname[MAXPDSTRING];
    pd_snprintf(methodname, MAXPDSTRING, "in_1_%s", s->s_name);
    pdpy_execute(o, methodname, s, argc, argv);
}

// ─────────────────────────────────────
// static void pdpy_pyobject(t_pdpy_pdobj *x, t_symbol *s, t_symbol *id) {
//     if (x->pyclass == NULL) {
//         post("pyclass is NULL");
//         return;
//     }
//
//     PyObject *pyclass = (PyObject *)x->pyclass;
//     PyObject *method = PyObject_GetAttrString(pyclass, "in_1_pyobj");
//     if (!method || !PyCallable_Check(method)) {
//         PyErr_Clear();
//         pd_error(x, "[%s] method 'in_1_pyobj' not found or not callable",
//         x->obj.te_g.g_pd->c_name->s_name); Py_XDECREF(method); return;
//     }
//
//     PyObject *pArg = pdpy_getoutptr(id); // borrowed
//     PyObject *pValue = PyObject_CallOneArg(method, pArg);
//     if (pValue == NULL) {
//         pdpy_printerror(x);
//         Py_DECREF(method);
//         return;
//     }
//
//     if (!Py_IsNone(pValue)) {
//         pd_error(x,
//                  "[%s] 'in_1_pyobj' did not return None. py4pd methods must return None; use "
//                  "'self.out' to output data.",
//                  x->obj.te_g.g_pd->c_name->s_name);
//     }
//
//     Py_DECREF(method);
//     Py_XDECREF(pValue);
// }

// ─────────────────────────────────────
static t_int *pdpy_perform(t_int *w) {
    t_pdpy_pdobj *x = (t_pdpy_pdobj *)(w[1]);
    int n = (int)w[2];

    if (x->dspfunction == NULL) {
        return w + x->siginlets + x->sigoutlets + 3;
    }

    PyObject *py_in = PyTuple_New(x->siginlets);
    for (int i = 0; i < (int)x->siginlets; i++) {
        t_sample *in = (t_sample *)w[3 + i];
        PyObject *py_list = PyList_New(n);
        for (int j = 0; j < n; j++) {
            PyList_SetItem(py_list, j, PyFloat_FromDouble(in[j]));
        }
        PyTuple_SetItem(py_in, i, py_list);
    }

    PyObject *pValue = PyObject_CallOneArg(x->dspfunction, py_in);
    if (pValue && PyTuple_Check(pValue)) {
        int size = (int)PyTuple_Size(pValue);
        if (size == (int)x->sigoutlets) {
            for (int i = 0; i < (int)x->sigoutlets; i++) {
                t_sample *out = (t_sample *)w[3 + x->siginlets + i];
                PyObject *pyout = PyTuple_GetItem(pValue, i);
                int arraysize = (int)PyTuple_Size(pyout);
                for (int j = 0; j < arraysize && j < n; j++) {
                    PyObject *pyf = PyTuple_GetItem(pyout, j);
                    out[j] = (t_sample)PyFloat_AsDouble(pyf);
                }
            }
        } else if (size == n) {
            PyObject *pyf0 = PyTuple_GetItem(pValue, 0);
            if (PyFloat_Check(pyf0)) {
                t_sample *out = (t_sample *)w[3 + x->siginlets];
                for (int j = 0; j < n; j++) {
                    PyObject *pyf = PyTuple_GetItem(pValue, j);
                    out[j] = (t_sample)PyFloat_AsDouble(pyf);
                }
            } else {
                PyErr_SetString(PyExc_RuntimeError, "Returned value inside Tuple is unknown!");
                pdpy_printerror(x);
            }
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Unknown Tuple size or way to process it");
            pdpy_printerror(x);
        }
    } else if (pValue) {
        PyErr_SetString(PyExc_RuntimeError, "Returned value is not a tuple");
        pdpy_printerror(x);
    } else {
        pdpy_printerror(x);
    }

    Py_XDECREF(pValue);
    Py_DECREF(py_in);

    return w + x->siginlets + x->sigoutlets + 3;
}

// ─────────────────────────────────────
static void pdpy_dsp(t_pdpy_pdobj *x, t_signal **sp) {
    int sum = (int)(x->siginlets + x->sigoutlets);
    int sigvecsize = sum + 2; // +1 for x, +1 for blocksize
    t_int *sigvec = getbytes((size_t)sigvecsize * sizeof(t_int));
    for (int i = (int)x->siginlets; i < sum; i++) {
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
    Py_DECREF(pysr);

    PyObject *pyvec = PyLong_FromDouble(sp[0]->s_n);
    if (PyObject_SetAttrString(pyclass, "blocksize", pyvec) < 0) {
        pd_error(x, "Failed to set blocksize");
        PyErr_Clear();
    }
    Py_DECREF(pyvec);

    PyObject *method = PyObject_GetAttrString(pyclass, "perform");
    if (!method || !PyCallable_Check(method)) {
        PyErr_Clear();
        pd_error(x, "[%s] No perform method defined or not callable",
                 x->obj.te_g.g_pd->c_name->s_name);
        for (int i = 0; i < sum; i++) {
            t_sample *out = (t_sample *)sp[i]->s_vec;
            dsp_add_zero(out, sp[i]->s_n);
        }
        if (method)
            Py_DECREF(method);
        freebytes(sigvec, (size_t)sigvecsize * sizeof(t_int));
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
            Py_DECREF(method);
            Py_DECREF(dsp_method);
            for (int i = 0; i < sum; i++) {
                t_sample *out = (t_sample *)sp[i]->s_vec;
                dsp_add_zero(out, sp[i]->s_n);
            }
            freebytes(sigvec, (size_t)sigvecsize * sizeof(t_int));
            return;
        }
        int truth = PyObject_IsTrue(r);
        Py_DECREF(r);
        if (!truth) {
            Py_DECREF(method);
            Py_DECREF(dsp_method);
            for (int i = 0; i < sum; i++) {
                t_sample *out = (t_sample *)sp[i]->s_vec;
                dsp_add_zero(out, sp[i]->s_n);
            }
            freebytes(sigvec, (size_t)sigvecsize * sizeof(t_int));
            return;
        }
    } else {
        PyErr_Clear();
        pd_error(x, "[%s] Object class has no callable dsp method",
                 x->obj.te_g.g_pd->c_name->s_name);
        for (int i = 0; i < sum; i++) {
            t_sample *out = (t_sample *)sp[i]->s_vec;
            dsp_add_zero(out, sp[i]->s_n);
        }
        Py_XDECREF(dsp_method);
        Py_DECREF(method);
        freebytes(sigvec, (size_t)sigvecsize * sizeof(t_int));
        return;
    }

    Py_XDECREF(dsp_method);
    PyErr_Clear();

    // Store perform method (INCREF to persist)
    x->dspfunction = method;
    // do NOT DECREF method here; hold reference

    dsp_addv(pdpy_perform, sigvecsize, sigvec);
    freebytes(sigvec, (size_t)sigvecsize * sizeof(t_int));
}

// ─────────────────────────────────────
static void pdpy_menu_open(t_pdpy_pdobj *o) {
    char name[MAXPDSTRING];
    pd_snprintf(name, MAXPDSTRING, "%s", o->script_filename);
    pdgui_vmess("::pd_menucommands::menu_openfile", "s", name);
}

// ─────────────────────────────────────
static void pdpy_proxyinlet_init(t_pdpy_proxyinlet *p, t_pdpy_pdobj *owner, unsigned int id) {
    p->pd = pdpy_proxyinlet_class;
    p->owner = owner;
    p->id = id;
}

// ─────────────────────────────────────
static t_class *pdpy_classnew(const char *n, bool dsp) {
    t_class *c = class_new(gensym(n), (t_newmethod)pdpy_new, (t_method)py4pdobj_free,
                           sizeof(t_pdpy_pdobj), CLASS_NOINLET | CLASS_MULTICHANNEL, A_GIMME, 0);

    class_addmethod(c, (t_method)pdpy_menu_open, gensym("menu-open"), A_NULL);
    if (dsp) {
        class_addmethod(c, (t_method)pdpy_dsp, gensym("dsp"), A_CANT, 0);
    }
    return c;
}

// ─────────────────────────────────────
static int pdpy_create_newpyobj(PyObject *subclass, const char *name, const char *filename) {

    bool havedsp = false;
    PyObject *dsp = PyObject_GetAttrString(subclass, "perform");
    if (dsp) {
        havedsp = PyCallable_Check(dsp);
    } else {
        PyErr_Clear();
    }
    Py_XDECREF(dsp);

    PyObject *pd_module = PyImport_ImportModule("puredata");
    if (pd_module == NULL) {
        pdpy_printerror(NULL);
        return 0;
    }

    PyObject *pyexternals = PyObject_GetAttrString(pd_module, "_objects");
    if (pyexternals == NULL) {
        pdpy_printerror(NULL);
        Py_DECREF(pd_module);
        return 0;
    }

    PyObject *externaldict = PyDict_New();
    if (!externaldict) {
        Py_DECREF(pyexternals);
        Py_DECREF(pd_module);
        return 0;
    }

    PyObject *val_script = PyUnicode_FromString(filename);
    PyObject *val_clocks = PyList_New(0);
    if (!val_script || !val_clocks) {
        Py_XDECREF(val_script);
        Py_XDECREF(val_clocks);
        Py_DECREF(externaldict);
        Py_DECREF(pyexternals);
        Py_DECREF(pd_module);
        return 0;
    }

    PyDict_SetItemString(externaldict, "py_class", subclass);
    PyDict_SetItemString(externaldict, "script_file", val_script);
    PyDict_SetItemString(externaldict, "clocks", val_clocks);
    Py_DECREF(val_script);
    Py_DECREF(val_clocks);

    t_class *pdclass = pdpy_classnew(name, havedsp);
    PyObject *capsule = PyCapsule_New((void *)pdclass, NULL, NULL);
    if (!capsule) {
        pdpy_printerror(NULL);
        Py_DECREF(externaldict);
        Py_DECREF(pyexternals);
        Py_DECREF(pd_module);
        return 0;
    }

    if (PyDict_SetItemString(externaldict, "pd_class", capsule) < 0) {
        Py_DECREF(capsule);
        pdpy_printerror(NULL);
        Py_DECREF(externaldict);
        Py_DECREF(pyexternals);
        Py_DECREF(pd_module);
        return 0;
    }
    Py_DECREF(capsule);

    if (PyDict_SetItemString(pyexternals, name, externaldict) < 0) {
        pdpy_printerror(NULL);
        Py_DECREF(externaldict);
        Py_DECREF(pyexternals);
        Py_DECREF(pd_module);
        return 0;
    }
    Py_DECREF(externaldict);
    Py_DECREF(pyexternals);
    Py_DECREF(pd_module);

    return 1;
}

// ─────────────────────────────────────
static int pdpy_validate_pd_subclasses(PyObject *module, const char *filename) {
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

    PyObject *module_dict = PyModule_GetDict(module);
    PyObject *items = PyDict_Items(module_dict);
    Py_ssize_t num_items = PyList_Size(items);

    int objects = 0;
    for (Py_ssize_t i = 0; i < num_items; i++) {
        PyObject *item = PyList_GetItem(items, i); // borrowed
        PyObject *obj = PyTuple_GetItem(item, 1);
        if (PyType_Check(obj)) {
            if (PyObject_IsSubclass(obj, new_object_type) && (obj != new_object_type)) {
                PyObject *objname = PyObject_GetAttrString(obj, "name");
                if (!objname) {
                    pd_error(NULL,
                             "[py4pd] cannot read class name inside %s, 'name' attribute missing",
                             filename);
                    continue;
                }

                PyObject *classname = PyObject_GetAttrString(obj, "__name__");
                if (!classname) {
                    PyErr_Clear();
                    pd_error(
                        NULL,
                        "[py4pd] cannot read class name inside %s, '__name__' attribute missing",
                        filename);
                    continue;
                }

                if (!PyUnicode_Check(objname)) {
                    const char *classname_c = PyUnicode_AsUTF8(classname);
                    pd_error(NULL, "[py4pd] 'name' attribute in %s is not a string", classname_c);
                    Py_DECREF(objname);
                    continue;
                }

                int ok = pdpy_create_newpyobj(obj, PyUnicode_AsUTF8(objname), filename);
                if (!ok) {
                    pdpy_printerror(NULL);
                } else {
                    objects++;
                }
                Py_DECREF(objname);
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
    char *source = (char *)malloc((size_t)size + 1);
    if (!source) {
        fclose(fp);
        pdpy_printerror(NULL);
        return 0;
    }
    if (fread(source, 1, (size_t)size, fp) != (size_t)size) {
        fclose(fp);
        free(source);
        pdpy_printerror(NULL);
        return 0;
    }
    source[size] = '\0';
    fclose(fp);

    PyObject *module = PyModule_New(modname);
    if (!module) {
        free(source);
        pdpy_printerror(NULL);
        return 0;
    }

    PyObject *code_obj = Py_CompileString(source, filename, Py_file_input);
    free(source);
    if (!code_obj) {
        pdpy_printerror(NULL);
        Py_DECREF(module);
        return 0;
    }

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
            Py_DECREF(code_obj);
            Py_DECREF(module);
            return 0;
        }
    }

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

    Py_DECREF(module);
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

    PyObject *obj_dict = PyObject_GetAttrString(pdmod, "_objects");
    if (!obj_dict || !PyDict_Check(obj_dict)) {
        Py_XDECREF(obj_dict);
        obj_dict = PyDict_New();
        if (obj_dict == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Failed to create _objects dictionary");
            Py_DECREF(pdmod);
            return -1;
        }
        if (PyObject_SetAttrString(pdmod, "_objects", obj_dict) < 0) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to set _objects in puredata module");
            Py_DECREF(obj_dict);
            Py_DECREF(pdmod);
            return -1;
        }
    }

    Py_DECREF(obj_dict);
    Py_DECREF(pdmod);
    return 0;
}

// ─────────────────────────────────────
static PyObject *pdpy_error(t_pdpy_pyclass *self, PyObject *args) {
    char msg[MAXPDSTRING] = "";
    size_t msg_len = 0;

    Py_ssize_t num_args = PyTuple_Size(args);
    for (Py_ssize_t i = 0; i < num_args; i++) {
        PyObject *arg = PyTuple_GetItem(args, i);
        PyObject *str_obj = PyObject_Str(arg);
        if (!str_obj)
            continue;
        const char *str = PyUnicode_AsUTF8(str_obj);
        if (str) {
            size_t str_len = strlen(str);
            if (msg_len + str_len + 4 >= MAXPDSTRING) {
                Py_DECREF(str_obj);
                pd_snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), "...");
                break;
            }
            pd_snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), "%s", str);
            msg_len += str_len;
            if (i < num_args - 1 && msg_len + 1 < MAXPDSTRING) {
                pd_snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), " ");
                msg_len++;
            }
        }
        Py_DECREF(str_obj);
    }
    pd_error(self->pdobj, "[%s]: %s", self->name, msg);
    Py_RETURN_TRUE;
}

// ─────────────────────────────────────
static PyObject *pdpy_logpost(t_pdpy_pyclass *self, PyObject *args) {
    char msg[MAXPDSTRING] = "";
    size_t msg_len = 0;

    if (PyTuple_Size(args) < 1) {
        PyErr_SetString(PyExc_TypeError, "logpost requires at least loglevel");
        return NULL;
    }
    PyObject *pyloglevel = PyTuple_GetItem(args, 0);
    int loglevel = (int)PyLong_AsLong(pyloglevel);

    Py_ssize_t num_args = PyTuple_Size(args);
    for (Py_ssize_t i = 1; i < num_args; i++) {
        PyObject *arg = PyTuple_GetItem(args, i);
        PyObject *str_obj = PyObject_Str(arg);
        if (!str_obj)
            continue;
        const char *str = PyUnicode_AsUTF8(str_obj);
        Py_DECREF(str_obj);
        if (str) {
            size_t str_len = strlen(str);
            if (msg_len + str_len + 4 >= MAXPDSTRING) {
                pd_snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), "...");
                break;
            }
            pd_snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), "%s", str);
            msg_len += str_len;
            if (i < num_args - 1 && msg_len + 1 < MAXPDSTRING) {
                pd_snprintf(msg + strlen(msg), sizeof(msg) - strlen(msg), " ");
                msg_len++;
            }
        }
    }
    logpost(self->pdobj, loglevel, "[%s]: %s", self->name, msg);
    Py_RETURN_TRUE;
}

// ─────────────────────────────────────
static PyObject *pdpy_getcurrentdir(t_pdpy_pyclass *self, PyObject *args) {
    t_symbol *dir = canvas_getdir(self->pdobj->canvas);
    if (!dir)
        Py_RETURN_NONE;
    return PyUnicode_FromString(dir->s_name);
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
        if (x->outobjptr->pValue) {
            Py_DECREF(x->outobjptr->pValue);
        }
        Py_INCREF(pValue);
        x->outobjptr->pValue = pValue;
        t_atom a[2];
        SETSYMBOL(&a[0], gensym(Py_TYPE(pValue)->tp_name));
        SETSYMBOL(&a[1], x->outobjptr->id);
        outlet_anything(out, gensym("PyObject"), 2, a);
        Py_RETURN_TRUE;
    } else if (pdtype == A_FLOAT) {
        t_float v = (t_float)PyFloat_AsDouble(pValue);
        outlet_float(out, v);
    } else if (pdtype == A_SYMBOL) {
        const char *s = PyUnicode_AsUTF8(pValue);
        outlet_symbol(out, gensym(s));
    } else if (pdtype == A_GIMME) {
        if (PyList_Check(pValue)) {
            int size = (int)PyList_Size(pValue);
            t_atom *list_array = (t_atom *)malloc((size_t)size * sizeof(t_atom));
            for (int i = 0; i < size; ++i) {
                PyObject *pValue_i = PyList_GetItem(pValue, i); // borrowed
                if (PyLong_Check(pValue_i)) {
                    SETFLOAT(&list_array[i], (t_float)PyLong_AsLong(pValue_i));
                } else if (PyFloat_Check(pValue_i)) {
                    SETFLOAT(&list_array[i], (t_float)PyFloat_AsDouble(pValue_i));
                } else if (PyUnicode_Check(pValue_i)) {
                    SETSYMBOL(&list_array[i], gensym(PyUnicode_AsUTF8(pValue_i)));
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
            PyErr_SetString(PyExc_TypeError, "Output with pd.LIST requires a list");
            return NULL;
        }
    }
    Py_RETURN_TRUE;
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
    char *source = (char *)malloc((size_t)size + 1);
    if (source == NULL) {
        fclose(fp);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for source");
        return NULL;
    }
    if (fread(source, 1, (size_t)size, fp) != (size_t)size) {
        fclose(fp);
        free(source);
        PyErr_SetString(PyExc_MemoryError, "Failed to read script file");
        return NULL;
    }
    source[size] = '\0';
    fclose(fp);

    PyObject *module = PyModule_New(name);
    if (!module) {
        free(source);
        PyErr_Print();
        return NULL;
    }

    PyObject *code_obj = Py_CompileString(source, filename, Py_file_input);
    free(source);
    if (!code_obj) {
        PyErr_Print();
        Py_DECREF(module);
        return NULL;
    }

    PyObject *mod_dict = PyModule_GetDict(module);
    if (!mod_dict) {
        PyErr_Print();
        Py_DECREF(code_obj);
        Py_DECREF(module);
        return NULL;
    }

    if (PyDict_GetItemString(mod_dict, "__builtins__") == NULL) {
        PyObject *builtins = PyEval_GetBuiltins();
        if (PyDict_SetItemString(mod_dict, "__builtins__", builtins) < 0) {
            PyErr_Print();
            Py_DECREF(code_obj);
            Py_DECREF(module);
            return NULL;
        }
    }

    PyObject *result = PyEval_EvalCode(code_obj, mod_dict, mod_dict);
    Py_DECREF(code_obj);
    if (!result) {
        PyErr_Print();
        Py_DECREF(module);
        return NULL;
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

    // update clocks
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
            Py_XDECREF(func);
        }
    }

    // dsp update: release previous dspfunction if invalid
    if (self->pdobj->dspfunction != NULL) {
        PyObject *dspmethod = PyObject_GetAttrString(pyclass, "perform");
        if (!dspmethod || !PyCallable_Check(dspmethod)) {
            PyErr_Clear();
            Py_DECREF(self->pdobj->dspfunction);
            self->pdobj->dspfunction = NULL;
        }
        Py_XDECREF(dspmethod);
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
    {"out", (PyCFunction)pdpy_out, METH_VARARGS | METH_KEYWORDS, "Send data to Pd outlet"},
    {"new_clock", (PyCFunction)pdpy_newclock, METH_VARARGS, "Return a clock object"},
    {"reload", (PyCFunction)pdpy_reload, METH_NOARGS, "Reload the current script and object"},

    {"get_current_dir", (PyCFunction)pdpy_getcurrentdir, METH_NOARGS, "Returns current canvas dir"},

    // {"tabwrite", (PyCFunction)pdpy_tabwrite, METH_VARARGS | METH_KEYWORDS, "Write to a Pd Array"},
    // {"tabread", (PyCFunction)pdpy_tabread, METH_VARARGS | METH_KEYWORDS, "Read a Pd Array"},
    {NULL, NULL, 0, NULL}};

// ─────────────────────────────────────
static void pdpy_dealloc(t_pdpy_pyclass *self) {
    Py_XDECREF(self->pyargs);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

PyTypeObject pdpy_type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "NewObject",
    .tp_doc = "It creates new PureData objects",
    .tp_basicsize = sizeof(t_pdpy_pyclass),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_methods = pdpy_methods,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)pdpyobj_init,
    .tp_dealloc = (destructor)pdpy_dealloc,
};

//╭─────────────────────────────────────╮
//│             pd methods              │
//╰─────────────────────────────────────╯
static PyObject *pdpy_post(PyObject *self, PyObject *args) {
    (void)self;
    if (PyTuple_Size(args) > 0) {
        startpost("[Python]: ");
        for (int i = 0; i < PyTuple_Size(args); i++) {
            PyObject *arg = PyTuple_GetItem(args, i);
            PyObject *str = PyObject_Str(arg);
            if (str) {
                startpost(PyUnicode_AsUTF8(str));
                startpost(" ");
                Py_DECREF(str);
            }
        }
        startpost("\n");
    }
    Py_RETURN_TRUE;
}

// ─────────────────────────────────────
static PyObject *pdpy_hasgui(PyObject *self, PyObject *args) {
    (void)self;
    (void)args;
    return PyLong_FromLong(sys_havegui());
}

// ─────────────────────────────────────
static PyObject *pdpy_sr(PyObject *self, PyObject *args) {
    (void)self;
    return PyLong_FromLong(sys_getsr());
}

// ─────────────────────────────────────
static PyObject *pdpy_tabread(PyObject *self, PyObject *args) {
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
        garray_getfloatwords(pdarray, &vecsize, &vec);
        PyObject *pAudio = PyTuple_New(vecsize);
        for (int i = 0; i < vecsize; i++) {
            PyTuple_SetItem(pAudio, i, PyFloat_FromDouble(vec[i].w_float));
        }
        return pAudio;
    }
}

// ─────────────────────────────────────
static PyObject *pdpy_tabwrite(PyObject *self, PyObject *args, PyObject *kwargs) {
    char *tabname;
    t_garray *pdarray;
    PyObject *array;
    t_word *vec;
    int vecsize;
    int resize = 0;

    // Parse arguments (positional: tabname, array)
    if (!PyArg_ParseTuple(args, "sO", &tabname, &array)) {
        PyErr_SetString(PyExc_TypeError, "Wrong arguments, expected string tabname and sequence");
        return NULL;
    }

    // Check for "resize" keyword in kwargs
    if (kwargs && PyDict_Check(kwargs)) {
        PyObject *r = PyDict_GetItemString(kwargs, "resize");
        if (r) {
            resize = PyObject_IsTrue(r);
        }
    }

    // Find Pd array
    t_symbol *pd_symbol = gensym(tabname);
    if (!(pdarray = (t_garray *)pd_findbyclass(pd_symbol, garray_class))) {
        PyErr_SetString(PyExc_TypeError, "self.tabwrite: array not found");
        return NULL;
    } else if (!garray_getfloatwords(pdarray, &vecsize, &vec)) {
        PyErr_SetString(PyExc_TypeError, "Bad template for tabwrite");
        return NULL;
    }

    int newsize = 0;

    if (PyList_Check(array)) {
        newsize = (int)PyList_Size(array);
        if (resize) {
            garray_resize_long(pdarray, newsize);
            garray_getfloatwords(pdarray, &vecsize, &vec);
        } else if (newsize > vecsize) {
            newsize = vecsize;
        }

        for (int i = 0; i < newsize; i++) {
            PyObject *PyFloatObj = PyList_GetItem(array, i);
            vec[i].w_float = (float)PyFloat_AsDouble(PyFloatObj);
        }
        garray_redraw(pdarray);
        Py_RETURN_TRUE;

    } else if (PyTuple_Check(array)) {
        newsize = (int)PyTuple_Size(array);
        if (resize) {
            garray_resize_long(pdarray, newsize);
            garray_getfloatwords(pdarray, &vecsize, &vec);
        } else if (newsize > vecsize) {
            newsize = vecsize;
        }

        for (int i = 0; i < newsize; i++) {
            PyObject *PyFloatObj = PyTuple_GetItem(array, i);
            vec[i].w_float = (float)PyFloat_AsDouble(PyFloatObj);
        }
        garray_redraw(pdarray);
        Py_RETURN_TRUE;

    } else {
        PyErr_SetString(PyExc_TypeError, "Input must be either a list or a tuple");
        return NULL;
    }
}

// ╭─────────────────────────────────────╮
// │            MODULE LOADER            │
// ╰─────────────────────────────────────╯
static int ObjectLoader_init(PyObject *self, PyObject *args, PyObject *kwargs) {
    const char *fullname = NULL;
    const char *path = NULL;
    static char *kwlist[] = {"fullname", "path", NULL};

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

    PyObject *util_module = PyImport_ImportModule("importlib.util");
    if (!util_module)
        return -1;

    PyObject *spec_func = PyObject_GetAttrString(util_module, "spec_from_loader");
    Py_DECREF(util_module);
    if (!spec_func)
        return -1;

    PyObject *spec_args = Py_BuildValue("(sO)", fullname, self);
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

    return 0;
}

// ─────────────────────────────────────
static PyObject *ObjectLoader_get_filename(PyObject *self, PyObject *args) {
    PyObject *fullname;
    if (!PyArg_ParseTuple(args, "O", &fullname))
        return NULL;
    (void)fullname;
    return PyObject_GetAttrString(self, "path");
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

    char *buffer = (char *)malloc((size_t)size);
    if (!buffer) {
        fclose(file);
        PyErr_NoMemory();
        return NULL;
    }

    size_t read = fread(buffer, 1, (size_t)size, file);
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
    (void)fullname;
    Py_RETURN_FALSE;
}

// ─────────────────────────────────────
static PyMethodDef ObjectLoader_methods[] = {
    {"get_filename", ObjectLoader_get_filename, METH_VARARGS, "Return the path to the source file"},
    {"get_data", ObjectLoader_get_data, METH_VARARGS, "Read the file's contents as bytes"},
    {"is_package", ObjectLoader_is_package, METH_VARARGS, "Check if the module is a package"},
    {NULL, NULL, 0, NULL}};

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
/*             MODULE INIT             */
// ╰─────────────────────────────────────╯
PyMethodDef pdpy_modulemethods[] = {
    {"post", pdpy_post, METH_VARARGS, "Print informations in PureData Console"},
    {"hasgui", pdpy_hasgui, METH_NOARGS, "Return False is pd is running in console"},
    {"tabwrite", (PyCFunction)pdpy_tabwrite, METH_VARARGS | METH_KEYWORDS, "Write to a Pd Array"},
    {"tabread", pdpy_tabread, METH_VARARGS, "Read a Pd Array"},
    
    {"get_sample_rate", pdpy_sr, METH_NOARGS, "Return sample rate"},
    // {"get_current_dir", pdpy_getcurrentdir, METH_NOARGS, "Returns current canvas dir"},
    {NULL, NULL, 0, NULL}};

// ─────────────────────────────────────
static PyObject *pd4pdmodule_init(PyObject *self) {
    if (PyType_Ready(&pdpy_type) < 0) {
        return NULL;
    }

    PyModule_AddObject(self, "FLOAT", PyLong_FromLong(A_FLOAT));
    PyModule_AddObject(self, "SYMBOL", PyLong_FromLong(A_SYMBOL));
    PyModule_AddObject(self, "LIST", PyLong_FromLong(A_GIMME));
    PyModule_AddObject(self, "PYOBJECT", PyLong_FromLong(PYOBJECT));

    PyModule_AddObject(self, "SIGNAL", PyUnicode_FromString(s_signal.s_name));
    PyModule_AddObject(self, "DATA", PyUnicode_FromString(s_anything.s_name));

    Py_INCREF(&pdpy_type);
    if (PyModule_AddObject(self, "NewObject", (PyObject *)&pdpy_type) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to add NewObject to module");
        return NULL;
    }

    PyObject *abc_module = PyImport_ImportModule("importlib.abc");
    if (!abc_module)
        return NULL;
    PyObject *source_loader = PyObject_GetAttrString(abc_module, "SourceLoader");
    Py_DECREF(abc_module);
    if (!source_loader)
        return NULL;

    PyObject *bases = PyTuple_Pack(1, source_loader);
    Py_DECREF(source_loader);
    if (!bases)
        return NULL;

    PyObject *object_loader_type = PyType_FromSpecWithBases(&ObjectLoaderSpec, bases);
    Py_DECREF(bases);
    if (!object_loader_type)
        return NULL;

    if (PyModule_AddObject(self, "_ObjectLoader", object_loader_type) < 0) {
        Py_DECREF(object_loader_type);
        return NULL;
    }

    PyObject *new_dict = PyDict_New();
    if (PyModule_AddObject(self, "_objects", new_dict) != 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to add _objects to module");
        return NULL;
    }

    return 0;
}

// ─────────────────────────────────────
static PyModuleDef_Slot pdpy_moduleslots[] = {{Py_mod_exec, pd4pdmodule_init}, {0, NULL}};

// ─────────────────────────────────────
static struct PyModuleDef pdpy_module_def = {
    PyModuleDef_HEAD_INIT,
    .m_name = "puredata",
    .m_doc = "pd module provide function to interact with PureData",
    .m_size = 0,
    .m_methods = pdpy_modulemethods,
    .m_slots = pdpy_moduleslots,
};

// ─────────────────────────────────────
PyMODINIT_FUNC pdpy_initpuredatamodule() {
    PyObject *m = PyModuleDef_Init(&pdpy_module_def);
    if (m == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create module");
        return NULL;
    }
    return m;
}

// ╭─────────────────────────────────────╮
// │            Py4pd Object             │
// ╰─────────────────────────────────────╯
static int pd4pd_loader_pathwise(t_canvas *canvas, const char *objectname, const char *path) {
    (void)canvas;
    char dirbuf[MAXPDSTRING], filename[MAXPDSTRING];
    char *ptr;
    const char *classname;
    int fd;
    if (!path) {
        return 0;
    }

    if ((classname = strrchr(objectname, '/'))) {
        classname++;
    } else {
        classname = objectname;
    }
    if ((fd = open_via_path(path, objectname, ".pd_py", dirbuf, &ptr, MAXPDSTRING, 1)) >= 0)
        if (pd4pd_loader_wrappath(fd, objectname, dirbuf)) {
            return 1;
        }

    pd_snprintf(filename, MAXPDSTRING, "%s", objectname);
    pd_snprintf(filename + strlen(filename), MAXPDSTRING - strlen(filename), "/");
    pd_snprintf(filename + strlen(filename), MAXPDSTRING - strlen(filename), "%s", classname);
    filename[MAXPDSTRING - 1] = 0;
    if ((fd = open_via_path(path, filename, ".pd_py", dirbuf, &ptr, MAXPDSTRING, 1)) >= 0)
        if (pd4pd_loader_wrappath(fd, objectname, dirbuf)) {
            return 1;
        }
    return 0;
}

// ─────────────────────────────────────
void py4pd_addpath2syspath(const char *path) {
    if (!path) {
        pd_error(NULL, "Invalid path: NULL");
        return;
    }

    PyObject *sys = PyImport_ImportModule("sys");
    if (!sys) {
        pd_error(NULL, "Failed to import sys module");
        return;
    }

    PyObject *sysPath = PyObject_GetAttrString(sys, "path");
    if (!sysPath || !PyList_Check(sysPath)) {
        pd_error(NULL, "Failed to access sys.path or it is not a list");
        Py_XDECREF(sysPath);
        Py_DECREF(sys);
        return;
    }

    PyObject *pathEntry = PyUnicode_FromString(path);
    if (!pathEntry) {
        pd_error(NULL, "Failed to create Python string from path");
        Py_DECREF(sysPath);
        Py_DECREF(sys);
        return;
    }

    // Prepend path if not already in sys.path
    if (PySequence_Contains(sysPath, pathEntry) == 0) {
        if (PyList_Insert(sysPath, 0, pathEntry) != 0) {
            pd_error(NULL, "Failed to insert path at start of sys.path");
        }
    }

    Py_DECREF(pathEntry);
    Py_DECREF(sysPath);
    Py_DECREF(sys);
}

// ─────────────────────────────────────
void py4pd_set_py4pdpath_env(const char *path) {
    if (!path) {
        pd_error(NULL, "Invalid path: NULL");
        return;
    }

    PyObject *os = PyImport_ImportModule("os");
    if (!os) {
        pd_error(NULL, "Failed to import os module");
        return;
    }

    PyObject *py_environ = PyObject_GetAttrString(os, "environ");
    if (!py_environ) {
        pd_error(NULL, "Failed to access os.environ");
        Py_DECREF(os);
        return;
    }

    PyObject *key = PyUnicode_FromString("PY4PD_PATH");
    PyObject *value = PyUnicode_FromString(path);

    if (!key || !value) {
        pd_error(NULL, "Failed to create Python strings for environment key/value");
        Py_XDECREF(key);
        Py_XDECREF(value);
        Py_DECREF(py_environ);
        Py_DECREF(os);
        return;
    }

    if (PyObject_SetItem(py_environ, key, value) != 0) {
        pd_error(NULL, "Failed to set PY4PD_PATH in os.environ");
    }

    Py_DECREF(key);
    Py_DECREF(value);
    Py_DECREF(py_environ);
    Py_DECREF(os);
}

// ─────────────────────────────────────
void *py4pd_new(t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    (void)argc;
    (void)argv;
    t_py *x = (t_py *)pd_new(py4pd_class);
    return (void *)x;
}

// ─────────────────────────────────────
void py4pd_setup(void) {
    int Major, Minor, Micro;
    sys_getversion(&Major, &Minor, &Micro);
    sys_register_loader((loader_t)pd4pd_loader_pathwise);

    if (Major < 0 && Minor < 54) {
        pd_error(NULL, "[py4pd] py4pd requires Pd version 0.54 or later.");
        return;
    }

    py4pd_class =
        class_new(gensym("py4pd"), (t_newmethod)py4pd_new, NULL, sizeof(t_py), 0, A_GIMME, 0);

    if (!Py_IsInitialized()) {
        objCount = 0;
        post("");
        post("[py4pd] by Charles K. Neimog | version %d.%d.%d", PY4PD_MAJOR_VERSION,
             PY4PD_MINOR_VERSION, PY4PD_MICRO_VERSION);
        post("[py4pd] Python version %d.%d.%d", PY_MAJOR_VERSION, PY_MINOR_VERSION,
             PY_MICRO_VERSION);
        post("");
        int r = PyImport_AppendInittab("puredata", pdpy_initpuredatamodule);
        if (r < 0) {
            pd_error(NULL, "[py4pd] PyInit_pd failed");
            return;
        }
        Py_Initialize();

        const char *py4pd_path = py4pd_class->c_externdir->s_name;
        py4pd_set_py4pdpath_env(py4pd_path);
        char py4pd_env[MAXPDSTRING];
        pd_snprintf(py4pd_env, MAXPDSTRING, "%s/py4pd-env", py4pd_path);
        py4pd_addpath2syspath(py4pd_env);
    }

    pdpy_proxyinlet_setup();
    pdpy_pyobjectoutput_setup();
    // TODO: receive
}

#ifdef __WIN64
__declspec(dllexport) void py4pd_setup(void);
#endif
