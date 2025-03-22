#include <m_pd.h>

#include <g_canvas.h>
#include <m_imp.h>
#include <s_stuff.h>

#include <Python.h>

typedef struct _pdpy_pyclass t_pdpy_pyclass;
typedef struct _pdpy_clock t_pdpy_clock;
static t_class *pdpy_proxyinlet_class = NULL;
static t_class *pdpy_pyobjectout_class = NULL;

#define PYOBJECT -1997
#define PY4PDSIGTOTAL(s) ((t_int)((s)->s_length * (s)->s_nchans))

// ╭─────────────────────────────────────╮
// │          Object Base Class          │
// ╰─────────────────────────────────────╯
typedef struct _pdpy_objptr {
    t_pd x_pd;
    t_symbol *Id;
    PyObject *pValue;
} t_pdpy_objptr;

// ─────────────────────────────────────
typedef struct _pdpy_pdobj {
    t_object obj;
    t_sample sample;
    t_pdpy_pyclass *pyclass;
    t_symbol *script;
    t_canvas *canvas;

    // dsp
    int nchs;
    int vecsize;
    PyObject *dspfunction;

    // output pointer
    t_pdpy_objptr *outobjptr;

    // clock
    t_pdpy_clock *clock;

    // in and outs
    t_outlet **outs;
    t_inlet **ins;
    int outletsize;
    int inletsize;
    int siginlets;
    int sigoutlets;

    struct pdpy_proxyinlet *proxy_in;

} t_pdpy_pdobj;

// ─────────────────────────────────────
typedef struct _pdpy_pyclass {
    PyObject_HEAD const char *name;
    PyObject *pyclass;
    t_pdpy_pdobj *pdobj;
    const char *script_name;
    PyObject *outlets;
    PyObject *inlets;
} t_pdpy_pyclass;

// ─────────────────────────────────────
typedef struct pdpy_proxyinlet {
    t_pd pd;
    t_pdpy_pdobj *owner;
    unsigned int id;
} t_pdpy_proxyinlet;

// ─────────────────────────────────────
typedef struct _pdpy_clock {
    PyObject_HEAD PyObject *pyclass;
    PyObject *function;
    t_pdpy_pdobj *pdobj;
    t_clock *clock;
    float delay_time;
} t_pdpy_clock;

// ╭─────────────────────────────────────╮
// │            Declarations             │
// ╰─────────────────────────────────────╯
static void pdpy_proxyinlet_init(t_pdpy_proxyinlet *p, t_pdpy_pdobj *owner, unsigned int id);
static void pdpy_proxy_anything(t_pdpy_proxyinlet *proxy, t_symbol *s, int argc, t_atom *argv);
static void pdpy_printerror(t_pdpy_pdobj *x);
void pdpy_execute(t_pdpy_pdobj *x, const char *methodname, t_symbol *s, int argc, t_atom *argv);

// ─────────────────────────────────────
static PyObject *getoutlets(t_pdpy_pyclass *self) {
    //
    return self->outlets;
}

// ─────────────────────────────────────
static int setoutlets(t_pdpy_pyclass *self, PyObject *value) {
    if (PyLong_Check(value)) {
        self->outlets = value;
        Py_INCREF(value);
        return 0;
    } else if (PyUnicode_Check(value)) {
        self->outlets = value;
        Py_INCREF(value);
        return 0;
    } else if (PyTuple_Check(value)) {
        self->outlets = value;
        Py_INCREF(value);
        return 0;
    }
    PyErr_SetString(PyExc_TypeError, "Auxiliary outlets must be an integer or tuple");
    return -1;
}

// ─────────────────────────────────────
static PyObject *getinlets(t_pdpy_pyclass *self) {
    //
    return self->inlets;
}

static int setinlets(t_pdpy_pyclass *self, PyObject *value) {
    if (PyLong_Check(value)) {
        self->inlets = value;
        Py_INCREF(value);
        return 0;
    } else if (PyUnicode_Check(value)) {
        self->inlets = value;
        Py_INCREF(value);
        return 0;
    } else if (PyTuple_Check(value)) {
        self->inlets = value;
        Py_INCREF(value);
        return 0;
    }

    PyErr_SetString(PyExc_TypeError, "Auxiliary inlets must be an integer or tuple");
    return -1;
}

// ─────────────────────────────────────
static PyGetSetDef pdpy_getset[] = {
    {"outlets", (getter)getoutlets, (setter)setoutlets, "Number of outlets", NULL},
    {"inlets", (getter)getinlets, (setter)setinlets, "Number of inlets", NULL},
    {NULL, NULL, NULL, NULL, NULL} /* Sentinel */
};

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
    self->delay_time = 0;
    // self->clock = clock_new(
    return (PyObject *)self;
}

// ─────────────────────────────────────
static void pdpy_clock_destruct(t_pdpy_clock *self) {
    //
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// ─────────────────────────────────────
static PyObject *pdpy_clock_delay(t_pdpy_clock *self) {
    clock_delay(self->clock, 2000);
    Py_RETURN_TRUE;
}

// ─────────────────────────────────────
static PyMethodDef Clock_methods[] = {
    {"delay", (PyCFunction)pdpy_clock_delay, METH_VARARGS, "Set the delay for the clock"},
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
    char buf[64];
    t_pdpy_objptr *x = (t_pdpy_objptr *)getbytes(sizeof(t_pdpy_objptr));
    x->x_pd = pdpy_pyobjectout_class;
    int ret = pd_snprintf(buf, sizeof(buf), "<%p>", (void *)x);
    if (ret < 0 || ret >= sizeof(buf)) {
        buf[sizeof(buf) - 1] = '\0';
    }
    x->Id = gensym(buf);
    pd_bind((t_pd *)x, x->Id);
    return x;
}

// ─────────────────────────────────────
static PyObject *pdpy_getoutptr(t_symbol *s) {
    t_pdpy_objptr *x = (t_pdpy_objptr *)pd_findbyclass(s, pdpy_pyobjectout_class);
    return x ? x->pValue : NULL;
}

// ─────────────────────────────────────
static PyObject *pypd_newpyclass(t_symbol *s, PyObject *pdmod) {
    // Get the NewObject class from the module
    const char *name = s->s_name;
    PyObject *pd_newobject = PyObject_GetAttrString(pdmod, "NewObject");
    if (!pd_newobject) {
        PyErr_SetString(PyExc_AttributeError, "Could not find 'NewObject' in 'puredata' module");
        PyErr_Print();
        Py_DECREF(pdmod);
        return 0;
    }

    PyObject *subclasses = PyObject_CallMethod(pd_newobject, "__subclasses__", NULL);
    if (!subclasses || !PyList_Check(subclasses)) {
        PyErr_SetString(PyExc_TypeError, "'NewObject.__subclasses__()' did not return a list");
        PyErr_Print();
        Py_DECREF(pd_newobject);
        Py_DECREF(pdmod);
        return 0;
    }

    Py_ssize_t num_subclasses = PyList_Size(subclasses);
    for (Py_ssize_t i = 0; i < num_subclasses; i++) {
        PyObject *subclass = PyList_GetItem(subclasses, i); // Borrowed reference
        if (!subclass) {
            continue;
        }

        PyObject *self = PyObject_CallNoArgs(subclass);
        if (!self) {
            pdpy_printerror(NULL);
            continue;
        }

        return self;
    }
    return NULL;
}

// ─────────────────────────────────────
t_class *pdpyobj_get_class(const char *classname) {
    PyObject *pdmod = PyImport_ImportModule("puredata");
    if (!pdmod) {
        PyErr_SetString(PyExc_ImportError, "Failed to import 'puredata' module");
        return NULL;
    }

    PyObject *obj_dict = PyObject_GetAttrString(pdmod, "_objects");
    if (!obj_dict || !PyDict_Check(obj_dict)) {
        PyErr_SetString(PyExc_RuntimeError, "Missing or invalid '_objects' dictionary");
        Py_XDECREF(obj_dict);
        Py_DECREF(pdmod);
        return NULL;
    }

    // Get the capsule from the dictionary
    PyObject *capsule = PyDict_GetItemString(obj_dict, classname);
    if (!capsule) {
        PyErr_SetString(PyExc_KeyError, "Class not found in _objects");
        Py_DECREF(obj_dict);
        Py_DECREF(pdmod);
        return NULL;
    }

    // Extract the t_class pointer from the capsule
    t_class *pdclass = (t_class *)PyCapsule_GetPointer(capsule, NULL);
    if (!pdclass) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid capsule for pdclass");
        Py_DECREF(obj_dict);
        Py_DECREF(pdmod);
        return NULL;
    }

    // Cleanup references (do NOT decrement the capsule—it's borrowed)
    Py_DECREF(obj_dict);
    Py_DECREF(pdmod);

    return pdclass;
}

// ─────────────────────────────────────
static void pdpy_inlets(t_pdpy_pdobj *x) {
    if (PyLong_Check(x->pyclass->inlets)) {
        int count = PyLong_AsLong(x->pyclass->inlets);
        x->ins = (t_inlet **)malloc((count) * sizeof(t_inlet *));
        x->proxy_in = (t_pdpy_proxyinlet *)malloc((count) * sizeof(t_pdpy_proxyinlet));
        for (int i = 0; i < count; i++) {
            pdpy_proxyinlet_init(&x->proxy_in[i], x, i + 1);
            x->ins[i] = inlet_new(&x->obj, &x->proxy_in[i].pd, 0, 0);
        }
        return;
    } else if (PyTuple_Check(x->pyclass->inlets)) {
        int count = PyTuple_Size(x->pyclass->inlets);
        x->ins = (t_inlet **)malloc(count * sizeof(t_inlet *));
        x->proxy_in = (t_pdpy_proxyinlet *)malloc((count) * sizeof(t_pdpy_proxyinlet));

        // NOTE:I think that on signal objects, the first inlets are always signals
        x->siginlets = 0;
        for (int i = 0; i < count; i++) {
            PyObject *config = PyTuple_GetItem(x->pyclass->inlets, i);
            if (config == NULL) {
                PyErr_SetString(PyExc_TypeError, "Invalid outlet type");
                continue;
            }
            const char *outtype = PyUnicode_AsUTF8(config);
            t_symbol *sym = strcmp(outtype, "anything") ? &s_signal : 0;
            pdpy_proxyinlet_init(&x->proxy_in[i], x, i + 1);
            x->ins[i] = inlet_new(&x->obj, &x->proxy_in[i].pd, sym, sym);
            if (strcmp(outtype, "signal") == 0) {
                x->siginlets++;
            }
        }
        return;
    } else if (PyUnicode_Check(x->pyclass->inlets)) {
        const char *outtype = PyUnicode_AsUTF8(x->pyclass->inlets);
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
    if (PyLong_Check(x->pyclass->outlets)) {
        int count = PyLong_AsLong(x->pyclass->outlets);
        x->outs = (t_outlet **)malloc(count * sizeof(t_outlet *));
        for (int i = 0; i < count; i++) {
            x->outs[i] = outlet_new(&x->obj, &s_anything);
        }
        x->outletsize = count;
        return;
    } else if (PyUnicode_Check(x->pyclass->outlets)) {
        const char *sym = PyUnicode_AsUTF8(x->pyclass->outlets);
        x->outs = (t_outlet **)malloc(1 * sizeof(t_outlet *));
        if (strcmp(sym, "signal") == 0) {
            x->outs[0] = outlet_new(&x->obj, &s_signal);
            x->sigoutlets++;
        } else {
            x->outs[0] = outlet_new(&x->obj, &s_anything);
        }
        x->outletsize = 1;
        return;
    } else if (PyTuple_Check(x->pyclass->outlets)) {
        int count = PyTuple_Size(x->pyclass->outlets);
        x->outs = (t_outlet **)malloc(count * sizeof(t_outlet *));
        x->sigoutlets = 0;
        for (int i = 0; i < count; i++) {
            PyObject *config = PyTuple_GetItem(x->pyclass->outlets, i);
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
    t_class *pyobj = pdpyobj_get_class(s->s_name);
    t_pdpy_pdobj *x = (t_pdpy_pdobj *)pd_new(pyobj);
    PyObject *pdmod = PyImport_ImportModule("puredata");
    if (pdmod == NULL) {
        PyErr_SetString(PyExc_ImportError, "Failed to import 'puredata' module");
        return NULL;
    }

    PyObject *pyclass = pypd_newpyclass(s, pdmod);
    t_pdpy_pyclass *self = (t_pdpy_pyclass *)pyclass;
    if (self == NULL) {
        pd_error(NULL, "Failed to create object");
        return NULL;
    }

    x->pyclass = self;
    self->pdobj = x;
    pdpy_inlets(x);
    pdpy_outlets(x);

    x->outobjptr = pdpy_createoutputptr();
    return (void *)x;
}

// ─────────────────────────────────────
static void py4pdobj_free(t_pdpy_pdobj *x) {
    free(x->outs);
    free(x->ins);
    free(x->proxy_in);
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
            Py_DECREF(pstr);
        }
    }
    Py_XDECREF(ptype);
    Py_XDECREF(pvalue);
    Py_XDECREF(ptraceback);
    PyErr_Clear();
}

// ─────────────────────────────────────
void pdpy_execute(t_pdpy_pdobj *x, const char *methodname, t_symbol *s, int argc, t_atom *argv) {
    if (x->pyclass == NULL) {
        post("pyclass is NULL");
        return;
    }

    PyObject *pyclass = (PyObject *)x->pyclass;
    PyObject *method = PyObject_GetAttrString(pyclass, methodname);
    if (!method || !PyCallable_Check(method)) {
        PyErr_Clear();
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
            return;
        }
        pValue = PyObject_CallOneArg(method, pArg);
    } else if (strcmp(s->s_name, "symbol") == 0) {
        t_symbol *sym = atom_getsymbolarg(0, argc, argv); // Rename to avoid shadowing
        pArg = PyUnicode_FromString(sym->s_name);
        if (!pArg) {
            return;
        }
        pValue = PyObject_CallOneArg(method, pArg);
    } else if (strcmp(s->s_name, "list") == 0) {
        pArg = py4pdobj_converttopy(argc, argv);
        if (!pArg) {
            return;
        }
        pValue = PyObject_CallOneArg(method, pArg);
    } else {
        PyObject *pTuple = PyTuple_New(1);
        pArg = py4pdobj_converttopy(argc, argv);
        if (!pArg) {
            return;
        }
        PyTuple_SetItem(pTuple, 0, pArg);
        pValue = PyObject_CallObject(method, pTuple);
    }

    // Check for Python call errors
    if (pValue == NULL) {
        pdpy_printerror(x);
    }

    // Cleanup
    Py_XDECREF(pArg);
    Py_XDECREF(pValue);
}

// ─────────────────────────────────────
void pdpy_proxy_anything(t_pdpy_proxyinlet *proxy, t_symbol *s, int argc, t_atom *argv) {
    t_pdpy_pdobj *o = proxy->owner;
    char methodname[MAXPDSTRING];
    pd_snprintf(methodname, MAXPDSTRING, "in_%d_%s", proxy->id, s->s_name);
    pdpy_execute(o, methodname, s, argc, argv);
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
    return;
}

// ─────────────────────────────────────
static t_int *pdpy_perform(t_int *w) {
    t_pdpy_pdobj *x = (t_pdpy_pdobj *)(w[1]);
    int n = (int)w[2];
    PyObject *py_in = PyTuple_New(x->siginlets);
    t_sample *in;

    // Converter vetores de entrada t_sample em tuplas Python
    for (int i = 0; i < x->siginlets; i++) {
        in = (t_sample *)w[3 + i]; // Ajuste do índice para capturar os ponteiros corretos
        PyObject *py_list = PyList_New(n);
        for (int j = 0; j < n; j++) {
            PyList_SetItem(py_list, j, PyFloat_FromDouble(in[j]));
        }
        PyTuple_SetItem(py_in, i, py_list);
    }

    // Chamada da função DSP em Python
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
                PyErr_SetString(PyExc_RuntimeError, "Return value is unknown!");
                pdpy_printerror(x);
            }
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Return value is unknown!");
            pdpy_printerror(x);
        }
    } else {
        PyErr_SetString(PyExc_RuntimeError, "Return value is not a tuple");
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
    PyObject *method = PyObject_GetAttrString(pyclass, "dsp_perform");
    if (!method || !PyCallable_Check(method)) {
        PyErr_Clear();
        pd_error(x, "[%s] no dsp_perform method defined or not callable",
                 x->obj.te_g.g_pd->c_name->s_name);
        return;
    }

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

// ─────────────────────────────────────
int pd4pd_loader_wrappath(int fd, const char *name, const char *dirbuf) {
    char fullpath[1024];
    if (!dirbuf || !name) {
        pd_error(NULL, "Error: dirbuf or name is NULL\n");
        return 0;
    }
    pd_snprintf(fullpath, sizeof(fullpath), "%s/%s.pd_py", dirbuf, name);
    FILE *file = fopen(fullpath, "r");
    if (file == NULL) {
        PyErr_SetString(PyExc_ImportError, "Failed to open module file");
        return 0;
    }

    // Run the Python file
    if (PyRun_SimpleFile(file, fullpath) != 0) {
        pdpy_printerror(NULL);
        fclose(file);
        return 0;
    }

    PyObject *pdmod = PyImport_ImportModule("puredata");
    if (!pdmod) {
        PyErr_SetString(PyExc_ImportError, "Could not import module 'puredata'");
        PyErr_Print();
        return 0;
    }

    // Get the NewObject class from the module
    PyObject *pd_newobject = PyObject_GetAttrString(pdmod, "NewObject");
    if (!pd_newobject) {
        PyErr_SetString(PyExc_AttributeError, "Could not find 'NewObject' in 'puredata' module");
        PyErr_Print();
        Py_DECREF(pdmod);
        return 0;
    }

    PyObject *subclasses = PyObject_CallMethod(pd_newobject, "__subclasses__", NULL);
    if (!subclasses || !PyList_Check(subclasses)) {
        PyErr_SetString(PyExc_TypeError, "'NewObject.__subclasses__()' did not return a list");
        PyErr_Print();
        Py_DECREF(pd_newobject);
        Py_DECREF(pdmod);
        return 0;
    }

    Py_ssize_t num_subclasses = PyList_Size(subclasses);
    for (Py_ssize_t i = 0; i < num_subclasses; i++) {
        PyObject *subclass = PyList_GetItem(subclasses, i); // Borrowed reference
        if (!subclass) {
            continue;
        }

        PyObject *objname = PyObject_GetAttrString(subclass, "name");
        if (!objname) {
            pd_error(NULL,
                     "[py4pd] not possible to read %s inside %s, 'name' class attribute is missing",
                     name, dirbuf);
            continue;
        }

        const char *objstring = PyUnicode_AsUTF8(objname);
        if (strcmp(objstring, name) == 0) {

            bool havedsp = false;
            PyObject *dsp = PyObject_GetAttrString(subclass, "dsp");
            if (dsp) {
                havedsp = PyObject_IsTrue(dsp);
            } else {
                havedsp = false;
            }

            // python object methods
            t_class *pdclass =
                class_new(gensym(name), (t_newmethod)pdpy_new, (t_method)py4pdobj_free,
                          sizeof(t_pdpy_pdobj), CLASS_NOINLET | CLASS_MULTICHANNEL, A_GIMME, 0);
            class_addmethod(pdclass, (t_method)pdpy_menu_open, gensym("menu-open"), A_NULL);
            class_addmethod(pdclass, (t_method)pdpy_pyobject, gensym("PyObject"), A_SYMBOL,
                            A_SYMBOL, 0);
            class_addanything(pdclass, pdpy_anything);
            if (havedsp) {
                class_addmethod(pdclass, (t_method)pdpy_dsp, gensym("dsp"), A_CANT, 0);
            }

            // save t_class on globals of puredata module
            char script[MAXPDSTRING];
            pd_snprintf(script, MAXPDSTRING, "%s/%s", dirbuf, name);
            pdclass->c_externdir = gensym(script);
            PyObject *pdmod = PyImport_ImportModule("puredata");
            if (!pdmod) {
                PyErr_SetString(PyExc_ImportError, "Could not import module 'puredata'");
                return -1;
            }
            PyObject *obj_dict = PyObject_GetAttrString(pdmod, "_objects");
            if (!obj_dict || !PyDict_Check(obj_dict)) {
                Py_XDECREF(obj_dict);
                obj_dict = PyDict_New();
                if (!obj_dict) {
                    PyErr_SetString(PyExc_MemoryError, "Failed to create _objects dictionary");
                    Py_DECREF(pdmod);
                    return -1;
                }

                // Set _objects in puredata module
                if (PyObject_SetAttrString(pdmod, "_objects", obj_dict) < 0) {
                    PyErr_SetString(PyExc_RuntimeError,
                                    "Failed to set _objects in puredata module");
                    Py_DECREF(obj_dict);
                    Py_DECREF(pdmod);
                    return -1;
                }
            }

            // Create a capsule to hold the pdclass pointer
            PyObject *capsule = PyCapsule_New((void *)pdclass, NULL, NULL);
            if (!capsule) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to create capsule for pdclass");
                Py_DECREF(obj_dict);
                Py_DECREF(pdmod);
                return -1;
            }

            // Add the capsule to the _objects dictionary using the class name as the key
            if (PyDict_SetItemString(obj_dict, name, capsule) < 0) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to store pdclass in _objects");
                Py_DECREF(capsule);
                Py_DECREF(obj_dict);
                Py_DECREF(pdmod);
                return -1;
            }

            // Cleanup references
            Py_DECREF(capsule);
            Py_DECREF(obj_dict);
            Py_DECREF(pdmod);
            return 1;
        }
    }
    return 0;
}

// ─────────────────────────────────────
static int pdpyobj_init(t_pdpy_pyclass *self, PyObject *args) {
    char *objname;

    if (!PyArg_ParseTuple(args, "s", &objname)) {
        return -1;
    }

    // python object methods
    t_class *pdclass = class_new(gensym(objname), (t_newmethod)pdpy_new, (t_method)py4pdobj_free,
                                 sizeof(t_pdpy_pyclass), 0, A_GIMME, 0);
    class_addmethod(pdclass, (t_method)pdpy_menu_open, gensym("menu-open"), A_NULL);
    class_addanything(pdclass, pdpy_anything);

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
        if (!obj_dict) {
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
    PyObject *capsule = PyCapsule_New((void *)pdclass, NULL, NULL);
    if (!capsule) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create capsule for pdclass");
        Py_DECREF(obj_dict);
        Py_DECREF(pdmod);
        return -1;
    }

    // Add the capsule to the _objects dictionary using the class name as the key
    if (PyDict_SetItemString(obj_dict, objname, capsule) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to store pdclass in _objects");
        Py_DECREF(capsule);
        Py_DECREF(obj_dict);
        Py_DECREF(pdmod);
        return -1;
    }

    // Cleanup references
    Py_DECREF(capsule);
    Py_DECREF(obj_dict);
    Py_DECREF(pdmod);

    return 0;
}
// ─────────────────────────────────────
static void pdpy_clock_execute(t_pdpy_clock *x) {
    if (x->function == NULL) {
        pd_error(x->pdobj, "Clock function is NULL");
        return;
    }

    if (!PyCallable_Check(x->function)) {
        pd_error(x->pdobj, "Clock function is not callable");
        return;
    }

    // PyObject_CallNoArgs(x->function);

    return;
}

// ─────────────────────────────────────
static PyObject *pdpy_new_clock(PyObject *self, PyObject *args) {
    if (PyType_Ready(&ClockType) < 0) {
        PyErr_SetString(PyExc_TypeError, "ClockType not ready");
        return NULL;
    }
    // get pyfunction to be executed
    PyObject *func;
    if (!PyArg_ParseTuple(args, "O", &func)) {
        return NULL;
    }

    if (!PyCallable_Check(func)) {
        return NULL;
    }

    t_pdpy_clock *clock_instance =
        (t_pdpy_clock *)PyObject_CallObject((PyObject *)&ClockType, NULL);
    if (!clock_instance) {
        return NULL;
    }
    clock_instance->pyclass = self;
    clock_instance->function = func;
    clock_instance->pdobj = ((t_pdpy_pyclass *)self)->pdobj;
    clock_instance->clock = clock_new(clock_instance->pdobj, (t_method)pdpy_clock_execute);
    return (PyObject *)clock_instance;
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

                // Check if adding this string would exceed MAXPDSTRING - 4
                if (msg_len + str_len + 4 >= MAXPDSTRING) {
                    strcat(msg, "...");
                    msg_len += 3;
                    Py_DECREF(str_obj);
                    break;
                }

                strcat(msg, str);
                msg_len += str_len;

                if (i < num_args - 1 && msg_len + 1 < MAXPDSTRING) {
                    strcat(msg, " ");
                    msg_len++;
                }
            }

            Py_DECREF(str_obj);
        }
    }
    pd_error(self->pdobj, "%s", msg);
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
            if (str) {
                size_t str_len = strlen(str);

                // Check if adding this string would exceed MAXPDSTRING - 4
                if (msg_len + str_len + 4 >= MAXPDSTRING) {
                    strcat(msg, "...");
                    msg_len += 3;
                    Py_DECREF(str_obj);
                    break;
                }

                strcat(msg, str);
                msg_len += str_len;

                if (i < num_args - 1 && msg_len + 1 < MAXPDSTRING) {
                    strcat(msg, " ");
                    msg_len++;
                }
            }
            Py_DECREF(str_obj);
        }
    }
    logpost(self->pdobj, loglevel, "%s", msg);
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
        SETSYMBOL(&args[1], x->outobjptr->Id);
        outlet_anything(out, gensym("PyObject"), 2, args);
        Py_RETURN_TRUE;
    } else if (pdtype == A_FLOAT) {
        PyObject *f = PyNumber_Float(pValue);
        t_float v = PyFloat_AsDouble(f);
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
            PyErr_SetString(PyExc_TypeError, "Output with pd.GIMME require a list output");
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
static PyMethodDef pdpy_methods[] = {
    {"logpost", (PyCFunction)pdpy_logpost, METH_VARARGS, "Post things on PureData console"},
    {"error", (PyCFunction)pdpy_error, METH_VARARGS,
     "Print error, same as logpost with error level"},

    {"out", (PyCFunction)pdpy_out, METH_VARARGS | METH_KEYWORDS, "Post things on PureData console"},

    {"new_clock", (PyCFunction)pdpy_new_clock, METH_VARARGS, "Return a clock object"},

    // arrays
    {"tabwrite", (PyCFunction)pdpy_tabwrite, METH_VARARGS | METH_KEYWORDS,
     "This read the content of a Pd Array"},
    {"tabread", (PyCFunction)pdpy_tabread, METH_VARARGS | METH_KEYWORDS,
     "This read the content of a Pd Array"},

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
    .tp_getset = pdpy_getset,
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
        }
        startpost("\n");
    }

    Py_RETURN_TRUE;
}

// ╭─────────────────────────────────────╮
// │             MODULE INIT             │
// ╰─────────────────────────────────────╯
PyMethodDef pdpy_modulemethods[] = {
    {"post", pdpy_post, METH_VARARGS, "Print informations in PureData Console"},
    {NULL, NULL, 0, NULL} //
};

// ─────────────────────────────────────
static PyObject *pd4pdmodule_init(PyObject *self) {
    if (PyType_Ready(&pdpy_type) < 0) {
        return NULL;
    }

    PyModule_AddObject(self, "FLOAT", PyLong_FromLong(A_FLOAT));
    PyModule_AddObject(self, "SYMBOL", PyLong_FromLong(A_SYMBOL));
    PyModule_AddObject(self, "GIMME", PyLong_FromLong(A_GIMME));
    PyModule_AddObject(self, "PYOBJECT", PyLong_FromLong(PYOBJECT));

    PyModule_AddObject(self, "SIGNAL", PyUnicode_FromString(s_signal.s_name));
    PyModule_AddObject(self, "DATA", PyUnicode_FromString(s_anything.s_name));

    // NewObject
    Py_INCREF(&pdpy_type);
    int r = PyModule_AddObject(self, "NewObject", (PyObject *)&pdpy_type);
    if (r != 0) {
        // TODO: clear memory
        PyErr_SetString(PyExc_RuntimeError, "Failed to add NewObject to module");
        return NULL;
    }

    PyObject *obj_dict = PyDict_New();
    r = PyModule_AddObject(self, "_objects", obj_dict);
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
static struct PyModuleDef pdpy_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "puredata",
    .m_doc = "pd module provide function to interact with PureData, see the "
             "docs in www.charlesneimog.github.io/py4pd",
    .m_size = 0,
    .m_methods = pdpy_modulemethods, // Methods of the module
    .m_slots = pdpy_moduleslots,
};

// ─────────────────────────────────────
PyMODINIT_FUNC PyInit_pd() {
    // TODO: I will remove numpy
    PyObject *m;
    m = PyModuleDef_Init(&pdpy_module);
    if (m == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create module");
        return NULL;
    }

    return m;
}
