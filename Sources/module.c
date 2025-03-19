#include <m_pd.h>

#include <g_canvas.h>
#include <m_imp.h>
#include <s_stuff.h>

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL PY4PD_NUMPYARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_25_API_VERSION
#include <numpy/arrayobject.h>

typedef struct _py4pd_obj py4pd_obj;
static t_class *pdpy_proxyinlet_class = NULL;

// ╭─────────────────────────────────────╮
// │          Object Base Class          │
// ╰─────────────────────────────────────╯
typedef struct _pyobj {
    t_object obj;
    py4pd_obj *pyclass;
    t_symbol *script;
    t_canvas *canvas;

    // in and outs
    t_outlet **outs;
    t_inlet **ins;
    struct pdpy_proxyinlet *proxy_in;
} t_pyobj;

typedef struct _py4pd_obj {
    PyObject_HEAD const char *name;
    t_class *pdclass;
    PyObject *pyclass;
    t_pyobj *pyobj;
    const char *script_name;
    int outlets;
    int inlets;
} py4pd_obj;

typedef struct pdpy_proxyinlet {
    t_pd pd;
    t_pyobj *owner;
    unsigned int id;
} t_pdpy_proxyinlet;

// Declarations
static void pdpy_proxyinlet_init(t_pdpy_proxyinlet *p, t_pyobj *owner, unsigned int id);
static void pdpy_proxy_anything(t_pdpy_proxyinlet *proxy, t_symbol *s, int argc, t_atom *argv);

// ─────────────────────────────────────
static PyObject *getoutlets(py4pd_obj *self) { return PyLong_FromLong(self->outlets); }
static int setoutlets(py4pd_obj *self, PyObject *value) {
    if (!PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Auxiliary outlets must be an integer");
        return -1;
    }
    self->outlets = PyLong_AsLong(value);
    return 0;
}

// ─────────────────────────────────────
static PyObject *getinlets(py4pd_obj *self) { return PyLong_FromLong(self->inlets); }
static int setinlets(py4pd_obj *self, PyObject *value) {
    if (!PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Auxiliary inlets must be an integer");
        return -1;
    }
    self->inlets = PyLong_AsLong(value);
    return 0;
}

// ─────────────────────────────────────
static PyGetSetDef pdpy_GetSet[] = {
    {"outlets", (getter)getoutlets, (setter)setoutlets, "Number of outlets", NULL},
    {"inlets", (getter)getinlets, (setter)setinlets, "Number of inlets", NULL},
    {NULL, NULL, NULL, NULL, NULL} /* Sentinel */
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
void pdpy_proxyinlet_setup(void) {
    pdpy_proxyinlet_class = class_new(gensym("_pdpy"), 0, 0, sizeof(t_pdpy_proxyinlet), 0, 0);
    if (pdpy_proxyinlet_class) {
        class_addanything(pdpy_proxyinlet_class, pdpy_proxy_anything);
    }
}

// ─────────────────────────────────────
static void *pdpy_new(t_symbol *s, int argc, t_atom *argv) {
    PyObject *pdmod = PyImport_ImportModule("puredata");
    if (!pdmod) {
        PyErr_SetString(PyExc_ImportError, "Could not import module 'puredata'");
        pd_error(NULL, "[%s], Could not import module 'puredata'", s->s_name);
        return NULL;
    }
    PyObject *obj_dict = PyObject_GetAttrString(pdmod, "_objects");
    if (!obj_dict || !PyDict_Check(obj_dict)) {
        PyErr_SetString(PyExc_AttributeError, "puredata._objects is missing or not a dictionary");
        Py_XDECREF(obj_dict);
        Py_DECREF(pdmod);
        pd_error(NULL, "[%s], Failed to get puredata._objects", s->s_name);
        return NULL;
    }

    PyObject *capsule = PyDict_GetItemString(obj_dict, s->s_name);
    if (!capsule) {
        PyErr_SetString(PyExc_KeyError, "Key not found in puredata._objects");
        pd_error(NULL, "[%s], Key not found in puredata._objects", s->s_name);
        Py_DECREF(obj_dict);
        Py_DECREF(pdmod);
        return NULL;
    }

    py4pd_obj *self = (py4pd_obj *)PyCapsule_GetPointer(capsule, s->s_name);
    if (!self) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to extract C pointer from capsule");
        pd_error(NULL, "[%s], Failed to find C pointer of the python object", s->s_name);
        return NULL;
    }

    //
    t_pyobj *x = (t_pyobj *)pd_new(self->pdclass);
    x->pyclass = self;
    self->pyobj = x;

    x->outs = (t_outlet **)malloc(self->outlets * sizeof(t_outlet *));
    for (int i = 0; i < self->outlets; i++) {
        x->outs[i] = outlet_new(&x->obj, &s_anything);
    }

    x->ins = (t_inlet **)malloc(self->inlets * sizeof(t_inlet *));
    x->proxy_in = (t_pdpy_proxyinlet *)malloc(self->inlets * sizeof(t_pdpy_proxyinlet));
    for (int i = 1; i < self->inlets; i++) {
        pdpy_proxyinlet_init(&x->proxy_in[i], x, i + 1);
        x->ins[i] = inlet_new(&x->obj, &x->proxy_in[i].pd, 0, 0);
    }

    Py_DECREF(obj_dict);
    Py_DECREF(pdmod);
    return (void *)x;
}
// ─────────────────────────────────────
static void py4pdobj_free(t_pyobj *x) {
    free(x->outs);
    free(x->ins);
    // free(x->proxy_in);
}

// ─────────────────────────────────────
void posterror(t_pyobj *x) {
    PyErr_Print();
    PyErr_Clear();
}

// ─────────────────────────────────────
void pdpy_execute(t_pyobj *x, const char *methodname, t_symbol *s, int argc, t_atom *argv) {
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
        PyObject *pValue = PyObject_CallObject(method, pTuple);
    }

    // Check for Python call errors
    if (!pValue) {
        PyErr_Print();
    }

    // Cleanup
    Py_XDECREF(pArg);
    Py_XDECREF(pValue);
}

// ─────────────────────────────────────
void pdpy_proxy_anything(t_pdpy_proxyinlet *proxy, t_symbol *s, int argc, t_atom *argv) {
    t_pyobj *o = proxy->owner;
    char methodname[MAXPDSTRING];
    pd_snprintf(methodname, MAXPDSTRING, "in_%d_%s", proxy->id, s->s_name);
    pdpy_execute(o, methodname, s, argc, argv);
    return;
}

// ─────────────────────────────────────
static void pdpy_anything(t_pyobj *o, t_symbol *s, int argc, t_atom *argv) {
    char methodname[MAXPDSTRING];
    pd_snprintf(methodname, MAXPDSTRING, "in_1_%s", s->s_name);
    pdpy_execute(o, methodname, s, argc, argv);
}

// ─────────────────────────────────────
static void pdpy_menu_open(t_pyobj *o) {
    // TODO:
    post("script is %s", o->script->s_name);
    return;
}

// ─────────────────────────────────────
static void pdpy_proxyinlet_init(t_pdpy_proxyinlet *p, t_pyobj *owner, unsigned int id) {
    p->pd = pdpy_proxyinlet_class;
    p->owner = owner;
    p->id = id;
}

// ─────────────────────────────────────
static int pdpyobj_init(py4pd_obj *self, PyObject *args, PyObject *kwds) {
    static char *keywords[] = {"param_str", NULL};
    char *param_str;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", keywords, &param_str)) {
        return -1;
    }

    // Create a Unicode string for the object name
    PyObject *objectName = PyUnicode_FromString(param_str);
    if (!objectName) {
        PyErr_SetString(PyExc_TypeError, "Object name not valid");
        return -1;
    }
    self->name = PyUnicode_AsUTF8(objectName);
    self->pdclass = class_new(gensym(self->name), (t_newmethod)pdpy_new, (t_method)py4pdobj_free,
                              sizeof(py4pd_obj), 0, A_GIMME, 0);

    class_addmethod(self->pdclass, (t_method)pdpy_menu_open, gensym("menu-open"), A_NULL);
    class_addanything(self->pdclass, pdpy_anything);

    // Import the 'puredata' module
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

    // Create a PyCapsule to store 'self' (C struct pointer) in the dictionary
    PyObject *capsule = PyCapsule_New(self, self->name, NULL); // Use param_str as the capsule name
    if (!capsule) {
        Py_DECREF(obj_dict);
        Py_DECREF(pdmod);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create capsule for object");
        return -1;
    }

    // Store the capsule in the _objects dictionary with the key 'param_str'
    if (PyDict_SetItemString(obj_dict, self->name, capsule) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to store object in _objects");
        Py_DECREF(capsule);
        Py_DECREF(obj_dict);
        Py_DECREF(pdmod);
        return -1;
    }

    PyObject *globals = PyEval_GetGlobals();
    PyObject *script_file = PyDict_GetItemString(globals, "__file__");
    if (script_file && PyUnicode_Check(script_file)) {
        const char *script_name = PyUnicode_AsUTF8(script_file);
        self->script_name = strdup(script_name); // Save the script path
    }

    // Clean up the references to objects
    Py_DECREF(capsule);
    Py_DECREF(obj_dict);
    Py_DECREF(pdmod);
    Py_INCREF(self);
    logpost(NULL, 4, "Object created: %s", self->name);

    return 0;
}

// ─────────────────────────────────────
static PyObject *pdpy_logpost(py4pd_obj *self, PyObject *args) {
    //
    Py_RETURN_TRUE;
}

// ─────────────────────────────────────
static PyMethodDef pdpy_methods[] = {
    {"logpost", (PyCFunction)pdpy_logpost, METH_NOARGS, "Post things on PureData console"},
    {NULL, NULL, 0, NULL} // Sentinel
};

// ─────────────────────────────────────
PyTypeObject pdpy_type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "NewObject",
    .tp_doc = "It creates new PureData objects",
    .tp_basicsize = sizeof(py4pd_obj),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // Allow inheritance
    .tp_methods = pdpy_methods,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)pdpyobj_init,
    .tp_getset = pdpy_GetSet,
};

// ─────────────────────────────────────
static PyObject *pdpy_post(PyObject *self, PyObject *args) {
    (void)self;
    char *string;

    if (PyArg_ParseTuple(args, "s", &string)) {
        post("%s", string);
        return PyLong_FromLong(0);
    } else {
        PyErr_SetString(PyExc_TypeError, "pd.post works with strings and numbers.");
        return NULL;
    }
}

// ╭─────────────────────────────────────╮
// │             MODULE INIT             │
// ╰─────────────────────────────────────╯
PyMethodDef PdMethods[] = {
    {"post", pdpy_post, METH_VARARGS, "Print informations in PureData Console"},
    {NULL, NULL, 0, NULL} //
};

// ─────────────────────────────────────
static PyObject *pd4pdmodule_init(PyObject *self) {
    if (PyType_Ready(&pdpy_type) < 0) {
        return NULL;
    }
    Py_INCREF(&pdpy_type);
    PyModule_AddObject(self, "NewObject", (PyObject *)&pdpy_type);
    return 0;
}

// ─────────────────────────────────────
static PyModuleDef_Slot pdmodule_slots[] = { //
    {Py_mod_exec, pd4pdmodule_init},
    {0, NULL}};

// ─────────────────────────────────────
static struct PyModuleDef pdModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "puredata",
    .m_doc = "pd module provide function to interact with PureData, see the "
             "docs in www.charlesneimog.github.io/py4pd",
    .m_size = 0,
    .m_methods = PdMethods, // Methods of the module
    .m_slots = pdmodule_slots,
};

// ─────────────────────────────────────
PyMODINIT_FUNC PyInit_pd() {
    import_array() PyObject *py4pdModule;
    py4pdModule = PyModuleDef_Init(&pdModule);
    if (py4pdModule == NULL) {
        return NULL;
    }

    return py4pdModule;
}
