#include "ext-class.h"
#include "object.h"
#include "py4pd.h"
#include "utils.h"

static t_class *InletsExtClassProxy;

// ====================================================
// ===================== Obj Methods ==================
// ====================================================
static void Py4pdNewObj_PdExecFloatMethod(t_py *x, t_float f) {
    (void)f;
    (void)x;
    Py4pdNewObj *pObjSelf;

    pObjSelf = (Py4pdNewObj *)x->objClass;
    PyObject *pFunc = (PyObject *)pObjSelf->pFuncFloat;
    PyObject *pArgs = PyTuple_New(0);

    (void)PyObject_CallObject(pFunc, pArgs);
}

// ====================================================
static void Py4pdNewObj_PdExecSymbolMethod(t_py *x, t_float f) {
    (void)f;
    (void)x;
    Py4pdNewObj *pObjSelf;

    pObjSelf = (Py4pdNewObj *)x->objClass;
    PyObject *pFunc = (PyObject *)pObjSelf->pFuncSymbol;
    PyObject *pArgs = PyTuple_New(0);

    (void)PyObject_CallObject(pFunc, pArgs);
}

// ====================================================
static void Py4pdNewObj_PdExecListMethod(t_py *x, t_symbol *s, int argc,
                                         t_atom *argv) {
    (void)x;
    Py4pdNewObj *pObjSelf;

    pObjSelf = (Py4pdNewObj *)x->objClass;
    PyObject *pFunc = (PyObject *)pObjSelf->pFuncList;
    PyObject *pArgs = PyTuple_New(0);

    (void)PyObject_CallObject(pFunc, pArgs);
}

// ====================================================
static void Py4pdNewObj_PdExecAnythingMethod(t_py *x, t_symbol *s, int argc,
                                             t_atom *argv) {
    (void)x;
    Py4pdNewObj *pObjSelf;

    pObjSelf = (Py4pdNewObj *)x->objClass;
    PyObject *pFunc = (PyObject *)pObjSelf->pFuncAnything;
    PyObject *pArgs = PyTuple_New(0);

    (void)PyObject_CallObject(pFunc, pArgs);
}

// ====================================================
static void Py4pdNewObj_PdExecBangMethod(t_py *x) {
    Py4pdNewObj *pObjSelf;

    pObjSelf = (Py4pdNewObj *)x->objClass;
    PyObject *pFunc = (PyObject *)pObjSelf->pFuncBang;
    PyObject *pArgs = PyTuple_New(0);

    (void)PyObject_CallObject(pFunc, pArgs);
}

// ====================================================
static void Py4pdNewObj_PdExecSelectorMethod(t_py *x, t_symbol *selector,
                                             int argc, t_atom *argv) {
    Py4pdNewObj *pObjSelf;

    pObjSelf = (Py4pdNewObj *)x->objClass;
    PyObject *pSelector = PyUnicode_FromString(selector->s_name);

    // get the Dict
    PyObject *pDictSelectors = (PyObject *)pObjSelf->pDictSelectors;
    PyObject *pFunc = PyDict_GetItem(pDictSelectors, pSelector);

    PyObject *pArgs = PyTuple_New(0);

    (void)PyObject_CallObject(pFunc, pArgs);
}

// ====================================================
// ===================== Create Obj ===================
// ====================================================
void *Py4pdNewObj_NewObj(t_symbol *s, int argc, t_atom *argv) {
    const char *objectName = s->s_name;
    char py4pd_objectName[MAXPDSTRING];
    snprintf(py4pd_objectName, sizeof(py4pd_objectName), "py4pd_ObjectDict_%s",
             objectName);
    PyObject *pd_module = PyImport_ImportModule("pd");

    if (pd_module == NULL) {
        pd_error(NULL,
                 "[py4pd] Not possible import the pd module, please report!");
        return NULL;
    }

    PyObject *py4pd_capsule =
        PyObject_GetAttrString(pd_module, py4pd_objectName);
    PyObject *PdDictCapsule = PyCapsule_GetPointer(py4pd_capsule, objectName);
    if (PdDictCapsule == NULL) {
        pd_error(NULL, "Error: PdDictCapsule is NULL, please report!");
        Py_DECREF(pd_module);
        Py_DECREF(py4pd_capsule);
        return NULL;
    }
    PyObject *PdDict = PyDict_GetItemString(PdDictCapsule, objectName);
    if (PdDict == NULL) {
        pd_error(NULL, "Error: PdDict is NULL");
        Py_DECREF(pd_module);
        Py_DECREF(py4pd_capsule);
        return NULL;
    }

    PyObject *PY_objectClass = PyDict_GetItemString(PdDict, "py4pdOBJ_CLASS");
    if (PY_objectClass == NULL) {
        pd_error(NULL, "Error: object Class is NULL");
        Py_DECREF(pd_module);
        Py_DECREF(py4pd_capsule);
        return NULL;
    }

    PyObject *ignoreOnNone = PyDict_GetItemString(PdDict, "py4pdOBJIgnoreNone");
    PyObject *playable = PyDict_GetItemString(PdDict, "py4pdOBJPlayable");
    PyObject *pyOUT = PyDict_GetItemString(PdDict, "py4pdOBJpyout");
    PyObject *nooutlet = PyDict_GetItemString(PdDict, "py4pdOBJnooutlet");
    PyObject *AuxOutletPy = PyDict_GetItemString(PdDict, "py4pdAuxOutlets");
    PyObject *RequireUserToSetOutletNumbers =
        PyDict_GetItemString(PdDict, "py4pdOBJrequireoutletn");
    PyObject *Py_ObjType = PyDict_GetItemString(PdDict, "py4pdOBJType");
    PyObject *pyFunction = PyDict_GetItemString(PdDict, "py4pdOBJFunction");
    PyObject *PyStructSelf = PyDict_GetItemString(PdDict, "objSelf");

    int AuxOutlet = PyLong_AsLong(AuxOutletPy);
    int requireNofOutlets = PyLong_AsLong(RequireUserToSetOutletNumbers);
    t_class *object_PY4PD_Class = (t_class *)PyLong_AsVoidPtr(PY_objectClass);
    t_py *x = (t_py *)pd_new(object_PY4PD_Class);

    Py4pdNewObj *objClass = PyLong_AsVoidPtr(PyStructSelf);
    x->objClass = (void *)objClass;
    x->visMode = 0;
    x->pyObject = 1;
    x->objArgsCount = argc;
    x->stackLimit = 100;
    x->canvas = canvas_getcurrent();
    t_canvas *c = x->canvas;
    t_symbol *patch_dir = canvas_getdir(c);
    x->objName = gensym(objectName);
    x->zoom = (int)x->canvas->gl_zoom;
    x->ignoreOnNone = PyLong_AsLong(ignoreOnNone);
    x->outPyPointer = PyLong_AsLong(pyOUT);
    x->funcCalled = 1;
    x->pFunction = pyFunction;
    x->pdPatchPath = patch_dir; // set name of the home path
    x->pkgPath = patch_dir;     // set name of the packages path
    x->playable = PyLong_AsLong(playable);

    return x;
    PyCodeObject *code = (PyCodeObject *)PyFunction_GetCode(pyFunction);
    x->pFuncName = gensym(PyUnicode_AsUTF8(code->co_name));
    x->pScriptName = gensym(PyUnicode_AsUTF8(code->co_filename));
    x->objType = PyLong_AsLong(Py_ObjType);
    Py4pdUtils_SetObjConfig(x);

    if (x->objType == PY4PD_VISOBJ)
        Py4pdUtils_CreatePicObj(x, PdDict, object_PY4PD_Class, argc, argv);

    x->pArgsCount = 0;
    int parseArgsRight =
        Py4pdUtils_ParseLibraryArguments(x, code, &argc, &argv);
    if (parseArgsRight == 0) {
        pd_error(NULL, "[%s] Error to parse arguments.", objectName);
        Py_DECREF(pd_module);
        Py_DECREF(py4pd_capsule);
        return NULL;
    }

    x->pdObjArgs = malloc(sizeof(t_symbol) * argc);
    for (int i = 0; i < argc; i++) {
        x->pdObjArgs[i] = argv[i];
    }
    x->objArgsCount = argc;
    if (x->pyObjArgs == NULL) {
        x->pyObjArgs = malloc(sizeof(t_py4pd_pValue *) * x->pArgsCount);
    }

    int inlets = Py4pdUtils_CreateObjInlets(pyFunction, x, InletsExtClassProxy,
                                            argc, argv);
    if (inlets != 0) {
        free(x->pdObjArgs);
        free(x->pyObjArgs);
        return NULL;
    }

    if (x->pArgsCount < 2) {
        x->audioError = 0;
    } else {
        x->audioError = 1;
    }

    if (x->objType > 1)
        x->pArgTuple = PyTuple_New(x->pArgsCount);

    if (!PyLong_AsLong(nooutlet)) {
        if (x->objType != PY4PD_AUDIOOUTOBJ && x->objType != PY4PD_AUDIOOBJ)
            x->mainOut = outlet_new(&x->obj, 0);
        else {
            x->mainOut = outlet_new(&x->obj, &s_signal);
            x->numOutlets = 1;
        }
    }
    if (requireNofOutlets) {
        if (x->numOutlets == -1) {
            pd_error(NULL,
                     "[%s]: This function require that you set the number of "
                     "outlets "
                     "using -outn {number_of_outlets} flag",
                     objectName);
            Py_DECREF(pd_module);
            Py_DECREF(py4pd_capsule);
            return NULL;
        } else {
            AuxOutlet = x->numOutlets;
        }
    }

    if (AuxOutlet > 0) {
        x->extrasOuts =
            (py4pdExtraOuts *)getbytes(AuxOutlet * sizeof(py4pdExtraOuts));
        x->extrasOuts->outCount = AuxOutlet;
        t_atom defarg[AuxOutlet];
        t_atom *ap;
        py4pdExtraOuts *u;
        int i;
        for (i = 0, u = x->extrasOuts, ap = defarg; i < AuxOutlet;
             i++, u++, ap++) {
            u->u_outlet = outlet_new(&x->obj, &s_anything);
        }
    }

    object_count++; // To clear memory when closing the patch
    Py_DECREF(pd_module);
    Py_DECREF(py4pd_capsule);
    return (x);
}

// ============================================================
static PyObject *Py4pdNewObj_AddFloatMethod(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *pFunc;

    if (!PyArg_ParseTuple(args, "O", &pFunc)) {
        // set the error string
        PyErr_SetString(
            PyExc_TypeError,
            "pd.new_object.addmethod_float must be a Python Function");
        return NULL;
    }

    if (!PyCallable_Check(pFunc)) {
        PyErr_SetString(PyExc_TypeError, "Function is not callable");
        return NULL;
    }

    Py4pdNewObj *selfStruct = (Py4pdNewObj *)self;
    selfStruct->pFuncFloat = pFunc;

    Py_RETURN_TRUE;
}

// ============================================================
static PyObject *Py4pdNewObj_AddSymbolMethod(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *pFunc;

    if (!PyArg_ParseTuple(args, "O", &pFunc)) {
        // set the error string
        PyErr_SetString(
            PyExc_TypeError,
            "pd.new_object.addmethod_float must be a Python Function");
        return NULL;
    }

    if (!PyCallable_Check(pFunc)) {
        PyErr_SetString(PyExc_TypeError, "Function is not callable");
        return NULL;
    }

    Py4pdNewObj *selfStruct = (Py4pdNewObj *)self;
    selfStruct->pFuncSymbol = pFunc;
    Py_RETURN_TRUE;
}
// ============================================================
static PyObject *Py4pdNewObj_AddListMethod(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *pFunc;

    if (!PyArg_ParseTuple(args, "O", &pFunc)) {
        // set the error string
        PyErr_SetString(
            PyExc_TypeError,
            "pd.new_object.addmethod_float must be a Python Function");
        return NULL;
    }

    if (!PyCallable_Check(pFunc)) {
        PyErr_SetString(PyExc_TypeError, "Function is not callable");
        return NULL;
    }

    Py4pdNewObj *selfStruct = (Py4pdNewObj *)self;
    selfStruct->pFuncList = pFunc;
    Py_RETURN_TRUE;
}

// ============================================================
static PyObject *Py4pdNewObj_AddAnythingMethod(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *pFunc;

    if (!PyArg_ParseTuple(args, "O", &pFunc)) {
        // set the error string
        PyErr_SetString(
            PyExc_TypeError,
            "pd.new_object.addmethod_float must be a Python Function");
        return NULL;
    }

    if (!PyCallable_Check(pFunc)) {
        PyErr_SetString(PyExc_TypeError, "Function is not callable");
        return NULL;
    }

    Py4pdNewObj *selfStruct = (Py4pdNewObj *)self;
    selfStruct->pFuncAnything = pFunc;
    Py_RETURN_TRUE;
}

// ============================================================
static PyObject *Py4pdNewObj_AddSelectorMethod(PyObject *self, PyObject *args,
                                               PyObject *keywords) {
    (void)self;

    PyObject *pFunc;
    PyObject *pArgs;
    const char *pSelector;

    if (!PyArg_ParseTuple(args, "sO", &pSelector, &pFunc)) {
        // set the error string
        PyErr_SetString(
            PyExc_TypeError,
            "pd.new_object.addmethod_float must be a Python Function");
        return NULL;
    }

    if (!PyCallable_Check(pFunc)) {
        PyErr_SetString(PyExc_TypeError, "Function is not callable");
        return NULL;
    }

    // check if keywords is a NULL
    if (keywords == NULL) {
        pArgs = PyList_New(1);
        PyObject *pArgsType = PyLong_FromLong(A_GIMME);
        PyList_SetItem(pArgs, 0, pArgsType);
        Py_DECREF(pArgsType);
    } else {
        if (!PyDict_Contains(keywords, PyUnicode_FromString("arg_types"))) {
            pArgs = PyList_New(1);
            PyObject *pArgsType = PyLong_FromLong(A_GIMME);
            PyList_SetItem(pArgs, 0, pArgsType);
            Py_DECREF(pArgsType);
        } else {
            pArgs = PyDict_GetItem(keywords, PyUnicode_FromString("arg_types"));
        }
    }
    Py4pdNewObj *selfStruct = (Py4pdNewObj *)self;

    if (!selfStruct->pDictSelectors) {
        selfStruct->pDictSelectors = PyDict_New();
    }

    PyObject *pSelectorPy = PyUnicode_FromString(pSelector);
    PyDict_SetItem(selfStruct->pDictSelectors, pSelectorPy, pFunc);
    Py_INCREF(pArgs);
    selfStruct->pSelectorArgs = pArgs;

    Py_RETURN_TRUE;
}
// ============================================================
static PyObject *Py4pdNewObj_AddBangMethod(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *pFunc;

    if (!PyArg_ParseTuple(args, "O", &pFunc)) {
        // set the error string
        PyErr_SetString(
            PyExc_TypeError,
            "pd.new_object.addmethod_float must be a Python Function");
        return NULL;
    }

    if (!PyCallable_Check(pFunc)) {
        PyErr_SetString(PyExc_TypeError, "Function is not callable");
        return NULL;
    }

    Py4pdNewObj *selfStruct = (Py4pdNewObj *)self;
    selfStruct->pFuncBang = pFunc;
    Py_RETURN_TRUE;
}
// ============================================================
static int Py4pdNewObj_Init(Py4pdNewObj *self, PyObject *args, PyObject *kwds) {
    (void)self;
    static char *keywords[] = {"param_str", NULL};
    char *param_str;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", keywords, &param_str)) {
        return -1;
    }
    PyObject *objectName = PyUnicode_FromString(param_str);
    if (!objectName) {
        PyErr_SetString(PyExc_TypeError, "Object name not valid");
        return -1;
    }
    self->objName = PyUnicode_AsUTF8(objectName);
    self->objType = PY4PD_NORMALOBJ;
    return 0;
}

// ============================================================
static PyObject *Py4pdNewObj_GetType(Py4pdNewObj *self) {
    return PyLong_FromLong(self->objType);
}

static int Py4pdNewObj_SetType(Py4pdNewObj *self, PyObject *value) {
    int typeValue = PyLong_AsLong(value);
    if (typeValue < 0 || typeValue > 3) {
        PyErr_SetString(PyExc_TypeError,
                        "Object type not supported, check the value");
        return -1;
    }
    self->objType = typeValue;
    return 0;
}
// ============================================================
static PyObject *Py4pdNewObj_GetName(Py4pdNewObj *self) {
    return PyUnicode_FromString(self->objName);
}

// ++++++++++
static int Py4pdNewObj_SetName(Py4pdNewObj *self, PyObject *value) {
    PyObject *objectName = PyUnicode_FromString(PyUnicode_AsUTF8(value));
    if (!objectName) {
        PyErr_SetString(PyExc_TypeError, "Object name not valid");
        return -1;
    }
    self->objName = PyUnicode_AsUTF8(objectName);
    Py_DECREF(objectName);
    return 0;
}

// ============================================================
static PyObject *Py4pdNewObj_GetPlayable(Py4pdNewObj *self) {
    return PyLong_FromLong(self->objIsPlayable);
}

// ++++++++++
static int Py4pdNewObj_SetPlayable(Py4pdNewObj *self, PyObject *value) {
    int playableValue = PyObject_IsTrue(value);
    if (playableValue < 0 || playableValue > 1) {
        PyErr_SetString(PyExc_TypeError,
                        "Playable value not supported, check the value");
        return -1;
    }
    self->objIsPlayable = playableValue;
    return 0;
}

// ============================================================
static PyObject *Py4pdNewObj_GetImage(Py4pdNewObj *self) {
    return PyUnicode_FromString(self->objImage);
}

// ++++++++++
static int Py4pdNewObj_SetImage(Py4pdNewObj *self, PyObject *value) {
    const char *imageValue = PyUnicode_AsUTF8(value);
    if (!imageValue) {
        PyErr_SetString(PyExc_TypeError, "Image path not valid");
        return -1;
    }
    self->objImage = imageValue;
    return 0;
}

// ============================================================
static PyObject *Py4pdNewObj_GetOutputPyObjs(Py4pdNewObj *self) {
    return PyLong_FromLong(self->objOutPyObjs);
}

// ++++++++++
static int Py4pdNewObj_SetOutputPyObjs(Py4pdNewObj *self, PyObject *value) {
    int outputPyObjsValue = PyObject_IsTrue(value);
    if (!outputPyObjsValue) {
        PyErr_SetString(PyExc_TypeError, "Image path not valid");
        return -1;
    }
    self->objOutPyObjs = outputPyObjsValue;
    return 0;
}

// ============================================================
static PyObject *Py4pdNewObj_GetFigSize(Py4pdNewObj *self) {
    return PyLong_FromLong(self->objIsPlayable);
}

// ++++++++++
static int Py4pdNewObj_SetFigSize(Py4pdNewObj *self, PyObject *value) {
    // see if object value is a tuple with two integers
    if (!PyTuple_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "Figure size must be a tuple with two integers");
        return -1; // Error handling
    }
    if (PyTuple_Size(value) != 2) {
        PyErr_SetString(PyExc_TypeError,
                        "Figure size must be a tuple with two integers");
        return -1; // Error handling
    }
    PyObject *width = PyTuple_GetItem(value, 0);
    PyObject *height = PyTuple_GetItem(value, 1);
    if (!PyLong_Check(width) || !PyLong_Check(height)) {
        PyErr_SetString(PyExc_TypeError,
                        "Figure size must be a tuple with two integers");
        return -1; // Error handling
    }
    self->objFigSize = value;

    return 0; // Success
}

// ============================================================
static PyObject *Py4pdNewObj_GetNoOutlets(Py4pdNewObj *self) {
    if (self->noOutlets) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

// ++++++++++
static int Py4pdNewObj_SetNoOutlets(
    Py4pdNewObj *self,
    PyObject *value) { // see if object value is a tuple with two integers
    int outpuNoOutlets = PyObject_IsTrue(value);
    self->noOutlets = outpuNoOutlets;
    return 0; // Success
}

// ============================================================
static PyObject *Py4pdNewObj_GetRequireNofOuts(Py4pdNewObj *self) {
    if (self->requireNofOutlets) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

// ++++++++++
static int Py4pdNewObj_SetRequireNofOuts(
    Py4pdNewObj *self,
    PyObject *value) { // see if object value is a tuple with two integers
    int outpuNoOutlets = PyObject_IsTrue(value);
    self->requireNofOutlets = outpuNoOutlets;
    return 0; // Success
}

// ============================================================
static PyObject *Py4pdNewObj_GetAuxOutNumbers(Py4pdNewObj *self) {
    return PyLong_FromLong(self->auxOutlets);
}

// ++++++++++
static int Py4pdNewObj_SetAuxOutNumbers(
    Py4pdNewObj *self,
    PyObject *value) { // see if object value is a tuple with two integers
    if (!PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "Auxiliary outlets must be an integer");
        return -1; // Error handling
    }
    self->auxOutlets = PyLong_AsLong(value);
    return 0; // Success
}

// ============================================================
static PyObject *Py4pdNewObj_GetIgnoreNone(Py4pdNewObj *self) {
    if (self->ignoreNoneOutputs) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

// ++++++++++
static int Py4pdNewObj_SetIgnoreNone(
    Py4pdNewObj *self,
    PyObject *value) { // see if object value is a tuple with two integers
    int ignoreNone = PyObject_IsTrue(value);
    self->ignoreNoneOutputs = ignoreNone;
    return 0; // Success
}

// ========================== GET/SET ==========================
static PyGetSetDef Py4pdNewObj_GetSet[] = {
    {"type", (getter)Py4pdNewObj_GetType, (setter)Py4pdNewObj_SetType,
     "pd.NORMALOBJ, Type attribute", NULL},
    {"name", (getter)Py4pdNewObj_GetName, (setter)Py4pdNewObj_SetName,
     "Name attribute", NULL},
    {"playable", (getter)Py4pdNewObj_GetPlayable,
     (setter)Py4pdNewObj_SetPlayable, "Playable attribute", NULL},
    {"figsize", (getter)Py4pdNewObj_GetFigSize, (setter)Py4pdNewObj_SetFigSize,
     "Figure size attribute", NULL},
    {"image", (getter)Py4pdNewObj_GetImage, (setter)Py4pdNewObj_SetImage,
     "Image patch", NULL},
    {"pyout", (getter)Py4pdNewObj_GetOutputPyObjs,
     (setter)Py4pdNewObj_SetOutputPyObjs, "Output or not PyObjs", NULL},
    {"no_outlet", (getter)Py4pdNewObj_GetNoOutlets,
     (setter)Py4pdNewObj_SetNoOutlets, "Number of outlets", NULL},
    {"require_n_of_outlets", (getter)Py4pdNewObj_GetRequireNofOuts,
     (setter)Py4pdNewObj_SetRequireNofOuts, "Require number of outlets", NULL},
    {"n_extra_outlets", (getter)Py4pdNewObj_GetAuxOutNumbers,
     (setter)Py4pdNewObj_SetAuxOutNumbers, "Number of auxiliary outlets", NULL},
    {"ignore_none", (getter)Py4pdNewObj_GetIgnoreNone,
     (setter)Py4pdNewObj_SetIgnoreNone, "Ignore None outputs", NULL},
    {NULL, NULL, NULL, NULL, NULL} /* Sentinel */
};

// =============================================================
static PyObject *Py4pdNewObj_Method_AddObj(Py4pdNewObj *self, PyObject *args) {
    (void)args;

    Py_INCREF(self); // TODO: NEED TO CLEAR THE MEMORY
    PyObject *objConfigDict = PyDict_New();
    const char *objectName;
    objectName = self->objName;

    t_class *objClass;

    if (self->objType < 5 && self->objType > -1) {
        objClass =
            class_new(gensym(objectName), (t_newmethod)Py4pdNewObj_NewObj,
                      (t_method)Py4pdUtils_FreeObj, sizeof(t_py), CLASS_DEFAULT,
                      A_GIMME, 0);
    } else {
        PyErr_SetString(PyExc_TypeError,
                        "Object type not supported, check the spelling");
        return NULL;
    }
    if (self->pFuncBang) {
        class_addbang(objClass, (t_method)Py4pdNewObj_PdExecBangMethod);
    }

    if (self->pFuncFloat) {
        class_addfloat(objClass, (t_method)Py4pdNewObj_PdExecFloatMethod);
    }

    if (self->pFuncSymbol) {
        class_addsymbol(objClass, (t_method)Py4pdNewObj_PdExecSymbolMethod);
    }

    if (self->pFuncList) {
        class_addlist(objClass, (t_method)Py4pdNewObj_PdExecListMethod);
    }

    if (self->pFuncAnything) {
        class_addanything(objClass, (t_method)Py4pdNewObj_PdExecAnythingMethod);
    }

    if (self->pDictSelectors) {
        // interate over the dict
        PyObject *pSelector, *pFunc;
        for (Py_ssize_t i = 0;
             PyDict_Next(self->pDictSelectors, &i, &pSelector, &pFunc);) {

            class_addmethod(objClass,
                            (t_method)Py4pdNewObj_PdExecSelectorMethod,
                            gensym(PyUnicode_AsUTF8(pSelector)), A_GIMME, 0);
        }
    }

    PyObject *Py_ClassLocal = PyLong_FromVoidPtr(objClass);
    PyDict_SetItemString(objConfigDict, "py4pdOBJ_CLASS", Py_ClassLocal);
    Py_DECREF(Py_ClassLocal);

    if (self->objFigSize != NULL) {
        PyObject *Py_Width = PyTuple_GetItem(self->objFigSize, 0);
        PyDict_SetItemString(objConfigDict, "py4pdOBJwidth",
                             Py_Width); // NOTE: change this names to width
        Py_DECREF(Py_Width);
        PyObject *Py_Height = PyTuple_GetItem(self->objFigSize, 1);
        PyDict_SetItemString(objConfigDict, "py4pdOBJheight", Py_Height);
        Py_DECREF(Py_Height);
    } else {
        PyObject *Py_Width = PyLong_FromLong(250);
        PyDict_SetItemString(objConfigDict, "py4pdOBJwidth", Py_Width);
        Py_DECREF(Py_Width);
        PyObject *Py_Height = PyLong_FromLong(250);
        PyDict_SetItemString(objConfigDict, "py4pdOBJheight", Py_Height);
        Py_DECREF(Py_Height);
    }

    PyObject *Py_Playable = PyLong_FromLong(self->objIsPlayable);
    PyDict_SetItemString(objConfigDict, "py4pdOBJPlayable", Py_Playable);
    Py_DECREF(Py_Playable);

    if (self->objImage != NULL) {
        PyObject *Py_GifImage = PyUnicode_FromString(self->objImage);
        PyDict_SetItemString(objConfigDict, "py4pdOBJGif", Py_GifImage);
        Py_DECREF(Py_GifImage);
    }

    PyObject *Py_ObjOuts = PyLong_FromLong(self->objOutPyObjs);
    PyDict_SetItemString(objConfigDict, "py4pdOBJpyout", Py_ObjOuts);
    Py_DECREF(Py_ObjOuts);

    PyObject *Py_NoOutlet = PyLong_FromLong(self->noOutlets);
    PyDict_SetItemString(objConfigDict, "py4pdOBJnooutlet", Py_NoOutlet);
    Py_DECREF(Py_NoOutlet);

    PyObject *Py_RequireOutletN = PyLong_FromLong(self->requireNofOutlets);
    PyDict_SetItemString(objConfigDict, "py4pdOBJrequireoutletn",
                         Py_RequireOutletN);
    Py_DECREF(Py_RequireOutletN);

    PyObject *Py_auxOutlets = PyLong_FromLong(self->auxOutlets);
    PyDict_SetItemString(objConfigDict, "py4pdAuxOutlets", Py_auxOutlets);
    Py_DECREF(Py_auxOutlets);

    PyObject *Py_ObjName = PyUnicode_FromString(objectName);
    PyDict_SetItemString(objConfigDict, "py4pdOBJname", Py_ObjName);
    Py_DECREF(Py_ObjName);

    PyObject *Py_IgnoreNoneReturn = PyLong_FromLong(self->ignoreNoneOutputs);
    PyDict_SetItemString(objConfigDict, "py4pdOBJIgnoreNone",
                         Py_IgnoreNoneReturn);
    Py_DECREF(Py_IgnoreNoneReturn);

    PyObject *PdObjStruct = PyLong_FromVoidPtr(self);
    PyDict_SetItemString(objConfigDict, "objSelf", PdObjStruct);
    Py_DECREF(PdObjStruct);

    // HERE THE OBJ WHERE WE SAVE EVERYTHING
    PyObject *objectDict = PyDict_New();
    PyDict_SetItemString(objectDict, objectName, objConfigDict);
    PyObject *py4pd_capsule = PyCapsule_New(objectDict, objectName, NULL);
    char py4pd_objectName[MAXPDSTRING];
    sprintf(py4pd_objectName, "py4pd_ObjectDict_%s", objectName);

    PyObject *pdModule = PyImport_ImportModule("pd");
    PyModule_AddObject(pdModule, py4pd_objectName, py4pd_capsule);
    Py_DECREF(pdModule);

    Py_RETURN_TRUE;
}

// ========================== METHODS ==========================
static PyMethodDef Py4pdNewObj_methods[] = {
    {"add_object", (PyCFunction)Py4pdNewObj_Method_AddObj, METH_NOARGS,
     "After the config of the Obj, use this to add the object to PureData"},

    {"addmethod_bang", (PyCFunction)Py4pdNewObj_AddBangMethod, METH_VARARGS,
     "Add Anything Method"},

    {"addmethod_float", (PyCFunction)Py4pdNewObj_AddFloatMethod, METH_VARARGS,
     "Add Float Method"},

    {"addmethod_symbol", (PyCFunction)Py4pdNewObj_AddSymbolMethod, METH_VARARGS,
     "Add Symbol Method"},

    {"addmethod_list", (PyCFunction)Py4pdNewObj_AddListMethod, METH_VARARGS,
     "Add List Method"},

    {"addmethod_anything", (PyCFunction)Py4pdNewObj_AddAnythingMethod,
     METH_VARARGS, "Add Anything Method"},

    {"addmethod", (PyCFunction)Py4pdNewObj_AddSelectorMethod,
     METH_VARARGS | METH_KEYWORDS, "Add Method with Selector"},

    // {"addmethod_bang"
    // {"addmethod"

    //     EXTERN void class_addmethod(t_class *c, t_method fn, t_symbol
    //     *sel, t_atomtype arg1, ...);
    // EXTERN void class_addbang(t_class *c, t_method fn);
    // EXTERN void class_addpointer(t_class *c, t_method fn);
    // EXTERN void class_doaddfloat(t_class *c, t_method fn);
    // EXTERN void class_addsymbol(t_class *c, t_method fn);
    // EXTERN void class_addlist(t_class *c, t_method fn);
    // EXTERN void class_addanything(t_class *c, t_method fn);

    {NULL, NULL, 0, NULL} // Sentinel
};

// ============================================================
static PyObject *Py4pdNewObj_Repr(PyObject *self) {
    Py4pdNewObj *obj = (Py4pdNewObj *)self;
    return PyUnicode_FromFormat("<%s object>", obj->objName);
}

// ========================== CLASS ============================
PyTypeObject Py4pdNewObj_Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "new_object",
    .tp_doc = "It creates new PureData objects",
    .tp_basicsize = sizeof(Py4pdNewObj),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)Py4pdNewObj_Init,
    .tp_methods = Py4pdNewObj_methods,
    .tp_getset = Py4pdNewObj_GetSet, // Add the GetSet descriptors here
    .tp_repr = Py4pdNewObj_Repr,     // Add the __repr__ method here

};
