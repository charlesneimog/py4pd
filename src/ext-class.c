#include "py4pd.h"

#include "dsp.h"
#include "ext-class.h"
#include "player.h"
#include "utils.h"

static t_class *InletsExtClassProxy;

// TODO: Adicionar
// Descrição: Forma de diferenciar methods entre argumentos que vão utilizar
// a mensagem como argumento e methods que vão utilizar os inlets como
// argumentos.

// ====================================================
// ===================== Obj Methods ==================
// ====================================================
static void Py4pdNewObj_PdExecBangMethod(t_py *x) {
    Py4pdNewObj *pObjSelf;
    pObjSelf = (Py4pdNewObj *)x->objClass;
    PyObject *pFunc = (PyObject *)pObjSelf->pFuncBang;
    if (pFunc == NULL) {
        pd_error(NULL, "[%s] No method defined for bang.", x->objName->s_name);
        return;
    }
    x->pFunction = pFunc;
    PyCodeObject *pFuncCode = (PyCodeObject *)PyFunction_GetCode(pFunc);
    int pFuncArgs = pFuncCode->co_argcount;

    PyObject *pInletValue = PyUnicode_FromString("bang");
    PyObject *pArgs = PyTuple_New(pFuncArgs);
    PyTuple_SetItem(pArgs, 0, pInletValue);
    for (int i = 1; i < pFuncArgs; i++) {
        PyTuple_SetItem(pArgs, i, x->pyObjArgs[i]->pValue);
        Py_INCREF(x->pyObjArgs[i]->pValue); // This keep the reference.
    }

    if (x->objType > PY4PD_VISOBJ) {
        Py_INCREF(pInletValue);
        x->pyObjArgs[0]->objectsUsing = 0;
        x->pyObjArgs[0]->pdout = 0;
        x->pyObjArgs[0]->objOwner = x->objName;
        x->pyObjArgs[0]->pValue = pInletValue;
        x->pArgTuple = pArgs;
        Py_INCREF(x->pArgTuple);
        return;
    }

    Py4pdUtils_RunPy(x, pArgs, x->kwargsDict);
    Py_DECREF(pArgs);
}

// ====================================================
static void Py4pdNewObj_PdExecFloatMethod(t_py *x, t_float f) {
    Py4pdNewObj *pObjSelf;

    pObjSelf = (Py4pdNewObj *)x->objClass;
    PyObject *pFunc = (PyObject *)pObjSelf->pFuncFloat;
    x->pFunction = pFunc;
    PyCodeObject *pFuncCode = (PyCodeObject *)PyFunction_GetCode(pFunc);
    int pFuncArgs = pFuncCode->co_argcount;

    PyObject *pInletValue;
    if ((int)f == f) {
        pInletValue = PyLong_FromLong((int)f);
    } else {
        pInletValue = PyFloat_FromDouble(f);
    }
    PyObject *pArgs = PyTuple_New(pFuncArgs);
    PyTuple_SetItem(pArgs, 0, pInletValue);
    for (int i = 1; i < pFuncArgs; i++) {
        PyTuple_SetItem(pArgs, i, x->pyObjArgs[i]->pValue);
        Py_INCREF(x->pyObjArgs[i]->pValue); // This keep the reference.
    }

    if (x->objType > PY4PD_VISOBJ) {
        Py_INCREF(pInletValue);
        x->pyObjArgs[0]->objectsUsing = 0;
        x->pyObjArgs[0]->pdout = 0;
        x->pyObjArgs[0]->objOwner = x->objName;
        x->pyObjArgs[0]->pValue = pInletValue;
        x->pArgTuple = pArgs;
        Py_INCREF(x->pArgTuple);
        return;
    }

    Py4pdUtils_RunPy(x, pArgs, x->kwargsDict);
    Py_DECREF(pArgs);
}

// ====================================================
static void Py4pdNewObj_PdExecSymbolMethod(t_py *x, t_symbol *s) {
    Py4pdNewObj *pObjSelf;

    pObjSelf = (Py4pdNewObj *)x->objClass;
    PyObject *pFunc = (PyObject *)pObjSelf->pFuncSymbol;
    x->pFunction = pFunc;
    PyCodeObject *pFuncCode = (PyCodeObject *)PyFunction_GetCode(pFunc);
    int pFuncArgs = pFuncCode->co_argcount;

    PyObject *pInletValue = PyUnicode_FromString(s->s_name);

    PyObject *pArgs = PyTuple_New(pFuncArgs);
    PyTuple_SetItem(pArgs, 0, pInletValue);
    for (int i = 1; i < pFuncArgs; i++) {
        PyTuple_SetItem(pArgs, i, x->pyObjArgs[i]->pValue);
        Py_INCREF(x->pyObjArgs[i]->pValue); // This keep the reference.
    }

    if (x->objType > PY4PD_VISOBJ) {
        Py_INCREF(pInletValue);
        x->pyObjArgs[0]->objectsUsing = 0;
        x->pyObjArgs[0]->pdout = 0;
        x->pyObjArgs[0]->objOwner = x->objName;
        x->pyObjArgs[0]->pValue = pInletValue;
        x->pArgTuple = pArgs;
        Py_INCREF(x->pArgTuple);
        return;
    }

    Py4pdUtils_RunPy(x, pArgs, x->kwargsDict);
    Py_DECREF(pArgs);
}

// ====================================================
static void Py4pdNewObj_PdExecListMethod(t_py *x, t_symbol *s, int argc,
                                         t_atom *argv) {
    Py4pdNewObj *pObjSelf;

    if (s == NULL || s == gensym("bang")) {
        Py4pdNewObj_PdExecBangMethod(x);
        return;
    }

    pObjSelf = (Py4pdNewObj *)x->objClass;
    PyObject *pFunc = (PyObject *)pObjSelf->pFuncList;

    x->pFunction = pFunc;
    PyCodeObject *pFuncCode = (PyCodeObject *)PyFunction_GetCode(pFunc);
    int pFuncArgs = pFuncCode->co_argcount;
    PyObject *pArgs = PyTuple_New(pFuncArgs);

    PyObject *pInletValue = Py4pdUtils_CreatePyObjFromPdArgs(s, argc, argv);
    PyTuple_SetItem(pArgs, 0, pInletValue);

    for (int i = 1; i < pFuncArgs; i++) {
        PyTuple_SetItem(pArgs, i, x->pyObjArgs[i]->pValue);
        Py_INCREF(x->pyObjArgs[i]->pValue); // This keep the reference.
    }

    if (x->objType > PY4PD_VISOBJ) {
        Py_INCREF(pArgs);
        x->pyObjArgs[0]->objectsUsing = 0;
        x->pyObjArgs[0]->pdout = 0;
        x->pyObjArgs[0]->objOwner = x->objName;
        x->pyObjArgs[0]->pValue = pArgs;
        x->pArgTuple = pArgs;
        Py_INCREF(x->pArgTuple);
        return;
    }

    Py4pdUtils_RunPy(x, pArgs, x->kwargsDict);
    Py_DECREF(pArgs);
}

// ====================================================
static void Py4pdNewObj_PdExecAnythingMethod(t_py *x, t_symbol *s, int argc,
                                             t_atom *argv) {
    Py4pdNewObj *pObjSelf;

    if (s == NULL || s == gensym("bang")) {
        Py4pdNewObj_PdExecBangMethod(x);
        return;
    }

    pObjSelf = (Py4pdNewObj *)x->objClass;
    PyObject *pFunc = (PyObject *)pObjSelf->pFuncAnything;
    x->pFunction = pFunc;
    PyCodeObject *pFuncCode = (PyCodeObject *)PyFunction_GetCode(pFunc);
    int pFuncArgs = pFuncCode->co_argcount;
    PyObject *pArgs = PyTuple_New(pFuncArgs);

    PyObject *pInletValue = Py4pdUtils_CreatePyObjFromPdArgs(s, argc, argv);
    PyTuple_SetItem(pArgs, 0, pInletValue);

    for (int i = 1; i < pFuncArgs; i++) {
        PyTuple_SetItem(pArgs, i, x->pyObjArgs[i]->pValue);
        Py_INCREF(x->pyObjArgs[i]->pValue); // This keep the reference.
    }

    if (x->objType > PY4PD_VISOBJ) {
        Py_INCREF(pArgs);
        x->pyObjArgs[0]->objectsUsing = 0;
        x->pyObjArgs[0]->pdout = 0;
        x->pyObjArgs[0]->objOwner = x->objName;
        x->pyObjArgs[0]->pValue = pArgs;
        x->pArgTuple = pArgs;
        Py_INCREF(x->pArgTuple);
        return;
    }
    Py4pdUtils_RunPy(x, pArgs, x->kwargsDict);
    Py_DECREF(pArgs);
}

// ====================================================
static void Py4pdNewObj_PdExecSelectorMethod(t_py *x, t_symbol *selector,
                                             int argc, t_atom *argv) {
    Py4pdNewObj *pObjSelf;

    pObjSelf = (Py4pdNewObj *)x->objClass;
    PyObject *pSelector = PyUnicode_FromString(selector->s_name);

    // get the Dict
    PyObject *pDictSelectors = (PyObject *)pObjSelf->pDictSelectors;
    PyObject *pDictArgs =
        PyDict_GetItem((PyObject *)pObjSelf->pSelectorArgs, pSelector);
    if (pDictArgs) {
        int argCount = PyTuple_Size(pDictArgs);
        for (int i = 0; i < argCount; i++) {
            if (i > argc) {
                continue;
            }
            PyObject *pArgType = PyTuple_GetItem(pDictArgs, i);
            t_atomtype pdArgType = (t_atomtype)PyLong_AsLong(pArgType);
            if (pdArgType == argv[i].a_type) {
                continue;
            } else if (pdArgType == 10) {
                continue;
            } else {
                PyObject *pRepr = PyObject_Repr(pDictArgs);
                const char *pdRepr = PyUnicode_AsUTF8(pRepr);
                pd_error(NULL,
                         "The method %s the following argument list %s | where "
                         "1 means Floats and 2 means symbols",
                         selector->s_name, pdRepr);
                return;
            }
        }
    }

    PyObject *pFunc = PyDict_GetItem(pDictSelectors, pSelector);
    x->pFunction = pFunc;
    if (!pFunc) {
        pd_error(NULL, "pFunc is NULL, please report in %s", PY4PD_GIT_ISSUES);
        return;
    }
    selector = NULL;

    PyCodeObject *pFuncCode = (PyCodeObject *)PyFunction_GetCode(pFunc);
    int pFuncArgs = pFuncCode->co_argcount;
    PyObject *pArgs = PyTuple_New(pFuncArgs);

    PyObject *pInletValue =
        Py4pdUtils_CreatePyObjFromPdArgs(selector, argc, argv);
    PyTuple_SetItem(pArgs, 0, pInletValue);

    for (int i = 1; i < pFuncArgs; i++) {
        PyTuple_SetItem(pArgs, i, x->pyObjArgs[i]->pValue);
        Py_INCREF(x->pyObjArgs[i]->pValue); // This keep the reference.
    }

    if (x->objType > PY4PD_VISOBJ) {
        Py_INCREF(pArgs);
        x->pyObjArgs[0]->objectsUsing = 0;
        x->pyObjArgs[0]->pdout = 0;
        x->pyObjArgs[0]->objOwner = x->objName;
        x->pyObjArgs[0]->pValue = pArgs;
        x->pArgTuple = pArgs;
        Py_INCREF(x->pArgTuple);
        return;
    }
    Py4pdUtils_RunPy(x, pArgs, x->kwargsDict);
    Py_DECREF(pArgs);
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
    if (selfStruct->pSelectorArgs == NULL) {
        selfStruct->pSelectorArgs = PyDict_New();
    }

    // check if keywords is a NULL
    if (keywords == NULL) {
        PyObject *pArgsType = PyLong_FromLong(A_GIMME);
        PyObject *pArgsTuple = PyTuple_New(1);
        PyTuple_SetItem(pArgsTuple, 0, pArgsType);
        PyDict_SetItemString(selfStruct->pSelectorArgs, pSelector, pArgsTuple);
        // TODO: MEMORY
        Py_DECREF(pArgsType);
    } else {
        if (!PyDict_Contains(keywords, PyUnicode_FromString("arg_types"))) {
            pArgs = PyTuple_New(1);
            PyObject *pArgsType = PyLong_FromLong(A_GIMME);
            PyTuple_SetItem(pArgs, 0, pArgsType);
            PyDict_SetItemString(selfStruct->pSelectorArgs, pSelector, pArgs);
            Py_DECREF(pArgsType);
            // TODO: Check how to manage the memory
        } else {
            PyObject *pArgsType = PyDict_GetItemString(keywords, "arg_types");
            if (PyTuple_Check(pArgsType)) {
                pArgs = pArgsType;
                PyDict_SetItemString(selfStruct->pSelectorArgs, pSelector,
                                     pArgs);
            } else {
                PyErr_SetString(PyExc_TypeError,
                                "arg_types must be a tuple of integers");
                return NULL;
            }
        }
    }

    if (!selfStruct->pDictSelectors) {
        selfStruct->pDictSelectors = PyDict_New();
    }

    PyObject *pSelectorPy = PyUnicode_FromString(pSelector);
    PyDict_SetItem(selfStruct->pDictSelectors, pSelectorPy, pFunc);

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
static PyObject *Py4pdNewObj_AddAudioMethod(PyObject *self, PyObject *args) {
    (void)self;

    PyObject *pFunc;

    if (!PyArg_ParseTuple(args, "O", &pFunc)) {
        PyErr_SetString(
            PyExc_TypeError,
            "pd.new_object.addmethod_audio must be a Python Function");
        return NULL;
    }

    if (!PyCallable_Check(pFunc)) {
        PyErr_SetString(PyExc_TypeError, "Function is not callable");
        return NULL;
    }

    Py4pdNewObj *selfStruct = (Py4pdNewObj *)self;
    if (selfStruct->pAudio) {
        PyErr_SetString(PyExc_TypeError, "Audio method already defined, you "
                                         "can't define more than one audio "
                                         "method per object");
        return NULL;
    }
    selfStruct->pAudio = pFunc;
    Py_RETURN_TRUE;
}

// ============================================================
/**
 * @brief This function initialize the object
 * @param self is the Class
 * @param argc is the number of arguments
 * @param argv is the arguments
 */
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
// ======================== Obj Attributes ====================
// ============================================================
static PyObject *Py4pdNewObj_GetType(Py4pdNewObj *self) {
    return PyLong_FromLong(self->objType);
}

static int Py4pdNewObj_SetType(Py4pdNewObj *self, PyObject *value) {
    int typeValue = PyLong_AsLong(value);
    if (typeValue < 0 || typeValue > 4) {
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

static int Py4pdNewObj_SetFigSize(Py4pdNewObj *self, PyObject *value) {
    if (!PyTuple_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "Figure size must be a tuple with two integers");
        return -1;
    }
    if (PyTuple_Size(value) != 2) {
        PyErr_SetString(PyExc_TypeError,
                        "Figure size must be a tuple with two integers");
        return -1;
    }
    PyObject *width = PyTuple_GetItem(value, 0);
    PyObject *height = PyTuple_GetItem(value, 1);
    if (!PyLong_Check(width) || !PyLong_Check(height)) {
        PyErr_SetString(PyExc_TypeError,
                        "Figure size must be a tuple with two integers");
        return -1;
    }
    self->objFigSize = value;
    return 0;
}

// ============================================================
static PyObject *Py4pdNewObj_GetNoOutlets(Py4pdNewObj *self) {
    if (self->noOutlets) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static int Py4pdNewObj_SetNoOutlets(Py4pdNewObj *self, PyObject *value) {
    int outpuNoOutlets = PyObject_IsTrue(value);
    self->noOutlets = outpuNoOutlets;
    return 0;
}

// ============================================================
static PyObject *Py4pdNewObj_GetRequireNofOuts(Py4pdNewObj *self) {
    if (self->requireNofOutlets) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static int Py4pdNewObj_SetRequireNofOuts(Py4pdNewObj *self, PyObject *value) {
    int outpuNoOutlets = PyObject_IsTrue(value);
    self->requireNofOutlets = outpuNoOutlets;
    return 0;
}

// ============================================================
static PyObject *Py4pdNewObj_GetAuxOutNumbers(Py4pdNewObj *self) {
    return PyLong_FromLong(self->auxOutlets);
}

static int Py4pdNewObj_SetAuxOutNumbers(Py4pdNewObj *self, PyObject *value) {
    if (!PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError,
                        "Auxiliary outlets must be an integer");
        return -1;
    }
    self->auxOutlets = PyLong_AsLong(value);
    return 0;
}

// ============================================================
static PyObject *Py4pdNewObj_GetIgnoreNone(Py4pdNewObj *self) {
    if (self->ignoreNoneOutputs) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static int Py4pdNewObj_SetIgnoreNone(Py4pdNewObj *self, PyObject *value) {
    int ignoreNone = PyObject_IsTrue(value);
    self->ignoreNoneOutputs = ignoreNone;
    return 0;
}

// ============================================================
static PyObject *Py4pdNewObj_GetEditorClick(Py4pdNewObj *self) {
    if (self->allowEdit) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static int Py4pdNewObj_SetEditorClick(Py4pdNewObj *self, PyObject *value) {
    int allowEditor = PyObject_IsTrue(value);
    self->allowEdit = allowEditor;
    return 0;
}
// ============================================================
static PyObject *Py4pdNewObj_GetHelpPatch(Py4pdNewObj *self) {
    if (self->helpPatch != NULL) {
        Py_RETURN_NONE;
    }
    return PyUnicode_FromString(self->helpPatch);
}

static int Py4pdNewObj_SetHelpPatch(Py4pdNewObj *self, PyObject *value) {
    if (!PyUnicode_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Help patch must be a string");
        return -1;
    }
    self->helpPatch = PyUnicode_AsUTF8(value);
    return 0;
}

// ========================== GET/SET ==========================
static PyGetSetDef Py4pdNewObj_GetSet[] = {
    {"type", (getter)Py4pdNewObj_GetType, (setter)Py4pdNewObj_SetType,
     "pd.NORMALOBJ, Type attribute", NULL},
    {"name", (getter)Py4pdNewObj_GetName, (setter)Py4pdNewObj_SetName,
     "Name attribute", NULL},
    {"playable", (getter)Py4pdNewObj_GetPlayable,
     (setter)Py4pdNewObj_SetPlayable, "Playable attribute", NULL},
    {"fig_size", (getter)Py4pdNewObj_GetFigSize, (setter)Py4pdNewObj_SetFigSize,
     "Figure size attribute", NULL},
    {"image", (getter)Py4pdNewObj_GetImage, (setter)Py4pdNewObj_SetImage,
     "Image patch", NULL},
    {"py_out", (getter)Py4pdNewObj_GetOutputPyObjs,
     (setter)Py4pdNewObj_SetOutputPyObjs, "Output or not PyObjs", NULL},
    {"no_outlet", (getter)Py4pdNewObj_GetNoOutlets,
     (setter)Py4pdNewObj_SetNoOutlets, "Number of outlets", NULL},
    {"require_n_of_outlets", (getter)Py4pdNewObj_GetRequireNofOuts,
     (setter)Py4pdNewObj_SetRequireNofOuts, "Require number of outlets", NULL},
    {"n_extra_outlets", (getter)Py4pdNewObj_GetAuxOutNumbers,
     (setter)Py4pdNewObj_SetAuxOutNumbers, "Number of auxiliary outlets", NULL},
    {"ignore_none", (getter)Py4pdNewObj_GetIgnoreNone,
     (setter)Py4pdNewObj_SetIgnoreNone, "Ignore None outputs", NULL},
    {"help_patch", (getter)Py4pdNewObj_GetHelpPatch,
     (setter)Py4pdNewObj_SetHelpPatch, "Ignore None outputs", NULL},
    {"allow_editor", (getter)Py4pdNewObj_GetEditorClick,
     (setter)Py4pdNewObj_SetEditorClick, "Ignore None outputs", NULL},

    {NULL, NULL, NULL, NULL, NULL} /* Sentinel */
};

// ====================================================
// ===================== Create Obj ===================
// ====================================================
void *Py4pdNewObj_NewObj(t_symbol *s, int argc, t_atom *argv) {
    const char *objectName = s->s_name;
    char py4pd_objectName[MAXPDSTRING];
    snprintf(py4pd_objectName, MAXPDSTRING, "py4pd_ObjectDict_%s", objectName);
    PyObject *pd_module = PyImport_ImportModule("pd");

    if (pd_module == NULL) {
        pd_error(
            NULL,
            "[py4pd] Not possible import the pd module, failed to create %s",
            s->s_name);
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

    PyObject *PY_objectClass = PyDict_GetItemString(PdDict, "CLASS");
    if (PY_objectClass == NULL) {
        pd_error(NULL, "Error: object Class is NULL");
        Py_DECREF(pd_module);
        Py_DECREF(py4pd_capsule);
        return NULL;
    }

    PyObject *ignoreOnNone = PyDict_GetItemString(PdDict, "IgnoreNone");
    PyObject *playable = PyDict_GetItemString(PdDict, "Playable");
    PyObject *pyOUT = PyDict_GetItemString(PdDict, "pyout");
    PyObject *nooutlet = PyDict_GetItemString(PdDict, "nooutlet");
    PyObject *AuxOutletPy = PyDict_GetItemString(PdDict, "py4pdAuxOutlets");
    PyObject *RequireUserToSetOutletNumbers =
        PyDict_GetItemString(PdDict, "requireoutletn");
    PyObject *pyFunction = PyDict_GetItemString(PdDict, "Function");
    PyObject *PyStructSelf = PyDict_GetItemString(PdDict, "objSelf");
    PyObject *pMaxArgFunc = PyDict_GetItemString(PdDict, "objFunc");

    int AuxOutlet = PyLong_AsLong(AuxOutletPy);
    int requireNofOutlets = PyLong_AsLong(RequireUserToSetOutletNumbers);
    t_class *object_PY4PD_Class = (t_class *)PyLong_AsVoidPtr(PY_objectClass);
    t_py *x = (t_py *)pd_new(object_PY4PD_Class);

    Py4pdNewObj *objClass = PyLong_AsVoidPtr(PyStructSelf);
    x->objClass = (void *)objClass;
    x->visMode = 0;
    x->objType = objClass->objType;
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
    x->vectorSize = sys_getblksize();
    x->pdPatchPath = patch_dir; // set name of the home path
    x->pkgPath = patch_dir;     // set name of the packages path
    x->playable = PyLong_AsLong(playable);

    PyCodeObject *code = (PyCodeObject *)PyFunction_GetCode(pMaxArgFunc);
    x->pFuncName = gensym(PyUnicode_AsUTF8(code->co_name));
    x->pScriptName = gensym(PyUnicode_AsUTF8(code->co_filename));
    x->numInlets = code->co_argcount;
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

    int inlets = Py4pdUtils_CreateObjInlets(pMaxArgFunc, x, InletsExtClassProxy,
                                            argc, argv);
    if (inlets != 0) {
        free(x->pdObjArgs);
        free(x->pyObjArgs);
        return NULL;
    }

    if (x->objType > PY4PD_VISOBJ) {
        if (objClass->pAudio == NULL) {
            pd_error(NULL,
                     "[%s] Error: Audio method not defined, use "
                     "pd.new_object.addmethod_audio",
                     objectName);
            return NULL;
        } else {
            x->pFunction = objClass->pAudio;
        }
    } else {
        if (objClass->pAudio != NULL) {
            pd_error(x,
                     "[%s] Audio method is defined but the object is "
                     "not a AUDIO object.",
                     objectName);
        }
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

    objCount++;
    Py_DECREF(pd_module);
    Py_DECREF(py4pd_capsule);
    return (x);
}
// =============================================================
static PyObject *Py4pdNewObj_Method_AddObj(Py4pdNewObj *self, PyObject *args) {
    (void)args;

    Py_INCREF(self); // TODO: NEED DECREF THIS SOMEWHERE
    PyObject *objConfigDict = PyDict_New();
    const char *objectName;
    objectName = self->objName;
    PyObject *PdModule = PyImport_ImportModule("pd");
    t_py *py4pd = Py4pdUtils_GetObject(PdModule);

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

    int pArgCount = 0;
    PyObject *pMaxArgFunction = NULL;

    if (self->pFuncBang) {
        class_addbang(objClass, (t_method)Py4pdNewObj_PdExecBangMethod);
        PyCodeObject *code =
            (PyCodeObject *)PyFunction_GetCode(self->pFuncBang);
        if (code->co_argcount > pArgCount || pMaxArgFunction == NULL) {
            pArgCount = code->co_argcount;
            pMaxArgFunction = self->pFuncBang;
        }
    }

    if (self->pFuncFloat) {
        class_addfloat(objClass, (t_method)Py4pdNewObj_PdExecFloatMethod);
        PyCodeObject *code =
            (PyCodeObject *)PyFunction_GetCode(self->pFuncFloat);
        if (code->co_argcount > pArgCount || pMaxArgFunction == NULL) {
            pArgCount = code->co_argcount;
            pMaxArgFunction = self->pFuncFloat;
        }
    }

    if (self->pFuncSymbol) {
        class_addsymbol(objClass, (t_method)Py4pdNewObj_PdExecSymbolMethod);
        PyCodeObject *code =
            (PyCodeObject *)PyFunction_GetCode(self->pFuncSymbol);
        if (code->co_argcount > pArgCount || pMaxArgFunction == NULL) {
            pArgCount = code->co_argcount;
            pMaxArgFunction = self->pFuncSymbol;
        }
    }

    if (self->pFuncList) {
        class_addlist(objClass, (t_method)Py4pdNewObj_PdExecListMethod);
        PyCodeObject *code =
            (PyCodeObject *)PyFunction_GetCode(self->pFuncList);
        if (code->co_argcount > pArgCount || pMaxArgFunction == NULL) {
            pArgCount = code->co_argcount;
            pMaxArgFunction = self->pFuncList;
        }
    }

    if (self->pFuncAnything) {
        class_addanything(objClass, (t_method)Py4pdNewObj_PdExecAnythingMethod);
        PyCodeObject *code =
            (PyCodeObject *)PyFunction_GetCode(self->pFuncAnything);
        if (code->co_argcount > pArgCount || pMaxArgFunction == NULL) {
            pArgCount = code->co_argcount;
            pMaxArgFunction = self->pFuncAnything;
        }
    }

    if (self->pDictSelectors) {
        PyObject *pSelector, *pFunc;
        for (Py_ssize_t i = 0;
             PyDict_Next(self->pDictSelectors, &i, &pSelector, &pFunc);) {
            class_addmethod(objClass,
                            (t_method)Py4pdNewObj_PdExecSelectorMethod,
                            gensym(PyUnicode_AsUTF8(pSelector)), A_GIMME, 0);

            PyObject *pMethodFunc = PyDict_GetItem(self->pDictSelectors,
                                                   pSelector); // TODO: CHECK
            PyCodeObject *code =
                (PyCodeObject *)PyFunction_GetCode(pMethodFunc);
            if (code->co_argcount > pArgCount || pMaxArgFunction == NULL) {
                pArgCount = code->co_argcount;
                pMaxArgFunction = pMethodFunc;
            }
        }
    }

    if (pMaxArgFunction == NULL && (self->pAudio == NULL)) {
        PyErr_SetString(PyExc_TypeError, "You must add a method to the object");
        return NULL;
    } else if (self->pAudio != NULL) {
        PyCodeObject *code = (PyCodeObject *)PyFunction_GetCode(self->pAudio);
        if (code->co_argcount > pArgCount) {
            pArgCount = code->co_argcount;
            pMaxArgFunction = self->pAudio;
        }
    }

    PyObject *Py_ClassLocal = PyLong_FromVoidPtr(objClass);
    PyDict_SetItemString(objConfigDict, "CLASS", Py_ClassLocal);
    Py_DECREF(Py_ClassLocal);

    if (self->objFigSize != NULL) {
        PyObject *Py_Width = PyTuple_GetItem(self->objFigSize, 0);
        PyDict_SetItemString(objConfigDict, "width",
                             Py_Width); // NOTE: change this names to width
        Py_DECREF(Py_Width);
        PyObject *Py_Height = PyTuple_GetItem(self->objFigSize, 1);
        PyDict_SetItemString(objConfigDict, "height", Py_Height);
        Py_DECREF(Py_Height);
    } else {
        PyObject *Py_Width = PyLong_FromLong(250);
        PyDict_SetItemString(objConfigDict, "width", Py_Width);
        Py_DECREF(Py_Width);
        PyObject *Py_Height = PyLong_FromLong(250);
        PyDict_SetItemString(objConfigDict, "height", Py_Height);
        Py_DECREF(Py_Height);
    }

    PyObject *Py_Playable = PyLong_FromLong(self->objIsPlayable);
    PyDict_SetItemString(objConfigDict, "Playable",
                         Py_Playable); // TODO: better names
    Py_DECREF(Py_Playable);

    if (self->objImage != NULL) {
        PyObject *Py_GifImage = PyUnicode_FromString(self->objImage);
        PyDict_SetItemString(objConfigDict, "Gif",
                             Py_GifImage); // TODO: Change var name
        Py_DECREF(Py_GifImage);
    }

    PyObject *Py_ObjOuts = PyLong_FromLong(self->objOutPyObjs);
    PyDict_SetItemString(objConfigDict, "pyout", Py_ObjOuts);
    Py_DECREF(Py_ObjOuts);

    PyObject *Py_NoOutlet = PyLong_FromLong(self->noOutlets);
    PyDict_SetItemString(objConfigDict, "nooutlet", Py_NoOutlet);
    Py_DECREF(Py_NoOutlet);

    PyObject *Py_RequireOutletN = PyLong_FromLong(self->requireNofOutlets);
    PyDict_SetItemString(objConfigDict, "requireoutletn", Py_RequireOutletN);
    Py_DECREF(Py_RequireOutletN);

    PyObject *Py_auxOutlets = PyLong_FromLong(self->auxOutlets);
    PyDict_SetItemString(objConfigDict, "py4pdAuxOutlets", Py_auxOutlets);
    Py_DECREF(Py_auxOutlets);

    PyObject *Py_ObjName = PyUnicode_FromString(objectName);
    PyDict_SetItemString(objConfigDict, "name", Py_ObjName);
    Py_DECREF(Py_ObjName);

    PyObject *Py_LibraryFolder =
        PyUnicode_FromString(py4pd->libraryFolder->s_name);
    PyDict_SetItemString(objConfigDict, "LibraryFolder", Py_LibraryFolder);
    Py_DECREF(Py_LibraryFolder);

    PyObject *Py_IgnoreNoneReturn = PyLong_FromLong(self->ignoreNoneOutputs);
    PyDict_SetItemString(objConfigDict, "IgnoreNone", Py_IgnoreNoneReturn);
    Py_DECREF(Py_IgnoreNoneReturn);

    PyObject *PdObjStruct = PyLong_FromVoidPtr(self);
    PyDict_SetItemString(objConfigDict, "objSelf", PdObjStruct);
    Py_DECREF(PdObjStruct);

    PyObject *pMaxArgs = PyLong_FromLong(pArgCount);
    PyDict_SetItemString(objConfigDict, "objMaxArgs", pMaxArgs);
    Py_DECREF(pMaxArgs);

    PyDict_SetItemString(objConfigDict, "objFunc", pMaxArgFunction);

    // HERE THE OBJ WHERE WE SAVE EVERYTHING
    PyObject *objectDict = PyDict_New();
    PyDict_SetItemString(objectDict, objectName, objConfigDict);
    PyObject *py4pd_capsule = PyCapsule_New(objectDict, objectName, NULL);
    char py4pd_objectName[MAXPDSTRING];
    sprintf(py4pd_objectName, "py4pd_ObjectDict_%s", objectName);

    PyModule_AddObject(PdModule, py4pd_objectName, py4pd_capsule);
    Py_DECREF(PdModule);

    if (self->objIsPlayable) {
        class_addmethod(objClass, (t_method)Py4pdPlayer_Play, gensym("play"),
                        A_GIMME, 0);
        class_addmethod(objClass, (t_method)Py4pdPlayer_Stop, gensym("stop"), 0,
                        0);
        class_addmethod(objClass, (t_method)Py4pdPlayer_Clear, gensym("clear"),
                        0, 0);
    }

    if (self->objType == PY4PD_AUDIOINOBJ) {
        class_addmethod(objClass, (t_method)Py4pdAudio_Dsp, gensym("dsp"),
                        A_CANT, 0);
        CLASS_MAINSIGNALIN(objClass, t_py, py4pdAudio);
    } else if (self->objType == PY4PD_AUDIOOUTOBJ) {
        class_addmethod(objClass, (t_method)Py4pdAudio_Dsp, gensym("dsp"),
                        A_CANT, 0);
    } else if (self->objType == PY4PD_AUDIOOBJ) {
        class_addmethod(objClass, (t_method)Py4pdAudio_Dsp, gensym("dsp"),
                        A_CANT, 0);
        CLASS_MAINSIGNALIN(objClass, t_py, py4pdAudio);
    }

    // TODO: Method to Reload
    if (self->allowEdit) {
        class_addmethod(objClass, (t_method)Py4pdUtils_Click, gensym("click"),
                        0, 0);
    }

    if (pArgCount != 0) {
        InletsExtClassProxy =
            class_new(gensym("_py4pdInlets_proxy"), 0, 0,
                      sizeof(t_py4pdInlet_proxy), CLASS_DEFAULT, 0);

        class_addanything(InletsExtClassProxy, Py4pdUtils_ExtraInletAnything);
        class_addmethod(InletsExtClassProxy,
                        (t_method)Py4pdUtils_ExtraInletPointer,
                        gensym("PyObject"), A_SYMBOL, A_POINTER, 0);
    }
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

    {"addmethod_audioin", (PyCFunction)Py4pdNewObj_AddAudioMethod, METH_VARARGS,
     "Add Audio Method"},

    {"addmethod_audioout", (PyCFunction)Py4pdNewObj_AddAudioMethod,
     METH_VARARGS, "Add Audio Method"},

    {"addmethod_audio", (PyCFunction)Py4pdNewObj_AddAudioMethod, METH_VARARGS,
     "Add Audio Method"},

    {"addmethod", (PyCFunction)Py4pdNewObj_AddSelectorMethod,
     METH_VARARGS | METH_KEYWORDS, "Add Method with Selector"},

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
