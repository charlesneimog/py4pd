#include "py4pd.h"

#include "dsp.h"
#include "ext-libraries.h"
#include "pic.h"
#include "player.h"
#include "utils.h"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PY4PD_NUMPYARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_25_API_VERSION
#include <numpy/arrayobject.h>

// ================= PUREDATA ================
void Py4pdLib_Bang(t_py *x);
static t_class *Py4pdInletsProxy;

// ===========================================
void Py4pdLib_ReloadObject(t_py *x) {
    LOG("Py4pdLib_Bang");
    if (x->pFunction != NULL && PyFunction_Check(x->pFunction)) {
        PyObject *pFunctionModule = PyObject_GetAttrString(x->pFunction, "__module__");
        if (pFunctionModule != NULL) {
            const char *pModuleStr = PyUnicode_AsUTF8(pFunctionModule);
            PyObject *pModule = PyImport_ImportModule(pModuleStr);
            if (pModule == NULL) {
                pd_error(x, "[Python] Failed to load module");
                return;
            }
            PyObject *pModuleReloaded = PyImport_ReloadModule(pModule);
            if (pModuleReloaded != NULL) {
                Py_DECREF(pModule);
                pModule = pModuleReloaded;
            } else {
                Py4pdUtils_PrintError(x);
                Py_XDECREF(pModule);
                Py_XDECREF(pFunctionModule);
                x->pFunction = NULL;
                x->pModule = NULL;
                return;
            }
            x->pFunction = PyObject_GetAttrString(pModule, x->pFuncName->s_name);
            if (x->pFunction == NULL) {
                Py4pdUtils_PrintError(x);
                Py_XDECREF(pModule);
                Py_XDECREF(pFunctionModule);
                x->pFunction = NULL;
                x->pModule = NULL;
                return;
            }
            Py_XDECREF(x->pModule);
            x->pModule = pModule;
            Py_XDECREF(pFunctionModule);
            post("[py4pd] Function reloaded: %s", x->pFuncName->s_name);
        }
        return;
    } else {
        pd_error(x, "[py4pd] The function is not set");
        return;
    }
}

// ===========================================
void Py4pdLib_Py4pdObjPicSave(t_gobj *z, t_binbuf *b) {
    LOG("Py4pdLib_Py4pdObjPicSave");
    t_py *x = (t_py *)z;
    if (x->VisMode) {
        binbuf_addv(b, "ssii", gensym("#X"), gensym("obj"), x->obj.te_xpix, x->obj.te_ypix);
        binbuf_addbinbuf(b, ((t_py *)x)->obj.te_binbuf);
        int objAtomsCount = binbuf_getnatom(((t_py *)x)->obj.te_binbuf);
        if (objAtomsCount == 1) {
            binbuf_addv(b, "ii", x->Width, x->Height);
        }
        binbuf_addsemi(b);
    }
    return;
}

// ===========================================
void Py4pdLib_SetKwargs(t_py *x, t_symbol *s, int ac, t_atom *av) {
    LOG("Py4pdLib_SetKwargs");
    (void)s;
    t_symbol *key;

    if (av[0].a_type != A_SYMBOL) {
        pd_error(x, "The first argument of the message 'kwargs' must be a symbol");
        return;
    }
    if (ac < 2) {
        pd_error(x, "You need to specify a value for the key");
        return;
    }
    key = atom_getsymbolarg(0, ac, av);

    if (x->KwArgsDict == NULL)
        x->KwArgsDict = PyDict_New();

    PyObject *oldItemFromDict = PyDict_GetItemString(x->KwArgsDict, s->s_name);
    if (oldItemFromDict != NULL) {
        PyDict_DelItemString(x->KwArgsDict, s->s_name);
        Py_DECREF(oldItemFromDict);
    }

    if (ac == 2) {
        if (av[1].a_type == A_FLOAT) {
            int isInt = atom_getintarg(1, ac, av) == atom_getfloatarg(1, ac, av);
            if (isInt)
                PyDict_SetItemString(x->KwArgsDict, key->s_name,
                                     PyLong_FromLong(atom_getintarg(1, ac, av)));
            else
                PyDict_SetItemString(x->KwArgsDict, key->s_name,
                                     PyFloat_FromDouble(atom_getfloatarg(1, ac, av)));

        } else if (av[1].a_type == A_SYMBOL)
            PyDict_SetItemString(x->KwArgsDict, key->s_name,
                                 PyUnicode_FromString(atom_getsymbolarg(1, ac, av)->s_name));
        else {
            pd_error(x, "The third argument of the message 'kwargs' must be a "
                        "symbol or a float");
            return;
        }
    } else if (ac > 2) {
        PyObject *pyInletValue = PyList_New(ac - 1);
        for (int i = 1; i < ac; i++) {
            if (av[i].a_type == A_FLOAT) {
                int isInt = atom_getintarg(i, ac, av) == atom_getfloatarg(i, ac, av);
                if (isInt)
                    PyList_SetItem(pyInletValue, i - 1, PyLong_FromLong(atom_getintarg(i, ac, av)));
                else
                    PyList_SetItem(pyInletValue, i - 1,
                                   PyFloat_FromDouble(atom_getfloatarg(i, ac, av)));
            } else if (av[i].a_type == A_SYMBOL)
                PyList_SetItem(pyInletValue, i - 1,
                               PyUnicode_FromString(atom_getsymbolarg(i, ac, av)->s_name));
        }
        PyDict_SetItemString(x->KwArgsDict, key->s_name, pyInletValue);
    }
    return;
}

// ===========================================
void Py4pdLib_ProxyPointer(t_py4pdInlet_proxy *x, t_symbol *s, t_symbol *id) {
    LOG("Py4pdLib_ProxyPointer");
    (void)s;
    t_py *py4pd = (t_py *)x->p_master;
    t_py4pd_pValue *pArg;
    pArg = Py4pdUtils_GetPyObjPtr(id);
    if (!pArg) {
        pd_error(py4pd, "The object %s doesn't exist", id->s_name);
        return;
    }
    if (!pArg->PdOutCount) {
        Py_DECREF(py4pd->PyObjArgs[x->inletIndex]->pValue);
    }
    Py4pdUtils_CopyPy4pdValueStruct(pArg, py4pd->PyObjArgs[x->inletIndex]);
    if (!pArg->PdOutCount) {
        Py_INCREF(py4pd->PyObjArgs[x->inletIndex]->pValue);
    }
    return;
}

// =============================================
// void Py4pdLib_Pointer(t_py *x, t_symbol *s, t_gpointer *gp) {

void Py4pdLib_Pointer(t_py *x, t_symbol *s, t_symbol *id) {
    LOG("Py4pdLib_Pointer");
    (void)s;

    t_py4pd_pValue *pArg;
    pArg = Py4pdUtils_GetPyObjPtr(id);
    if (!pArg) {
        pd_error(x, "The object %s doesn't exist", id->s_name);
        return;
    }

    if (x->ObjType > PY4PD_AUDIOINOBJ) {
        if (!pArg->PdOutCount)
            Py_DECREF(x->PyObjArgs[0]->pValue);

        Py4pdUtils_CopyPy4pdValueStruct(pArg, x->PyObjArgs[0]);

        if (!pArg->PdOutCount)
            Py_INCREF(x->PyObjArgs[0]->pValue);
        x->AudioError = 0;
        return;
    }

    if (x->ObjType > PY4PD_AUDIOINOBJ)
        return;

    PyObject *pArgs = PyTuple_New(x->pArgsCount);
    PyTuple_SetItem(pArgs, 0, pArg->pValue);
    Py_INCREF(pArg->pValue);

    for (int i = 1; i < x->pArgsCount; i++) {
        PyTuple_SetItem(pArgs, i, x->PyObjArgs[i]->pValue);
        Py_INCREF(x->PyObjArgs[i]->pValue);
    }

    Py4pdUtils_RunPy(x, pArgs, x->KwArgsDict);
    Py_DECREF(pArgs);

    return;
}

// =====================================
void Py4pdLib_ProxyAnything(t_py4pdInlet_proxy *x, t_symbol *s, int ac, t_atom *av) {
    LOG("Py4pdLib_ProxyAnything");
    (void)s;
    t_py *py4pd = (t_py *)x->p_master;

    PyObject *pInletValue = Py4pdUtils_CreatePyObjFromPdArgs(s, ac, av);

    if (!py4pd->PyObjArgs[x->inletIndex]->PdOutCount)
        Py_DECREF(py4pd->PyObjArgs[x->inletIndex]->pValue);

    // obj
    py4pd->PyObjArgs[x->inletIndex]->PdOutCount = 0;
    py4pd->PyObjArgs[x->inletIndex]->ObjOwner = py4pd->ObjName;
    py4pd->PyObjArgs[x->inletIndex]->pValue = pInletValue;

    if (py4pd->ObjType == PY4PD_AUDIOOBJ || py4pd->ObjType == PY4PD_AUDIOINOBJ) {
        py4pd->AudioError = 0;
        return;
    }

    return;
}

// =====================================
void Py4pdLib_Anything(t_py *x, t_symbol *s, int ac, t_atom *av) {
    LOG("Py4pdLib_Anything");

    if (x->pFunction == NULL) {
        pd_error(x, "[%s]: No function defined", x->ObjName->s_name);
        return;
    }
    if (s == gensym("bang")) {
        Py4pdLib_Bang(x);
        return;
    }

    PyObject *pInletValue = Py4pdUtils_CreatePyObjFromPdArgs(s, ac, av);

    if (x->ObjType == PY4PD_AUDIOOUTOBJ) {
        x->PyObjArgs[0]->pValue = pInletValue;
        x->AudioError = 0;
        return;
    }

    if (x->ObjType > PY4PD_AUDIOINOBJ)
        return;

    PyObject *pArgs = PyTuple_New(x->pArgsCount);
    PyTuple_SetItem(pArgs, 0, pInletValue);

    for (int i = 1; i < x->pArgsCount; i++) {
        PyTuple_SetItem(pArgs, i, x->PyObjArgs[i]->pValue);
        Py_INCREF(x->PyObjArgs[i]->pValue); // This keep the reference.
    }

    if (x->ObjType > PY4PD_VISOBJ) {
        Py_INCREF(pInletValue);
        x->PyObjArgs[0]->PdOutCount = 0;
        x->PyObjArgs[0]->ObjOwner = x->ObjName;
        x->PyObjArgs[0]->pValue = pInletValue;
        x->pArgTuple = pArgs;
        Py_INCREF(x->pArgTuple);
        return;
    }

    Py4pdUtils_RunPy(x, pArgs, x->KwArgsDict);
    Py_DECREF(pArgs);
    PyErr_Clear();
    return;
}

// =====================================
void Py4pdLib_Bang(t_py *x) {
    LOG("Py4pdLib_Bang");
    if (x->pArgsCount != 0) {
        pd_error(x, "[%s] Bang can be used only with no arguments Function.", x->ObjName->s_name);
        return;
    }
    if (x->pFunction == NULL) {
        pd_error(x, "[%s] Function not defined", x->ObjName->s_name);
        return;
    }
    if (x->AudioOut) {
        return;
    }
    PyObject *pArgs = PyTuple_New(0);
    Py4pdUtils_RunPy(x, pArgs, x->KwArgsDict);
    Py_DECREF(pArgs);
}

// ====================================================
// ===================== Create Obj ===================
// ====================================================
void *Py4pdLib_NewObj(t_symbol *s, int argc, t_atom *argv) {
    LOG("Py4pdLib_NewObj");
    const char *objectName = s->s_name;
    char py4pd_objectName[MAXPDSTRING];
    snprintf(py4pd_objectName, sizeof(py4pd_objectName), "py4pd_ObjectDict_%s", objectName);
    PyObject *pd_module = PyImport_ImportModule("pd");

    if (pd_module == NULL) {
        pd_error(NULL, "[py4pd] Not possible import the pd module for %s, please report!",
                 s->s_name);
        Py4pdUtils_PrintError(NULL);
        return NULL;
    }

    PyObject *py4pd_capsule = PyObject_GetAttrString(pd_module, py4pd_objectName);
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

    PyObject *PY_objectClass = PyDict_GetItemString(PdDict, "ObjStruct");
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
    PyObject *RequireUserToSetOutletNumbers = PyDict_GetItemString(PdDict, "requireoutletn");
    PyObject *Py_ObjType = PyDict_GetItemString(PdDict, "Type");
    PyObject *pyFunction = PyDict_GetItemString(PdDict, "Function");

    int AuxOutlet = PyLong_AsLong(AuxOutletPy);
    int requireNofOutlets = PyLong_AsLong(RequireUserToSetOutletNumbers);
    PyCodeObject *code = (PyCodeObject *)PyFunction_GetCode(pyFunction);
    t_class *object_PY4PD_Class = (t_class *)PyLong_AsVoidPtr(PY_objectClass);
    t_py *x = (t_py *)pd_new(object_PY4PD_Class);

    x->VisMode = 0;
    x->PyObject = 1;
    x->ObjArgsCount = argc;
    x->StackLimit = 100;
    x->Canvas = canvas_getcurrent();
    t_canvas *c = x->Canvas;
    t_symbol *patch_dir = canvas_getdir(c);
    x->ObjName = gensym(objectName);
    x->Zoom = (int)x->Canvas->gl_zoom;
    x->IgnoreOnNone = PyLong_AsLong(ignoreOnNone);
    x->OutPyPointer = PyLong_AsLong(pyOUT);
    x->PyObjectPtr = Py4pdUtils_CreatePyObjPtr();
    x->FuncCalled = 1;
    x->pFunction = pyFunction;
    x->PdPatchPath = patch_dir; // set name of the home path
    x->PkgPath = patch_dir;     // set name of the packages path
    x->Playable = PyLong_AsLong(playable);
    x->pFuncName = gensym(PyUnicode_AsUTF8(code->co_name));
    x->pScriptName = gensym(PyUnicode_AsUTF8(code->co_filename));
    x->ObjType = PyLong_AsLong(Py_ObjType);
    x->nInlets = code->co_argcount;
    x->VectorSize = sys_getblksize();
    Py4pdUtils_SetObjConfig(x);
    x->pArgsCount = 0;
    int parseArgsRight = Py4pdUtils_ParseLibraryArguments(x, code, &argc, &argv);
    if (parseArgsRight == 0) {
        pd_error(NULL, "[%s] Error to parse arguments.", objectName);
        Py_DECREF(pd_module);
        Py_DECREF(py4pd_capsule);
        return NULL;
    }

    if (x->ObjType == PY4PD_VISOBJ)
        Py4pdUtils_CreatePicObj(x, PdDict, object_PY4PD_Class, argc, argv);

    x->PdObjArgs = malloc(sizeof(t_atom) * argc);
    for (int i = 0; i < argc; i++) {
        x->PdObjArgs[i] = argv[i];
    }
    x->ObjArgsCount = argc;
    if (x->PyObjArgs == NULL) {
        x->PyObjArgs = malloc(sizeof(t_py4pd_pValue *) * x->pArgsCount);
    }
    int inlets = Py4pdUtils_CreateObjInlets(pyFunction, x, Py4pdInletsProxy, argc, argv);
    if (inlets != 0) {
        free(x->PdObjArgs);
        free(x->PyObjArgs);
        return NULL;
    }

    if (x->pArgsCount < 2) {
        x->AudioError = 0;
    } else {
        x->AudioError = 1;
    }

    if (x->ObjType > 1)
        x->pArgTuple = PyTuple_New(x->pArgsCount);

    if (!PyLong_AsLong(nooutlet)) {
        if (x->ObjType != PY4PD_AUDIOOUTOBJ && x->ObjType != PY4PD_AUDIOOBJ)
            x->MainOut = outlet_new(&x->obj, 0);
        else {
            x->MainOut = outlet_new(&x->obj, &s_signal);
            x->nOutlets = 1;
        }
    }
    if (requireNofOutlets) {
        if (x->nOutlets == -1) {
            pd_error(NULL,
                     "[%s]: This function require that you set the number of "
                     "auxiliar outlets "
                     "using -outn {number_of_outlets} flag",
                     objectName);
            Py_DECREF(pd_module);
            Py_DECREF(py4pd_capsule);
            return NULL;
        } else {
            AuxOutlet = x->nOutlets - 1;
            if (AuxOutlet < 0) {
                AuxOutlet = 0;
            }
        }
    }

    if (AuxOutlet > 0) {
        x->ExtrasOuts = (py4pdExtraOuts *)getbytes(AuxOutlet * sizeof(py4pdExtraOuts));
        x->ExtrasOuts->outCount = AuxOutlet;
        t_atom defarg[AuxOutlet];
        t_atom *ap;
        py4pdExtraOuts *u;
        int i;
        for (i = 0, u = x->ExtrasOuts, ap = defarg; i < AuxOutlet; i++, u++, ap++) {
            u->u_outlet = outlet_new(&x->obj, &s_anything);
        }
    }

    objCount++; // To clear memory when closing the patch
    Py_DECREF(pd_module);
    Py_DECREF(py4pd_capsule);
    return (x);
}

// ====================================================
// ===================== pd.add_object ================
// ====================================================
PyObject *Py4pdLib_AddObj(PyObject *self, PyObject *args, PyObject *keywords) {
    (void)self;
    char *objectName;
    const char *helpPatch;
    PyObject *Function;
    int w = 250, h = 250;
    int objpyout = 0;
    int nooutlet = 0;
    int added2pd_info = 0;
    int personalisedHelp = 0;
    int ignoreNoneReturn = 0;
    const char *gifImage = NULL;
    int auxOutlets = 0;
    int require_outlet_n = 0;
    int playableInt = 0;
    t_py *py4pd = Py4pdUtils_GetObject(self);

    size_t totalLength;
    const char *helpFolder = "/help/";
    if (py4pd->LibraryFolder != NULL) {
        totalLength = strlen(py4pd->LibraryFolder->s_name) + strlen(helpFolder) + 1;
    } else {
        pd_error(py4pd, "[py4pd] Library Folder is NULL, some help patches may "
                        "not be found");
        return NULL;
    }

    char *helpFolderCHAR = (char *)malloc(totalLength * sizeof(char));
    if (helpFolderCHAR == NULL) {
        pd_error(py4pd, "[py4pd] Error allocating memory (code 001)");
        return NULL;
    }

    pd_snprintf(helpFolderCHAR, totalLength, "%s", py4pd->LibraryFolder->s_name);
    strcat(helpFolderCHAR, helpFolder);

    if (!PyArg_ParseTuple(args, "Os", &Function, &objectName)) {
        post("[Python]: Error parsing arguments");
        free(helpFolderCHAR);
        return NULL;
    }

    int objectType = 0;
    if (keywords != NULL) {
        if (PyDict_Contains(keywords, PyUnicode_FromString("type"))) {
            PyObject *type = PyDict_GetItemString(keywords, "type");
            objectType = PyLong_AsLong(type);
        }
        if (PyDict_Contains(keywords, PyUnicode_FromString("fig_size"))) {
            PyObject *figsize = PyDict_GetItemString(keywords, "fig_size");
            PyObject *width = PyTuple_GetItem(figsize, 0);
            PyObject *height = PyTuple_GetItem(figsize, 1);
            w = PyLong_AsLong(width);
            h = PyLong_AsLong(height);
        }
        if (PyDict_Contains(keywords, PyUnicode_FromString("py_out"))) {
            PyObject *output = PyDict_GetItemString(keywords, "py_out");
            if (output == Py_True) {
                objpyout = 1;
            }
        }
        if (PyDict_Contains(keywords, PyUnicode_FromString("no_outlet"))) {
            PyObject *output = PyDict_GetItemString(keywords, "no_outlet");
            if (output == Py_True) {
                nooutlet = 1;
            }
        }
        if (PyDict_Contains(keywords, PyUnicode_FromString("require_outlet_n"))) {
            PyObject *output = PyDict_GetItemString(keywords, "require_outlet_n");
            if (output == Py_True) {
                require_outlet_n = 1;
            }
        }
        if (PyDict_Contains(keywords, PyUnicode_FromString("added2pd_info"))) {
            PyObject *output = PyDict_GetItemString(keywords, "added2pd_info");
            if (output == Py_True) {
                added2pd_info = 1;
            }
        }
        if (PyDict_Contains(keywords, PyUnicode_FromString("help_patch"))) {
            PyObject *helpname = PyDict_GetItemString(keywords, "help_patch");
            helpPatch = PyUnicode_AsUTF8(helpname);
            const char *suffix = "-help.pd";
            int suffixLen = strlen(suffix);
            int helpPatchLen = strlen(helpPatch);
            if (helpPatchLen >= suffixLen) {
                if (strcmp(helpPatch + (helpPatchLen - suffixLen), suffix) == 0) {
                    personalisedHelp = 1;
                    char helpPatchName[MAXPDSTRING];
                    pd_snprintf(helpPatchName, sizeof(helpPatchName), "%.*s",
                                helpPatchLen - suffixLen, helpPatch);
                    helpPatch = helpPatchName;
                } else {
                    pd_error(NULL, "Help patch must end with '-help.pd'");
                }
            } else {
                pd_error(NULL, "Help patch must end with '-help.pd'");
            }
        }
        if (PyDict_Contains(keywords, PyUnicode_FromString("ignore_none"))) {
            PyObject *noneReturn = PyDict_GetItemString(keywords, "ignore_none");
            if (noneReturn == Py_True) {
                ignoreNoneReturn = 1;
            }
        }
        if (PyDict_Contains(keywords, PyUnicode_FromString("image"))) {
            PyObject *type = PyDict_GetItemString(keywords, "image");
            gifImage = PyUnicode_AsUTF8(type);
        }
        if (PyDict_Contains(keywords, PyUnicode_FromString("n_extra_outlets"))) {
            PyObject *type = PyDict_GetItemString(keywords, "n_extra_outlets");
            auxOutlets = PyLong_AsLong(type) + 1;
            // to keep compatibility
        }
        if (PyDict_Contains(keywords, PyUnicode_FromString("n_outlets"))) {
            PyObject *type = PyDict_GetItemString(keywords, "n_outlets");
            auxOutlets = PyLong_AsLong(type) - 1;
            // need to thing about this, it is weird
        }
        if (PyDict_Contains(keywords, PyUnicode_FromString("playable"))) {
            PyObject *playable = PyDict_GetItemString(keywords, "playable");
            if (playable == Py_True) {
                playableInt = 1;
            }
        }
    }

    class_set_extern_dir(gensym(helpFolderCHAR));
    t_class *localClass;

    if (objectType == PY4PD_NORMALOBJ) {
        localClass =
            class_new(gensym(objectName), (t_newmethod)Py4pdLib_NewObj,
                      (t_method)Py4pdUtils_FreeObj, sizeof(t_py), CLASS_DEFAULT, A_GIMME, 0);
    } else if (objectType == PY4PD_VISOBJ) {
        localClass =
            class_new(gensym(objectName), (t_newmethod)Py4pdLib_NewObj,
                      (t_method)Py4pdUtils_FreeObj, sizeof(t_py), CLASS_DEFAULT, A_GIMME, 0);
    } else if (objectType == PY4PD_AUDIOINOBJ) {
        localClass =
            class_new(gensym(objectName), (t_newmethod)Py4pdLib_NewObj,
                      (t_method)Py4pdUtils_FreeObj, sizeof(t_py), CLASS_MULTICHANNEL, A_GIMME, 0);
    } else if (objectType == PY4PD_AUDIOOUTOBJ) {
        localClass =
            class_new(gensym(objectName), (t_newmethod)Py4pdLib_NewObj,
                      (t_method)Py4pdUtils_FreeObj, sizeof(t_py), CLASS_MULTICHANNEL, A_GIMME, 0);
    } else if (objectType == PY4PD_AUDIOOBJ) {
        localClass =
            class_new(gensym(objectName), (t_newmethod)Py4pdLib_NewObj,
                      (t_method)Py4pdUtils_FreeObj, sizeof(t_py), CLASS_MULTICHANNEL, A_GIMME, 0);
    } else {
        PyErr_SetString(PyExc_TypeError, "Object type not supported, check the spelling");
        return NULL;
    }

    // Add configs to the object
    PyObject *nestedDict = PyDict_New(); // New
    PyDict_SetItemString(nestedDict, "Function", Function);

    PyObject *Py_ObjType = PyLong_FromLong(objectType);
    PyDict_SetItemString(nestedDict, "Type", Py_ObjType);
    Py_DECREF(Py_ObjType);

    PyObject *Py_LibraryFolder = PyUnicode_FromString(py4pd->LibraryFolder->s_name);
    PyDict_SetItemString(nestedDict, "LibraryFolder", Py_LibraryFolder);
    Py_DECREF(Py_LibraryFolder);

    PyObject *Pd_ObjStruct = PyLong_FromVoidPtr(localClass);
    PyDict_SetItemString(nestedDict, "ObjStruct", Pd_ObjStruct);
    Py_DECREF(Pd_ObjStruct);

    PyObject *Py_Width = PyLong_FromLong(w);
    PyDict_SetItemString(nestedDict, "width", Py_Width);
    Py_DECREF(Py_Width);

    PyObject *Py_Height = PyLong_FromLong(h);
    PyDict_SetItemString(nestedDict, "height", Py_Height);
    Py_DECREF(Py_Height);

    PyObject *Py_Playable = PyLong_FromLong(playableInt);
    PyDict_SetItemString(nestedDict, "Playable", Py_Playable);
    Py_DECREF(Py_Playable);

    if (gifImage != NULL) {
        PyObject *Py_GifImage = PyUnicode_FromString(gifImage);
        PyDict_SetItemString(nestedDict, "Gif", Py_GifImage);
        Py_DECREF(Py_GifImage);
    }

    PyObject *Py_ObjOuts = PyLong_FromLong(objpyout);
    PyDict_SetItemString(nestedDict, "pyout", PyLong_FromLong(objpyout));
    Py_DECREF(Py_ObjOuts);

    PyObject *Py_NoOutlet = PyLong_FromLong(nooutlet);
    PyDict_SetItemString(nestedDict, "nooutlet", Py_NoOutlet);
    Py_DECREF(Py_NoOutlet);

    PyObject *Py_RequireOutletN = PyLong_FromLong(require_outlet_n);
    PyDict_SetItemString(nestedDict, "requireoutletn", Py_RequireOutletN);
    Py_DECREF(Py_RequireOutletN);

    PyObject *Py_auxOutlets = PyLong_FromLong(auxOutlets);
    PyDict_SetItemString(nestedDict, "py4pdAuxOutlets", Py_auxOutlets);
    Py_DECREF(Py_auxOutlets);

    PyObject *Py_ObjName = PyUnicode_FromString(objectName);
    PyDict_SetItemString(nestedDict, "name", Py_ObjName);
    Py_DECREF(Py_ObjName);

    PyObject *Py_IgnoreNoneReturn = PyLong_FromLong(ignoreNoneReturn);
    PyDict_SetItemString(nestedDict, "IgnoreNone", Py_IgnoreNoneReturn);
    Py_DECREF(Py_IgnoreNoneReturn);

    PyObject *objectDict = PyDict_New();
    PyDict_SetItemString(objectDict, objectName, nestedDict);
    PyObject *py4pd_capsule = PyCapsule_New(objectDict, objectName, NULL);
    char py4pd_objectName[MAXPDSTRING];
    pd_snprintf(py4pd_objectName, MAXPDSTRING, "py4pd_ObjectDict_%s", objectName);

    PyObject *pdModule = PyImport_ImportModule("pd");
    PyModule_AddObject(pdModule, py4pd_objectName, py4pd_capsule);
    Py_DECREF(pdModule);

    // =====================================
    PyCodeObject *code = (PyCodeObject *)PyFunction_GetCode(Function);
    int py_args = code->co_argcount;

    // special methods
    if (objectType == PY4PD_NORMALOBJ) {
        class_addmethod(localClass, (t_method)Py4pdUtils_Click, gensym("click"), 0, 0);
        if (playableInt == 1) {
            class_addmethod(localClass, (t_method)Py4pdPlayer_Play, gensym("play"), A_GIMME, 0);
            class_addmethod(localClass, (t_method)Py4pdPlayer_Stop, gensym("stop"), 0, 0);
            class_addmethod(localClass, (t_method)Py4pdPlayer_Clear, gensym("clear"), 0, 0);
        }
    } else if (objectType == PY4PD_VISOBJ) {
        if (playableInt == 1) {
            class_addmethod(localClass, (t_method)Py4pdPlayer_Play, gensym("play"), A_GIMME, 0);
            class_addmethod(localClass, (t_method)Py4pdPlayer_Stop, gensym("stop"), 0, 0);
            class_addmethod(localClass, (t_method)Py4pdPlayer_Clear, gensym("clear"), 0, 0);
        }
        class_addmethod(localClass, (t_method)Py4pdPic_Zoom, gensym("zoom"), A_CANT, 0);
        class_addmethod(localClass, (t_method)Py4pd_SetPythonPointersUsage, gensym("pointers"),
                        A_FLOAT, 0);
        // class_setsavefn(localClass, &Py4pdLib_PicSave);
    }
    // AUDIOIN
    else if (objectType == PY4PD_AUDIOINOBJ) {
        class_addmethod(localClass, (t_method)Py4pdUtils_Click, gensym("click"), 0, 0);
        class_addmethod(localClass, (t_method)Py4pdAudio_Dsp, gensym("dsp"), A_CANT,
                        0); // add a method to a class
        CLASS_MAINSIGNALIN(localClass, t_py, Py4pdAudio);
    }
    // AUDIOOUT
    else if (objectType == PY4PD_AUDIOOUTOBJ) {
        class_addmethod(localClass, (t_method)Py4pdUtils_Click, gensym("click"), 0, 0);
        class_addmethod(localClass, (t_method)Py4pdAudio_Dsp, gensym("dsp"), A_CANT,
                        0); // add a method to a class

    }
    // AUDIO
    else if (objectType == PY4PD_AUDIOOBJ) {
        class_addmethod(localClass, (t_method)Py4pdUtils_Click, gensym("click"), 0, 0);
        class_addmethod(localClass, (t_method)Py4pdAudio_Dsp, gensym("dsp"), A_CANT,
                        0); // add a method to a class
        CLASS_MAINSIGNALIN(localClass, t_py, Py4pdAudio);
    } else {
        PyErr_SetString(PyExc_TypeError, "Object type not supported, check the spelling");
        return NULL;
    }
    // add methods to the class
    class_addanything(localClass, Py4pdLib_Anything);
    class_addmethod(localClass, (t_method)Py4pdLib_Pointer, gensym("PyObject"), A_SYMBOL, A_SYMBOL,
                    0);
    class_addmethod(localClass, (t_method)Py4pd_PrintDocs, gensym("doc"), 0, 0);
    class_addmethod(localClass, (t_method)Py4pdLib_SetKwargs, gensym("kwargs"), A_GIMME, 0);
    class_addmethod(localClass, (t_method)Py4pd_SetPythonPointersUsage, gensym("pointers"), A_FLOAT,
                    0);
    class_addmethod(localClass, (t_method)Py4pdLib_ReloadObject, gensym("reload"), 0, 0);

    // add help patch
    if (personalisedHelp == 1) {
        class_sethelpsymbol(localClass, gensym(helpPatch));
    } else {
        class_sethelpsymbol(localClass, gensym(objectName));
    }
    free(helpFolderCHAR);

    if (py_args != 0) {
        Py4pdInletsProxy = class_new(gensym("_py4pdInlets_proxy"), 0, 0, sizeof(t_py4pdInlet_proxy),
                                     CLASS_DEFAULT, 0);
        class_addanything(Py4pdInletsProxy, Py4pdLib_ProxyAnything);
        class_addmethod(Py4pdInletsProxy, (t_method)Py4pdLib_ProxyPointer, gensym("PyObject"),
                        A_SYMBOL, A_SYMBOL, 0);
    }
    if (added2pd_info == 1) {
        post("[py4pd]: Object {%s} added to PureData", objectName);
    }
    class_set_extern_dir(&s_);
    Py_RETURN_TRUE;
}
