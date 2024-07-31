#include "py4pd.h"

#include "ext-class.h"
#include "module.h"
#include "pic.h"
#include "utils.h"

#include <m_imp.h>

#include <string.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PY4PD_NUMPYARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_25_API_VERSION
#include <numpy/arrayobject.h>

/*
This file serves as a versatile collection of functions utilized within the
Py4pd object, providing a diverse set of utilities for various tasks and
functionalities. The functions are organized following distinct subsets, each
dedicated to specific purposes.
*/

// ====================================================
// ====================== Utilities ===================
// ====================================================
/*
 * @brief See if str is a number or a dot
 * @param str is the string to check
 * @return 1 if it is a number or a dot, 0 otherwise
 */
int Py4pdUtils_IsNumericOrDot(const char *str) {
    int hasDot = 0;
    while (*str) {
        if (isdigit(*str)) {
            str++;
        } else if (*str == '.' && !hasDot) {
            hasDot = 0;
            str++;
        } else {
            return 0;
        }
    }
    return 1;
}

// =====================================================================
/*
 * @brief Remove some char from a string
 * @param str is the string to remove the char
 * @param c is the char to remove
 * @return the string without the char
 */
void Py4pdUtils_RemoveChar(char *str, char c) {
    int i, j;
    for (i = 0, j = 0; str[i] != '\0'; i++) {
        if (str[i] != c) {
            str[j] = str[i];
            j++;
        }
    }
    str[j] = '\0';
}

// =====================================================================
/*
 * @brief Replace one char by the other
 * @param str
 * @param char2replace char that will be replaced
 * @param newchar new char
 */
void Py4pdUtils_ReplaceChar(char *str, char char2replace, char newchar) {
    if (str == NULL) {
        printf("Error: Input string is NULL.\n");
        return;
    }
    while (*str != '\0') {
        if (*str == char2replace) {
            *str = newchar;
        }
        str++;
    }
}
// =====================================================================
/*
 * @brief Convert and output Python Values to PureData values
 * @param x is the py4pd object
 * @param pValue is the Python value to convert
 * @return nothing, but output the value to the outlet
 */
void Py4pdUtils_FromSymbolSymbol(t_py *x, t_symbol *s, t_outlet *outlet) {
    (void)x;
    // new and redone - Derek Kwan
    long unsigned int seplen = strlen(" ");
    seplen++;
    char *sep = t_getbytes(seplen * sizeof(*sep));
    memset(sep, '\0', seplen);
    Py4pdUtils_Strlcpy(sep, " ", seplen);
    if (s) {
        long unsigned int iptlen = strlen(s->s_name);
        t_atom *out = t_getbytes(iptlen * sizeof(*out));
        iptlen++;
        char *newstr = t_getbytes(iptlen * sizeof(*newstr));
        memset(newstr, '\0', iptlen);
        Py4pdUtils_Strlcpy(newstr, s->s_name, iptlen);
        int atompos = 0; // position in atom
        char *ret = Py4pdUtils_Mtok(newstr, sep);
        char *err; // error pointer
        while (ret != NULL) {
            if (strlen(ret) > 0) {
                int allnums = Py4pdUtils_IsNumericOrDot(ret); // flag if all nums
                if (allnums) { // if errpointer is at beginning, that means
                               // we've got a float
                    double f = strtod(ret, &err);
                    SETFLOAT(&out[atompos], (t_float)f);
                } else { // else we're dealing with a symbol
                    t_symbol *cursym = gensym(ret);
                    SETSYMBOL(&out[atompos], cursym);
                };
                atompos++; // increment position in atom
            };
            ret = Py4pdUtils_Mtok(NULL, sep);
        };
        if (out->a_type == A_SYMBOL) {
            outlet_anything(outlet, out->a_w.w_symbol, atompos - 1, out + 1);
        } else if (out->a_type == A_FLOAT && atompos >= 1) {
            outlet_list(outlet, &s_list, atompos, out);
        }
        t_freebytes(out, iptlen * sizeof(*out));
        t_freebytes(newstr, iptlen * sizeof(*newstr));
    };
    t_freebytes(sep, seplen * sizeof(*sep));
}

// ====================================================
// ====================== Libraries ===================
// ====================================================
/*
 * @brief This function parse the arguments for pd Objects created with the
 * library
 * @param x is the py4pd object
 * @param code is the code object of the function
 * @param argc is the number of arguments
 * @param argv is the arguments
 * @return 1 if all arguments are ok, 0 if not
 */
int Py4pdUtils_ParseLibraryArguments(t_py *x, PyCodeObject *code, int *argcPtr, t_atom **argvPtr) {
    int argsNumberDefined = 0;
    x->nOutlets = -1;
    x->nChs = 1;

    int argc = *argcPtr;
    t_atom *argv = *argvPtr;

    int i, j;
    for (i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            if (strcmp(argv[i].a_w.w_symbol->s_name, "-n_args") == 0 ||
                strcmp(argv[i].a_w.w_symbol->s_name, "-a") == 0) {

                if (argc > (i + 2)) {
                    pd_error(x, "The -a or -n_args must be created in the end "
                                "of the object paramerts");
                    return 0; // TODO: error should be -1 NOT 0;
                }

                if (i + 1 < argc) {
                    if (argv[i + 1].a_type == A_FLOAT) {
                        x->pArgsCount = atom_getintarg(i + 1, argc, argv);
                        argsNumberDefined = 1;
                        for (j = i; j < argc; j++) {
                            argv[j] = argv[j + 2];
                            (*argvPtr)[j] = (*argvPtr)[j + 2];
                            *argcPtr = *argcPtr - 1;
                        }
                    }
                }
            } else if (strcmp(argv[i].a_w.w_symbol->s_name, "-outn") == 0) {
                if (argv[i + 1].a_type == A_FLOAT) {
                    x->nOutlets = atom_getintarg(i + 1, argc, argv);
                    // remove -outn and the number of outlets from the arguments
                    // list
                    for (j = i; j < argc; j++) {
                        argv[j] = argv[j + 2];
                        (*argvPtr)[j] = (*argvPtr)[j + 2];
                        // *argcPtr = *argcPtr - 2;
                    }
                } else {
                    x->nOutlets = -1; // -1 means that the number of outlets
                                      // is not defined
                }
            } else if (strcmp(argv[i].a_w.w_symbol->s_name, "-ch") == 0 ||
                       strcmp(argv[i].a_w.w_symbol->s_name, "-channels") == 0) {
                if (argv[i + 1].a_type == A_FLOAT) {
                    x->nChs = atom_getintarg(i + 1, argc, argv);
                    for (j = i; j < argc; j++) {
                        argv[j] = argv[j + 2];
                        (*argvPtr)[j] = (*argvPtr)[j + 2];
                        // *argcPtr = *argcPtr - 2;
                    }
                }
            }
        }
    }
    if (code->co_flags & CO_VARARGS) {
        if (argsNumberDefined == 0) {
            pd_error(x,
                     "[%s] this function uses *args, "
                     "you need to specify the number of arguments "
                     "using -n_args (-a for short) {number}",
                     x->ObjName->s_name);
            return 0;
        }
        x->UsepArgs = 1;
    }
    if (code->co_flags & CO_VARKEYWORDS) {
        x->UsepKwargs = 1;
    }
    if (code->co_argcount != 0) {
        if (x->pArgsCount == 0) {
            x->pArgsCount = code->co_argcount;
        } else {
            x->pArgsCount = x->pArgsCount + code->co_argcount;
        }
    }
    return 1;
}

// ====================================================
int Py4pdUtils_CreateObjInlets(PyObject *function, t_py *x, t_class *py4pdInlets_proxy_class,
                               int argc, t_atom *argv) {
    (void)function;
    t_pd **py4pdInlet_proxies;
    int i;
    int pyFuncArgs = x->pArgsCount - 1;

    PyObject *defaults =
        PyObject_GetAttrString(function, "__defaults__"); // TODO:, WHERE CLEAR THIS?
    int defaultsCount = PyTuple_Size(defaults);

    if (x->UsepArgs && defaultsCount > 0) {
        pd_error(x, "[py4pd] You can't use *args and defaults at the same time");
        return -1;
    }
    int indexWhereStartDefaults = x->pArgsCount - defaultsCount;

    t_py4pd_pValue *PyPtrValueMain = (t_py4pd_pValue *)malloc(sizeof(t_py4pd_pValue));
    PyPtrValueMain->PdOutCount = 0;
    PyPtrValueMain->ObjOwner = x->ObjName;
    if (indexWhereStartDefaults == 0) {
        PyPtrValueMain->pValue = PyTuple_GetItem(defaults, 0);
        x->PyObjArgs[0] = PyPtrValueMain;
    } else {
        Py_INCREF(Py_None);
        PyPtrValueMain->pValue = Py_None;
        x->PyObjArgs[0] = PyPtrValueMain;
    }

    if (pyFuncArgs != 0) {
        py4pdInlet_proxies = (t_pd **)getbytes((pyFuncArgs + 1) * sizeof(*py4pdInlet_proxies));
        for (i = 0; i < pyFuncArgs; i++) {
            py4pdInlet_proxies[i] = pd_new(py4pdInlets_proxy_class);
            t_py4pdInlet_proxy *y = (t_py4pdInlet_proxy *)py4pdInlet_proxies[i];
            y->p_master = x;
            y->inletIndex = i + 1;
            inlet_new((t_object *)x, (t_pd *)y, 0, 0);
            t_py4pd_pValue *PyPtrValue = (t_py4pd_pValue *)malloc(sizeof(t_py4pd_pValue));
            PyPtrValue->PdOutCount = 0;
            PyPtrValue->ObjOwner = x->ObjName;
            if (i + 1 >= indexWhereStartDefaults) {
                PyPtrValue->pValue = PyTuple_GetItem(defaults, (i + 1) - indexWhereStartDefaults);
            } else {
                PyPtrValue->pValue = Py_None;
                Py_INCREF(Py_None);
            }
            x->PyObjArgs[i + 1] = PyPtrValue;
        }
        int argNumbers = x->pArgsCount;

        for (i = 0; i < argNumbers; i++) {
            if (i <= argc) {
                if (argv[i].a_type == A_FLOAT) {
                    int isInt = atom_getintarg(i, argc, argv) == atom_getfloatarg(i, argc, argv);
                    if (isInt) {
                        x->PyObjArgs[i]->pValue = PyLong_FromLong(atom_getfloatarg(i, argc, argv));
                    } else {
                        x->PyObjArgs[i]->pValue =
                            PyFloat_FromDouble(atom_getfloatarg(i, argc, argv));
                    }
                }

                else if (argv[i].a_type == A_SYMBOL) {
                    if (strcmp(atom_getsymbolarg(i, argc, argv)->s_name, "None") == 0) {
                        Py_INCREF(Py_None);
                        x->PyObjArgs[i]->pValue = Py_None;
                    } else {
                        x->PyObjArgs[i]->pValue =
                            PyUnicode_FromString(atom_getsymbolarg(i, argc, argv)->s_name);
                    }
                } else if (x->PyObjArgs[i]->pValue == NULL) {
                    Py_INCREF(Py_None);
                    x->PyObjArgs[i]->pValue = Py_None;
                }
            } else if (x->PyObjArgs[i]->pValue == NULL) {
                Py_INCREF(Py_None);
                x->PyObjArgs[i]->pValue = Py_None;
            }
        }
    }
    return 0;
}

// ====================================================
void Py4pdUtils_ExtraInletAnything(t_py4pdInlet_proxy *x, t_symbol *s, int ac, t_atom *av) {
    (void)s;
    t_py *py4pd = (t_py *)x->p_master;
    PyObject *pyInletValue = NULL;

    if (ac == 0)
        pyInletValue = PyUnicode_FromString(s->s_name);
    else if ((s == gensym("list") || s == gensym("anything")) && ac > 1) {
        pyInletValue = PyList_New(ac);
        for (int i = 0; i < ac; i++) {
            if (av[i].a_type == A_FLOAT) {
                int isInt = atom_getintarg(i, ac, av) == atom_getfloatarg(i, ac, av);
                if (isInt)
                    PyList_SetItem(pyInletValue, i, PyLong_FromLong(atom_getintarg(i, ac, av)));
                else
                    PyList_SetItem(pyInletValue, i,
                                   PyFloat_FromDouble(atom_getfloatarg(i, ac, av)));
            } else if (av[i].a_type == A_SYMBOL)
                PyList_SetItem(pyInletValue, i,
                               PyUnicode_FromString(atom_getsymbolarg(i, ac, av)->s_name));
        }
    } else if ((s == gensym("float") || s == gensym("symbol")) && ac == 1) {
        if (av[0].a_type == A_FLOAT) {
            int isInt = atom_getintarg(0, ac, av) == atom_getfloatarg(0, ac, av);
            if (isInt)
                pyInletValue = PyLong_FromLong(atom_getintarg(0, ac, av));
            else
                pyInletValue = PyFloat_FromDouble(atom_getfloatarg(0, ac, av));
        } else if (av[0].a_type == A_SYMBOL)
            pyInletValue = PyUnicode_FromString(atom_getsymbolarg(0, ac, av)->s_name);
    } else {
        pyInletValue = PyList_New(ac + 1);
        PyList_SetItem(pyInletValue, 0, PyUnicode_FromString(s->s_name));
        for (int i = 0; i < ac; i++) {
            if (av[i].a_type == A_FLOAT) {
                int isInt = atom_getintarg(i, ac, av) == atom_getfloatarg(i, ac, av);
                if (isInt)
                    PyList_SetItem(pyInletValue, i + 1, PyLong_FromLong(atom_getintarg(i, ac, av)));
                else
                    PyList_SetItem(pyInletValue, i + 1,
                                   PyFloat_FromDouble(atom_getfloatarg(i, ac, av)));
            } else if (av[i].a_type == A_SYMBOL)
                PyList_SetItem(pyInletValue, i + 1,
                               PyUnicode_FromString(atom_getsymbolarg(i, ac, av)->s_name));
        }
    }
    if (!py4pd->PyObjArgs[x->inletIndex]->PdOutCount)
        Py_DECREF(py4pd->PyObjArgs[x->inletIndex]->pValue);

    py4pd->PyObjArgs[x->inletIndex]->PdOutCount = 0;
    py4pd->PyObjArgs[x->inletIndex]->ObjOwner = py4pd->ObjName;
    py4pd->PyObjArgs[x->inletIndex]->pValue = pyInletValue;

    if (py4pd->ObjType == PY4PD_AUDIOOBJ || py4pd->ObjType == PY4PD_AUDIOINOBJ) {
        py4pd->AudioError = 0;
        return;
    }

    return;
}

// ====================================================
PyObject *Py4pdUtils_CreatePyObjFromPdArgs(t_symbol *s, int argc, t_atom *argv) {

    PyObject *pArgs = NULL;

    if (argc == 0) {
        if (s != NULL) {
            pArgs = PyUnicode_FromString(s->s_name);
        }
    } else if ((s == gensym("list") || s == gensym("anything") || s == NULL) && argc > 1) {
        pArgs = PyList_New(argc);
        for (int i = 0; i < argc; i++) {
            t_atomtype aType = argv[i].a_type;
            if (aType == A_FLOAT) {
                int pdInt = atom_getintarg(i, argc, argv);
                float pdFloat = atom_getfloatarg(i, argc, argv);
                int isInt = pdInt == pdFloat;
                if (isInt) {
                    PyObject *pInt = PyLong_FromLong(pdInt);
                    PyList_SetItem(pArgs, i, pInt);
                } else {
                    PyObject *pFloat = PyFloat_FromDouble(pdFloat);
                    PyList_SetItem(pArgs, i, pFloat);
                }
            } else if (aType == A_SYMBOL) {
                t_symbol *pdSymbol = atom_getsymbolarg(i, argc, argv);
                PyObject *pSymbol = PyUnicode_FromString(pdSymbol->s_name);
                PyList_SetItem(pArgs, i, pSymbol);
            } else {
                pd_error(NULL, "[py4pd] py4pd just support floats or symbols");
            }
        }
    } else if ((s == gensym("float") || s == gensym("symbol") || s == NULL) && argc == 1) {
        if (argv[0].a_type == A_FLOAT) {
            int isInt = atom_getintarg(0, argc, argv) == atom_getfloatarg(0, argc, argv);
            if (isInt)
                pArgs = PyLong_FromLong(atom_getintarg(0, argc, argv));
            else
                pArgs = PyFloat_FromDouble(atom_getfloatarg(0, argc, argv));
        } else if (argv[0].a_type == A_SYMBOL)
            pArgs = PyUnicode_FromString(atom_getsymbolarg(0, argc, argv)->s_name);
    } else {
        pArgs = PyList_New(argc + 1);
        PyList_SetItem(pArgs, 0, PyUnicode_FromString(s->s_name));
        for (int i = 0; i < argc; i++) {
            if (argv[i].a_type == A_FLOAT) {
                int isInt = atom_getintarg(i, argc, argv) == atom_getfloatarg(i, argc, argv);
                if (isInt)
                    PyList_SetItem(pArgs, i + 1, PyLong_FromLong(atom_getintarg(i, argc, argv)));
                else
                    PyList_SetItem(pArgs, i + 1,
                                   PyFloat_FromDouble(atom_getfloatarg(i, argc, argv)));
            } else if (argv[i].a_type == A_SYMBOL)
                PyList_SetItem(pArgs, i + 1,
                               PyUnicode_FromString(atom_getsymbolarg(i, argc, argv)->s_name));
        }
    }

    return pArgs;
}

// ====================================================
void Py4pdUtils_ExtraInletPointer(t_py4pdInlet_proxy *x, t_symbol *s, t_gpointer *gp) {
    (void)s;
    t_py *py4pd = (t_py *)x->p_master;
    t_py4pd_pValue *pArg;
    pArg = (t_py4pd_pValue *)gp;
    if (!pArg->PdOutCount)
        Py_DECREF(py4pd->PyObjArgs[x->inletIndex]->pValue);

    Py4pdUtils_CopyPy4pdValueStruct(pArg, py4pd->PyObjArgs[x->inletIndex]);

    if (!pArg->PdOutCount)
        Py_INCREF(py4pd->PyObjArgs[x->inletIndex]->pValue);

    return;
}

// ====================================================
// ========== Iteration between Pd2py|Py2pd ===========
// ====================================================
/*
 * @brief add PureData Object struct to Python Module, because this we are able
 * to config Pd Obj from Python
 * @param x is the py4pd object
 * @param capsule is the PyObject (capsule)
 * @return the pointer to the py capsule
 */

PyObject *Py4pdUtils_AddPdObject(t_py *x) {
    LOG("Py4pdUtils_AddPdObject");
    PyObject *PdModule = PyModule_GetDict(PyImport_AddModule("pd"));
    PyObject *objCapsule;
    if (PdModule != NULL) {
        objCapsule = PyDict_GetItemString(PdModule, "py4pd"); // borrowed reference
        if (objCapsule != NULL) {
            PyCapsule_SetPointer(objCapsule, x);
        } else {
            objCapsule = PyCapsule_New(x, "py4pd", NULL);
            PyObject *pdModule = PyImport_ImportModule("pd");
            int addSucess = PyModule_AddObject(pdModule, "py4pd", objCapsule);
            if (addSucess != 0) {
                pd_error(x, "[py4pd] Failed to add object to Python");
                return NULL;
            }

            Py_DECREF(pdModule);
        }
    } else {
        pd_error(x, "[py4pd] Could not get the main module");
        objCapsule = NULL;
    }
    return objCapsule;
}

// =====================================================================
/*
 * @brief Convert one PyObject pointer to a PureData pointer
 * @param pValue is the PyObject pointer to convert
 * @return the PureData pointer
 */
PyObject *Py4pdUtils_PyObjectToPointer(PyObject *pValue) {
    PyObject *pValuePointer = PyLong_FromVoidPtr(pValue);
    return pValuePointer;
}

// =====================================================================
/*
 * @brief Convert one PureData pointer to a PyObject pointer
 * @param p is the PureData pointer to convert
 * @return the PyObject pointer
 */
PyObject *Py4pdUtils_PointerToPyObject(PyObject *p) {
    PyObject *pValue = PyLong_AsVoidPtr(p);
    return pValue;
}

// =====================================================================
void Py4pdUtils_CopyPy4pdValueStruct(t_py4pd_pValue *src, t_py4pd_pValue *dest) {
    dest->pValue = src->pValue;
    dest->PdOutCount = src->PdOutCount;
    dest->ObjOwner = src->ObjOwner;
}

// ====================================================
/*
 * @brief This function parse the arguments for the py4pd object
 * @param x is the py4pd object
 * @param c is the canvas of the object
 * @param argc is the number of arguments
 * @param argv is the arguments
 * @return return void
 */
t_py *Py4pdUtils_GetObject(PyObject *pd_module) {
    PyObject *py4pd_capsule = PyObject_GetAttrString(pd_module, "py4pd");
    if (py4pd_capsule == NULL) {
        return NULL;
    }
    t_py *py4pd = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
    Py_DECREF(py4pd_capsule);
    return py4pd;
}

// ====================================================
// ========================= Files ====================
// ====================================================
/*
 * @brief get the folder name of something
 * @param x is the py4pd object
 * @return save the py4pd folder in x->py4pdPath
 */
char *Py4pdUtils_GetFolderName(char *Path) {
    LOG("Py4pdUtils_GetFolderName)");

    char *Folder = NULL;
    char *FolderSeparator = NULL;

// Find the last occurrence of a path separator
#ifdef _WIN32
    FolderSeparator = strrchr(path, '\\');
#else
    FolderSeparator = strrchr(Path, '/');
#endif

    // If a separator is found, extract the folder name
    if (FolderSeparator != NULL) {
        size_t FolderLen = FolderSeparator - Path;
        Folder = malloc(FolderLen + 1);
        strncpy(Folder, Path, FolderLen);
        Folder[FolderLen] = '\0';
    }

    return Folder;
}

// ====================================================
/*
 * @brief get the folder name of something
 * @param x is the py4pd object
 * @return save the py4pd folder in x->py4pdPath
 */

const char *Py4pdUtils_GetFilename(const char *Path) {
    const char *Filename = NULL;

    // Find the last occurrence of a path separator
    const char *LastSeparator = strrchr(Path, '/');

#ifdef _WIN64
    const char *LastSeparatorWin = strrchr(Path, '\\');
    if (LastSeparatorWin != NULL && LastSeparatorWin > LastSeparator) {
        LastSeparator = LastSeparatorWin;
    }
#endif

    // If a separator is found, extract the filename
    if (LastSeparator != NULL) {
        Filename = LastSeparator + 1;
    } else {
        // No separator found, use the entire path as the filename
        Filename = Path;
    }

    // remove .py from filename
    const char *LastDot = strrchr(Filename, '.');
    if (LastDot != NULL) {
        size_t FilenameLen = LastDot - Filename;
        char *FileNoExt = malloc(FilenameLen + 1);
        strncpy(FileNoExt, Filename, FilenameLen);
        FileNoExt[FilenameLen] = '\0';
        Filename = FileNoExt;
    }

    return Filename;
}

// ====================================================
int Py4pdUtils_CheckPkgNameConflict(t_py *x, char *folderToCheck, t_symbol *script_file_name) {
#ifdef _WIN64
    WIN32_FIND_DATAA findData;
    HANDLE hFind;

    char searchPath[MAX_PATH];
    snprintf(searchPath, sizeof(searchPath), "%s\\*", folderToCheck);

    hFind = FindFirstFileA(searchPath, &findData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                const char *entryName = findData.cFileName;
                if (strcmp(entryName, ".") != 0 && strcmp(entryName, "..") != 0) {
                    if (strcmp(entryName, script_file_name->s_name) == 0) {
                        // Process the conflict
                        post("");
                        pd_error(x,
                                 "[py4pd] The library '%s' "
                                 "conflicts with a Python package name.",
                                 script_file_name->s_name);
                        pd_error(x, "[py4pd] This can cause "
                                    "problems related to py4pdLoadObjects.");
                        pd_error(x, "[py4pd] Rename the library.");
                        return 1;
                    }
                }
            }
        } while (FindNextFileA(hFind, &findData) != 0);
        FindClose(hFind);
    }
#else
    DIR *dir;
    struct dirent *entry;

    if (folderToCheck == NULL) {
        printf("folderToCheck is NULL\n");
        return 0;
    }

    dir = opendir(folderToCheck);

    if (dir == NULL) {
        return 0;
    }

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 &&
            strcmp(entry->d_name, "..") != 0) {
            if (strcmp(entry->d_name, script_file_name->s_name) == 0) {
                post("");
                pd_error(x,
                         "[py4pd] The library '%s' conflicts with a Python "
                         "package name.",
                         script_file_name->s_name);
                pd_error(x, "[py4pd] This can cause problems related with "
                            "py4pdLoadObjects.");
                pd_error(x, "[py4pd] Rename the library.");
                return 1;
            }
        }
    }
    closedir(dir);
#endif
    return 0;
}

// ====================================================
/*
 * @brief Get the py4pd folder object, it creates the folder for scripts inside
 * resources
 * @param x is the py4pd object
 * @return save the py4pd folder in x->py4pdPath
 */

void Py4pdUtils_CreateNewThread(t_py *x) {}

// ====================================================
/*
 * @brief Get the py4pd folder object, it creates the folder for scripts inside
 * resources
 * @param x is the py4pd object
 * @return save the py4pd folder in x->py4pdPath
 */

void Py4pdUtils_FindObjFolder(t_py *x) {
    x->Py4pdPath = x->obj.te_g.g_pd->c_externdir;
    logpost(x, 4, "Obj path is %s", x->Py4pdPath->s_name);
}

// ===================================================================
/**
 * @brief Get the temp path object (inside Users/.py4pd), it creates the folder
 * if it not exist
 * @param x is the py4pd object
 * @return save the temp path in x->tempPath
 */

void Py4pdUtils_CreateTempFolder(t_py *x) {
#ifdef _WIN64
    // get user folder
    char *user_folder = getenv("USERPROFILE");
    LPSTR home = (LPSTR)malloc(256 * sizeof(char));
    memset(home, 0, 256);
    sprintf(home, "%s\\.py4pd\\", user_folder);
    x->TempPath = gensym(home);
    if (access(home, F_OK) == -1) {
        char *command = (char *)malloc(256 * sizeof(char));
        memset(command, 0, 256);
        if (!CreateDirectory(home, NULL)) {
            post("Failed to create directory, Report, this create "
                 "instabilities: %d\n",
                 GetLastError());
        }
        if (!SetFileAttributes(home, FILE_ATTRIBUTE_HIDDEN)) {
            post("Failed to set hidden attribute: %d\n", GetLastError());
        }
        free(command);
    }
    free(home);
#else
    const char *Home = getenv("HOME");
    char *TmpFolder = (char *)malloc(256 * sizeof(char));
    sprintf(TmpFolder, "%s/.py4pd/", Home);
    x->TempPath = gensym(TmpFolder);
    if (access(TmpFolder, F_OK) == -1) {
        char *Command = (char *)malloc(256 * sizeof(char));
        sprintf(Command, "mkdir -p %s", TmpFolder);
        int Result = system(Command);
        if (Result != 0) {
            pd_error(NULL,
                     "Failed to create directory, Report, this create "
                     "instabilities: %d\n",
                     Result);
        }
        free(Command);
    }
    free(TmpFolder);
#endif
}

// ====================================================
// ===================== User Interation ==============
// ====================================================
/*
 * @brief Run a Python function
 * @param x is the py4pd object
 * @param pArgs is the arguments to pass to the function
 * @return the return value of the function
 */
void Py4pdUtils_PrintError(t_py *x) {
    LOG("Py4pdUtils_PrintError");
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

// ===========================================
void Py4pdUtils_Click(t_py *x) {
    if (x->pFunction) {
        PyCodeObject *code = (PyCodeObject *)PyFunction_GetCode(x->pFunction);
        int line = PyCode_Addr2Line(code, 0);
        char command[MAXPDSTRING];
        Py4pdUtils_GetEditorCommand(x, command, line);
        Py4pdUtils_ExecuteSystemCommand(command, 1);
    } else {
        if (x->ObjClass == NULL) {
            pd_error(NULL, "Any Python function or class was defined");
            return;
        } else {
            PyObject *pFunction;
            Py4pdNewObj *pObjSelf;
            pObjSelf = (Py4pdNewObj *)x->ObjClass;
            int pFuncDefined = 0;
            if (pObjSelf->pFuncFloat != NULL) {
                pFunction = pObjSelf->pFuncFloat;
                pFuncDefined += 1;
            }
            if (pObjSelf->pFuncSymbol != NULL) {
                pFunction = pObjSelf->pFuncSymbol;
                pFuncDefined += 1;
            }
            if (pObjSelf->pFuncList != NULL) {
                pFunction = pObjSelf->pFuncList;
                pFuncDefined += 1;
            }
            if (pObjSelf->pFuncAnything != NULL) {
                pFunction = pObjSelf->pFuncAnything;
                pFuncDefined += 1;
            }
            if (pObjSelf->pFuncBang != NULL) {
                pFunction = pObjSelf->pFuncBang;
                pFuncDefined += 1;
            }

            if (pFuncDefined == 0) {
                pd_error(NULL, "Any Python function was defined");
                return;
            } else if (pFuncDefined > 1) {
                pd_error(NULL, "More than one Python function was defined");
                return;
            } else {
                PyCodeObject *code = (PyCodeObject *)PyFunction_GetCode(pFunction);
                int line = PyCode_Addr2Line(code, 0);
                char command[MAXPDSTRING];
                Py4pdUtils_GetEditorCommand(x, command, line);
                Py4pdUtils_ExecuteSystemCommand(command, 1);
            }
        }
    }
    return;
}

// ====================================================
// ===================== Run ==========================
// ====================================================
/*
 * @brief Run a Python function
 * @param x is the py4pd object
 * @param pArgs is the arguments to pass to the function
 * @return the return value of the function
 */
int Py4pdUtils_RunPy(t_py *x, PyObject *pArgs, PyObject *pKwargs) {
    LOG("Py4pdUtils_RunPy");

    t_py *PrevObj = NULL;
    int IsPrevObj = 0;
    PyObject *PdModule = PyImport_ImportModule("pd");
    PyObject *oldObjectCapsule = NULL;
    PyObject *pValue = NULL;
    PyObject *objectCapsule = NULL;

    if (PdModule != NULL) {
        oldObjectCapsule = PyObject_GetAttrString(PdModule, "py4pd"); // borrowed reference
        if (oldObjectCapsule != NULL) {
            PyObject *ObjCapsule = PyObject_GetAttrString(PdModule, "py4pd");
            PrevObj = (t_py *)PyCapsule_GetPointer(ObjCapsule, "py4pd");
            IsPrevObj = 1;
        } else {
            IsPrevObj = 0;
            Py_XDECREF(oldObjectCapsule);
        }
    } else {
        pd_error(x, "[%s] Failed to import pd module when Running Python function",
                 x->pFuncName->s_name);
        Py4pdUtils_PrintError(x);
        Py_XDECREF(PdModule);
        return -1;
    }
    objectCapsule = Py4pdUtils_AddPdObject(x);
    if (objectCapsule == NULL) {
        pd_error(x, "[Python] Failed to add object to Python");
        Py_XDECREF(PdModule);
        return -1;
    }

    LOG("    Before PyObjectCall");
    pValue = PyObject_Call(x->pFunction, pArgs, pKwargs);
    LOG("    After PyObjectCall");

    t_py4pd_pValue *PyPtrValue = NULL;
    if (x->ObjType < 3) {
        PyPtrValue = (t_py4pd_pValue *)malloc(sizeof(t_py4pd_pValue));
        PyPtrValue->pValue = pValue;
        PyPtrValue->PdOutCount = 0;
        PyPtrValue->ObjOwner = x->ObjName;
    }

    if (IsPrevObj == 1 && pValue != NULL) {
        objectCapsule = Py4pdUtils_AddPdObject(PrevObj);
        if (objectCapsule == NULL) {
            if (PyPtrValue != NULL) {
                free(PyPtrValue);
            }
            pd_error(x, "[Python] Failed to add object to Python");
            return -1;
        }
    }

    if (x->AudioOut == 1) {
        if (x->ObjType < 3) {
            free(PyPtrValue);
        }
        return -1;
    }
    if (pValue != NULL && (x->ObjType < 3)) {
        Py4pdUtils_ConvertToPd(x, PyPtrValue, x->MainOut);
        Py_DECREF(pValue);
        Py_XDECREF(PdModule);
        free(PyPtrValue);
        PyErr_Clear();
        return 0;
    } else if (pValue == NULL) {
        Py4pdUtils_PrintError(x);
        Py_XDECREF(pValue);
        Py_XDECREF(PdModule);
        free(PyPtrValue);
        return 0;
    }

    else if (x->ObjType > 2) {
        Py_XDECREF(PdModule);
        free(PyPtrValue);
        PyErr_Clear();
        return 0;
    } else {
        pd_error(x, "[%s] Unknown error, please report", x->ObjName->s_name);
        PyErr_Clear();
        return -1;
    }
}

// ====================================================
PyObject *Py4pdUtils_RunPyAudioOut(t_py *x, PyObject *pArgs, PyObject *pKwargs) {
    t_py *prev_obj = NULL;
    int prev_obj_exists = 0;
    PyObject *MainModule = PyImport_ImportModule("pd");
    PyObject *oldObjectCapsule = NULL;
    PyObject *pValue;
    PyObject *objectCapsule = NULL;

    if (MainModule != NULL) {
        oldObjectCapsule = PyObject_GetAttrString(MainModule, "py4pd"); // borrowed reference
        if (oldObjectCapsule != NULL) {
            PyObject *py4pd_capsule =
                PyObject_GetAttrString(MainModule, "py4pd"); // borrowed reference
            prev_obj = (t_py *)PyCapsule_GetPointer(py4pd_capsule,
                                                    "py4pd"); // borrowed reference
            prev_obj_exists = 1;
            Py_DECREF(oldObjectCapsule);
            Py_DECREF(py4pd_capsule);
        } else {
            prev_obj_exists = 0;
            Py_XDECREF(oldObjectCapsule);
        }
    } else {
        pd_error(x, "[%s] Failed to import pd module when Running Python function",
                 x->pFuncName->s_name);
        PyErr_Print();
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[Python] %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        Py_XDECREF(MainModule);
        PyErr_Clear();
        return NULL;
    }
    objectCapsule = Py4pdUtils_AddPdObject(x);
    if (objectCapsule == NULL) {
        pd_error(x, "[Python] Failed to add object to Python");
        Py_XDECREF(MainModule);
        return NULL;
    }
    pValue = PyObject_Call(x->pFunction, pArgs, pKwargs);
    if (prev_obj_exists == 1 && pValue != NULL) {
        objectCapsule = Py4pdUtils_AddPdObject(prev_obj);
        if (objectCapsule == NULL) {
            pd_error(x, "[Python] Failed to add object to Python");
            return NULL;
        }
    }
    Py_XDECREF(MainModule);
    return pValue;
}

// ====================================================
// ===================== Memory =======================
// ====================================================
int Py4pdUtils_Snprintf(char *buffer, size_t size, const char *format, ...) {
    int result;
    va_list args;

    va_start(args, format);

    // Use vsnprintf with the provided size to prevent buffer overflow
    result = vsnprintf(buffer, size, format, args);

    // Check for errors or truncation (result is negative)
    if (result < 0 || (size_t)result >= size) {
        // Handle the error or truncation as needed
        result = -1; // Set to an error code or take appropriate action
    }

    va_end(args);

    return result;
}

// ====================================================
/*
 * @brief This function returns a deepcopy of a PythonObject
 * @param Source is the object to copy
 * @return Return a deepcopy of the object
 */

PyObject *Py4pdUtils_DeepCopy(PyObject *pValue) {
    PyObject *deepCopyModule = PyImport_ImportModule("copy");
    PyObject *deepCopyFunction = PyObject_GetAttrString(deepCopyModule, "deepcopy");
    PyObject *deepCopyArgs = PyTuple_Pack(1, pValue);
    PyObject *obj = PyObject_CallObject(deepCopyFunction, deepCopyArgs);
    Py_DECREF(deepCopyModule);
    Py_DECREF(deepCopyFunction);
    Py_DECREF(deepCopyArgs);

    return obj;
}

// ====================================================
/*
 * @brief It warnings if there is a memory leak
 * @param pValue is the value to check
 * @return nothing
 */
void Py4pdUtils_MemLeakCheck(PyObject *pValue, int refcnt, char *where) {
    LOG("Py4pdUtils_MemLeakCheck");

    if (Py_IsNone(pValue)) {
        return;
    } else if (Py_IsTrue(pValue)) {
        return;
    } else if (PyLong_Check(pValue)) {
        long value = PyLong_AsLong(pValue);
        if (value >= -5 && value <= 255) {
            return;
        } else {
            if (pValue->ob_refcnt != refcnt) {
                if (refcnt < pValue->ob_refcnt) {
                    pd_error(NULL,
                             "[DEV] Memory Leak inside %s, Ref Count should be "
                             "%d but is %d",
                             where, refcnt, (int)pValue->ob_refcnt);
                } else {
                    pd_error(NULL,
                             "[DEV] Ref Count error in %s, Ref Count should be "
                             "%d but is %d",
                             where, refcnt, (int)pValue->ob_refcnt);
                }
            }
        }
    } else {
        if (pValue->ob_refcnt != refcnt) {
            if (refcnt < pValue->ob_refcnt) {
                pd_error(NULL,
                         "[DEV] Memory Leak inside %s, Ref Count should be %d "
                         "but is %d",
                         where, refcnt, (int)pValue->ob_refcnt);
            } else {
                pd_error(NULL,
                         "[DEV] Ref Count error in %s, Ref Count should be %d "
                         "but is %d",
                         where, refcnt, (int)pValue->ob_refcnt);
            }
        }
    }
    return;
}

// ============================================
/**
 * @brief Free the memory of the object
 * @param x
 * @return void*
 */

void *Py4pdUtils_FreeObj(t_py *x) {
    LOG("Py4pdUtils_FreeObj");
    objCount--;
    if (objCount == 0) {
        objCount = 0;
        char command[1000];
#ifdef _WIN32
        sprintf(command, "cmd /C del /Q /S %s*.*", x->TempPath->s_name);
        (void)Py4pdUtils_ExecuteSystemCommand(command, 0);
#else
        sprintf(command, "rm -rf %s", x->TempPath->s_name);
        (void)Py4pdUtils_ExecuteSystemCommand(command, 0);
#endif
    }
    if (x->VisMode != 0)
        Py4pdPic_Free(x);

    if (x->PdCollect != NULL)
        Py4pdMod_FreePdcollectHash(x->PdCollect);

    if (x->pArgsCount > 1 && x->PyObjArgs != NULL) {
        for (int i = 1; i < x->pArgsCount; i++) {
            if (!x->PyObjArgs[i]->PdOutCount)
                Py_DECREF(x->PyObjArgs[i]->pValue);
            free(x->PyObjArgs[i]);
        }
        free(x->PyObjArgs);
    }
    if (x->PdObjArgs != NULL) {
        free(x->PdObjArgs);
    }
    return NULL;
}

// ====================================================
// ===================== Convertions ==========================
// ====================================================
/*
 * @brief Convert and output Python Values to PureData values
 * @param x is the py4pd object
 * @param pValue is the Python value to convert
 * @return nothing, but output the value to the outlet
 */
inline void *Py4pdUtils_ConvertToPd(t_py *x, t_py4pd_pValue *pValueStruct, t_outlet *outlet) {
    LOG("Py4pdUtils_ConvertToPd");
    PyObject *pValue = pValueStruct->pValue;

    if (pValue->ob_refcnt < 1) {
        pd_error(NULL, "[FATAL]: When converting to pd, pValue "
                       "refcnt < 1");
        return NULL;
    }

    if (x->OutPyPointer) {
        if (pValue == Py_None && x->IgnoreOnNone == 1) {
            return NULL;
        }
        t_atom args[2];
        SETSYMBOL(&args[0], gensym(Py_TYPE(pValue)->tp_name));
        SETPOINTER(&args[1], (t_gpointer *)pValueStruct);
        outlet_anything(outlet, gensym("PyObject"), 2, args);
        return NULL;
    }

    if (PyTuple_Check(pValue)) {
        if (PyTuple_Size(pValue) == 1) {
            pValue = PyTuple_GetItem(pValue, 0);
        }
    }

    if (PyList_Check(pValue)) { // If the function return a list list
        int list_size = PyList_Size(pValue);
        t_atom *list_array = (t_atom *)malloc(list_size * sizeof(t_atom));
        int i;
        int listIndex = 0;
        PyObject *pValue_i;
        for (i = 0; i < list_size; ++i) {
            pValue_i = PyList_GetItem(pValue, i); // borrowed reference
            if (PyLong_Check(pValue_i)) {
                float result = (float)PyLong_AsLong(pValue_i);
                SETFLOAT(&list_array[listIndex], result);
                listIndex++;
            } else if (PyFloat_Check(pValue_i)) {
                float result = PyFloat_AsDouble(pValue_i);
                SETFLOAT(&list_array[listIndex], result);
                listIndex++;
            } else if (PyUnicode_Check(pValue_i)) { // If the function return a
                const char *result = PyUnicode_AsUTF8(pValue_i);
                SETSYMBOL(&list_array[listIndex], gensym(result));
                listIndex++;

            } else if (Py_IsNone(pValue_i)) {
                // not possible to represent None in Pd, so we just skip it
            } else {
                if (list_array != NULL) {
                    free(list_array);
                }
                pd_error(x,
                         "[py4pd] py4pd just convert int, float, string, and lists! "
                         "Received: list of %ss",
                         Py_TYPE(pValue_i)->tp_name);
                return 0;
            }
        }
        if (list_array[0].a_type == A_SYMBOL) {
            outlet_anything(outlet, list_array[0].a_w.w_symbol, listIndex - 1, list_array + 1);
        } else {
            outlet_list(outlet, &s_list, listIndex, list_array);
        }
        free(list_array);
    } else {
        if (PyLong_Check(pValue)) {
            long result = PyLong_AsLong(pValue); // If the function return a integer
            outlet_float(outlet, result);

        } else if (PyFloat_Check(pValue)) {
            double result = PyFloat_AsDouble(pValue); // If the function return a float
            float result_float = (float)result;
            outlet_float(outlet, result_float);
        } else if (PyUnicode_Check(pValue)) {
            const char *result = PyUnicode_AsUTF8(pValue); // If the function return a string
            Py4pdUtils_FromSymbolSymbol(x, gensym(result), outlet);
        } else if (Py_IsNone(pValue)) {
            // Not possible to represent None in Pd, so we just skip it
        } else {
            pd_error(x,
                     "[py4pd] py4pd just convert int, float and string or list "
                     "of this atoms! Received: %s",
                     Py_TYPE(pValue)->tp_name);
        }
    }
    return 0;
}

// ============================================
/*
 * @brief Convert PureData Values to Python values
 * @param listsArrays were the lists are stored
 * @param argc is the number of arguments
 * @param argv is the arguments
 * @return the Python tuple with the values
 */
PyObject *Py4pdUtils_ConvertToPy(PyObject *listsArrays[], int argc, t_atom *argv) {
    LOG("Py4pdUtils_ConvertToPy");
    PyObject *ArgsTuple = PyTuple_New(0); // start new tuple with 1 element
    int listStarted = 0;
    int argCount = 0;
    int listCount = 0;

    for (int i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            if (strchr(argv[i].a_w.w_symbol->s_name, '[') != NULL) {
                char *str = (char *)malloc(strlen(argv[i].a_w.w_symbol->s_name) + 1);
                Py4pdUtils_Strlcpy(str, argv[i].a_w.w_symbol->s_name,
                                   strlen(argv[i].a_w.w_symbol->s_name) + 1);
                Py4pdUtils_RemoveChar(str, '[');
                listsArrays[listCount] = PyList_New(0);
                int isNumeric = Py4pdUtils_IsNumericOrDot(str);
                if (isNumeric == 1) {
                    if (strchr(str, '.') != NULL) {
                        PyList_Append(listsArrays[listCount], PyFloat_FromDouble(atof(str)));
                    } else {
                        PyList_Append(listsArrays[listCount], PyLong_FromLong(atol(str)));
                    }
                } else {
                    PyList_Append(listsArrays[listCount], PyUnicode_FromString(str));
                }
                free(str);
                listStarted = 1;
            }

            else if ((strchr(argv[i].a_w.w_symbol->s_name, '[') != NULL) &&
                     (strchr(argv[i].a_w.w_symbol->s_name, ']') != NULL)) {
                char *str = (char *)malloc(strlen(argv[i].a_w.w_symbol->s_name) + 1);
                strcpy(str, argv[i].a_w.w_symbol->s_name);
                Py4pdUtils_RemoveChar(str, '[');
                Py4pdUtils_RemoveChar(str, ']');
                listsArrays[listCount] = PyList_New(0);
                int isNumeric = Py4pdUtils_IsNumericOrDot(str);
                if (isNumeric == 1) {
                    if (strchr(str, '.') != NULL) {
                        PyList_Append(listsArrays[listCount], PyFloat_FromDouble(atof(str)));
                    } else {
                        PyList_Append(listsArrays[listCount], PyLong_FromLong(atol(str)));
                    }
                } else {
                    PyList_Append(listsArrays[listCount], PyUnicode_FromString(str));
                }
                free(str);
                listStarted = 1;
            }

            // ========================================
            else if (strchr(argv[i].a_w.w_symbol->s_name, ']') != NULL) {
                char *str = (char *)malloc(strlen(argv[i].a_w.w_symbol->s_name) + 1);
                strcpy(str, argv[i].a_w.w_symbol->s_name);
                Py4pdUtils_RemoveChar(str, ']');
                int isNumeric = Py4pdUtils_IsNumericOrDot(str);
                _PyTuple_Resize(&ArgsTuple, argCount + 1);
                // TODO: This is old code, fix it!
                if (isNumeric == 1) {
                    if (strchr(str, '.') != NULL) {
                        PyList_Append(listsArrays[listCount], PyFloat_FromDouble(atof(str)));
                    } else {
                        PyList_Append(listsArrays[listCount], PyLong_FromLong(atol(str)));
                    }
                    PyTuple_SetItem(ArgsTuple, argCount, listsArrays[listCount]);
                } else {
                    PyList_Append(listsArrays[listCount], PyUnicode_FromString(str));
                    PyTuple_SetItem(ArgsTuple, argCount, listsArrays[listCount]);
                }
                free(str);
                listStarted = 0;
                listCount++;
                argCount++;
            }

            // ========================================
            else {
                if (listStarted == 1) {
                    char *str = (char *)malloc(strlen(argv[i].a_w.w_symbol->s_name) + 1);
                    strcpy(str, argv[i].a_w.w_symbol->s_name);
                    // TODO: This is old code, fix it!
                    int isNumeric = Py4pdUtils_IsNumericOrDot(str);
                    if (isNumeric == 1) {
                        if (strchr(str, '.') != NULL) {
                            PyList_Append(listsArrays[listCount], PyFloat_FromDouble(atof(str)));
                        } else {
                            PyList_Append(listsArrays[listCount], PyLong_FromLong(atol(str)));
                        }
                    } else {
                        PyList_Append(listsArrays[listCount], PyUnicode_FromString(str));
                    }
                    free(str);
                } else {
                    _PyTuple_Resize(&ArgsTuple, argCount + 1);
                    PyTuple_SetItem(ArgsTuple, argCount,
                                    PyUnicode_FromString(argv[i].a_w.w_symbol->s_name));
                    argCount++;
                }
            }
        } else {
            if (listStarted == 1) {
                PyList_Append(listsArrays[listCount],
                              PyFloat_FromDouble(atom_getfloatarg(i, argc, argv)));
            } else {
                _PyTuple_Resize(&ArgsTuple, argCount + 1);
                PyTuple_SetItem(ArgsTuple, argCount,
                                PyFloat_FromDouble(atom_getfloatarg(i, argc, argv)));
                argCount++;
            }
        }
    }
    return ArgsTuple;
}

// ====================================================
// ===================== ObjConfig ====================
// ====================================================
/*
 * @brief This function parse the arguments for the py4pd object
 * @param x is the py4pd object
 * @param c is the canvas of the object
 * @param argc is the number of arguments
 * @param argv is the arguments
 * @return return void
 */

void Py4pdUtils_ParseArguments(t_py *x, t_canvas *c, int argc, t_atom *argv) {
    LOG("Py4pdUtils_ParseArguments");
    int i;
    for (i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            t_symbol *py4pdArgs = atom_getsymbolarg(i, argc, argv);
            if (py4pdArgs == gensym("-picture") || py4pdArgs == gensym("-score") ||
                py4pdArgs == gensym("-pic") || py4pdArgs == gensym("-canvas")) {
                Py4pdPic_InitVisMode(x, c, py4pdArgs, i, argc, argv, NULL);
                int j;
                for (j = i; j < argc; j++) {
                    argv[j] = argv[j + 1];
                }
                argc--;
            } else if (py4pdArgs == gensym("-nvim") || py4pdArgs == gensym("-vscode") ||
                       py4pdArgs == gensym("-sublime") || py4pdArgs == gensym("-emacs")) {
                // remove the '-' from the name of the editor
                const char *editor = py4pdArgs->s_name;
                editor++;
                x->EditorName = gensym(editor);
                int j;
                for (j = i; j < argc; j++) {
                    argv[j] = argv[j + 1];
                }
                argc--;
            } else if (py4pdArgs == gensym("-audioin")) {
                x->AudioIn = 1;
                x->AudioOut = 0;
                x->UseNumpy = 0;
                int j;
                for (j = i; j < argc; j++) {
                    argv[j] = argv[j + 1];
                }
                argc--;
            } else if (py4pdArgs == gensym("-audioout")) {
                // post("[py4pd] Audio Outlets enabled");
                x->AudioOut = 1;
                x->AudioIn = 0;
                x->UseNumpy = 0;
                x->MainOut = outlet_new(&x->obj, gensym("signal")); // create a signal outlet
                int j;
                for (j = i; j < argc; j++) {
                    argv[j] = argv[j + 1];
                }
                argc--;
            } else if (py4pdArgs == gensym("-audio")) {
                x->AudioIn = 1;
                x->AudioOut = 1;
                x->MainOut = outlet_new(&x->obj, gensym("signal")); // create a signal outlet
                x->UseNumpy = 0;
                int j;
                for (j = i; j < argc; j++) {
                    argv[j] = argv[j + 1];
                }
                argc--;
            }
        }
    }
    if (x->AudioOut == 0) {
        x->MainOut = outlet_new(&x->obj,
                                0); // cria um outlet caso o objeto nao contenha audio
    }
}

/*

* @brief Get the config from py4pd.cfg file
* @param x is the py4pd object
* @return the pointer to the py4pd object with the config

*/

void Py4pdUtils_SetObjConfig(t_py *x) {
    LOG("Py4pdUtils_SetObjConfig");
    int folderLen = strlen("/py4pd-env/") + 1;
    char *PADRAO_packages_path =
        (char *)malloc(sizeof(char) * (strlen(x->PdPatchPath->s_name) + folderLen)); //
    snprintf(PADRAO_packages_path, strlen(x->PdPatchPath->s_name) + folderLen, "%s/py4pd-env/",
             x->PdPatchPath->s_name);
    x->CondaPath = gensym(PADRAO_packages_path);
    if (x->EditorName == NULL) {
        const char *editor = PY4PD_EDITOR;
        x->EditorName = gensym(editor);
    }
    if (x->Py4pdPath == NULL) {
        Py4pdUtils_FindObjFolder(x);
    }

    char config_path[MAXPDSTRING];
    snprintf(config_path, sizeof(config_path), "%s/py4pd.cfg", x->Py4pdPath->s_name);
    if (access(config_path, F_OK) != -1) {
        FILE *file = fopen(config_path, "r");
        char line[256];
        while (fgets(line, sizeof(line), file)) {
            if (strstr(line, "conda_env_packages =") != NULL) {
                char *packages_path = (char *)malloc(
                    sizeof(char) * (strlen(line) - strlen("conda_env_packages = ") + 1));
                strcpy(packages_path, line + strlen("conda_env_packages = ")); // TODO: Need review
                if (strlen(packages_path) > 0) {
                    if (packages_path[strlen(packages_path) - 1] == '\n') {
                        packages_path[strlen(packages_path) - 1] = '\0';
                    }
                    if (packages_path[0] == '.') {
                        char *new_packages_path =
                            (char *)malloc(sizeof(char) * (strlen(x->PdPatchPath->s_name) +
                                                           strlen(packages_path) + 1)); //
                        Py4pdUtils_Strlcpy(new_packages_path, x->PdPatchPath->s_name,
                                           strlen(x->PdPatchPath->s_name) + 1);
                        Py4pdUtils_Strlcat(new_packages_path, packages_path + 1,
                                           strlen(packages_path) + 1);
                        Py4pdUtils_RemoveChar(new_packages_path, '\n');
                        Py4pdUtils_RemoveChar(new_packages_path, '\r');
                        x->CondaPath = gensym(new_packages_path);
                        free(new_packages_path);
                    } else {
                        Py4pdUtils_RemoveChar(packages_path, '\n');
                        Py4pdUtils_RemoveChar(packages_path, '\r');
                        x->CondaPath = gensym(packages_path);
                    }
                }
                free(packages_path);
            } else if (strstr(line, "packages =") != NULL) {
                pd_error(NULL, "The key 'packages' in py4pd.cfg is deprecated, "
                               "use 'conda_env_packages' instead.");

            } else if (strstr(line, "editor =") != NULL) {
                char *editor =
                    (char *)malloc(sizeof(char) * (strlen(line) - strlen("editor = ") + 1)); //
                Py4pdUtils_Strlcpy(editor, line + strlen("editor = "),
                                   strlen(line) - strlen("editor = ") + 1);
                Py4pdUtils_RemoveChar(editor, '\n');
                Py4pdUtils_RemoveChar(editor, '\r');
                Py4pdUtils_RemoveChar(editor, ' ');
                x->EditorName = gensym(editor);
                free(editor);
            } else if (strstr(line, "editor_command =") != NULL) {
                char *editor_command = (char *)malloc(
                    sizeof(char) * (strlen(line) - strlen("editor_command = ") + 1)); //
                Py4pdUtils_Strlcpy(editor_command, line + strlen("editor_command = "),
                                   strlen(line) - strlen("editor_command = ") + 1);
                Py4pdUtils_RemoveChar(editor_command, '\n');
                Py4pdUtils_RemoveChar(editor_command, '\r');
                x->EditorCommand = gensym(editor_command);
                free(editor_command);
            }
        }
        fclose(file);
    }
    free(PADRAO_packages_path);
    Py4pdUtils_CreateTempFolder(x);
    return;
}

// ============================================
void Py4pdUtils_AddPathsToPythonPath(t_py *x) {
    LOG("Py4pdUtils_AddPathsToPythonPath");

    char Py4pdMod[MAXPDSTRING];
    int ret = snprintf(Py4pdMod, MAXPDSTRING, "%s/py4pd", x->Py4pdPath->s_name);
    if (ret < 0) {
        pd_error(x, "[py4pd] Error when adding py4pd path to sys.path");
        return;
    }
    char Py4pdPkgs[MAXPDSTRING];
    ret = snprintf(Py4pdPkgs, MAXPDSTRING, "%s/py4pd-env", x->Py4pdPath->s_name);
    if (ret < 0) {
        pd_error(x, "[py4pd] Error when adding py4pd path to sys.path");
        return;
    }

    PyObject *HomePath = PyUnicode_FromString(x->PdPatchPath->s_name);
    PyObject *SitePkg = PyUnicode_FromString(x->PkgPath->s_name);
    PyObject *CondaPkg = PyUnicode_FromString(x->CondaPath->s_name);
    PyObject *Py4pdGlobalPkg = PyUnicode_FromString(Py4pdPkgs);

    if (HomePath && SitePkg && CondaPkg && Py4pdGlobalPkg) {
        PyObject *SysPath = PySys_GetObject("path"); // Borrowed reference
        int SysPathLen = PyList_Size(SysPath);
        if (SysPath && PyList_Check(SysPath)) {
            PyList_Insert(SysPath, SysPathLen, HomePath);
            PyList_Insert(SysPath, SysPathLen, SitePkg);
            PyList_Insert(SysPath, SysPathLen, CondaPkg);
            PyList_Insert(SysPath, SysPathLen, Py4pdGlobalPkg);
        }
    }

    return;
}

// ============================================
void Py4pdUtils_ConfigurePythonPaths() {
    PyObject *SysPath = PySys_GetObject("path"); // Borrowed reference
    Py_ssize_t PathsLen = PyList_Size(SysPath);

#ifdef __linux__
    const int PathsToRemove = 2;
#elif __APPLE__
    const int PathsToRemove = 1;
#else
    const int PathsToRemove = 1;
#endif

    if (PathsLen > 0 && PathsToRemove > 0) {
        for (int i = 0; i < PathsToRemove; ++i) {
            Py_ssize_t index = PathsLen - 1 - i;
            PyObject *path_entry = PyList_GetItem(SysPath, index);
            PyList_SetSlice(SysPath, index, index + 1, NULL);
            Py_DECREF(path_entry);
        }
    }
}

// ===================================================================
int Py4pdUtils_CheckNumpyInstall(t_py *x) {
    PyObject *NumpyUmath = PyImport_ImportModule("numpy");
    if (NumpyUmath == NULL) {
        pd_error(x, "[py4pd] Numpy not installed, send [pip install numpy] to "
                    "the py4pd object");
        PyErr_Print();
        Py4pdUtils_PrintError(x);
        return 0;
    }

    // check numpy version
    // PyObject *Numpy = PyImport_ImportModule("numpy");
    // const char *NumpyVersion = PyUnicode_AsUTF8(PyObject_GetAttrString(Numpy, "__version__"));
    // if (NumpyVersion[0] != '2') {
    //     pd_error(x,
    //              "[py4pd] Numpy version %s is not supported, please update to "
    //              "version 2.0.0 or higher",
    //              NumpyVersion);
    //     return 0;
    // }

    // Py_DECREF(Numpy);
    PyErr_Clear();
    return 1;
}

// ===================================================================
/*
 * @brief It creates the commandline to open the editor
 * @param x is the py4pd object
 * @param command is the commandline to open the editor
 * @param line is the line to open the editor
 * @return the commandline to open the editor
 */
void Py4pdUtils_GetEditorCommand(t_py *x, char *command, int line) {
    const char *editor = x->EditorName->s_name;
    const char *filename = x->pScriptName->s_name;
    const char *home = x->PdPatchPath->s_name;
    char completePath[MAXPDSTRING];

    if (x->PyObject) {
        sprintf(completePath, "'%s'", filename);
    } else if (x->IsLib) {
        PyCodeObject *code = (PyCodeObject *)PyFunction_GetCode(x->pFunction);
        t_symbol *pScriptName = gensym(PyUnicode_AsUTF8(code->co_filename));
        sprintf(completePath, "'%s'", pScriptName->s_name);
    } else {
        sprintf(completePath, "'%s%s.py'", home, filename);
    }

    if (x->EditorCommand) {
        sprintf(command, x->EditorCommand->s_name, completePath, line);
        return;
    }

    // check if there is .py in filename
    if (strcmp(editor, PY4PD_EDITOR) == 0) {
        sprintf(command, "%s %s", PY4PD_EDITOR, completePath);
    } else if (strcmp(editor, "vscode") == 0) {
#ifdef __APPLE__
        sprintf(command,
                "\"/Applications/Visual Studio Code.app/Contents/"
                "Resources/app/bin/code\" -g %s:%d",
                completePath, line);
#else
        sprintf(command, "code -g %s:%d", completePath, line);
#endif

    } else if (strcmp(editor, "nvim") == 0) {
// if it is linux
#ifdef __linux__
        char *env_var = getenv("XDG_CURRENT_DESKTOP");
        if (env_var == NULL) {
            pd_error(x, "[py4pd] Your desktop environment is not supported, "
                        "please report.");
            sprintf(command, "ls ");
        } else {
            if (strcmp(env_var, "GNOME") == 0) {
                int gnomeConsole = system("kgx --version");
                int gnomeTerminal = system("gnome-terminal --version");
                char GuiTerminal[MAXPDSTRING];
                if (gnomeConsole == 0) {
                    sprintf(GuiTerminal, "kgx");
                } else if (gnomeTerminal == 0) {
                    sprintf(GuiTerminal, "gnome-terminal");
                } else {
                    pd_error(x, "[py4pd] You seems to be using GNOME, but "
                                "gnome-terminal or gnome-console is not installed, "
                                "please install one of them.");
                }
                sprintf(command, "%s -- nvim +%d %s", GuiTerminal, line, completePath);
            } else if (strcmp(env_var, "KDE") == 0) {
                pd_error(x, "[py4pd] This is untested, please report if it works.");
                sprintf(command, "konsole -e \"nvim +%d %s\"", line, completePath);
            } else {
                pd_error(x,
                         "[py4pd] Your desktop environment %s is not "
                         "supported, please report.",
                         env_var);
                sprintf(command, "ls ");
            }
        }
#else
        sprintf(command, "nvim +%d %s", line, completePath);
#endif
    } else if (strcmp(editor, "gvim") == 0) {
        sprintf(command, "gvim +%d %s", line, completePath);
    } else if (strcmp(editor, "sublime") == 0) {
        sprintf(command, "subl --goto %s:%d", completePath, line);
    } else if (strcmp(editor, "emacs") == 0) {
        sprintf(command, "emacs --eval '(progn (find-file \"%s\") (goto-line %d))'", completePath,
                line);
    } else {
        pd_error(x, "[py4pd] editor %s not supported.", editor);
    }
    return;
}

// ====================================================
// ===================== SubInterpreters ==============
// ====================================================
#if PYTHON_REQUIRED_VERSION(3, 12)

struct Py4pd_ObjSubInterp {
    PyObject *pFunc;
    PyObject *pModule;
    t_py *x;
};

// =================================================================
void *Py4pdUtils_CreateSubInterpreter(void *arg) {
    (void)arg;
    PyInterpreterConfig config = {
        .check_multi_interp_extensions = 1,
        .gil = PyInterpreterConfig_OWN_GIL,
    };

    PyThreadState *tstate = NULL;
    PyStatus status = Py_NewInterpreterFromConfig(&tstate, &config);
    if (PyStatus_Exception(status)) {
        PyErr_SetString(PyExc_RuntimeError, "Interpreter creation failed");
        return NULL;
    }

    if (tstate == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Interpreter creation failed");
        return NULL;
    }
    // WE ARE IN THE NEW SUBINTERPRETER

    // run function
    struct Py4pd_ObjSubInterp *objSubInterp = arg;
    t_py *x = objSubInterp->x;
    x->FuncCalled = 0;
    return 0;
}

// ===============================================================
/*
 * @brief This function will create a new Python interpreter and initialize
 * it.
 * @param x is the py4pd object
 * @return It will return 0 if the interpreter was created successfully,
 * otherwise it will return 1.
 */
void Py4pdUtils_CreatePythonInterpreter(t_py *x) {
    if (x->pSubInterpRunning) {
        pd_error(x, "[Python] Subinterpreter already running");
        return;
    }

    struct Py4pd_ObjSubInterp *objSubInterp = malloc(sizeof(struct Py4pd_ObjSubInterp));
    objSubInterp->x = x;
    x->pSubInterpRunning = 1;

    pthread_t PyInterpId;
    pthread_create(&PyInterpId, NULL, Py4pdUtils_CreateSubInterpreter, objSubInterp);
    pthread_detach(PyInterpId);
    return;
}
#endif

// ====================================================
// ================= Show images in PureData  =========
// ====================================================
/*
 * @brief Py4pdUtils_Mtok is a function separated from the tokens of one
 * string
 * @param input is the string to be separated
 * @param delimiter is the string to be separated
 * @return the string separated by the delimiter
 */
char *Py4pdUtils_Mtok(char *input, char *delimiter) { // TODO: WRONG PLACE
    static char *string;
    if (input != NULL)
        string = input;
    if (string == NULL)
        return string;
    char *end = strstr(string, delimiter);
    while (end == string) {
        *end = '\0';
        string = end + strlen(delimiter);
        end = strstr(string, delimiter);
    };
    if (end == NULL) {
        char *temp = string;
        string = NULL;
        return temp;
    }
    char *temp = string;
    *end = '\0';
    string = end + strlen(delimiter);
    return (temp);
}

// =====================================================================
size_t Py4pdUtils_Strlcpy(char *dst, const char *src, size_t size) {
    LOG("Py4pdUtils_Strlcpy");
#if defined(__APPLE__)
    return strlcpy(dst, src, size);
#else
    size_t srclen = strlen(src);
    if (srclen + 1 < size) {
        memcpy(dst, src, srclen + 1);
    } else {
        memcpy(dst, src, size - 1);
        dst[size - 1] = '\0';
    }
    return srclen;
#endif
}

// =====================================================================
size_t Py4pdUtils_Strlcat(char *dst, const char *src, size_t size) {
#if defined(__APPLE__)
    return strlcat(dst, src, size);
#else
    size_t dest_len = strlen(dst);
    size_t src_len = strlen(src);
    if (size <= dest_len) {
        return dest_len + src_len;
    }
    size_t space_left = size - dest_len - 1;
    strncat(dst, src, space_left);
    dst[size - 1] = '\0';
    return dest_len + src_len;
#endif
}

// =====================================================================
void Py4pdUtils_CreatePicObj(t_py *x, PyObject *PdDict, t_class *object_PY4PD_Class, int argc,
                             t_atom *argv) {
    t_canvas *c = x->Canvas;
    PyObject *pyLibraryFolder = PyDict_GetItemString(PdDict, "LibraryFolder");
    t_symbol *py4pdArgs = gensym("-canvas");
    PyObject *py4pdOBJwidth = PyDict_GetItemString(PdDict, "width");
    x->Width = PyLong_AsLong(py4pdOBJwidth);
    PyObject *py4pdOBJheight = PyDict_GetItemString(PdDict, "height");
    x->Height = PyLong_AsLong(py4pdOBJheight);
    if (argc > 1) {
        if (argv[0].a_type == A_FLOAT) {
            x->Width = atom_getfloatarg(0, argc, argv);
        }
        if (argv[1].a_type == A_FLOAT) {
            x->Height = atom_getfloatarg(1, argc, argv);
        }
    }

    PyObject *gifFile = PyDict_GetItemString(PdDict, "Gif");
    if (gifFile == NULL) {
        x->ImageBase64 = PY4PD_IMAGE;
    } else {
        char *gifFileCHAR = (char *)PyUnicode_AsUTF8(gifFile);
        if (gifFileCHAR[0] == '.' && gifFileCHAR[1] == '/') {
            char completeImagePath[MAXPDSTRING];
            gifFileCHAR++; // remove the first dot
            sprintf(completeImagePath, "%s%s", PyUnicode_AsUTF8(pyLibraryFolder), gifFileCHAR);
            char *ext = strrchr(completeImagePath, '.');
            if (strcmp(ext, ".gif") == 0) {
                Py4pdUtils_ReadGifFile(x, completeImagePath);
            } else if (strcmp(ext, ".png") == 0) {
                Py4pdUtils_ReadPngFile(x, completeImagePath);
            } else {
                pd_error(x,
                         "[%s] File extension not supported (uses just .png "
                         "and .gif), using empty image.",
                         x->ObjName->s_name);
            }
        } else {
            pd_error(NULL, "Image file bad format, the file must be relative "
                           "to library folder and start with './'.");
        }
    }
    Py4pdPic_InitVisMode(x, c, py4pdArgs, 0, argc, argv, object_PY4PD_Class);
}

// =====================================================================
/*
 * @brief This function read the gif file and return the base64 string
 * @param x is the object
 * @param filename is the gif file name
 * @return void
 */
static char *Py4pdUtils_Gif2Base64(const unsigned char *data, size_t dataSize) {
    const char base64Chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t outputSize = 4 * ((dataSize + 2) / 3); // Calculate the output size
    char *encodedData = (char *)malloc(outputSize + 1);
    if (!encodedData) {
        pd_error(NULL, "Memory allocation failed.\n");
        return NULL;
    }

    size_t i, j;
    for (i = 0, j = 0; i < dataSize; i += 3, j += 4) {
        unsigned char byte1 = data[i];
        unsigned char byte2 = (i + 1 < dataSize) ? data[i + 1] : 0;
        unsigned char byte3 = (i + 2 < dataSize) ? data[i + 2] : 0;

        unsigned char charIndex1 = byte1 >> 2;
        unsigned char charIndex2 = ((byte1 & 0x03) << 4) | (byte2 >> 4);
        unsigned char charIndex3 = ((byte2 & 0x0F) << 2) | (byte3 >> 6);
        unsigned char charIndex4 = byte3 & 0x3F;

        encodedData[j] = base64Chars[charIndex1];
        encodedData[j + 1] = base64Chars[charIndex2];
        encodedData[j + 2] = (i + 1 < dataSize) ? base64Chars[charIndex3] : '=';
        encodedData[j + 3] = (i + 2 < dataSize) ? base64Chars[charIndex4] : '=';
    }

    encodedData[outputSize] = '\0'; // Null-terminate the encoded data

    return encodedData;
}
// =====================================================================
/*
 * @brief This function read the gif file and return the base64 string
 * @param x is the object
 * @param filename is the gif file name
 * @return void
 */

void Py4pdUtils_ReadGifFile(t_py *x, const char *filename) {
    (void)x;
    FILE *file = fopen(filename, "rb");
    if (!file) {
        pd_error(NULL, "Unable to open file.\n");
        return;
    }

    // pixel size
    fseek(file, 6, SEEK_SET);
    int readIntWidth = fread(&x->Width, 2, 1, file);
    int readIntHeight = fread(&x->Height, 2, 1, file);
    if (readIntWidth != 1 || readIntHeight != 1) {
        pd_error(NULL, "Failed to read file.\n");
        fclose(file);
        return;
    }
    fseek(file, 0, SEEK_END);
    size_t fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);
    // Allocate memory to store file contents
    unsigned char *fileContents = (unsigned char *)malloc(fileSize);
    if (!fileContents) {
        pd_error(NULL, "Memory allocation failed.\n");
        fclose(file);
        return;
    }

    // Read file contents
    size_t bytesRead = fread(fileContents, 1, fileSize, file);
    if (bytesRead != fileSize) {
        pd_error(NULL, "Failed to read file.\n");
        free(fileContents);
        fclose(file);
        return;
    }
    fclose(file);
    char *base64Data = Py4pdUtils_Gif2Base64(fileContents, fileSize);
    free(fileContents);
    x->ImageBase64 = base64Data;
    if (!base64Data) {
        free(base64Data);
        x->ImageBase64 = PY4PD_IMAGE;
        pd_error(NULL, "Base64 encoding failed.\n");
    }

    return;
}

// =====================================================================
/*
 * @brief This convert get the png file and convert to base64, that can be
 * readed for pd-gui
 * @param data is the png file
 * @param input_length is the size of the png file
 * @param encoded_data is the base64 string
 * @return void
 */
static void Py4pdUtils_Png2Base64(const uint8_t *data, size_t input_length, char *encoded_data) {
    const char base64_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    size_t output_length = 4 * ((input_length + 2) / 3);
    size_t padding_length = (3 - (input_length % 3)) % 3;
    size_t encoded_length = output_length + padding_length + 1;

    for (size_t i = 0, j = 0; i < input_length;) {
        uint32_t octet_a = i < input_length ? data[i++] : 0;
        uint32_t octet_b = i < input_length ? data[i++] : 0;
        uint32_t octet_c = i < input_length ? data[i++] : 0;

        uint32_t triple = (octet_a << 0x10) + (octet_b << 0x08) + octet_c;

        encoded_data[j++] = base64_chars[(triple >> 3 * 6) & 0x3F];
        encoded_data[j++] = base64_chars[(triple >> 2 * 6) & 0x3F];
        encoded_data[j++] = base64_chars[(triple >> 1 * 6) & 0x3F];
        encoded_data[j++] = base64_chars[(triple >> 0 * 6) & 0x3F];
    }

    // Add padding if necessary
    for (size_t i = 0; i < padding_length; i++) {
        encoded_data[output_length - padding_length + i] = '=';
    }

    encoded_data[encoded_length - 1] = '\0';
}
// ===============================================
/*
 * @brief This function read the png file and convert to base64
 * @param x is the object
 * @param filename is the png file
 * @return void
 */
void Py4pdUtils_ReadPngFile(t_py *x, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        pd_error(x, "Failed to open file\n");
        return;
    }

    int width, height;
    fseek(file, 16, SEEK_SET);
    int resultIntWidth = fread(&width, 4, 1, file);
    int resultIntHeight = fread(&height, 4, 1, file);
    if (resultIntWidth != 1 || resultIntHeight != 1) {
        pd_error(x, "Failed to read file\n");
        fclose(file);
        return;
    }
    width = Py4pdUtils_Ntohl(width);
    height = Py4pdUtils_Ntohl(height);
    x->Width = width;
    x->Height = height;

    // Determine the file size
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    rewind(file);

    // Allocate memory for the file data
    uint8_t *file_data = (uint8_t *)malloc(file_size);
    if (file_data == NULL) {
        pd_error(x, "Failed to allocate memory\n");
        fclose(file);
        return;
    }

    // Read the file into memory
    size_t bytes_read = fread(file_data, 1, file_size, file);
    fclose(file);

    if (bytes_read != file_size) {
        pd_error(x, "Failed to read file\n");
        free(file_data);
        return;
    }

    // Encode the file data as Base64
    size_t encoded_length = 4 * ((bytes_read + 2) / 3) + 1;
    char *base64_data = (char *)malloc(encoded_length);
    if (base64_data == NULL) {
        pd_error(x, "Failed to allocate memory\n");
        free(file_data);
        return;
    }
    Py4pdUtils_Png2Base64(file_data, bytes_read, base64_data);
    x->ImageBase64 = base64_data;
    free(file_data);
    return;
}

// =====================================================================
/*
 * @brief Run a Python function
 * @param x is the py4pd object
 * @param pArgs is the arguments to pass to the function
 * @return the return value of the function
 */
void Py4pdUtils_INCREF(PyObject *pValue) {
    if (pValue->ob_refcnt < 0) {
        pd_error(NULL, "[DEV] pValue refcnt < 0, Memory Leak, please report!");
        return;
    }

    if (Py_IsNone(pValue)) {
        return;
    }
    // check if pValue is between -5 and 255
    else if (PyLong_Check(pValue)) {
        long value = PyLong_AsLong(pValue);
        if (value >= -5 && value <= 255) {
            return;
        }
    }
    Py_INCREF(pValue);
    return;
}

// ====================================================
// ================= Compatibility between OS =========
// ====================================================
/*
 * @brief get the size (width and height) of the png file
 * @param pngfile is the path to the png file
 * @return the width and height of the png file
 */
inline uint32_t Py4pdUtils_Ntohl(uint32_t netlong) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return ((netlong & 0xff) << 24) | ((netlong & 0xff00) << 8) | ((netlong & 0xff0000) >> 8) |
           ((netlong & 0xff000000) >> 24);
#else
    return netlong;
#endif
}

// ====================================
void *Py4pdUtils_DetachCommand(void *arg) {
    char *command = (char *)arg;
    int result = system(command);
    if (result != 0) {
        fprintf(stderr, "[py4pd] Failed to execute command: %s\n", command);
    }
    return NULL;
}

// ====================================
/*
 * @brief Run system command and check for errors
 * @param command is the command to run
 * @return void, but it prints the error if it fails
 */

int Py4pdUtils_ExecuteSystemCommand(const char *command, int thread) { // TODO: WRONG PLACE
#ifdef _WIN64
    STARTUPINFO si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(STARTUPINFO));
    si.cb = sizeof(STARTUPINFO);
    ZeroMemory(&pi, sizeof(PROCESS_INFORMATION));

    DWORD exitCode;
    if (CreateProcess(NULL, (LPSTR)command, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si,
                      &pi)) {
        WaitForSingleObject(pi.hProcess, INFINITE);
        if (GetExitCodeProcess(pi.hProcess, &exitCode)) {
            if (exitCode != 0) {
                post("HELP: Try to run: '%s' from the terminal/cmd", command);
            }
        } else {
            pd_error(NULL, "[py4pd] Unable to retrieve exit code from command!");
        }
    } else {
        pd_error(NULL, "Error: Process creation failed!");
        return -1;
    }
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return exitCode;
#else

    // run system(command in another thread detached)
    if (thread == 0) {
        int result = system(command);
        if (result != 0) {
            fprintf(stderr, "[py4pd] Failed to create thread\n");
            return -1;
        }
        return 0;
    }

    pthread_t pthread;
    int pthread_create_result =
        pthread_create(&pthread, NULL, Py4pdUtils_DetachCommand, (void *)command);
    if (pthread_create_result != 0) {
        fprintf(stderr, "[py4pd] Failed to create thread\n");
        return -1;
    }
    // Detach the thread to allow it to run independently
    pthread_detach(pthread);
    return 0;
#endif
}
