#define PY_ARRAY_UNIQUE_SYMBOL py4pd_ARRAY_API
#include "py4pd.h"

// ============================================
t_class *py4pd_class;        // For for normal objects, almost unused
t_class *py4pd_classLibrary; // For libraries
int object_count = 0;

// ============================================
// =========== PY4PD LOAD LIBRARIES ===========
// ============================================
/**
 * @brief function to load the libraries (when -lib is used)
 * @param x pointer to the object
 * @param argc number of arguments
 * @param argv array of arguments
 * @return void
 */

static int Py4pd_LibraryLoad(t_py *x, int argc, t_atom *argv) {
    if (argc > 2) {
        pd_error(
            x, "[py4pd] Too many arguments! Usage: py4pd -lib <library_name>");
        return -1;
    }
    t_symbol *scriptFileName = atom_gensym(argv + 1);

    char script_file_path[MAXPDSTRING];
    snprintf(script_file_path, MAXPDSTRING, "%s/%s.py", x->pdPatchPath->s_name,
             scriptFileName->s_name);

    char script_inside_py4pd_path[MAXPDSTRING];
    snprintf(script_inside_py4pd_path, MAXPDSTRING,
             "%s/resources/scripts/%s.py", x->py4pdPath->s_name,
             scriptFileName->s_name);

    int thereIsRequirements = 0;
    PyObject *sysPath = PySys_GetObject("path");
    if (access(script_file_path, F_OK) == -1 &&
        access(script_inside_py4pd_path, F_OK) == -1) {
        Py_XDECREF(x->pFunction);
        int libraryNotFound = 1;
        for (int i = 0; 1; i++) {
            char const *pathelem = namelist_get(STUFF->st_searchpath, i);
            if (!pathelem) {
                break;
            }
            char *library_path =
                malloc(strlen(pathelem) + strlen(scriptFileName->s_name) + 1);
            snprintf(library_path, MAXPDSTRING, "%s/%s/", pathelem,
                     scriptFileName->s_name);

            if (access(library_path, F_OK) != -1) { // Library founded
                libraryNotFound = 0;
                PyList_Append(sysPath, PyUnicode_FromString(library_path));
                int requirementsSize =
                    strlen(library_path) + strlen("/requirements.txt") + 1;
                char *requirements_path = malloc(requirementsSize);
                snprintf(requirements_path, requirementsSize,
                         "%s/requirements.txt", library_path);
                if (access(requirements_path, F_OK) != -1) {
                    post("There is a requirements.txt file inside the "
                         "library "
                         "folder");
                    thereIsRequirements = 1;
                }
                free(requirements_path);
            }
            free(library_path);
        }
        if (libraryNotFound) {
            pd_error(x, "[py4pd] Library file '%s.py' not found!",
                     scriptFileName->s_name);
            return -1;
        }
    }
    if (!thereIsRequirements) {
        char localRequirements[MAXPDSTRING];
        snprintf(localRequirements, MAXPDSTRING, "%s/requirements.txt",
                 x->pdPatchPath->s_name);
        if (access(localRequirements, F_OK) != -1) {
            thereIsRequirements = 1; //  TODO: Add requirments
        }
    }

    PyObject *pModule, *pFunc; // Create the variables of the python objects
    char pyScriptsFolder[MAXPDSTRING];
    char pyGlobalFolder[MAXPDSTRING];
    snprintf(pyScriptsFolder, MAXPDSTRING, "%s/resources/scripts",
             x->py4pdPath->s_name);
    snprintf(pyGlobalFolder, MAXPDSTRING, "%s/resources/py-modules",
             x->py4pdPath->s_name);

    // conver const char* to char*
    char *pkgPathchar = malloc(strlen(x->pkgPath->s_name) + 1);
    strcpy(pkgPathchar, x->pkgPath->s_name);

    Py4pdUtils_CheckPkgNameConflict(x, pkgPathchar, scriptFileName);
    Py4pdUtils_CheckPkgNameConflict(x, pyGlobalFolder, scriptFileName);

    t_py *prev_obj = NULL;
    int prev_obj_exists = 0;
    PyObject *MainModule = PyImport_ImportModule("pd");
    PyObject *oldObjectCapsule;

    if (MainModule != NULL) {
        oldObjectCapsule =
            PyDict_GetItemString(MainModule, "py4pd"); // borrowed reference
        if (oldObjectCapsule != NULL) {
            PyObject *py4pd_capsule =
                PyObject_GetAttrString(MainModule, "py4pd");
            prev_obj = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
            prev_obj_exists = 1;
        } else {
            prev_obj_exists = 0;
        }
    }
    PyObject *objectCapsule = Py4pdUtils_AddPdObject(x);

    if (objectCapsule == NULL) {
        pd_error(x, "[Python] Failed to add object to Python");
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyObject *ptype_str = PyObject_Str(ptype);
        PyObject *pvalue_str = PyObject_Str(pvalue);
        PyObject *ptraceback_str = PyObject_Str(ptraceback);
        char const *ptype_c = PyUnicode_AsUTF8(ptype_str);
        char const *pvalue_c = PyUnicode_AsUTF8(pvalue_str);
        char const *ptraceback_c = PyUnicode_AsUTF8(ptraceback_str);
        pd_error(x, "[Python] %s: %s\n%s", ptype_c, pvalue_c, ptraceback_c);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        Py_XDECREF(ptype_str);
        Py_XDECREF(pvalue_str);
        Py_XDECREF(ptraceback_str);
        Py_XDECREF(MainModule);
        return -1;
    }
    pModule =
        PyImport_ImportModule(scriptFileName->s_name); // Import the script file
    if (pModule == NULL) {
        pd_error(x, "[Python] Failed to load script file %s",
                 scriptFileName->s_name);
        Py4pdUtils_PrintError(x);
        Py_XDECREF(pModule);
        Py_XDECREF(MainModule);
        return -1;
    }

    // reload the module if it already exists
    PyObject *pModuleReloaded = PyImport_ReloadModule(pModule);
    if (pModuleReloaded != NULL) {
        Py_DECREF(pModule);
        pModule = pModuleReloaded;
    } else {
        Py4pdUtils_PrintError(x);
        Py_XDECREF(pModule);
        Py_XDECREF(MainModule);
        return -1;
    }

    PyObject *pModuleDict = PyModule_GetDict(pModule);

    if (pModuleDict == NULL) {
        pd_error(x, "[Python] Failed to get script file dictionary");
        return -1;
    }

    PyObject *pFilenameObj = PyDict_GetItemString(pModuleDict, "__file__");
    if (!pFilenameObj) {
        pd_error(x, "[Python] Failed to get script file path");
        Py_DECREF(pModule);
        Py_XDECREF(MainModule);
        return -1;
    }
    // convert const char * to char *
    char *libraryFolder = malloc(strlen(PyUnicode_AsUTF8(pFilenameObj)) + 1);
    strcpy(libraryFolder, PyUnicode_AsUTF8(pFilenameObj));
    x->libraryFolder = gensym(Py4pdUtils_GetFolderName(libraryFolder));
    free(libraryFolder);

    if (pModule == NULL) {
        Py4pdUtils_PrintError(x);
        Py_XDECREF(pModule);
        Py_XDECREF(MainModule);
        return -1;
    }

    // check if module has the function Py4pdLoadObjects or "script_file_name +
    // setup"
    char *setupFuncName = malloc(strlen(scriptFileName->s_name) + 7);
    snprintf(setupFuncName, strlen(scriptFileName->s_name) + 7, "%s_setup",
             scriptFileName->s_name);
    PyObject *pFuncName = PyUnicode_FromString(setupFuncName);
    t_symbol *pFuncNameSymbol = gensym(setupFuncName);
    pFunc = PyObject_GetAttr(pModule, pFuncName);
    if (pFunc == NULL) {
        Py_DECREF(pFuncName);
        pFuncName = PyUnicode_FromString("py4pdLoadObjects");
        Py_XDECREF(pFunc);
        pFunc = PyObject_GetAttr(pModule, pFuncName);
        if (pFunc == NULL) {
            pd_error(x, "[Python] Failed to load function %s", setupFuncName);
            Py4pdUtils_PrintError(x);
            Py_XDECREF(pModule);
            free(setupFuncName);
            return -1;
        }
        pFuncNameSymbol = gensym("Py4pdLoadObjects");
    }
    free(setupFuncName);

    if (pFunc && PyCallable_Check(pFunc)) {
        if (objectCapsule == NULL) {
            pd_error(x, "[Python] Failed to add object to Python, capsule "
                        "not found.");
            return -1;
        }
        PyObject *pValue = PyObject_CallNoArgs(pFunc); // Call the function

        if (prev_obj_exists == 1 && pValue != NULL) {
            objectCapsule = Py4pdUtils_AddPdObject(prev_obj);
            if (objectCapsule == NULL) {
                pd_error(x, "[Python] Failed to add object to Python");
                return -1;
            }
        }

        if (pValue == NULL) {
            Py4pdUtils_PrintError(x);
            Py_XDECREF(MainModule);
            Py_XDECREF(pModule);
            Py_XDECREF(pModuleReloaded);
            Py_XDECREF(pFunc);
            Py_XDECREF(pValue);
            return -1;
        }
        // odd code, but solve the bug
        if (prev_obj_exists == 1 && pValue != NULL) {
            objectCapsule = Py4pdUtils_AddPdObject(prev_obj);
            if (objectCapsule == NULL) {
                pd_error(x, "[Python] Failed to add object to Python");
                return -1;
            }
        }
        x->pModule = pModule;
        x->pFunction = pFunc;
        x->pScriptName = scriptFileName;
        x->pFuncName = pFuncNameSymbol;
        x->funcCalled = 1;
        x->isLib = 1;
        Py_XDECREF(MainModule);
        Py_XDECREF(pModuleReloaded);
        Py_XDECREF(pValue);
        logpost(x, 3, "[py4pd] Library %s loaded!", scriptFileName->s_name);
    } else {
        x->funcCalled = 1; // set the flag to 0 because it crash Pd if
        Py4pdUtils_PrintError(x);
        Py_XDECREF(pModule);
        Py_XDECREF(MainModule);
        Py_XDECREF(pFunc);
    }
    return 0;
}

// ========================================
struct pipInstallArgs {
    t_py *x;
    t_symbol *pipPackage;
};

// ========================================
void *Py4pd_PipInstallDetach(void *Args) {
    struct pipInstallArgs *args = (struct pipInstallArgs *)Args;
    t_py *x = args->x;
    char const *pipPackage = args->pipPackage->s_name;

    if (x->pipGlobalInstall) {
#ifdef __linux__
        size_t commandSize = snprintf(NULL, 0,
                                      "python%d.%d -m pip install --target "
                                      "%sresources/py-modules %s --upgrade",
                                      PY_MAJOR_VERSION, PY_MINOR_VERSION,
                                      x->py4pdPath->s_name, pipPackage) +
                             1;

#elif defined __WIN32
        size_t commandSize = snprintf(NULL, 0,
                                      "py -%d.%d -m pip install --target "
                                      "%sresources/py-modules %s --upgrade",
                                      PY_MAJOR_VERSION, PY_MINOR_VERSION,
                                      x->py4pdPath->s_name, pipPackage) +
                             1;
#elif defined(__APPLE__) || defined(__MACH__)
        if (x->pipGlobalInstall) {
            size_t commandSize =
                snprintf(NULL, 0,
                         "/usr/local/bin/python%d.%d -m pip install --target "
                         "%sresources/py-modules %s --upgrade",
                         PY_MAJOR_VERSION, PY_MINOR_VERSION,
                         x->py4pdPath->s_name, pipPackage) +
                1;
#endif

        char *COMMAND = malloc(commandSize);
#ifdef __linux__
        snprintf(COMMAND, commandSize,
                 "python%d.%d -m pip install --target %sresources/py-modules "
                 "%s --upgrade",
                 PY_MAJOR_VERSION, PY_MINOR_VERSION, x->py4pdPath->s_name,
                 pipPackage);

#elif defined __WIN32
        snprintf(COMMAND, commandSize,
                 "py -%d.%d -m pip install --target %sresources/py-modules "
                 "%s --upgrade",
                 PY_MAJOR_VERSION, PY_MINOR_VERSION, x->py4pdPath->s_name,
                 pipPackage);

#elif defined(__APPLE__) || defined(__MACH__)
            snprintf(COMMAND, commandSize,
                     "/usr/local/bin/python%d.%d -m pip install --target "
                     "%sresources/py-modules %s --upgrade",
                     PY_MAJOR_VERSION, PY_MINOR_VERSION, x->py4pdPath->s_name,
                     pipPackage);
#endif
        pd_error(NULL, "Installing %s in background, please WAIT ...",
                 pipPackage);
        int result = system(COMMAND);
        if (result == -1) {
            pd_error(NULL, "The instalation seems to have failed. Restart "
                           "PureData and try again.\n");
            free(COMMAND);
            return NULL;
        } else {
            pd_error(NULL, "The installation has been completed.\n");
            free(COMMAND);
            return NULL;
        }
    }
    return NULL;
}

// ============================================
/**
 * @brief it installs a package using pip
 * @param x pointer to the object
 * @param s symbol
 * @param argc number of arguments
 * @param argv array of arguments
 * @return void
 */
static void Py4pd_PipInstall(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;

    struct pipInstallArgs pipArgs;
    pipArgs.x = x;
    pipArgs.pipPackage = atom_getsymbolarg(0, argc, argv);

    // run using pthread
    pthread_t threadId;
    pthread_create(&threadId, NULL, Py4pd_PipInstallDetach, &pipArgs);
    pthread_detach(threadId);
    return;
}

// ============================================
/**
 * @brief it prints the version of py4pd and python
 * @param x pointer to the object
 * @return void
 */
static void Py4pd_PrintPy4pdVersion(t_py *x) {
    int major, minor, micro;
    major = PY4PD_MAJOR_VERSION;
    minor = PY4PD_MINOR_VERSION;
    micro = PY4PD_MICRO_VERSION;
    t_atom py4pdVersionArray[3];
    SETFLOAT(&py4pdVersionArray[0], major);
    SETFLOAT(&py4pdVersionArray[1], minor);
    SETFLOAT(&py4pdVersionArray[2], micro);
    outlet_anything(x->mainOut, gensym("py4pd"), 3, py4pdVersionArray);
    t_atom pythonVersionArray[3];
    major = PY_MAJOR_VERSION;
    minor = PY_MINOR_VERSION;
    micro = PY_MICRO_VERSION;
    SETFLOAT(&pythonVersionArray[0], major);
    SETFLOAT(&pythonVersionArray[1], minor);
    SETFLOAT(&pythonVersionArray[2], micro);
    outlet_anything(x->mainOut, gensym("python"), 3, pythonVersionArray);
}

// ============================================
/**
 * @brief set the home path to py4pd
 * @brief Get the config from py4pd.cfg file
 * @param x is the py4pd object
 * @param s is the symbol (message) that was sent to the object
 * @param argc is the number of arguments
 * @param argv is the arguments
 * @return void, but it sets the home path
 */

static void Py4pd_SetPy4pdHomePath(t_py *x, t_symbol *s, int argc,
                                   t_atom *argv) {
    (void)s;
    if (argc < 1) {
        post("[py4pd] The home path is: %s", x->pdPatchPath->s_name);
    } else {
        x->pdPatchPath = atom_getsymbol(argv);
        post("[py4pd] The home path set to: %s", x->pdPatchPath->s_name);
    }
    return;
}

// ============================================
/**
 * @brief set the packages path to py4pd, if start with . then build the
 * complete path
 * @brief set the packages path to py4pd, if start with . then build the
 * complete path
 * @param x is the py4pd object
 * @param s is the symbol (message) that was sent to the object
 * @param argc is the number of arguments
 * @param argv is the arguments
 * @return void, but it sets the packages path
 */

static void Py4pd_SetPackages(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    if (argc < 1) {
        post("[py4pd] The packages path is: %s", x->pkgPath->s_name);
        return;
    } else {
        if (argc == 1) {
            if (argv[0].a_type == A_SYMBOL) {
                t_symbol *path = atom_getsymbol(argv);
                // It checks relative path
                if (path->s_name[0] == '.' && path->s_name[1] == '/') {
                    char *new_path = malloc(strlen(x->pdPatchPath->s_name) +
                                            strlen(path->s_name) + 1);
                    strcpy(new_path, x->pdPatchPath->s_name);
                    strcat(new_path, path->s_name + 1);
                    post("[py4pd] Packages path set to: %s", new_path);
                    x->pkgPath = gensym(new_path);
                    free(new_path);
                } else {
                    x->pkgPath = atom_getsymbol(argv);
                    post("[py4pd] Packages path set to: %s",
                         x->pkgPath->s_name);
                }
                char cfgFile[MAXPDSTRING];
                char const *py4pdDir = x->py4pdPath->s_name;
                snprintf(cfgFile, MAXPDSTRING, "%s/py4pd.cfg", py4pdDir);
                FILE *file = fopen(cfgFile, "w");
                if (x->editorName != NULL) {
                    fprintf(file, "editor = %s\n", x->editorName->s_name);
                }
                fprintf(file, "packages = %s", x->pkgPath->s_name);
                fclose(file);
            } else {
                pd_error(x, "[py4pd] The packages path must be a string");
                return;
            }
            // check if path exists and is valid
            if (access(x->pkgPath->s_name, F_OK) == -1) {
                pd_error(x, "[py4pd] The packages path is not valid");
                return;
            }
        } else {
            pd_error(x, "It seems that your package folder has |spaces|.");
            return;
        }
        return;
    }
}

// ====================================
/**
 * @brief print all the functions in the module
 * @param x is the py4pd object
 * @param s is the symbol (message) that was sent to the object
 * @param argc is the number of arguments
 * @param argv is the arguments
 * @return void, but it prints the functions


*/
static void Py4pd_PrintModuleFunctions(t_py *x, t_symbol *s, int argc,
                                       t_atom *argv) {
    (void)s;
    (void)argc;
    (void)argv;

    PyObject *module_dict = PyModule_GetDict(x->pModule);
    Py_ssize_t pos = 0;
    PyObject *key, *value;

    post("[py4pd] Functions in module %s:", x->pScriptName->s_name);
    while (PyDict_Next(module_dict, &pos, &key, &value)) {
        if (PyCallable_Check(value)) {
            post("[py4pd] Function: %s", PyUnicode_AsUTF8(key));
        }
    }
}

// ====================================
/**
 * @brief print the documentation of the function
 * @param x is the py4pd object
 * @param s is the symbol (message) that was sent to the object
 * @param argc is the number of arguments
 * @param argv is the arguments
 * @return void, but it prints the documentation
 */
void Py4pd_PrintDocs(t_py *x) {
    if (x->funcCalled == 0) {
        pd_error(x, "[py4pd] To see the documentaion you need to set the "
                    "function first!");
        return;
    }
    if (x->pFunction &&
        PyCallable_Check(
            x->pFunction)) { // Check if the function exists and is callable
        PyObject *pDoc = PyObject_GetAttrString(
            x->pFunction,
            "__doc__"); // Get the documentation of the function
        if (pDoc != NULL) {
            char const *Doc = PyUnicode_AsUTF8(pDoc);
            if (Doc != NULL) {
                post("");
                post("==== %s documentation ====", x->pFuncName->s_name);
                post("");
                post("%s", Doc);
                post("");
                post("==== %s documentation ====", x->pFuncName->s_name);
                post("");
                return;
            } else {
                pd_error(x, "[py4pd] No documentation found!");
                return;
            }
        } else {
            pd_error(x, "[py4pd] No documentation found!");
            return;
        }
    }
}

// ============================================
/**
 * @brief open the script  in the editor
 * @param x is the py4pd object
 * @param s is the symbol (message) that was sent to the object
 * @param argc is the number of arguments
 * @param argv is the arguments
 * @return void, but it opens the script in the editor
 */
static void Py4pd_OpenScript(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    (void)argc;

    if (argv[0].a_type != A_SYMBOL) {
        pd_error(x, "[py4pd] The script name must be a symbol");
        return;
    }
    x->pScriptName = argv[0].a_w.w_symbol;
    char command[MAXPDSTRING];
    Py4pdUtils_GetEditorCommand(x, command, 0);
    Py4pdUtils_ExecuteSystemCommand(command);
    return;
}

// ====================================
/**
 * @brief set the editor
 * @param x is the py4pd object
 * @param s is the symbol (message) that was sent to the object
 * @param argc is the number of arguments
 * @param argv is the arguments
 * @return void, but it sets the editor
 */
void Py4pd_SetEditor(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    // check if argv[0] is a symbol and it is diferent from "float"
    if (argc != 0 && argv[0].a_type == A_SYMBOL &&
        strcmp(argv[0].a_w.w_symbol->s_name, "float") != 0) {
        x->editorName = atom_getsymbol(argv + 0);
        post("[py4pd] Editor set to: %s", x->editorName->s_name);
        char cfgFile[MAXPDSTRING];
        char const *py4pdDir = x->py4pdPath->s_name;
        snprintf(cfgFile, MAXPDSTRING, "%s/py4pd.cfg", py4pdDir);
        FILE *file = fopen(cfgFile, "w");
        fprintf(file, "editor = %s\n", x->editorName->s_name);
        fprintf(file, "packages = %s", x->pkgPath->s_name);
        fclose(file);
        return;
    }
    if (x->funcCalled == 0) { // if the set method was not called, then we
        pd_error(x, "[py4pd] To open the editor you need to set the "
                    "function first!");
        return;
    }

    PyCodeObject *code = (PyCodeObject *)PyFunction_GetCode(x->pFunction);
    int line = PyCode_Addr2Line(code, 0);
    char command[MAXPDSTRING];
    Py4pdUtils_GetEditorCommand(x, command, line);
    Py4pdUtils_ExecuteSystemCommand(command);
    return;
}

// ====================================
/**
 * @brief reload the Python Script
 * @param x is the py4pd object
 * @param s is the symbol (message) that was sent to the object
 * @param argc is the number of arguments
 * @param argv is the arguments
 * @return void, but it reloads the script
 */
void Py4pd_ReloadPy4pdFunction(t_py *x) {
    PyObject *pName, *pFunc, *pModule, *pReload;
    if (x->funcCalled == 0) { // if the set method was not called, then we
                              // can not run the function :)
        pd_error(x, "To reload the script you need to set the function first!");
        return;
    }
    pFunc = x->pFunction;

    // reload the module
    pName = PyUnicode_DecodeFSDefault(
        x->pScriptName->s_name); // Name of script file
    pModule = PyImport_Import(pName);
    if (pModule == NULL) {
        pd_error(x, "Error importing the module!");
        x->funcCalled = 0;
        Py_DECREF(pFunc);
        Py_DECREF(pName);
        return;
    }

    pReload = PyImport_ReloadModule(pModule);
    if (pReload == NULL) {
        pd_error(x, "Error reloading the module!");
        x->funcCalled = 0;
        Py_DECREF(pFunc);
        Py_DECREF(pModule);
        return;
    } else {
        pFunc = PyObject_GetAttrString(
            pModule,
            x->pFuncName->s_name); // Function name inside the script file
        Py_DECREF(pName);
        Py_DECREF(pReload);
        if (pFunc &&
            PyCallable_Check(
                pFunc)) { // Check if the function exists and is callable
            x->pFunction = pFunc;
            x->funcCalled = 1;
            PyCodeObject *code =
                (PyCodeObject *)PyFunction_GetCode(x->pFunction);
            int argCount = code->co_argcount;
            x->pArgsCount = argCount;
            post("The module was reloaded!");
            return;
        } else {
            pd_error(x, "Error reloading the module!");
            x->funcCalled = 0;
            Py_DECREF(x->pFunction);
            return;
        }
    }
}

// ====================================
/**
 * @brief set the python function and save it on x->pFunction
 * @param x is the py4pd object
 * @param s is the symbol (message) that was sent to the object
 * @param argc is the number of arguments
 * @param argv is the arguments
 * @return void, but it sets the function
 */
void Py4pd_SetFunction(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    // =====================
    // check number of arguments
    if (argc < 2) { // check is the number of arguments is correct | set
        pd_error(x,
                 "[py4pd] 'set' message needs two arguments! The 'Script Name' "
                 "and the 'Function Name'!");
        return;
    }

    t_symbol *script_file_name = atom_gensym(argv + 0);
    t_symbol *pFuncNameSymbol = atom_gensym(argv + 1);

    if (x->funcCalled == 1) {
        int function_is_equal =
            strcmp(pFuncNameSymbol->s_name,
                   x->pFuncName->s_name); // if string is equal strcmp returns 0
        if (function_is_equal == 0) {
            pd_error(x, "[py4pd] The function was already set!");
            return;
        } else {
            Py_XDECREF(x->pFunction);
            x->funcCalled = 0;
        }
    }

    // Check if there is extension (not to use it)
    char *extension = strrchr(script_file_name->s_name, '.');
    if (extension != NULL) {
        pd_error(x, "[py4pd] Don't use extensions in the script file name!");
        Py_XDECREF(x->pFunction);
        return;
    }

    // check if script file exists
    char script_file_path[MAXPDSTRING];
    snprintf(script_file_path, MAXPDSTRING, "%s/%s.py", x->pdPatchPath->s_name,
             script_file_name->s_name);

    char script_inside_py4pd_path[MAXPDSTRING];
    snprintf(script_inside_py4pd_path, MAXPDSTRING,
             "%s/resources/scripts/%s.py", x->py4pdPath->s_name,
             script_file_name->s_name);

    if (access(script_file_path, F_OK) == -1 &&
        access(script_inside_py4pd_path, F_OK) == -1) {
        pd_error(x, "[py4pd] The script file %s was not found!",
                 script_file_name->s_name);
        Py_XDECREF(x->pFunction);
        return;
    }
    PyObject *pModule, *pFunc; // Create the variables of the python objects

    // odd code, but solve the bug
    t_py *prev_obj;
    int prev_obj_exists = 0;
    PyObject *MainModule = PyImport_ImportModule("pd");
    PyObject *oldObjectCapsule;

    if (MainModule != NULL) {
        oldObjectCapsule =
            PyDict_GetItemString(MainModule, "py4pd"); // borrowed reference
        if (oldObjectCapsule != NULL) {
            PyObject *py4pd_capsule =
                PyObject_GetAttrString(MainModule, "py4pd");
            prev_obj = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
            prev_obj_exists = 1;
        } else {
            prev_obj_exists = 0;
        }
    }

    PyObject *objectCapsule = Py4pdUtils_AddPdObject(x);

    // =====================
    pModule = PyImport_ImportModule(
        script_file_name->s_name); // Import the script file with the function
    // =====================

    // check if the module was loaded
    if (pModule == NULL) {
        Py4pdUtils_PrintError(x);
        Py_XDECREF(pModule);
        return;
    }
    pFunc = PyObject_GetAttrString(
        pModule,
        pFuncNameSymbol->s_name); // Function name inside the script file
    if (pFunc && PyCallable_Check(
                     pFunc)) { // Check if the function exists and is callable
        PyCodeObject *code = (PyCodeObject *)PyFunction_GetCode(pFunc);

        if (prev_obj_exists == 1 && pFunc != NULL) {
            objectCapsule = Py4pdUtils_AddPdObject(prev_obj);
            if (objectCapsule == NULL) {
                pd_error(x, "[Python] Failed to add object to Python");
                return;
            }
        }

        if (code->co_flags & CO_VARARGS) {
            pd_error(
                x, "[py4pd] The '%s' function has variable arguments (*args)!",
                pFuncNameSymbol->s_name);
            Py_XDECREF(pFunc);
            Py_XDECREF(pModule);
            return;
        } else if (code->co_flags & CO_VARKEYWORDS) {
            pd_error(x,
                     "[py4pd] The '%s' function has variable keyword arguments "
                     "(**kwargs)!",
                     pFuncNameSymbol->s_name);
            Py_XDECREF(pFunc);
            Py_XDECREF(pModule);
            return;
        }
        x->pArgsCount = code->co_argcount;
        if (x->isLib == 0) {
            post("[py4pd] The '%s' function has %d arguments!",
                 pFuncNameSymbol->s_name, x->pArgsCount);
        }
        x->pModule = pModule;
        x->pFunction = pFunc;
        x->pScriptName = script_file_name;
        x->pFuncName = pFuncNameSymbol;
        x->funcCalled = 1;
    } else {
        pd_error(x, "[py4pd] Function %s not loaded!", pFuncNameSymbol->s_name);
        x->funcCalled = 0;
        Py4pdUtils_PrintError(x);
        Py_XDECREF(pModule);
        Py_XDECREF(pFunc);
    }
    return;
}

// ============================================
/**
 * @brief Run the function setted, it works with lists too.
 * @param x
 * @param s
 * @param argc
 * @param argv
 * @param x
 * @param s
 * @param argc
 * @param argv
 */

static void Py4pd_RunFunction(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    int OpenList_count = 0;
    int CloseList_count = 0;
    PyObject *ArgsTuple;
    ArgsTuple = NULL;

    if (argc != 0) {
        for (int i = 0; i < argc; i++) {
            if (argv[i].a_type == A_SYMBOL) {
                if (strchr(argv[i].a_w.w_symbol->s_name, '[') != NULL) {
                    CloseList_count++;
                }
                if (strchr(argv[i].a_w.w_symbol->s_name, ']') != NULL) {
                    OpenList_count++;
                }
            }
        }
        if (OpenList_count != CloseList_count) {
            pd_error(x, "[py4pd] The number of '[' and ']' is not the same!");
            return;
        }
        PyObject **lists =
            (PyObject **)malloc(OpenList_count * sizeof(PyObject *));

        ArgsTuple =
            Py4pdUtils_ConvertToPy(lists, argc,
                                   argv); // convert the arguments to python
        int argCount = PyTuple_Size(ArgsTuple); // get the number of arguments
        if (argCount != x->pArgsCount) {
            pd_error(x,
                     "[py4pd] Wrong number of arguments! The function %s "
                     "needs %i "
                     "arguments, received %i!",
                     x->pFuncName->s_name, (int)x->pArgsCount, argCount);
            return;
        }
        free(lists);
    } else {
        ArgsTuple = PyTuple_New(0);
    }
    Py4pdUtils_RunPy(x, ArgsTuple, NULL);
    return;
}

// ============================================
/**
 * @brief This function will control were the Python will run, with PEP 684,
 * I want to make possible using parallelism in Python
 * @param x
 * @param s
 * @param argc
 * @param argv
 * @brief This function will control were the Python will run, with PEP 684,
 * I want to make possible using parallelism in Python
 * @param x
 * @param s
 * @param argc
 * @param argv
 * @return It will return nothing but will run the Python function
 */

static void Py4pd_ExecuteFunction(t_py *x, t_symbol *s, int argc,
                                  t_atom *argv) {
    (void)s;
    if (x->funcCalled == 0) {
        pd_error(x, "[py4pd] You need to call a function before run!");
        return;
    }
    Py4pd_RunFunction(x, s, argc, argv); // Implement here the functions
    return;
}

// ============================================
/**
 * @brief This will enable or disable the numpy array support and start
 * numpy import if it is not imported.
 * @brief This will enable or disable the numpy array support and start
 * numpy import if it is not imported.
 * @param x is the py4pd object
 * @param f is the status of the numpy array support
 * @return It will return void.
 */
void Py4pd_SetPythonPointersUsage(t_py *x, t_floatarg f) {
    int usepointers = (int)f;
    if (usepointers == 1) {
        post("[py4pd] Python Pointers enabled.");
        x->outPyPointer = 1;
    } else if (usepointers == 0) {
        x->outPyPointer = 0;
        post("[py4pd] Python Pointers disabled");
    } else {
        pd_error(x, "[py4pd] Python Pointers status must be 0 (disable) or 1 "
                    "(enable)");
    }
    return;
}

// ============================================
// =========== SETUP OF OBJECT ================
// ============================================
/**
 * @brief This function create the py4pd object.
 * @param s is the name of the object
 * @param argc is the number of arguments
 * @param argv is the arguments
 * @return It will return the py4pd object.
 */

void *Py4pd_Py4pdNew(t_symbol *s, int argc, t_atom *argv) {
    int i;
    t_py *x;
    int libraryMODE = 0;
    int normalMODE = 1;
    t_symbol *scriptName = NULL;
    int major, minor, micro;
    sys_getversion(&major, &minor, &micro);
    if (major < 0 && minor < 54) {
        pd_error(NULL, "[py4pd] You need to use Pd 0.54 or higher");
        return NULL;
    }

    // Get what will be the type of the object
    for (i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            t_symbol *py4pdArgs = atom_getsymbolarg(i, argc, argv);
            if (py4pdArgs == gensym("-library") ||
                py4pdArgs == gensym("-lib")) {
                libraryMODE = 1;
                normalMODE = 0;
                scriptName = atom_getsymbolarg(i + 1, argc, argv);
            }
        }
    }

    if (normalMODE == 1) {
        x = (t_py *)pd_new(py4pd_class); // create a new py4pd object
        x->canvas = canvas_getcurrent();
        t_canvas *c = x->canvas;
        t_symbol *patch_dir = canvas_getdir(c);
        x->zoom = (int)x->canvas->gl_zoom;
        x->audioInput = 0;
        x->audioOutput = 0;
        x->visMode = 0;
        x->editorName = NULL;
        x->pipGlobalInstall = 1;
        x->pyObject = 0;
        x->vectorSize = 0;
        Py4pdUtils_ParseArguments(x, c, argc, argv); // parse arguments
        x->pdPatchPath = patch_dir; // set name of the home path
        x->pkgPath = patch_dir;     // set name of the packages path
        x->pArgsCount = 0;
        Py4pdUtils_SetObjConfig(x);

        if (object_count == 0) {
            Py4pdUtils_AddPathsToPythonPath(x);
        }
        if (argc > 1) { // check if there are two arguments
            if (_import_array() != 0) {
                pd_error(NULL,
                         "\n!!!!!!\n [py4pd] Unable to import NumPy! Send "
                         "[pipinstall "
                         "global numpy] to py4pd object to install it.] "
                         "\n!!!!!!\n");
                x->numpyImported = 0;
                return (x);
            }
            Py4pd_SetFunction(x, s, argc, argv);
            x->numpyImported = 1;
        }
        object_count++;
        return (x);
    } else if (libraryMODE == 1) { // library
        x = (t_py *)pd_new(py4pd_classLibrary);
        x->canvas = canvas_getcurrent();
        t_canvas *c = x->canvas;
        x->zoom = (int)x->canvas->gl_zoom;
        t_symbol *patch_dir = canvas_getdir(c);
        x->pdPatchPath = patch_dir;
        x->pkgPath = patch_dir;
        x->visMode = 0;
        Py4pdUtils_SetObjConfig(x);
        if (object_count == 0) {
            Py4pdUtils_AddPathsToPythonPath(x);
            if (_import_array() != 0) {
                pd_error(NULL,
                         "\n!!!!!!\n [py4pd] Unable to import NumPy! Send "
                         "[pipinstall "
                         "global numpy] to py4pd object to install it.] "
                         "\n!!!!!!\n");
                x->numpyImported = 0;
                return NULL;
            }
        }
        x->numpyImported = 1;
        int libraryLoaded = Py4pd_LibraryLoad(x, argc, argv);
        if (libraryLoaded == -1) {
            return NULL;
        }
        x->pScriptName = scriptName;
        object_count++;
        return (x);
    } else {
        pd_error(NULL,
                 "Error in py4pdNew, you can not use more than one flag at "
                 "the same time.");
        return NULL;
    }
}

// ====================================================
/**
 * @brief Setup the py4pd object
 */

void py4pd_setup(void) {
    if (!Py_IsInitialized()) {
        object_count = 0;
        post("");
        post("[py4pd] by Charles K. Neimog");
        post("[py4pd] Version %d.%d.%d", PY4PD_MAJOR_VERSION,
             PY4PD_MINOR_VERSION, PY4PD_MICRO_VERSION);
        post("[py4pd] Python version %d.%d.%d", PY_MAJOR_VERSION,
             PY_MINOR_VERSION, PY_MICRO_VERSION);
        post("");
        PyImport_AppendInittab("pd", PyInit_pd);
        Py_Initialize();
    }

    py4pd_class =
        class_new(gensym("py4pd"), // cria o objeto quando escrevemos py4pd
                  (t_newmethod)Py4pd_Py4pdNew, // metodo de criação do objeto
                  (t_method)Py4pdLib_FreeObj,  // quando voce deleta o objeto
                  sizeof(t_py), // quanta memoria precisamos para esse objeto
                  0,            // nao há uma GUI especial para esse objeto???
                  A_GIMME,      // os podem ser qualquer coisa
                  0);           // fim de argumentos

    py4pd_classLibrary = class_new(gensym("py4pd"), (t_newmethod)Py4pd_Py4pdNew,
                                   (t_method)Py4pdLib_FreeObj, sizeof(t_py),
                                   CLASS_NOINLET, A_GIMME, 0);

    // this is like have lot of objects with the same name, add all methods
    // for
    class_addmethod(py4pd_class, (t_method)Py4pd_SetPy4pdHomePath,
                    gensym("home"), A_GIMME,
                    0); // set home path
    class_addmethod(py4pd_class, (t_method)Py4pd_SetPackages,
                    gensym("packages"), A_GIMME,
                    0); // set packages path
    class_addmethod(py4pd_class, (t_method)Py4pd_PipInstall,
                    gensym("pipinstall"), A_GIMME,
                    0); // on/off threading
    class_addmethod(py4pd_class, (t_method)Py4pd_SetPythonPointersUsage,
                    gensym("pointers"), A_FLOAT,
                    0); // set home path
    class_addmethod(py4pd_class, (t_method)Py4pd_ReloadPy4pdFunction,
                    gensym("reload"), 0,
                    0); // reload python script

    // Object INFO
    class_addmethod(py4pd_class, (t_method)Py4pd_PrintPy4pdVersion,
                    gensym("version"), 0,
                    0); // show version
    class_addmethod(py4pd_class, (t_method)Py4pd_SetEditor, gensym("editor"),
                    A_GIMME,
                    0); // open code
    class_addmethod(py4pd_class, (t_method)Py4pd_OpenScript, gensym("open"),
                    A_GIMME, 0);
    class_addmethod(py4pd_class, (t_method)Py4pd_SetEditor, gensym("click"), 0,
                    0); // when click open editor
    class_addmethod(py4pd_classLibrary, (t_method)Py4pd_SetEditor,
                    gensym("click"), 0,
                    0); // when click open editor

    // User
    class_addmethod(py4pd_class, (t_method)Py4pd_PrintDocs, gensym("doc"), 0,
                    0); // open documentation
    class_addmethod(py4pd_class, (t_method)Py4pd_ExecuteFunction, gensym("run"),
                    A_GIMME,
                    0); // run function
    class_addmethod(py4pd_class, (t_method)Py4pd_SetFunction, gensym("set"),
                    A_GIMME,
                    0); // set function to be called
    class_addmethod(py4pd_class, (t_method)Py4pd_PrintModuleFunctions,
                    gensym("functions"), A_GIMME, 0);

// test functions
#if PYTHON_REQUIRED_VERSION(3, 12)
    class_addmethod(py4pd_class, (t_method)Py4pdUtils_CreatePythonInterpreter,
                    gensym("detach"), 0);
#endif
}
