#include "py4pd.h"
#include "module.h"
#include "utils.h"

// ============================================
t_class *Py4pdObjClass; // For for normal objects, almost unused
t_class *Py4pdLibClass; // For libraries
int objCount = 0;       // To keep track of the number of objects created

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

/* TODO: Add way to load local modules inside some folders.
 * Isso é importante para que, caso uma peça seja feita com um módulo (versão específica)
 * seja possível usar aquela versão para a obra. Assim também posso atualizar os módulos sem pensar
em quebras entre outros.
*/

static int Py4pd_LibraryLoad(t_py *Py4pd, int Argc, t_atom *Argv) {
    LOG("Py4pd_LibraryLoad");
    if (Argc > 2) {
        pd_error(Py4pd, "[py4pd] Too many arguments! Usage: py4pd -lib <library_name>");
        return -1;
    }
    printf("Loading library %s\n", atom_gensym(Argv + 1)->s_name);
    const char *Py4pdLibName = atom_gensym(Argv + 1)->s_name;
    char Py4pdCorrectLibName[MAXPDSTRING];
    int j;
    for (j = 0; Py4pdLibName[j] != '\0'; j++) {
        Py4pdCorrectLibName[j] = (Py4pdLibName[j] == '-') ? '_' : Py4pdLibName[j];
    }
    Py4pdCorrectLibName[j] = '\0';
    t_symbol *ScriptFileName = gensym(Py4pdCorrectLibName);

    char ScriptFilePath[MAXPDSTRING];
    snprintf(ScriptFilePath, MAXPDSTRING, "%s/%s.py", Py4pd->PdPatchPath->s_name,
             atom_gensym(Argv + 1)->s_name);

    char ScriptInsidePy4pdFolder[MAXPDSTRING];
    snprintf(ScriptInsidePy4pdFolder, MAXPDSTRING, "%s/%s.py", Py4pd->Py4pdPath->s_name,
             atom_gensym(Argv + 1)->s_name);

    PyObject *SysPath = PySys_GetObject("path");
    if (access(ScriptFilePath, F_OK) == -1 && access(ScriptInsidePy4pdFolder, F_OK) == -1) {
        Py_XDECREF(Py4pd->pFunction);
        int LibNotFound = 1;
        for (int i = 0; 1; i++) {
            char const *pathelem = namelist_get(STUFF->st_searchpath, i);
            if (!pathelem) {
                break;
            }
            char *LibPath = (char *)malloc(strlen(pathelem) + strlen(ScriptFileName->s_name) + 1);
            snprintf(LibPath, MAXPDSTRING, "%s/%s/", pathelem, atom_gensym(Argv + 1)->s_name);
            if (access(LibPath, F_OK) != -1) { // Library found
                LibNotFound = 0;
                PyList_Append(SysPath, PyUnicode_FromString(LibPath));
                post("[py4pd] Library path added: %s", LibPath);
            }
            free(LibPath);
        }
        if (LibNotFound) {
            pd_error(Py4pd, "[py4pd] Library '%s' not found!", ScriptFileName->s_name);
            pd_error(Py4pd,
                     "[py4pd] Please, make sure the library is in the search path of PureData");
            return -1;
        }
    }

    PyObject *pModule, *pFunc; // Create the variables of the python objects
    char pyGlobalFolder[MAXPDSTRING];
    snprintf(pyGlobalFolder, MAXPDSTRING, "%s/py4pd-env", Py4pd->Py4pdPath->s_name);

    // convert const char* to char*
    char *GlobalFolderChar = malloc(strlen(pyGlobalFolder) + 1);
    Py4pdUtils_Strlcpy(GlobalFolderChar, pyGlobalFolder, strlen(pyGlobalFolder) + 1);

    // sound file path
    char *pkgPathchar = malloc(strlen(Py4pd->PkgPath->s_name) + 1);
    Py4pdUtils_Strlcpy(pkgPathchar, Py4pd->PkgPath->s_name, strlen(Py4pd->PkgPath->s_name) + 1);

    t_py *PrevObj = NULL;
    int ThereIsPrevObj = 0;
    PyObject *MainModule = PyImport_ImportModule("pd");
    PyObject *PrevObjectCapsule;

    if (MainModule != NULL) {
        PrevObjectCapsule = PyDict_GetItemString(MainModule, "py4pd"); // borrowed reference
        if (PrevObjectCapsule != NULL) {
            PyObject *py4pd_capsule = PyObject_GetAttrString(MainModule, "py4pd");
            PrevObj = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
            ThereIsPrevObj = 1;
        } else {
            ThereIsPrevObj = 0;
        }
    }
    PyObject *objectCapsule = Py4pdUtils_AddPdObject(Py4pd);

    if (objectCapsule == NULL) {
        Py4pdUtils_PrintError(Py4pd);
        Py_XDECREF(MainModule);
        return -1;
    }
    pModule = PyImport_ImportModule(atom_gensym(Argv + 1)->s_name);
    if (pModule == NULL) {
        pd_error(Py4pd, "[Python] Failed to load script file %s", ScriptFileName->s_name);
        Py4pdUtils_PrintError(Py4pd);
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
        Py4pdUtils_PrintError(Py4pd);
        Py_XDECREF(pModule);
        Py_XDECREF(MainModule);
        return -1;
    }

    PyObject *pModuleDict = PyModule_GetDict(pModule);

    if (pModuleDict == NULL) {
        pd_error(Py4pd, "[Python] Failed to get script file dictionary");
        return -1;
    }

    PyObject *pFilenameObj = PyDict_GetItemString(pModuleDict, "__file__");
    if (!pFilenameObj) {
        pd_error(Py4pd, "[Python] Failed to get script file path");
        Py_DECREF(pModule);
        Py_XDECREF(MainModule);
        return -1;
    }
    // convert const char * to char *
    char *libraryFolder = malloc(strlen(PyUnicode_AsUTF8(pFilenameObj)) + 1);
    Py4pdUtils_Strlcpy(libraryFolder, PyUnicode_AsUTF8(pFilenameObj),
                       strlen(PyUnicode_AsUTF8(pFilenameObj)) + 1);

    Py4pd->LibraryFolder = gensym(Py4pdUtils_GetFolderName(libraryFolder));
    free(libraryFolder);

    if (pModule == NULL) {
        Py4pdUtils_PrintError(Py4pd);
        Py_XDECREF(pModule);
        Py_XDECREF(MainModule);
        return -1;
    }

    // check if module has the function Py4pdLoadObjects or
    // "sScriptFilename + setup"
    char *setupFuncName = malloc(strlen(ScriptFileName->s_name) + 7);
    snprintf(setupFuncName, strlen(ScriptFileName->s_name) + 7, "%s_setup", ScriptFileName->s_name);
    PyObject *pFuncName = PyUnicode_FromString(setupFuncName);
    t_symbol *pFuncNameSymbol = gensym(setupFuncName);
    pFunc = PyObject_GetAttr(pModule, pFuncName);
    if (pFunc == NULL) {
        Py_DECREF(pFuncName);
        pFuncName = PyUnicode_FromString("py4pdLoadObjects");
        Py_XDECREF(pFunc);
        pFunc = PyObject_GetAttr(pModule, pFuncName);
        if (pFunc == NULL) {
            pd_error(Py4pd, "[Python] Failed to load function %s", setupFuncName);
            Py4pdUtils_PrintError(Py4pd);
            Py_XDECREF(pModule);
            free(setupFuncName);
            return -1;
        }
        pFuncNameSymbol = gensym("Py4pdLoadObjects");
    }
    free(setupFuncName);

    if (pFunc && PyCallable_Check(pFunc)) {
        if (objectCapsule == NULL) {
            pd_error(Py4pd, "[Python] Failed to add object to Python, capsule "
                            "not found.");
            return -1;
        }
        PyObject *pValue = PyObject_CallNoArgs(pFunc); // Call the function

        if (ThereIsPrevObj && pValue != NULL) {
            objectCapsule = Py4pdUtils_AddPdObject(PrevObj);
            if (objectCapsule == NULL) {
                pd_error(Py4pd, "[Python] Failed to add object to Python");
                return -1;
            }
        }

        if (pValue == NULL) {
            Py4pdUtils_PrintError(Py4pd);
            Py_XDECREF(MainModule);
            Py_XDECREF(pModule);
            Py_XDECREF(pModuleReloaded);
            Py_XDECREF(pFunc);
            Py_XDECREF(pValue);
            return -1;
        }
        // odd code, but solve the bug
        if (ThereIsPrevObj && pValue != NULL) {
            objectCapsule = Py4pdUtils_AddPdObject(PrevObj);
            if (objectCapsule == NULL) {
                pd_error(Py4pd, "[Python] Failed to add object to Python");
                return -1;
            }
        }
        Py4pd->pModule = pModule;
        Py4pd->pFunction = pFunc;
        Py4pd->pScriptName = ScriptFileName;
        Py4pd->pFuncName = pFuncNameSymbol;
        Py4pd->FuncCalled = 1;
        Py4pd->IsLib = 1;
        Py_XDECREF(MainModule);
        Py_XDECREF(pModuleReloaded);
        Py_XDECREF(pValue);
        logpost(Py4pd, 3, "[py4pd] Library %s loaded!", ScriptFileName->s_name);
    } else {
        Py4pd->FuncCalled = 1; // set the flag to 0 because it crash Pd if
        Py4pdUtils_PrintError(Py4pd);
        Py_XDECREF(pModule);
        Py_XDECREF(MainModule);
        Py_XDECREF(pFunc);
    }
    return 0;
}

// ========================================
/*
 * @brief Struct to pass args to the new thread
 */

struct pipInstallArgs {
    t_py *x;
    t_symbol *pipPackage;
};

// ========================================
/**
 * @brief This function call pip install from a new thread
 * @param pointer to the struct pipInstallArgs
 * @return void
 */
static void *Py4pd_PipInstallRequirementsDetach(void *Args) {
    struct pipInstallArgs *args = (struct pipInstallArgs *)Args;
    t_py *x = args->x;
    char const *pipPackage = args->pipPackage->s_name;
    t_symbol *pipTarget;
    if (x->PipGlobalInstall) {
        pipTarget = x->Py4pdPath;
    } else {
        pipTarget = x->PdPatchPath;
    }

    // loop for all pipTarget and check if there is some space char, if yes, get
    // error
    for (int i = 0; i < (int)strlen(pipTarget->s_name); i++) {
        if (isspace(pipTarget->s_name[i])) {
            pd_error(x, "[py4pd] Spaces are not supported in the path yet, try "
                        "to remove it");
            // outlet_float(x->mainOut, 0);
            // return NULL;
        }
    }

#ifdef __linux__
    size_t commandSize =
        snprintf(NULL, 0,
                 "python%d.%d -m pip install --target "
                 "'%s/py4pd-env' -r %s --upgrade",
                 PY_MAJOR_VERSION, PY_MINOR_VERSION, pipTarget->s_name, pipPackage) +
        1;
#elif defined _WIN32
    size_t commandSize =
        snprintf(NULL, 0,
                 "py -%d.%d -m pip install --target "
                 "\"%s/py4pd-env\" -r %s --upgrade",
                 PY_MAJOR_VERSION, PY_MINOR_VERSION, pipTarget->s_name, pipPackage) +
        1;

#elif defined(__APPLE__) || defined(__MACH__)
    size_t commandSize =
        snprintf(NULL, 0,
                 "/usr/local/bin/python%d.%d -m pip install --target "
                 "'%s/py4pd-env' -r %s --upgrade",
                 PY_MAJOR_VERSION, PY_MINOR_VERSION, pipTarget->s_name, pipPackage) +
        1;
#endif
    char *COMMAND = malloc(commandSize);
#ifdef __linux__
    snprintf(COMMAND, commandSize,
             "python%d.%d -m pip install --target '%s/py4pd-env' -r %s --upgrade", PY_MAJOR_VERSION,
             PY_MINOR_VERSION, pipTarget->s_name, pipPackage);

#elif defined __WIN64
    snprintf(COMMAND, commandSize,
             "py -%d.%d -m pip install --target \"%s/py4pd-env\" "
             "-r %s --upgrade",
             PY_MAJOR_VERSION, PY_MINOR_VERSION, pipTarget->s_name, pipPackage);

#elif defined(__APPLE__) || defined(__MACH__)
    snprintf(COMMAND, commandSize,
             "/usr/local/bin/python%d.%d -m pip install --target "
             "-r '%s/py4pd-env' %s --upgrade",
             PY_MAJOR_VERSION, PY_MINOR_VERSION, pipTarget->s_name, pipPackage);
#endif
    pd_error(NULL,
             "Installing %s in the background. Please DO NOT close PureData until you receive a "
             "complete message... Requirements can take a while!",
             pipPackage);
    int result = Py4pdUtils_ExecuteSystemCommand(COMMAND, 0);
    if (result != 0) {
        pd_error(NULL, "The instalation failed with error code %d", result);
        free(COMMAND);
        return NULL;
    } else {
        pd_error(NULL, "The installation has been completed.\nPlease, RESTART "
                       "PureData!");
        free(COMMAND);
        return NULL;
    }
    return NULL;
}
// ========================================
/**
 * @brief This function call pip install from a new thread
 * @param pointer to the struct pipInstallArgs
 * @return void
 */
static void *Py4pd_PipInstallDetach(void *Args) {
    struct pipInstallArgs *args = (struct pipInstallArgs *)Args;
    t_py *x = args->x;
    char const *pipPackage = args->pipPackage->s_name;
    t_symbol *pipTarget;
    if (x->PipGlobalInstall) {
        // add resources to the global path
        pipTarget = x->Py4pdPath;
    } else {
        pipTarget = x->PdPatchPath;
    }

    // loop for all pipTarget and check if there is some space char, if yes, get
    // error
    for (int i = 0; i < (int)strlen(pipTarget->s_name); i++) {
        if (isspace(pipTarget->s_name[i])) {
            pd_error(x, "[py4pd] Spaces are not supported in the path yet, try "
                        "to remove it");
            // outlet_float(x->mainOut, 0);
            // return NULL;
        }
    }

#ifdef __linux__
    size_t commandSize =
        snprintf(NULL, 0,
                 "python%d.%d -m pip install --target "
                 "'%s/py4pd-env' %s --upgrade",
                 PY_MAJOR_VERSION, PY_MINOR_VERSION, pipTarget->s_name, pipPackage) +
        1;
#elif defined _WIN32
    size_t commandSize =
        snprintf(NULL, 0,
                 "py -%d.%d -m pip install --target "
                 "\"%s/py4pd-env\" %s --upgrade",
                 PY_MAJOR_VERSION, PY_MINOR_VERSION, pipTarget->s_name, pipPackage) +
        1;

#elif defined(__APPLE__) || defined(__MACH__)
    size_t commandSize =
        snprintf(NULL, 0,
                 "/usr/local/bin/python%d.%d -m pip install --target "
                 "'%s/py4pd-env' %s --upgrade",
                 PY_MAJOR_VERSION, PY_MINOR_VERSION, pipTarget->s_name, pipPackage) +
        1;
#endif
    char *COMMAND = malloc(commandSize);
#ifdef __linux__
    snprintf(COMMAND, commandSize,
             "python%d.%d -m pip install --target '%s/py4pd-env' %s --upgrade", PY_MAJOR_VERSION,
             PY_MINOR_VERSION, pipTarget->s_name, pipPackage);
    post("Folder is %s", pipTarget->s_name);

#elif defined _WIN32
    snprintf(COMMAND, commandSize,
             "py -%d.%d -m pip install --target \"%s/py4pd-env\" "
             "%s --upgrade",
             PY_MAJOR_VERSION, PY_MINOR_VERSION, pipTarget->s_name, pipPackage);

#elif defined(__APPLE__) || defined(__MACH__)
    snprintf(COMMAND, commandSize,
             "/usr/local/bin/python%d.%d -m pip install --target "
             "'%s/py4pd-env' %s --upgrade",
             PY_MAJOR_VERSION, PY_MINOR_VERSION, pipTarget->s_name, pipPackage);
#endif
    pd_error(NULL,
             "Installing %s in the background. Please DO NOT close PureData until you receive a "
             "complete message... This can take a while!",
             pipPackage);
    int result = Py4pdUtils_ExecuteSystemCommand(COMMAND, 0);
    if (result != 0) {
        pd_error(NULL, "The instalation failed with error code %d", result);
        free(COMMAND);
        outlet_float(x->MainOut, 0);
        return NULL;
    } else {
        pd_error(NULL, "The installation has been completed.\nPlease, RESTART "
                       "PureData!");
        free(COMMAND);
        outlet_float(x->MainOut, 1);
        return NULL;
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
void Py4pd_Pip(t_py *Py4pd, t_symbol *S, int Argc, t_atom *Argv) {
    (void)S;

    // check if first argv is equal to install
    if (Argc < 1) {
        pd_error(Py4pd, "[py4pd] Usage: You need to specify the method for pip: "
                        "install or target.");
        return;
    }

    if (atom_getsymbolarg(0, Argc, Argv) == gensym("install")) {
        if (atom_getsymbolarg(1, Argc, Argv) == gensym("-r")) {
            Py_BEGIN_ALLOW_THREADS struct pipInstallArgs pipArgs;
            pipArgs.x = Py4pd;
            pipArgs.pipPackage = atom_getsymbolarg(2, Argc, Argv);
            pthread_t threadId;
            pthread_create(&threadId, NULL, Py4pd_PipInstallRequirementsDetach, &pipArgs);
            pthread_detach(threadId);
            Py_END_ALLOW_THREADS return;
        }

        for (int j = 1; j < Argc; j++) {
            if (Argv[j].a_type == A_SYMBOL) {
                struct pipInstallArgs pipArgs;
                pipArgs.x = Py4pd;
                pipArgs.pipPackage = atom_getsymbolarg(j, Argc, Argv);
                pthread_t threadId;
                pthread_create(&threadId, NULL, Py4pd_PipInstallDetach, &pipArgs);
                pthread_detach(threadId);
            } else {
                pd_error(Py4pd, "[py4pd] The package name must be a symbol");
                return;
            }
        }
        return;
    } else if (atom_getsymbolarg(0, Argc, Argv) == gensym("target")) {
        t_symbol *folder = atom_getsymbolarg(1, Argc, Argv);
        if (folder == gensym("global")) {
            Py4pd->PipGlobalInstall = 1;
            post("[py4pd] pip target set to global.");
        } else if (folder == gensym("local")) {
            Py4pd->PipGlobalInstall = 0;
            post("[py4pd] pip target set to local.");
        }
    }
    // TODO: add pip install requirements.txt
}

// ============================================
/**
 * @brief Function to warning about deprecated methods
 * @param x pointer to the object
 * @param s symbol
 * @param argc number of arguments
 * @param argv array of arguments
 * @return void
 */
static void Py4pd_Deprecated(t_py *Py4pd, t_symbol *S, int Argc, t_atom *Argv) {
    (void)Argc;
    (void)Argv;

    if (S == gensym("pipinstall")) {
        pd_error(Py4pd, "[py4pd] The [pipinstall] method is deprecated, use [pip "
                        "install] instead.");
        return;
    }
}

// ============================================
/**
 * @brief it prints the version of py4pd and python
 * @param x pointer to the object
 * @return void
 */
static void Py4pd_PrintPy4pdVersion(t_py *x) {
    int Major, Minor, Micro;
    Major = PY4PD_MAJOR_VERSION;
    Minor = PY4PD_MINOR_VERSION;
    Micro = PY4PD_MICRO_VERSION;
    t_atom Py4pdVersion[3];
    SETFLOAT(&Py4pdVersion[0], Major);
    SETFLOAT(&Py4pdVersion[1], Minor);
    SETFLOAT(&Py4pdVersion[2], Micro);
    outlet_anything(x->MainOut, gensym("py4pd"), 3, Py4pdVersion);

    t_atom PythonVersion[3];
    Major = PY_MAJOR_VERSION;
    Minor = PY_MINOR_VERSION;
    Micro = PY_MICRO_VERSION;
    SETFLOAT(&PythonVersion[0], Major);
    SETFLOAT(&PythonVersion[1], Minor);
    SETFLOAT(&PythonVersion[2], Micro);
    outlet_anything(x->MainOut, gensym("python"), 3, PythonVersion);
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

static void Py4pd_SetPy4pdHomePath(t_py *Py4pd, t_symbol *S, int Argc, t_atom *Argv) {
    (void)S;
    if (Argc < 1) {
        post("[py4pd] The home path is: %s", Py4pd->PdPatchPath->s_name);
    } else {
        Py4pd->PdPatchPath = atom_getsymbol(Argv);
        post("[py4pd] The home path set to: %s", Py4pd->PdPatchPath->s_name);
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
        post("[py4pd] The packages path is: %s", x->PkgPath->s_name);
        return;
    } else {
        if (argc == 1) {
            if (argv[0].a_type == A_SYMBOL) {
                t_symbol *path = atom_getsymbol(argv);
                // It checks relative path
                if (path->s_name[0] == '.' && path->s_name[1] == '/') {
                    char *new_path =
                        malloc(strlen(x->PdPatchPath->s_name) + strlen(path->s_name) + 1);
                    Py4pdUtils_Strlcpy(new_path, x->PdPatchPath->s_name,
                                       strlen(x->PdPatchPath->s_name) + 1);
                    Py4pdUtils_Strlcat(new_path, path->s_name, strlen(new_path) + 1);
                    post("[py4pd] Packages path set to: %s", new_path);
                    x->PkgPath = gensym(new_path);
                    free(new_path);
                } else {
                    x->PkgPath = atom_getsymbol(argv);
                    post("[py4pd] Packages path set to: %s", x->PkgPath->s_name);
                }
                char cfgFile[MAXPDSTRING];
                char const *py4pdDir = x->Py4pdPath->s_name;
                snprintf(cfgFile, MAXPDSTRING, "%s/py4pd.cfg", py4pdDir);
                FILE *file = fopen(cfgFile, "w");
                if (x->EditorName != NULL) {
                    fprintf(file, "editor = %s\n", x->EditorName->s_name);
                }
                fprintf(file, "packages = %s", x->PkgPath->s_name);
                fclose(file);
            } else {
                pd_error(x, "[py4pd] The packages path must be a string");
                return;
            }
            // check if path exists and is valid
            if (access(x->PkgPath->s_name, F_OK) == -1) {
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
static void Py4pd_PrintModuleFunctions(t_py *x, t_symbol *s, int argc, t_atom *argv) {
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
    if (x->FuncCalled == 0) {
        pd_error(x, "[py4pd] To see the documentaion you need to set the "
                    "function first!");
        return;
    }
    if (x->pFunction &&
        PyCallable_Check(x->pFunction)) { // Check if the function exists and is callable
        PyObject *pDoc = PyObject_GetAttrString(x->pFunction,
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
    Py4pdUtils_ExecuteSystemCommand(command, 1);
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
static void Py4pd_SetEditor(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    // check if argv[0] is a symbol and it is diferent from "float"
    if (argc != 0 && argv[0].a_type == A_SYMBOL &&
        strcmp(argv[0].a_w.w_symbol->s_name, "float") != 0) {
        x->EditorName = atom_getsymbol(argv + 0);
        post("[py4pd] Editor set to: %s", x->EditorName->s_name);
        char cfgFile[MAXPDSTRING];
        char const *py4pdDir = x->Py4pdPath->s_name;
        snprintf(cfgFile, MAXPDSTRING, "%s/py4pd.cfg", py4pdDir);
        FILE *file = fopen(cfgFile, "w");
        fprintf(file, "editor = %s\n", x->EditorName->s_name);
        fprintf(file, "packages = %s", x->PkgPath->s_name);
        fclose(file);
        return;
    }
    if (x->FuncCalled == 0) { // if the set method was not called, then we
        pd_error(x, "[py4pd] To open the editor you need to set the "
                    "function first!");
        return;
    }

    PyCodeObject *code = (PyCodeObject *)PyFunction_GetCode(x->pFunction);
    int line = PyCode_Addr2Line(code, 0);
    char command[MAXPDSTRING];
    Py4pdUtils_GetEditorCommand(x, command, line);
    Py4pdUtils_ExecuteSystemCommand(command, 0);
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
static void Py4pd_ReloadPy4pdFunction(t_py *x) {
    PyObject *pName, *pFunc, *pModule, *pReload;
    if (x->FuncCalled == 0) { // if the set method was not called, then we
                              // can not run the function :)
        pd_error(x, "To reload the script you need to set the function first!");
        return;
    }
    pFunc = x->pFunction;

    // reload the module
    pName = PyUnicode_DecodeFSDefault(x->pScriptName->s_name); // Name of script file
    pModule = PyImport_Import(pName);
    if (pModule == NULL) {
        pd_error(x, "Error importing the module!");
        x->FuncCalled = 0;
        Py_DECREF(pFunc);
        Py_DECREF(pName);
        return;
    }

    pReload = PyImport_ReloadModule(pModule);
    if (pReload == NULL) {
        pd_error(x, "Error reloading the module!");
        x->FuncCalled = 0;
        Py_DECREF(pFunc);
        Py_DECREF(pModule);
        return;
    } else {
        pFunc =
            PyObject_GetAttrString(pModule,
                                   x->pFuncName->s_name); // Function name inside the script file
        Py_DECREF(pName);
        Py_DECREF(pReload);
        if (pFunc && PyCallable_Check(pFunc)) { // Check if the function exists and is callable
            x->pFunction = pFunc;
            x->FuncCalled = 1;
            PyCodeObject *code = (PyCodeObject *)PyFunction_GetCode(x->pFunction);
            int argCount = code->co_argcount;
            x->pArgsCount = argCount;
            post("The module was reloaded!");
            return;
        } else {
            pd_error(x, "Error reloading the module!");
            x->FuncCalled = 0;
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
        pd_error(x, "[py4pd] 'set' message needs two arguments! The 'Script Name' "
                    "and the 'Function Name'!");
        return;
    }

    t_symbol *ScriptFilename = atom_gensym(argv + 0);
    t_symbol *pFuncNameSymbol = atom_gensym(argv + 1);
    if (ScriptFilename == NULL) {
        pd_error(x, "[py4pd] Script Name is NULL!");
        return;
    }

    x->NumpyImported = Py4pdUtils_CheckNumpyInstall(x);
    if (!x->NumpyImported) {
        return;
    }

    if (x->FuncCalled == 1) {
        int function_is_equal = strcmp(pFuncNameSymbol->s_name,
                                       x->pFuncName->s_name); // if string is equal strcmp returns 0
        if (function_is_equal == 0) {
            pd_error(x, "[py4pd] The function was already set!");
            return;
        } else {
            Py_XDECREF(x->pFunction);
            x->FuncCalled = 0;
        }
    }

    // Check if there is extension (not to use it)
    char *extension = strrchr(ScriptFilename->s_name, '.');
    if (extension != NULL) {
        pd_error(x, "[py4pd] Don't use extensions in the script file name!");
        Py_XDECREF(x->pFunction);
        return;
    }

    // check if script file exists
    char ScriptFilePath[MAXPDSTRING];
    snprintf(ScriptFilePath, MAXPDSTRING, "%s/%s.py", x->PdPatchPath->s_name,
             ScriptFilename->s_name);

    PyObject *pModule, *pFunc; // Create the variables of the python objects

    // odd code, but solve the bug
    t_py *PrevObj;
    int ThereIsPrevObj = 0;
    PyObject *MainModule;
    MainModule = PyImport_ImportModule("pd");
    PyObject *OldObjCapsule;
    if (MainModule != NULL) {
        OldObjCapsule = PyDict_GetItemString(MainModule, "py4pd");
        if (OldObjCapsule != NULL) {
            PyObject *py4pd_capsule = PyObject_GetAttrString(MainModule, "py4pd");
            PrevObj = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
            ThereIsPrevObj = 1;
        } else {
            ThereIsPrevObj = 0;
        }
    } else {
        Py4pdUtils_PrintError(x);
        return;
    }

    PyObject *ObjCapsule = Py4pdUtils_AddPdObject(x);

    // Import module
    pModule = PyImport_ImportModule(ScriptFilename->s_name);

    if (pModule == NULL) {
        Py4pdUtils_PrintError(x);
        Py_XDECREF(pModule);
        return;
    }
    pFunc = PyObject_GetAttrString(pModule, pFuncNameSymbol->s_name);
    if (pFunc && PyCallable_Check(pFunc)) {
        PyCodeObject *Code = (PyCodeObject *)PyFunction_GetCode(pFunc);

        if (ThereIsPrevObj == 1 && pFunc != NULL) {
            ObjCapsule = Py4pdUtils_AddPdObject(PrevObj);
            if (ObjCapsule == NULL) {
                pd_error(x, "[Python] Failed to add object to Python");
                return;
            }
        }

        if (Code->co_flags & CO_VARARGS) {
            pd_error(x, "[py4pd] The '%s' function has variable arguments (*args)!",
                     pFuncNameSymbol->s_name);
            Py_XDECREF(pFunc);
            Py_XDECREF(pModule);
            return;
        } else if (Code->co_flags & CO_VARKEYWORDS) {
            pd_error(x,
                     "[py4pd] The '%s' function has variable keyword arguments "
                     "(**kwargs)!",
                     pFuncNameSymbol->s_name);
            Py_XDECREF(pFunc);
            Py_XDECREF(pModule);
            return;
        }
        x->pArgsCount = Code->co_argcount;
        if (x->IsLib == 0) {
            post("[py4pd] The '%s' function has %d arguments!", pFuncNameSymbol->s_name,
                 x->pArgsCount);
        }
        x->pModule = pModule;
        x->pFunction = pFunc;
        x->pScriptName = ScriptFilename;
        x->pFuncName = pFuncNameSymbol;
        x->FuncCalled = 1;
    } else {
        pd_error(x, "[py4pd] Function %s not loaded!", pFuncNameSymbol->s_name);
        x->FuncCalled = 0;
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

        PyObject **lists;
        if (OpenList_count == 0) {
            lists = (PyObject **)malloc(sizeof(PyObject *));
        } else {
            lists = (PyObject **)malloc(OpenList_count * sizeof(PyObject *));
        }

        ArgsTuple = Py4pdUtils_ConvertToPy(lists, argc,
                                           argv); // convert the arguments to python
        int argCount = PyTuple_Size(ArgsTuple);   // get the number of arguments
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
 * @return It will return nothing but will run the Python function
 */
static void Py4pd_ExecuteFunction(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    if (x->FuncCalled == 0) {
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
 * @param x is the py4pd object
 * @param f is the status of the numpy array support
 * @return It will return void.
 */
void Py4pd_SetPythonPointersUsage(t_py *x, t_floatarg f) {
    int usepointers = (int)f;
    if (usepointers == 1) {
        post("[py4pd] Python Pointers enabled.");
        x->OutPyPointer = 1;
    } else if (usepointers == 0) {
        x->OutPyPointer = 0;
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
static void *Py4pd_Py4pdNew(t_symbol *s, int argc, t_atom *argv) {
    LOG("Py4pd_Py4pdNew");
    int i;
    t_py *x;
    int LibMode = 0;
    int ObjMode = 1;
    t_symbol *ScriptName = NULL;

    // Get what will be the type of the object
    for (i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            t_symbol *py4pdArgs = atom_getsymbolarg(i, argc, argv);
            if (py4pdArgs == gensym("-library") || py4pdArgs == gensym("-lib")) {
                LibMode = 1;
                ObjMode = 0;
                ScriptName = atom_getsymbolarg(i + 1, argc, argv);
            }
        }
    }

    if (ObjMode == 1) {
        x = (t_py *)pd_new(Py4pdObjClass); // create a new py4pd object
        x->Canvas = canvas_getcurrent();
        t_canvas *c = x->Canvas;
        t_symbol *patchDir = canvas_getdir(c);
        size_t strLen = strlen(patchDir->s_name);

        if (strLen > 0 && (patchDir->s_name)[strLen - 1] != '/') {
            char *new_path = malloc(strLen + 2);
            Py4pdUtils_Strlcpy(new_path, patchDir->s_name, strLen + 1);
            Py4pdUtils_Strlcat(new_path, "/", strLen + 2);
            patchDir = gensym(new_path);
            free(new_path);
        }
        x->Zoom = (int)x->Canvas->gl_zoom;
        x->AudioIn = 0;
        x->AudioOut = 0;
        x->VisMode = 0;
        x->EditorName = NULL;
        x->PipGlobalInstall = 1;
        x->PyObject = 0;
        x->VectorSize = 0;
        Py4pdUtils_ParseArguments(x, c, argc, argv);
        x->PdPatchPath = patchDir;
        x->PkgPath = patchDir;
        x->pArgsCount = 0;
        Py4pdUtils_SetObjConfig(x);
        if (objCount == 0) {
            Py4pdUtils_AddPathsToPythonPath(x);
        }
        if (argc > 1) {
            Py4pd_SetFunction(x, s, argc, argv);
            x->NumpyImported = 1;
        }
        objCount++;
        return (x);
    } else if (LibMode == 1) {
        x = (t_py *)pd_new(Py4pdLibClass);
        x->Canvas = canvas_getcurrent();
        t_canvas *c = x->Canvas;
        x->Zoom = (int)x->Canvas->gl_zoom;
        t_symbol *PatchDir = canvas_getdir(c);
        if (PatchDir->s_name[strlen(PatchDir->s_name) - 1] != '/') {
            char *NewPath = malloc(strlen(PatchDir->s_name) + 2);
            Py4pdUtils_Strlcpy(NewPath, PatchDir->s_name, strlen(PatchDir->s_name) + 1);
            Py4pdUtils_Strlcat(NewPath, "/\0", strlen(NewPath) + 1);
            PatchDir = gensym(NewPath);
            free(NewPath);
        }

        x->PdPatchPath = PatchDir;
        x->PkgPath = PatchDir;
        x->VisMode = 0;
        Py4pdUtils_SetObjConfig(x);
        if (objCount == 0) {
            Py4pdUtils_AddPathsToPythonPath(x);
        }
        x->NumpyImported = Py4pdUtils_CheckNumpyInstall(x);
        if (!x->NumpyImported) {
            return NULL;
        }
        int libraryLoaded = Py4pd_LibraryLoad(x, argc, argv);
        if (libraryLoaded == -1) {
            return NULL;
        }
        x->pScriptName = ScriptName;
        objCount++;
        return (x);
    } else {
        pd_error(NULL, "Error in py4pdNew, you can not use more than one flag at "
                       "the same time.");
        return NULL;
    }
}

// ====================================================
/**
 * @brief Setup the py4pd object, pd call this
 */
void py4pd_setup(void) {
    int Major, Minor, Micro;
    sys_getversion(&Major, &Minor, &Micro);

    if (Major < 0 && Minor < 54) {
        pd_error(NULL, "[py4pd] py4pd requires Pd version 0.54 or later.");
        return;
    }

    Py4pdObjClass = class_new(gensym("py4pd"), (t_newmethod)Py4pd_Py4pdNew,
                              (t_method)Py4pdUtils_FreeObj, sizeof(t_py), 0, A_GIMME, 0);

    Py4pdLibClass =
        class_new(gensym("py4pd"), (t_newmethod)Py4pd_Py4pdNew, (t_method)Py4pdUtils_FreeObj,
                  sizeof(t_py), CLASS_NOINLET, A_GIMME, 0);

    class_addmethod(Py4pdObjClass, (t_method)Py4pd_SetPy4pdHomePath, gensym("home"), A_GIMME,
                    0); // set home path
    class_addmethod(Py4pdObjClass, (t_method)Py4pd_SetPackages, gensym("packages"), A_GIMME,
                    0); // set packages path
    class_addmethod(Py4pdObjClass, (t_method)Py4pd_Pip, gensym("pip"), A_GIMME,
                    0); // pip install
    class_addmethod(Py4pdObjClass, (t_method)Py4pd_SetPythonPointersUsage, gensym("pointers"),
                    A_FLOAT,
                    0); // set home path
    class_addmethod(Py4pdObjClass, (t_method)Py4pd_ReloadPy4pdFunction, gensym("reload"), 0,
                    0); // reload python script

    // DEPRECATED
    class_addmethod(Py4pdObjClass, (t_method)Py4pd_Deprecated, gensym("pipinstall"), A_GIMME,
                    0); // run function

    // Object INFO
    class_addmethod(Py4pdObjClass, (t_method)Py4pd_PrintPy4pdVersion, gensym("version"), 0,
                    0); // show version
    class_addmethod(Py4pdObjClass, (t_method)Py4pd_SetEditor, gensym("editor"), A_GIMME,
                    0); // open code
    class_addmethod(Py4pdObjClass, (t_method)Py4pd_OpenScript, gensym("open"), A_GIMME, 0);
    class_addmethod(Py4pdObjClass, (t_method)Py4pd_OpenScript, gensym("create"), A_GIMME, 0);
    class_addmethod(Py4pdObjClass, (t_method)Py4pd_SetEditor, gensym("click"), 0,
                    0); // when click open editor
    class_addmethod(Py4pdLibClass, (t_method)Py4pd_SetEditor, gensym("click"), 0,
                    0); // when click open editor

    // User
    class_addmethod(Py4pdObjClass, (t_method)Py4pd_PrintDocs, gensym("doc"), 0,
                    0); // open documentation
    class_addmethod(Py4pdObjClass, (t_method)Py4pd_ExecuteFunction, gensym("run"), A_GIMME,
                    0); // run function
    class_addmethod(Py4pdObjClass, (t_method)Py4pd_SetFunction, gensym("set"), A_GIMME,
                    0); // set function to be called
    class_addmethod(Py4pdObjClass, (t_method)Py4pd_PrintModuleFunctions, gensym("functions"),
                    A_GIMME, 0);

    if (!Py_IsInitialized()) {
        objCount = 0;
        post("");
        post("[py4pd] by Charles K. Neimog");
        post("[py4pd] Version %d.%d.%d", PY4PD_MAJOR_VERSION, PY4PD_MINOR_VERSION,
             PY4PD_MICRO_VERSION);
        post("[py4pd] Python version %d.%d.%d", PY_MAJOR_VERSION, PY_MINOR_VERSION,
             PY_MICRO_VERSION);
        post("");
        PyImport_AppendInittab("pd", PyInit_pd);
        Py_Initialize();
        Py4pdUtils_ConfigurePythonPaths();
    }
}

#ifdef __WIN64
__declspec(dllexport) void py4pd_setup(void);
#endif
