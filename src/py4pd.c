#include "py4pd.h"
#include "m_pd.h"

#define NPY_NO_DEPRECATED_API NPY_1_25_API_VERSION
#include <numpy/arrayobject.h>

// chat gpt chat to great name functions conventions https://chat.openai.com/c/9000f606-6f15-4fd3-be89-1577c2087624

// ============================================
t_class *py4pd_class;          // For for normal objects, almost unused
t_class *py4pd_classLibrary;   // For libraries
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

static int Py4pd_LibraryLoad(t_py *x, int argc, t_atom *argv){

    if (argc > 2) {
        pd_error(x, "[py4pd] Too many arguments! Usage: py4pd -lib <library_name>");
        return -1;
    }

    int numpyImporter = _import_array(); // import numpy
    if (numpyImporter < 0) {
        pd_error(x, "[py4pd] Error importing numpy!");
        return -1;
    }


    t_symbol *script_file_name = atom_gensym(argv + 1);

    // check if script file exists
    char script_file_path[MAXPDSTRING];
    snprintf(script_file_path, MAXPDSTRING, "%s/%s.py", x->pdPatchFolder->s_name, script_file_name->s_name);

    char script_inside_py4pd_path[MAXPDSTRING];
    snprintf(script_inside_py4pd_path, MAXPDSTRING, 
             "%s/resources/scripts/%s.py", x->py4pdPath->s_name, script_file_name->s_name); // TODO: Change this to other name

    PyObject *sys_path = PySys_GetObject("path");
    if (access(script_file_path, F_OK) == -1 && access(script_inside_py4pd_path, F_OK) == -1) {
        Py_XDECREF(x->function);
        int libraryNotFound = 1; // search if it is possible to found library  in search path
        for (int i = 0; 1; i++){ 
            const char *pathelem = namelist_get(STUFF->st_searchpath, i);
            if (!pathelem){
                break;
            }
            char library_path[MAXPDSTRING];
            snprintf(library_path, MAXPDSTRING, "%s/%s/", pathelem, script_file_name->s_name); 
            /* The library folder must have the same name as the library file */
            if (access(library_path, F_OK) != -1) {
                libraryNotFound = 0;
                PyObject *library_path_py = PyUnicode_FromString(library_path);
                PyList_Insert(sys_path, 0, library_path_py);
            }
        }
        if (libraryNotFound){
            pd_error(x, "[py4pd] Library file '%s.py' not found!", script_file_name->s_name);
            return -1;
        }
    }

    PyObject *pModule, *pFunc;  // Create the variables of the python objects
    char *pyScriptsFolder = malloc(strlen(x->py4pdPath->s_name) + 40); // allocate extra space
    char *pyGlobalFolder = malloc(strlen(x->py4pdPath->s_name) + 40); // allocate extra space
    snprintf(pyScriptsFolder, strlen(x->py4pdPath->s_name) + 40, "%s/resources/scripts", x->py4pdPath->s_name);
    snprintf(pyGlobalFolder, strlen(x->py4pdPath->s_name) + 40, "%s/resources/py-modules", x->py4pdPath->s_name);

    // conver const char* to char*
    char *pkgPathchar = malloc(strlen(x->pkgPath->s_name) + 1);
    strcpy(pkgPathchar, x->pkgPath->s_name);

    Py4pdUtils_CheckPkgNameConflict(x, pkgPathchar, script_file_name);
    Py4pdUtils_CheckPkgNameConflict(x, pyGlobalFolder, script_file_name);

    PyObject *home_path = PyUnicode_FromString(x->pdPatchFolder->s_name);  // Place where script file will probably be
    PyObject *site_package = PyUnicode_FromString(x->pkgPath->s_name);  // Place where the packages will be
    PyObject *globalPackages = PyUnicode_FromString(pyGlobalFolder);  // Place where the py4pd scripts will be
    PyObject *py4pdScripts = PyUnicode_FromString(pyScriptsFolder);  // Place where the py4pd scripts will be
    
    PyList_Insert(sys_path, 0, home_path);
    PyList_Insert(sys_path, 0, site_package);
    PyList_Insert(sys_path, 0, py4pdScripts);
    PyList_Insert(sys_path, 0, globalPackages);
    Py_DECREF(home_path);
    Py_DECREF(site_package);
    Py_DECREF(py4pdScripts);
    Py_DECREF(globalPackages);
    free(pyScriptsFolder);
    free(pyGlobalFolder);
    free(pkgPathchar);

    // odd code, but solve the bug
    t_py *prev_obj = NULL;
    int prev_obj_exists = 0;
    PyObject *MainModule = PyImport_ImportModule("pd");
    PyObject *oldObjectCapsule;

    if (MainModule != NULL) {
        oldObjectCapsule = PyDict_GetItemString(MainModule, "py4pd"); // borrowed reference
        if (oldObjectCapsule != NULL) {
            PyObject *py4pd_capsule = PyObject_GetAttrString(MainModule, "py4pd");
            prev_obj = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
            prev_obj_exists = 1;
        }
        else {
            prev_obj_exists = 0;
        }
    }
    PyObject *objectCapsule = Py4pdUtils_AddPdObject(x);

    if (objectCapsule == NULL){
        pd_error(x, "[Python] Failed to add object to Python");
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyObject *ptype_str = PyObject_Str(ptype);
        PyObject *pvalue_str = PyObject_Str(pvalue);
        PyObject *ptraceback_str = PyObject_Str(ptraceback);
        const char *ptype_c = PyUnicode_AsUTF8(ptype_str);
        const char *pvalue_c = PyUnicode_AsUTF8(pvalue_str);
        const char *ptraceback_c = PyUnicode_AsUTF8(ptraceback_str);
        pd_error(x, "[Python] %s: %s\n%s", ptype_c, pvalue_c, ptraceback_c);
        return -1;
    }
    pModule = PyImport_ImportModule(script_file_name->s_name);  // Import the script file
    if (pModule == NULL){
        pd_error(x, "[Python] Failed to load script file %s", script_file_name->s_name);
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[py4pd] Call failed: %s", PyUnicode_AsUTF8(pstr));
        Py_XDECREF(pstr);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        Py_XDECREF(pModule);
        return -1;
    }
    
    // reload the module if it already exists
    PyObject *pModuleReloaded = PyImport_ReloadModule(pModule);
    if (pModuleReloaded != NULL){
        Py_DECREF(pModule);
        pModule = pModuleReloaded;
    }
    else{
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[py4pd] Call failed: %s", PyUnicode_AsUTF8(pstr));
        Py_XDECREF(pstr);
        Py_XDECREF(pModule);
        return -1;
    }

    PyObject *pModuleDict = PyModule_GetDict(pModule);

    if (pModuleDict == NULL){
        pd_error(x, "[Python] Failed to get script file dictionary");
        return -1;
    }

    PyObject *pFilenameObj = PyDict_GetItemString(pModuleDict, "__file__");
    if (!pFilenameObj) {
        pd_error(x, "[Python] Failed to get script file path");
        Py_DECREF(pModule);
        return -1;
    }
    // convert const char * to char *
    char *libraryFolder = malloc(strlen(PyUnicode_AsUTF8(pFilenameObj)) + 1);
    strcpy(libraryFolder, PyUnicode_AsUTF8(pFilenameObj));
    x->libraryFolder = gensym(Py4pdUtils_GetFolderName(libraryFolder));
    free(libraryFolder);

    if (pModule == NULL) {
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[py4pd] Call failed: %s", PyUnicode_AsUTF8(pstr));
        Py_XDECREF(pstr);
        Py_XDECREF(pModule);
        return -1;
    }

    // check if module has the function Py4pdLoadObjects or "script_file_name + setup"
    char *setupFuncName = malloc(strlen(script_file_name->s_name) + 7);
    snprintf(setupFuncName, strlen(script_file_name->s_name) + 7, "%s_setup", script_file_name->s_name);
    PyObject *pFuncName = PyUnicode_FromString(setupFuncName);
    t_symbol *function_name = gensym(setupFuncName);
    pFunc = PyObject_GetAttr(pModule, pFuncName);
    if (pFunc == NULL){
        Py_DECREF(pFuncName);
        pFuncName = PyUnicode_FromString("py4pdLoadObjects");
        Py_XDECREF(pFunc);
        pFunc = PyObject_GetAttr(pModule, pFuncName);
        if (pFunc == NULL){
            pd_error(x, "[Python] Failed to load function %s", setupFuncName);
            PyObject *ptype, *pvalue, *ptraceback;
            PyErr_Fetch(&ptype, &pvalue, &ptraceback);
            PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
            PyObject *pstr = PyObject_Str(pvalue);
            pd_error(x, "[py4pd] Call failed: %s", PyUnicode_AsUTF8(pstr));
            Py_XDECREF(pstr);
            Py_XDECREF(ptype);
            Py_XDECREF(pvalue);
            Py_XDECREF(ptraceback);
            Py_XDECREF(pModule);
            return -1;
        }
        function_name = gensym("Py4pdLoadObjects");
    }
    free(setupFuncName);

    if (pFunc && PyCallable_Check(pFunc)) {  
        if (objectCapsule == NULL){
            pd_error(x, "[Python] Failed to add object to Python, capsule not found.");
            return -1;
        }
        PyObject *pValue = PyObject_CallNoArgs(pFunc);  // Call the function

        if (prev_obj_exists == 1 && pValue != NULL) {
            objectCapsule = Py4pdUtils_AddPdObject(prev_obj);
            if (objectCapsule == NULL){
                pd_error(x, "[Python] Failed to add object to Python");
                return -1;
            }
        }

        if (pValue == NULL) {
            PyObject *ptype, *pvalue, *ptraceback;
            PyErr_Fetch(&ptype, &pvalue, &ptraceback);
            PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
            PyObject *pstr = PyObject_Str(pvalue);
            pd_error(x, "[py4pd] Call failed: %s", PyUnicode_AsUTF8(pstr));
            Py_XDECREF(pstr);
            Py_XDECREF(ptype);
            Py_XDECREF(pvalue);
            Py_XDECREF(ptraceback);
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
            if (objectCapsule == NULL){
                pd_error(x, "[Python] Failed to add object to Python");
                return -1;
            }
        }
        x->module = pModule;
        x->function = pFunc;
        x->script_name = script_file_name;
        x->function_name = function_name;
        x->function_called = 1;
        x->py4pd_lib = 1;
        Py_XDECREF(MainModule);
        Py_XDECREF(pModuleReloaded);
        Py_XDECREF(pValue);
        logpost(x, 3, "[py4pd] Library %s loaded!", script_file_name->s_name);
    } 
    else {
        x->function_called = 1;  // set the flag to 0 because it crash Pd if
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[%s] %s.", script_file_name->s_name, PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        Py_XDECREF(pModule);
        Py_XDECREF(MainModule);
        Py_XDECREF(pFunc);
        PyErr_Clear();
    }
    return 0;
}


// ============================================
// ========= PY4PD METHODS FUNCTIONS ==========
// ============================================
static void Py4pd_PipInstall(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    const char *pipPackage;
    const char *localORglobal;

    PyObject *py4pdModule = PyImport_ImportModule("py4pd");
    if (py4pdModule == NULL) {
        pd_error(x, "[Python] pipInstall: py4pd module not found");
        return;
    }
    PyObject *pipInstallFunction = PyObject_GetAttrString(py4pdModule, "pipinstall");
    if (pipInstallFunction == NULL) {
        PyErr_SetString(PyExc_TypeError, "[Python] pd.pipInstall: pipinstall function not found");
        return;
    }
    PyObject *ObjFunction = x->function;
    x->function = pipInstallFunction;

    localORglobal = atom_getsymbolarg(0, argc, argv)->s_name;
    pipPackage = atom_getsymbolarg(1, argc, argv)->s_name;

    // the function is executed using pipinstall([localORglobal, pipPackage])
    PyObject *argsList = PyList_New(2);
    PyList_SetItem(argsList, 0, Py_BuildValue("s", localORglobal));
    PyList_SetItem(argsList, 1, Py_BuildValue("s", pipPackage));
    PyObject *argTuple = Py_BuildValue("(O)", argsList);
    
    PyObject *pipInstallResult = Py4pdUtils_RunPy(x, argTuple, NULL);
    if (pipInstallResult == NULL) {
        x->function = ObjFunction;
        Py_DECREF(argTuple);
        Py_DECREF(pipInstallResult);
        Py_DECREF(pipInstallFunction);
        Py_DECREF(py4pdModule);
        return;
    }
    x->function = ObjFunction;
    Py_DECREF(argTuple);
    Py_DECREF(pipInstallResult);
    Py_DECREF(pipInstallFunction);
    Py_DECREF(py4pdModule);
    sys_vgui("tk_messageBox -icon warning -type ok -title \"%s installed!\" -message \"%s installed! \nYou need to restart PureData!\"\n", pipPackage, pipPackage);
    return;
}

// ============================================
/**
 * @brief it prints the version of py4pd and python
 * @param x pointer to the object
 * @return void
*/
static void Py4pd_PrintPy4pdVersion(t_py *x){
    int major, minor, micro;
    major = PY4PD_MAJOR_VERSION;
    minor = PY4PD_MINOR_VERSION;
    micro = PY4PD_MICRO_VERSION;
    t_atom py4pdVersionArray[3];
    SETFLOAT(&py4pdVersionArray[0], major);
    SETFLOAT(&py4pdVersionArray[1], minor);
    SETFLOAT(&py4pdVersionArray[2], micro);
    outlet_anything(x->out1, gensym("py4pd"), 3, py4pdVersionArray);
    t_atom pythonVersionArray[3];
    major = PY_MAJOR_VERSION;
    minor = PY_MINOR_VERSION;
    micro = PY_MICRO_VERSION;
    SETFLOAT(&pythonVersionArray[0], major);
    SETFLOAT(&pythonVersionArray[1], minor);
    SETFLOAT(&pythonVersionArray[2], micro);
    outlet_anything(x->out1, gensym("python"), 3, pythonVersionArray);
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

static void Py4pd_SetPy4pdHomePath(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;  
    if (argc < 1) {
        post("[py4pd] The home path is: %s", x->pdPatchFolder->s_name);
    } else {
        x->pdPatchFolder = atom_getsymbol(argv);
        post("[py4pd] The home path set to: %s", x->pdPatchFolder->s_name);
    }
    return;
}

// ============================================
/**
 * @brief set the packages path to py4pd, if start with . then build the complete path
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
        if (argc < 2 && argc > 0) {
            if (argv[0].a_type == A_SYMBOL) {
                t_symbol *path = atom_getsymbol(argv);
                // It checks relative path
                if (path->s_name[0] == '.' && path->s_name[1] == '/') {
                    char *new_path = malloc(strlen(x->pdPatchFolder->s_name) +
                                            strlen(path->s_name) + 1);
                    strcpy(new_path, x->pdPatchFolder->s_name);
                    strcat(new_path, path->s_name + 1);
                    post("[py4pd] The packages path set to: %s", new_path);
                    x->pkgPath = gensym(new_path);
                    free(new_path);
                } else {
                    x->pkgPath = atom_getsymbol(argv);
                    post("[py4pd] The packages path set to: %s",
                         x->pkgPath->s_name);
                }
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
static void Py4pd_PrintModuleFunctions(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    (void)argc;
    (void)argv;

    PyObject *module_dict = PyModule_GetDict(x->module);
    Py_ssize_t pos = 0;
    PyObject *key, *value;
    
    post("[py4pd] Functions in module %s:", x->script_name->s_name);
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
    if (x->function_called == 0) { 
        pd_error(x, "[py4pd] To see the documentaion you need to set the function first!");
        return;
    }
    if (x->function && PyCallable_Check(x->function)) {  // Check if the function exists and is callable
        PyObject *pDoc = PyObject_GetAttrString(x->function, "__doc__");  // Get the documentation of the function
        if (pDoc != NULL) {
            const char *Doc = PyUnicode_AsUTF8(pDoc);
            if (Doc != NULL) {
                post("");
                post("==== %s documentation ====", x->function_name->s_name);
                post("");
                post("%s", Doc);
                post("");
                post("==== %s documentation ====", x->function_name->s_name);
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
    x->script_name = argv[0].a_w.w_symbol;
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
    if (argc != 0) {
        x->editorName = atom_getsymbol(argv + 0);
        post("[py4pd] Editor set to: %s", x->editorName->s_name);
        char cfgFile[MAXPDSTRING];
        const char *py4pdDir = x->py4pdPath->s_name;
        snprintf(cfgFile, MAXPDSTRING, "%s/py4pd.cfg", py4pdDir);
        FILE* file = fopen(cfgFile, "w");
        fprintf(file, "editor = %s", x->editorName->s_name);
        fclose(file);
        return;
    }
    if (x->function_called == 0) {  // if the set method was not called, then we
        pd_error(x, "[py4pd] To open the editor you need to set the function first!");
        return;
    }

    PyCodeObject *code = (PyCodeObject *)PyFunction_GetCode(x->function);
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
    if (x->function_called == 0) {  // if the set method was not called, then we
                                    // can not run the function :)
        pd_error(x, "To reload the script you need to set the function first!");
        return;
    }
    pFunc = x->function;

    // reload the module
    pName = PyUnicode_DecodeFSDefault(x->script_name->s_name);  // Name of script file
    pModule = PyImport_Import(pName);
    if (pModule == NULL) {
        pd_error(x, "Error importing the module!");
        x->function_called = 0;
        Py_DECREF(pFunc);
        Py_DECREF(pName);
        return;
    }


    pReload = PyImport_ReloadModule(pModule);
    if (pReload == NULL) {
        pd_error(x, "Error reloading the module!");
        x->function_called = 0;
        Py_DECREF(pFunc);
        Py_DECREF(pModule);
        return;
    } 
    else {
        pFunc = PyObject_GetAttrString(pModule, x->function_name->s_name);  // Function name inside the script file
        Py_DECREF(pName);
        Py_DECREF(pReload);
        if (pFunc && PyCallable_Check(pFunc)) {  // Check if the function exists and is callable
            x->function = pFunc;
            x->function_called = 1;
            // get new number of python function
            PyObject *inspect = NULL, *getfullargspec = NULL;
            PyObject *argspec = NULL, *args = NULL;
            inspect = PyImport_ImportModule("inspect");
            getfullargspec = PyObject_GetAttrString(inspect, "getfullargspec");
            argspec = PyObject_CallFunctionObjArgs(getfullargspec, pFunc, NULL);
            args = PyObject_GetAttrString(argspec, "args");
            x->py_arg_numbers = PyList_Size(args);
            post("The module was reloaded!");
            Py_DECREF(inspect);
            Py_DECREF(getfullargspec);
            Py_DECREF(argspec);
            return;
        } 
        else {
            pd_error(x, "Error reloading the module!");
            x->function_called = 0;
            Py_DECREF(x->function);
            return;
        }
    }
}

// ====================================
/**
 * @brief set the python function and save it on x->function
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
    if (argc < 2) {  // check is the number of arguments is correct | set
        pd_error(x, "[py4pd] 'set' message needs two arguments! The 'Script Name' and the 'Function Name'!");
        return;
    }

    t_symbol *script_file_name = atom_gensym(argv + 0);
    t_symbol *function_name = atom_gensym(argv + 1);

    if (x->function_called == 1) {
        int function_is_equal = strcmp(function_name->s_name, x->function_name->s_name);  // if string is equal strcmp returns 0
        if (function_is_equal == 0) {
            pd_error(x, "[py4pd] The function was already set!");
            return;
        } 
        else {
            Py_XDECREF(x->function);
            x->function_called = 0;
        }
    }

    // Check if there is extension (not to use it)
    char *extension = strrchr(script_file_name->s_name, '.');
    if (extension != NULL) {
        pd_error(x, "[py4pd] Don't use extensions in the script file name!");
        Py_XDECREF(x->function);
        return;
    }

    // check if script file exists
    char script_file_path[MAXPDSTRING];
    snprintf(script_file_path, MAXPDSTRING, "%s/%s.py", x->pdPatchFolder->s_name, script_file_name->s_name);

    char script_inside_py4pd_path[MAXPDSTRING];
    snprintf(script_inside_py4pd_path, MAXPDSTRING, "%s/resources/scripts/%s.py", x->py4pdPath->s_name, script_file_name->s_name);

    if (access(script_file_path, F_OK) == -1 && access(script_inside_py4pd_path, F_OK) == -1) {
        pd_error(x, "[py4pd] The script file %s was not found!", script_file_name->s_name);
        Py_XDECREF(x->function);
        return;
    }
    PyObject *pModule, *pFunc;  // Create the variables of the python objects

    // odd code, but solve the bug
    t_py *prev_obj;
    int prev_obj_exists = 0;
    PyObject *MainModule = PyImport_ImportModule("pd");
    PyObject *oldObjectCapsule;

    if (MainModule != NULL) {
        oldObjectCapsule = PyDict_GetItemString(MainModule, "py4pd"); // borrowed reference
        if (oldObjectCapsule != NULL) {
            PyObject *py4pd_capsule = PyObject_GetAttrString(MainModule, "py4pd");
            prev_obj = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
            prev_obj_exists = 1;
        }
        else {
            prev_obj_exists = 0;
        }
    }

    PyObject *objectCapsule = Py4pdUtils_AddPdObject(x);

    // =====================
    pModule = PyImport_ImportModule(script_file_name->s_name);  // Import the script file with the function
    // =====================

    // check if the module was loaded
    if (pModule == NULL) {
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[py4pd] Call failed: %s", PyUnicode_AsUTF8(pstr));
        Py_XDECREF(pstr);
        Py_XDECREF(pModule);
        return;
    }
    pFunc = PyObject_GetAttrString(pModule, function_name->s_name);  // Function name inside the script file
    if (pFunc && PyCallable_Check(pFunc)) {  // Check if the function exists and is callable
        PyCodeObject* code = (PyCodeObject*)PyFunction_GetCode(pFunc);

        if (prev_obj_exists == 1 && pFunc != NULL) {
            objectCapsule = Py4pdUtils_AddPdObject(prev_obj);
            if (objectCapsule == NULL){
                pd_error(x, "[Python] Failed to add object to Python");
                return;
            }
        }

        if (code->co_flags & CO_VARARGS) {
            pd_error(x, "[py4pd] The '%s' function has variable arguments (*args)!", function_name->s_name);
            Py_XDECREF(pFunc);
            Py_XDECREF(pModule);
            return;
        }
        else if (code->co_flags & CO_VARKEYWORDS) {
            pd_error(x, "[py4pd] The '%s' function has variable keyword arguments (**kwargs)!", function_name->s_name);
            Py_XDECREF(pFunc);
            Py_XDECREF(pModule);
            return;
        }
        x->py_arg_numbers = code->co_argcount;
        if (x->py4pd_lib == 0) {
            post("[py4pd] The '%s' function has %d arguments!", function_name->s_name, x->py_arg_numbers);
        }
        x->module = pModule;
        x->function = pFunc;
        x->script_name = script_file_name;
        x->function_name = function_name;
        x->function_called = 1;
    } 
    else {
        pd_error(x, "[py4pd] Function %s not loaded!", function_name->s_name);
        x->function_called = 0; 
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[py4pd] Set function had failed: %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        Py_XDECREF(pModule);
        Py_XDECREF(pFunc);
        PyErr_Clear();
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
        PyObject **lists = (PyObject **)malloc(OpenList_count * sizeof(PyObject *));

        ArgsTuple = Py4pdUtils_ConvertToPy(lists, argc, argv);  // convert the arguments to python
        int argCount = PyTuple_Size(ArgsTuple);  // get the number of arguments
        if (argCount != x->py_arg_numbers) {
            pd_error(x, "[py4pd] Wrong number of arguments! The function %s needs %i arguments, received %i!", 
                        x->function_name->s_name, (int)x->py_arg_numbers,
                        argCount);
            return;
        }
    } 
    else {
        ArgsTuple = PyTuple_New(0);
    }
    Py4pdUtils_RunPy(x, ArgsTuple, NULL);

    return;
}

// ============================================
/**
 * @brief This function will control were the Python will run, with PEP 684, I want to make possible using parallelism in Python
 * @param x 
 * @param s 
 * @param argc 
 * @param argv 
 * @return It will return nothing but will run the Python function
 */

static void Py4pd_ExecuteFunction(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    if (x->function_called == 0) {
        pd_error(x, "[py4pd] You need to call a function before run!");
        return;
    }
    if (x->runmode == 0) {
        Py4pd_RunFunction(x, s, argc, argv);
    }
    else if(x->runmode == 1) {
        pd_error(x, "[py4pd] Thread is not implemented yet, wait for Python 3.12");
        return;
    }

    return;
}

// ===========================================

// TODO: I will change this when PEP 684 is implemented

static void Py4pd_Py4pdThread(t_py *x, t_floatarg f) {
    int mode = (int)f;
    if (mode == 0) {
        x->runmode = 0;
        return;
    }
    else if (mode == 1) {
        x->runmode = 1; //fork
        post("[py4pd] Thread mode activated");
        return;
    }   
    else {
        pd_error(x, "[py4pd] Invalid runmode, use 0 for normal mode and 1 for threading mode");
    }
}

// ============================================
/**
 * @brief This function import the numpy module. We can not use import_array() directly, because it is a macro.
 * @param NULL 
 * @return It will return NULL.
 */
void *Py4pd_ImportNumpyForPy4pd() {
    import_array();
    return NULL;
}

// ============================================
/**
 * @brief This will enable or disable the numpy array support and start numpy import if it is not imported.
 * @param x is the py4pd object
 * @param f is the status of the numpy array support
 * @return It will return void.
 */
void Py4pd_SetPythonPointersUsage(t_py *x, t_floatarg f) {
    int usepointers = (int)f;
    if (usepointers == 1) {
        post("[py4pd] Python Pointers enabled.");
        x->outPyPointer = 1;
    } 
    else if (usepointers == 0) {
        x->outPyPointer = 0;
        post("[py4pd] Python Pointers disabled");
    } 
    else {
        pd_error(x, "[py4pd] Python Pointers status must be 0 (disable) or 1 (enable)");
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
    int visMODE = 0;
    int audioOUT = 0;
    int audioIN = 0;
    int libraryMODE = 0;
    int normalMODE = 1;
    t_symbol *scriptName = NULL;

    int major, minor, micro;
    sys_getversion(&major, &minor, &micro);
    if (major < 0 && minor < 54){
        pd_error(NULL, "[py4pd] You need to use Pd 0.54 or higher");
        return NULL;
    }
    logpost(NULL, 3, "[py4pd] Pd version: %d.%d.%d", major, minor, micro);
    
    // Get what will be the type of the object
    for (i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            t_symbol *py4pdArgs = atom_getsymbolarg(i, argc, argv);
            if (py4pdArgs == gensym("-picture") ||
                py4pdArgs == gensym("-score") ||
                py4pdArgs == gensym("-pic") ||
                py4pdArgs == gensym("-canvas")) {
                visMODE = 1;
                if (argv[i + 1].a_type == A_FLOAT && argv[i + 2].a_type == A_FLOAT){
                    pd_error(NULL, "[py4pd] -picture, -score, -pic and -canvas was removed in version 0.8.0");
                    pd_error(NULL, "[py4pd] Please transfor your code in one Pd Object, it is simple");
                    pd_error(NULL, "[py4pd] Check: https://py4pd.readthedocs.io/en/latest/python-users/#pdaddobject");
                    return NULL;
                    
                }
            } 
            else if (py4pdArgs == gensym("-audio") || py4pdArgs == gensym("-audioout")) {
                pd_error(NULL, "[py4pd] -audio option was removed in version 0.8.0");
                pd_error(NULL, "[py4pd] Please transfor your code in one Pd Object, it is simple");
                pd_error(NULL, "[py4pd] Check: https://py4pd.readthedocs.io/en/latest/python-users/#pdaddobject");
                return NULL;
            }
            else if (py4pdArgs == gensym("-audioin")) {
                pd_error(NULL, "[py4pd] -audioin was removed in version 0.8.0");
                pd_error(NULL, "[py4pd] Please transfor your code in one Pd Object, it is simple");
                pd_error(NULL, "[py4pd] Check: https://py4pd.readthedocs.io/en/latest/python-users/#pdaddobject");
                return NULL;
            }
            else if (py4pdArgs == gensym("-library") || py4pdArgs == gensym("-lib")) {
                libraryMODE = 1;
                normalMODE = 0;
                scriptName = atom_getsymbolarg(i + 1, argc, argv);
            }
        }
    }
    
    // INIT PYTHON
    if (!Py_IsInitialized()) {
        object_count = 0; 
        post("");
        post("[py4pd] by Charles K. Neimog");
        post("[py4pd] Version %d.%d.%d", PY4PD_MAJOR_VERSION, PY4PD_MINOR_VERSION, PY4PD_MICRO_VERSION);
        post("[py4pd] Python version %d.%d.%d", PY_MAJOR_VERSION, PY_MINOR_VERSION, PY_MICRO_VERSION);
        post("");
        PyImport_AppendInittab("pd", PyInit_pd);  
        Py_Initialize();  
    }

    // =================
    // INIT PY4PD OBJECT
    // =================
    if (normalMODE == 1 && visMODE == 0 && audioOUT == 0 && audioIN == 0) {
        x = (t_py *)pd_new(py4pd_class);  // create a new py4pd object
    } 
    else if (libraryMODE == 1 && visMODE == 0 && audioOUT == 0 && audioIN == 0) {  // library
        x = (t_py *)pd_new(py4pd_classLibrary);  
        x->canvas = canvas_getcurrent();      
        t_canvas *c = x->canvas;             
        t_symbol *patch_dir = canvas_getdir(c);
        x->runmode = 0;
        x->object_number = object_count; 
        x->pdPatchFolder = patch_dir;       
        x->pkgPath = patch_dir;  
        Py4pdUtils_SetObjConfig(x); 
        int libraryLoaded = Py4pd_LibraryLoad(x, argc, argv);
        if (libraryLoaded == -1){
            return NULL;
        }
        x->script_name = scriptName;
        object_count++;
        Py4pd_ImportNumpyForPy4pd();
        return (x);
    }
    else {
        pd_error(NULL, "Error in py4pdNew, you can not use more than one flag at the same time.");
        return NULL;
    }
    x->canvas = canvas_getcurrent();      
    t_canvas *c = x->canvas;             
    t_symbol *patch_dir = canvas_getdir(c);
    x->audioInput = 0;
    x->audioOutput = 0;
    x->visMode = 0;
    x->editorName = NULL;
    x->pyObject = 0;
    x->vectorSize = 0;
    Py4pdUtils_ParseArguments(x, c, argc, argv);  // parse arguments
    x->runmode = 0;
    x->object_number = object_count;  // save object number
    x->pdPatchFolder = patch_dir; // set name of the home path
    x->pkgPath = patch_dir;     // set name of the packages path
    Py4pdUtils_SetObjConfig(x);          // set the config file (in py4pd.cfg, make this be
    if (argc > 1) {             // check if there are two arguments
        Py4pd_SetFunction(x, s, argc, argv);
        Py4pd_ImportNumpyForPy4pd();
    }

    // Add additional paths to the python path
    char *pyScripts_folder = malloc(strlen(x->py4pdPath->s_name) + 20); // allocate extra space
    snprintf(pyScripts_folder, strlen(x->py4pdPath->s_name) + 20, "%s/resources/scripts", x->py4pdPath->s_name);
    char *pyGlobal_packages = malloc(strlen(x->py4pdPath->s_name) + 20); // allocate extra space
    snprintf(pyGlobal_packages, strlen(x->py4pdPath->s_name) + 20, "%s/resources/py-modules", x->py4pdPath->s_name);
    PyObject *home_path = PyUnicode_FromString(x->pdPatchFolder->s_name);  // Place where script file will probably be
    PyObject *site_package = PyUnicode_FromString(x->pkgPath->s_name);  // Place where the packages will be
    PyObject *py4pdScripts = PyUnicode_FromString(pyScripts_folder);  // Place where the py4pd scripts will be
    PyObject *py4pdGlobalPackages = PyUnicode_FromString(pyGlobal_packages);  // Place where the py4pd global packages will be
    PyObject *sys_path = PySys_GetObject("path");
    PyList_Insert(sys_path, 0, home_path);
    PyList_Insert(sys_path, 0, site_package);
    PyList_Insert(sys_path, 0, py4pdScripts);
    PyList_Insert(sys_path, 0, py4pdGlobalPackages);
    Py_DECREF(home_path);
    Py_DECREF(site_package);
    Py_DECREF(py4pdScripts);
    free(pyScripts_folder);
    free(pyGlobal_packages);
    object_count++;
    return (x);
}

// ====================================================
/**
 * @brief Setup the py4pd object
 */

void py4pd_setup(void) {
    py4pd_class =
        class_new(gensym("py4pd"),  // cria o objeto quando escrevemos py4pd
                  (t_newmethod)Py4pd_Py4pdNew,  // metodo de criação do objeto
                  (t_method)Py4pdLib_FreeObj,    // quando voce deleta o objeto
                  sizeof(t_py),  // quanta memoria precisamos para esse objeto
                  0,             // nao há uma GUI especial para esse objeto???
                  A_GIMME,       // os argumentos são um símbolo
                  0);            // fim de argumentos
    py4pd_classLibrary = class_new(gensym("py4pd"), (t_newmethod)Py4pd_Py4pdNew, (t_method)Py4pdLib_FreeObj, sizeof(t_py), CLASS_NOINLET, A_GIMME, 0);

    // this is like have lot of objects with the same name, add all methods for
    class_addmethod(py4pd_class, (t_method)Py4pd_SetPy4pdHomePath, gensym("home"), A_GIMME, 0);  // set home path
    class_addmethod(py4pd_class, (t_method)Py4pd_SetPackages, gensym("packages"), A_GIMME, 0);  // set packages path
    class_addmethod(py4pd_class, (t_method)Py4pd_PipInstall, gensym("pipinstall"), A_GIMME, 0);  // on/off threading
    class_addmethod(py4pd_class, (t_method)Py4pd_Py4pdThread, gensym("thread"), A_FLOAT, 0);  // on/off threading
    class_addmethod(py4pd_class, (t_method)Py4pd_SetPythonPointersUsage, gensym("pointers"), A_FLOAT, 0);  // set home path
    class_addmethod(py4pd_class, (t_method)Py4pd_ReloadPy4pdFunction, gensym("reload"), 0, 0);  // reload python script

    // Object INFO
    class_addmethod(py4pd_class, (t_method)Py4pd_PrintPy4pdVersion, gensym("version"), 0, 0);  // show version
    class_addmethod(py4pd_class, (t_method)Py4pd_SetEditor, gensym("editor"), A_GIMME, 0);  // open code
    class_addmethod(py4pd_class, (t_method)Py4pd_OpenScript, gensym("open"), A_GIMME, 0); 
    class_addmethod(py4pd_class, (t_method)Py4pd_SetEditor, gensym("click"), 0, 0);  // when click open editor
    class_addmethod(py4pd_classLibrary, (t_method)Py4pd_SetEditor, gensym("click"), 0, 0);  // when click open editor
    
    // User 
    class_addmethod(py4pd_class, (t_method)Py4pd_PrintDocs, gensym("doc"), 0, 0);  // open documentation
    class_addmethod(py4pd_class, (t_method)Py4pd_ExecuteFunction, gensym("run"), A_GIMME, 0);  // run function
    class_addmethod(py4pd_class, (t_method)Py4pd_SetFunction, gensym("set"), A_GIMME, 0);  // set function to be called
    class_addmethod(py4pd_class, (t_method)Py4pd_PrintModuleFunctions, gensym("functions"), A_GIMME, 0); 
}
