#include "py4pd.h"
#include "m_pd.h"
#include "pd_module.h"
#include "py4pd_pic.h"
#include "py4pd_utils.h"

#define NPY_NO_DEPRECATED_API NPY_1_25_API_VERSION
#include <numpy/arrayobject.h>

// ============================================
t_class *py4pd_class;          // For audioin and without audio
t_class *py4pd_class_VIS;      // For visualisation | pic object by pd-else
t_class *py4pd_classAudioIn;   // For audio in
t_class *py4pd_classAudioOut;  // For audio out
t_class *py4pd_classLibrary;   // For libraries
int pipePy4pdNum = 0;
int object_count = 0;

// ============================================
// =========== PY4PD LOAD LIBRARIES ===========
// ============================================

static void libraryLoad(t_py *x, int argc, t_atom *argv){

    if (argc > 2) {
        pd_error(x, "[py4pd] Too many arguments! Usage: py4pd -lib <library_name>");
        return;
    }

    t_symbol *script_file_name = atom_gensym(argv + 1);
    t_symbol *function_name = gensym("py4pdLoadObjects");

    // check if script file exists
    char script_file_path[MAXPDSTRING];
    snprintf(script_file_path, MAXPDSTRING, "%s/%s.py", x->pdPatchFolder->s_name, script_file_name->s_name);


    char script_inside_py4pd_path[MAXPDSTRING];
    snprintf(script_inside_py4pd_path, MAXPDSTRING, "%s/resources/scripts/%s.py", x->py4pdPath->s_name, script_file_name->s_name);

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
            snprintf(library_path, MAXPDSTRING, "%s/%s/", pathelem, script_file_name->s_name); // NOTE: The library folder must have the same name as the library file
            if (access(library_path, F_OK) != -1) {
                libraryNotFound = 0;
                PyObject *library_path_py = PyUnicode_FromString(library_path);
                PyList_Insert(sys_path, 0, library_path_py);
            }
        }
        if (libraryNotFound){
            pd_error(x, "[py4pd] Library file %s not found in search path", script_file_name->s_name);
            return;
        }
    }

    // return;

    PyObject *pModule, *pFunc;  // Create the variables of the python objects
    char *pyScriptsFolder = malloc(strlen(x->py4pdPath->s_name) + 40); // allocate extra space
    char *pyGlobalFolder = malloc(strlen(x->py4pdPath->s_name) + 40); // allocate extra space
    snprintf(pyScriptsFolder, strlen(x->py4pdPath->s_name) + 40, "%s/resources/scripts/", x->py4pdPath->s_name);
    snprintf(pyGlobalFolder, strlen(x->py4pdPath->s_name) + 40, "%s/resources/py-modules/", x->py4pdPath->s_name);

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


    t_py *prev_obj;
    int prev_obj_exists = 0;
    PyObject *MainModule = PyImport_ImportModule("__main__");
    PyObject *oldObjectCapsule;
    if (MainModule != NULL) {
        oldObjectCapsule = PyDict_GetItemString(MainModule, "py4pd"); // borrowed reference
        if (oldObjectCapsule != NULL) {
            PyObject *py4pd_capsule = PyObject_GetAttrString(MainModule, "py4pd");
            prev_obj = (t_py *)PyCapsule_GetPointer(py4pd_capsule, "py4pd");
            prev_obj_exists = 1;
        }
    }
    PyObject *objectCapsule = py4pd_add_pd_object(x);

    pModule = PyImport_ImportModule(script_file_name->s_name);  // Import the script file
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
    if (pFunc && PyCallable_Check(pFunc)) {  

        if (objectCapsule == NULL){
            pd_error(x, "[Python] Failed to add object to Python");
            return;
        }
        PyObject *pValue = PyObject_CallNoArgs(pFunc);  // Call the function
        if (pValue == NULL) {
            PyObject *ptype, *pvalue, *ptraceback;
            PyErr_Fetch(&ptype, &pvalue, &ptraceback);
            PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
            PyObject *pstr = PyObject_Str(pvalue);
            pd_error(x, "[py4pd] Call failed: %s", PyUnicode_AsUTF8(pstr));
            Py_XDECREF(pstr);
            Py_XDECREF(pModule);
            Py_XDECREF(pFunc);
            return;
        }
        // odd code, but solve the bug
        if (prev_obj_exists == 1 && pValue != NULL) {
            objectCapsule = py4pd_add_pd_object(prev_obj);
            if (objectCapsule == NULL){
                pd_error(x, "[Python] Failed to add object to Python");
                return;
            }
        }
        x->module = pModule;
        x->function = pFunc;
        x->script_name = script_file_name;
        x->function_name = function_name;
        x->function_called = 1;
        logpost(x, 3, "[py4pd] Library %s loaded!", script_file_name->s_name);
    } 
    else {
        pd_error(x, "[py4pd] Library %s not loaded!", function_name->s_name);
        x->function_called = 1;  // set the flag to 0 because it crash Pd if
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[py4pd] ERROR %s", PyUnicode_AsUTF8(pstr));
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
// ========= PY4PD METHODS FUNCTIONS ==========
// ============================================
static void out_py4pdVersion(t_py *x){
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

static void home(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;  // unused but required by pd
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

static void packages(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    if (argc < 1) {
        post("[py4pd] The packages path is: %s", x->pkgPath->s_name);
        return;  // is this necessary?
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
static void getmoduleFunction(t_py *x, t_symbol *s, int argc, t_atom *argv) {
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
// ALWAYS DESCRIPTION OF NEXT FUNCTION
/**
 * @brief print the documentation of the function
 * @param x is the py4pd object
 * @param s is the symbol (message) that was sent to the object
 * @param argc is the number of arguments
 * @param argv is the arguments
 * @return void, but it prints the documentation
 */
void documentation(t_py *x) {
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
static void openscript(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    (void)argc;

    if (argv[0].a_type != A_SYMBOL) {
        pd_error(x, "[py4pd] The script name must be a symbol");
        return;
    }

    x->script_name = argv[0].a_w.w_symbol;

    // Open VsCode in Windows
    #ifdef _WIN64
        char *command = malloc(strlen(x->pdPatchFolder->s_name) + strlen(x->script_name->s_name) + 20);
        command = get_editor_command(x);
        SHELLEXECUTEINFO sei = {0};
        sei.cbSize = sizeof(sei);
        sei.fMask = SEE_MASK_NOCLOSEPROCESS;
        sei.lpFile = "cmd.exe ";
        sei.lpParameters = command;
        sei.nShow = SW_HIDE;
        ShellExecuteEx(&sei);
        CloseHandle(sei.hProcess);
        return;
    #else  
        char *command = malloc(strlen(x->pdPatchFolder->s_name) + strlen(x->script_name->s_name) + 20);
        command = get_editor_command(x);
        pd4py_system_func(command);
        return;
    #endif
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
static void editor(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    if (argc != 0) {
        x->editorName = atom_getsymbol(argv + 0);
        post("[py4pd] Editor set to: %s", x->editorName->s_name);
        return;
    }
    if (x->function_called == 0) {  // if the set method was not called, then we
        pd_error(x, "[py4pd] To open vscode you need to set the function first!");
        return;
    }
    post("[py4pd] Opening editor...");

    // Open VsCode in Windows
    #ifdef _WIN64
        char *command = get_editor_command(x);
        command = get_editor_command(x);
        // use get_editor_command
        SHELLEXECUTEINFO sei = {0};
        sei.cbSize = sizeof(sei);
        sei.fMask = SEE_MASK_NOCLOSEPROCESS;
        sei.lpFile = "cmd.exe ";
        sei.lpParameters = command;
        sei.nShow = SW_HIDE;
        ShellExecuteEx(&sei);
        CloseHandle(sei.hProcess);
        return;

    // Not Windows OS
    #else  // if not windows 64bits
        char *command = malloc(strlen(x->pdPatchFolder->s_name) + strlen(x->script_name->s_name) + 20);
        command = get_editor_command(x);
        pd4py_system_func(command);
        return;
    #endif
}

// ====================================
/**
 * @brief DEPRECATED: open the script  in the editor
*/
static void vscode(t_py *x) {
    pd_error(x, "This method is deprecated, please use the editor method instead!");
}

// ====================================
/**
 * @brief set parameters to and PyDict from embedded module pd
 * @param x is the py4pd object
 * @param s is the symbol (message) that was sent to the object
 * @param argc is the number of arguments
 * @param argv is the arguments
 * @return void, but it sets the editor
 */
void set_param(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    if (argc > 2) {
        pd_error(x, "[py4pd] For now, just one parameter at a time!");
        return;
    }
    if (x->Dict == NULL) {
        x->Dict = PyDict_New();
    }
    // Add key and value to the dictionary
    PyObject *key = PyUnicode_FromString(argv[0].a_w.w_symbol->s_name);
    PyObject *value = NULL;
    if (argv[1].a_type == A_SYMBOL) {
        const char *pyS = argv[1].a_w.w_symbol->s_name;
        value = PyUnicode_FromString(pyS);
    }
    // check if the value is a float
    else if (argv[1].a_type == A_FLOAT) {
        float f = argv[1].a_w.w_float;
        value = PyFloat_FromDouble(f);
    }
    else {
        pd_error(x, "[py4pd] The value must be a symbol or a float!");
        return;
    }
    
    int result = PyDict_SetItem(x->Dict, key, value);

    if (result == -1) {
        pd_error(x, "[py4pd] Error setting the parameter!");
        return;
    }
    else {
        // get key from x->Dict
        post("[py4pd] Parameter set in key %s", argv[0].a_w.w_symbol->s_name);

    }

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
void reload(t_py *x) {
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
 * @brief set the function and save it on x->function
 * @param x is the py4pd object
 * @param s is the symbol (message) that was sent to the object
 * @param argc is the number of arguments
 * @param argv is the arguments
 * @return void, but it sets the function
 */
void set_function(t_py *x, t_symbol *s, int argc, t_atom *argv) {
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

    // =====================
    char *pyScripts_folder = malloc(strlen(x->py4pdPath->s_name) + 20); // allocate extra space
    snprintf(pyScripts_folder, strlen(x->py4pdPath->s_name) + 20, "%s/resources/scripts", x->py4pdPath->s_name);
    // =====================
    char *pyGlobal_packages = malloc(strlen(x->py4pdPath->s_name) + 20); // allocate extra space
    snprintf(pyGlobal_packages, strlen(x->py4pdPath->s_name) + 20, "%s/resources/py-modules", x->py4pdPath->s_name);

    // Add aditional path to python to work with Pure Data
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
        x->function_called = 1;  // set the flag to 0 because it crash Pd if
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

static void run_function(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    //  TODO: Check for memory leaks
    (void)s;
    int OpenList_count = 0;
    int CloseList_count = 0;

    PyObject *pValue, *ArgsTuple;
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

        ArgsTuple = py4pd_convert_to_py(lists, argc, argv);  // convert the arguments to python
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

    // odd code, but solve the bug
    t_py *prev_obj;
    int prev_obj_exists = 0;
    PyObject *MainModule = PyImport_ImportModule("__main__");
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

    PyObject *objectCapsule = py4pd_add_pd_object(x);

    if (objectCapsule == NULL){
        pd_error(x, "[Python] Failed to add object to Python");
        return;
    }

    pValue = PyObject_CallObject(x->function, ArgsTuple);

    // odd code, but solve the bug
    if (prev_obj_exists == 1 && pValue != NULL) {
        objectCapsule = py4pd_add_pd_object(prev_obj);
        if (objectCapsule == NULL){
            pd_error(x, "[Python] Failed to add object to Python");
            return;
        }
    }

    if (pValue != NULL) { // if the function returns a value  TODO: add pointer output when x->Python is 1;
        py4pd_convert_to_pd(x, pValue);  // convert the value to pd
    } 
    else {                             // if the function returns a error
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[%s] Call failed: %s", x->function_name->s_name, PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        PyErr_Clear();
    }
    Py_XDECREF(pValue);
    Py_DECREF(ArgsTuple);
    return;
}
// ============================================

struct thread_arg_struct {
    t_py x;
    PyObject *process;
} thread_arg;

// ============================================
static void thread_run(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    (void)argc;
    (void)argv;
    PyObject *multiprocessing = PyImport_ImportModule("multiprocessing");
    PyObject* Process = PyObject_GetAttrString(multiprocessing, "Process");
    PyObject* kwargs = Py_BuildValue("{s:O}", "target", x->function);
    PyObject* process = PyObject_Call(Process, Py_BuildValue("()"), kwargs);
    PyObject_SetAttrString(process, "daemon", Py_True);
    PyObject_CallMethod(process, "start", NULL);
    Py_DECREF(process);
    Py_DECREF(Process);
    Py_DECREF(multiprocessing);
    return;
}

// =====================================
struct py4pdEXE{
    t_py            *x;
    t_symbol        *s;
    int             argc;
    t_atom          *argv;
}t_py4pdEXE;


// ============================================
void *py4pdThreadExe(void *arg) {
    char *command = (char *)arg;
    int status = system(command);
    if (status != 0) {
        pd_error(NULL, "[py4pd] py4pd detached thread exited with status %d", status);
    }
    pthread_exit(NULL);
    return NULL;
}


// ============================================
void *independed_run(void *arg) {
    struct py4pdEXE *py4pdEXE = (struct py4pdEXE *)arg;
    t_py *x = py4pdEXE->x;
    int argc = py4pdEXE->argc;
    t_atom *argv = py4pdEXE->argv;
    
    // # run just if is is not Windows
    #ifndef _WIN32
    
    const char *pipe_PDARGS = "/tmp/py4pd_PDARGS"; // It sends the arguments list to the py4pd (exe)
    const char *pipe_PDARGS_SIZE = "/tmp/py4pd_PDARGS_SIZE"; // It sends the arguments list size to the py4pd (exe)
    const char *pipe_PDATOM_SIZE = "/tmp/py4pd_PDATOM_SIZE"; // It sends the arguments list size to the py4pd (exe)
    const char *pipe_PY4PDHOME = "/tmp/py4pd_PY4PDHOME"; // It sends the Pd home path to the py4pd (exe)
    const char *pipe_PATCHHOME = "/tmp/py4pd_PATCHHOME"; // It sends the Pd patch path to the py4pd (exe)
    const char *pipe_SITEPACKAGES = "/tmp/py4pd_SITEPACKAGES";
    const char *pipe_PYMODULE = "/tmp/py4pd_PYMODULE"; // The name of the named pipe
    const char *pipe_PYFUNCTION = "/tmp/py4pd_PYFUNCTION"; // The name of the named pipe
    const char *pipe_PYRETURN = "/tmp/py4pd_PYRETURN"; // The name of the named pipe
    const char *pipe_RETURNSIZE = "/tmp/py4pd_RETURNSIZE"; // The name of the named pipe
    
    mkfifo(pipe_PDARGS, 0666);
    mkfifo(pipe_PDARGS_SIZE, 0666);
    mkfifo(pipe_PDATOM_SIZE, 0666);
    mkfifo(pipe_PY4PDHOME, 0666);
    mkfifo(pipe_PATCHHOME, 0666);
    mkfifo(pipe_SITEPACKAGES, 0666);
    mkfifo(pipe_PYMODULE, 0666);
    mkfifo(pipe_PYFUNCTION, 0666);
    mkfifo(pipe_PYRETURN, 0666);
    mkfifo(pipe_RETURNSIZE, 0666);

    // ============================================
    unlink(pipe_PYRETURN);
    unlink(pipe_RETURNSIZE);
    const char *py4pd_PATH = x->py4pdPath->s_name;
    const char py4pd_EXEC[] = "py4pd";

    // create exec path
    char *exec_path = (char *)malloc(strlen(py4pd_PATH) + strlen(py4pd_EXEC) + 2);
    strcpy(exec_path, py4pd_PATH);
    strcat(exec_path, "/");
    strcat(exec_path, py4pd_EXEC);
    pthread_t thread;
    pthread_create(&thread, NULL, py4pdThreadExe, exec_path); // this execute where pipes in read mode are opened

    int fd_PY4PDHOME = open(pipe_PY4PDHOME, O_WRONLY);
    int fd_PATCHHOME = open(pipe_PATCHHOME, O_WRONLY);
    int fd_PYMODULE = open(pipe_PYMODULE, O_WRONLY);
    int fd_PYFUNCTION = open(pipe_PYFUNCTION, O_WRONLY);
    int fd_SITEPACKAGES = open(pipe_SITEPACKAGES, O_WRONLY);

    if (fd_PY4PDHOME < 0 || fd_PYMODULE < 0 || fd_PYFUNCTION < 0 || fd_SITEPACKAGES < 0 || fd_PATCHHOME < 0){ 
        pd_error(x, "[py4pd] Failed to open pipe");
        pthread_cancel(thread);
        return 0;
    }
    const char *home = canvas_getdir(x->x_canvas)->s_name;
    write(fd_PY4PDHOME, x->py4pdPath->s_name, strlen(x->py4pdPath->s_name) + 1);
    write(fd_PATCHHOME, home, strlen(home) + 1);
    write(fd_PYMODULE, x->script_name->s_name, strlen(x->script_name->s_name) + 1);
    write(fd_PYFUNCTION, x->function_name->s_name, strlen(x->function_name->s_name) + 1);
    write(fd_SITEPACKAGES, x->pkgPath->s_name, strlen(x->pkgPath->s_name) + 1);
    
    close(fd_PY4PDHOME);
    close(fd_PATCHHOME);
    close(fd_PYMODULE);
    close(fd_PYFUNCTION);
    close(fd_SITEPACKAGES);

    int fd_PDARGS_SIZE = open(pipe_PDARGS_SIZE, O_WRONLY);
    write(fd_PDARGS_SIZE, &argc, sizeof(int));
    close(fd_PDARGS_SIZE);
    
    char pd_atoms[64 * argc]; // Buffer to hold all arguments
    int offset = 0; // Offset within buffer
    int *sizes = (int *)malloc(argc * sizeof(int));
    for (int i = 0; i < argc; i++) {
        char pd_atom[64];
        atom_string(&argv[i], pd_atom, 64);
        int len = strlen(pd_atom) + 1; // Add 1 to include null terminator
        memcpy(pd_atoms + offset, pd_atom, len); // Copy argument to buffer
        offset += len; // Update offset
        sizes[i] = len;
    }

    int fd_PDARGS = open(pipe_PDARGS, O_WRONLY);
    int fd_PDATOM_SIZE = open(pipe_PDATOM_SIZE, O_WRONLY);
    int bytes_written = write(fd_PDARGS, pd_atoms, offset);
    if (bytes_written == -1) {
        pd_error(NULL, "Error to convert\n");
    }
    write(fd_PDATOM_SIZE, sizes, argc * sizeof(int));
    close(fd_PDARGS);

    mkfifo(pipe_PYRETURN, 0666);
    mkfifo(pipe_RETURNSIZE, 0666);
    int pipeVALUES = open(pipe_PYRETURN, O_RDONLY);
    int pipeSIZE = open(pipe_RETURNSIZE, O_RDONLY);

    int size;
    read(pipeSIZE, &size, sizeof(int));
    close(pipeSIZE);

    py4pd_atom *value = (py4pd_atom *)malloc(size * sizeof(py4pd_atom));
    read(pipeVALUES, value, size * sizeof(py4pd_atom));
    if (size == 1){
        if (value[0].a_type == PY4PD_FLOAT){
            outlet_float(x->x_obj.ob_outlet, value[0].floatvalue);
        }
        else if (value[0].a_type == PY4PD_SYMBOL){
            outlet_symbol(x->x_obj.ob_outlet, gensym(value[0].symbolvalue));
        }
        else{
            pd_error(x, "[py4pd] Error: Unknown type");
        }
    }
    else if (size > 1){
        t_atom *atoms = (t_atom *)malloc(size * sizeof(t_atom));
        for (int i = 0; i < size; i++){
            if (value[i].a_type == PY4PD_FLOAT){
                SETFLOAT(&atoms[i], value[i].floatvalue);
            }
            else if (value[i].a_type == PY4PD_SYMBOL){
                SETSYMBOL(&atoms[i], gensym(value[i].symbolvalue));
            }
            else{
                pd_error(x, "[py4pd] Error: Unknown type");
            }
        }
        outlet_list(x->x_obj.ob_outlet, &s_list, size, atoms);
        free(atoms);
    }
    else{
        pd_error(x, "[py4pd] Error: Unknown type");
    }
    free(value);
    close(pipeVALUES);
    #endif
    return 0;
}

// ============================================
static void py4pdexe_run(t_py *x, t_symbol *s, int argc, t_atom *argv){
    struct py4pdEXE *x_py4pdEXE = (struct py4pdEXE *)malloc(sizeof(struct py4pdEXE));
    x_py4pdEXE->x = x;
    x_py4pdEXE->s = s;
    x_py4pdEXE->argc = argc;
    x_py4pdEXE->argv = argv;
    pthread_t thread;
    pthread_create(&thread, NULL, independed_run, x_py4pdEXE);
    pthread_detach(thread);
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

static void run(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
    if (x->function_called == 0) {
        pd_error(x, "[py4pd] You need to call a function before run!");
        return;
    }
    if (x->runmode == 0) {
        run_function(x, s, argc, argv);
    }
    else if(x->runmode == 1) {
        thread_run(x, s, argc, argv);
    }
    else if(x->runmode == 2){
        py4pdexe_run(x, s, argc, argv);
    }
    return;
}

// ===========================================

static void py4pdThread(t_py *x, t_floatarg f) {
    int mode = (int)f;
    if (mode == 0) {
        x->runmode = 0;
        return;
    }
    else if (mode == 1) {
        #ifdef _WIN32
            pd_error(x, "[py4pd] Thread is not implemented in Windows OS, wait for PEP 684");
        #else
        x->runmode = 1; //fork
        post("[py4pd] Thread mode activated");
        #endif
        return;
    }   
    else if (mode == 2){
        x->runmode = 2; //my own .exe
        post("[py4pd] Independed mode activated");
        return;
    }
    else {
        pd_error(x, "[py4pd] Invalid runmode, use 0 for normal mode and 1 for threading mode");
    }
}

// ============================================
/**
 * @brief This function Python for all py4pd object. I remove it because I had some weird errors and crasches with python3.11.
 * @param x is the py4pd object
 * @return It will void.
 */
static void restartPython(t_py *x) {
    pd_error(x, "[py4pd] This function is not implemented yet!");
    return;

    // Py_Finalize();
    // x->function_called = 0;
    // x->function_name = NULL;
    // x->script_name = NULL;
    // x->module = NULL;
    // x->function = NULL;
    // int i;
    // for (i = 0; i < 100; i++) {
    //     char object_name[20];
    //     sprintf(object_name, "py4pd_%d", i);
    //     post("object name: %s", object_name);
    // y = (t_py *)pd_findbyclass((x->object_name = gensym(object_name)),
    // py4pd_class); post("object pointer: %p", y);

    // if (y != NULL) {
    //     y->function_called = 0;
    //     y->function_name = NULL;
    //     y->script_name = NULL;
    //     y->module = NULL;
    //     y->function = NULL;
    //     y->packages_path = gensym("./py-modules");
    //     y->thread = 2;
    //     y->editorName = gensym("code");
    // }
    // PyImport_AppendInittab("pd", PyInit_pd); // Add the pd module to the
    // python interpreter Py_Initialize(); return;
}

// ============================================
// =========== AUDIO FUNCTIONS ================
// ============================================
/**
 * @brief This function run audio functions in Python. It send the input audio to Python and receive the output audio from Python
 * @param w is the signal vector
 * @return It will return the output, this is for audio analisys (like sigmund~). 
 */

t_int *py4pd_perform(t_int *w) {
    t_py *x = (t_py *)(w[1]);  // this is the object itself
    if (x->audioInput == 0) {
        return (w + 4);
    }

        // ======= TO AVOID CRASHES =======
    if (x->audioInput == 0 && x->audioOutput == 0) {
        return (w + 4);
    }
    if (x->function_called == 0) {
        pd_error(x, "[py4pd] You need to call a function before run!");
        return (w + 4);
    }

    if (x->audioInput == 1 && x->audioOutput == 0 && x->py_arg_numbers != 1){
        pd_error(x, "[py4pd] When -audioin is used, the function must just one argument, function have %d arguments", (int)x->py_arg_numbers);
        return (w + 4);
    }

    // ======= TO AVOID CRASHES =======

    t_sample *audioIn = (t_sample *)(w[2]);  // this is the input vector (the sound)
    int n = (int)(w[3]);     
    PyObject *ArgsTuple, *pValue, *pAudio, *pSample;

    if (x->function_called == 0) {
        pd_error(x, "[py4pd] You need to call a function before run!");
        return (w + 4);
    }
    pSample = NULL;  //

    if (x->numpyImported == 1 && x->audioInput == 1 && x->use_NumpyArray == 1) { // BUG: Mess up, fix this later.
        const npy_intp dims = n;
        pAudio = PyArray_SimpleNewFromData(1, &dims, NPY_FLOAT, audioIn);
        ArgsTuple = PyTuple_New(1);
        PyTuple_SetItem(ArgsTuple, 0, pAudio);
    } 
    else {
        pAudio = PyList_New(n);
        for (int i = 0; i < n; i++) {
            pSample = PyFloat_FromDouble(audioIn[i]);
            PyList_SetItem(pAudio, i, pSample);
        }
        ArgsTuple = PyTuple_New(1);
        PyTuple_SetItem(ArgsTuple, 0, pAudio);
    }

    PyObject *objectCapsule = py4pd_add_pd_object(x);
    if (objectCapsule == NULL){
        pd_error(x, "[Python] Failed to add object to Python");
        return (w + 4);
    }
    pValue = PyObject_CallObject(x->function, ArgsTuple);

    if (pValue != NULL) {
        py4pd_convert_to_pd(x, pValue);  // convert the value to pd
    } 
    else {                             // if the function returns a error
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[py4pd] Call failed: %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        PyErr_Clear();
    }

    Py_XDECREF(pValue);
    Py_DECREF(ArgsTuple);
    if (pSample != NULL) {
        Py_DECREF(pSample);
    }
    return (w + 4);
}

// ============================================
/**
 * @brief This function run audio functions in Python and return the audio output. 
 * @param w is the signal vector
 * @return It will return the output the audio output.
 */
t_int *py4pd_performAudioOutput(t_int *w) {

    t_py *x = (t_py *)(w[1]);  // this is the object itself

    // ======= TO AVOID CRASHES =======
    if (x->audioInput == 0 && x->audioOutput == 0) {
        return (w + 5);
    }
    if (x->function_called == 0) {
        pd_error(x, "[py4pd] You need to call a function before run!");
        return (w + 5);
    }

    if (x->audioInput == 1 && x->audioOutput == 0 && x->py_arg_numbers != 1){
        pd_error(x, "[py4pd] When -audioin is used, the function must just one argument!");
        return (w + 5);
    }
    else if (x->audioOutput == 1 && x->audioInput == 0 &&  x->py_arg_numbers != 0){
        pd_error(x, "[py4pd] When -audioout is used, the function must not have arguments!");
        return (w + 5);
    }
    // ======= TO AVOID CRASHES =======

    t_sample *audioIn = (t_sample *)(w[2]);  // this is the input vector (the sound)
    t_sample *audioOut = (t_sample *)(w[3]);  // this is the output vector (the sound)
    int n = (int)(w[4]);     // this is the vector size (number of samples, for example 64)
   
    PyObject *ArgsTuple, *pValue, *pAudio, *pSample;

    pSample = NULL;  // NOTE: This is the way to not distorce the audio output
    pAudio = NULL; 

    if (x->audioInput == 1) {
        if (x->numpyImported == 1) {
            const npy_intp dims = n;
            pAudio = PyArray_SimpleNewFromData(1, &dims, NPY_FLOAT, audioIn);
            ArgsTuple = PyTuple_New(1);
            PyTuple_SetItem(ArgsTuple, 0, pAudio);
        } 
        else {
            // pSample = NULL;  NOTE: this distorce the sound.
            pAudio = PyList_New(n);
            for (int i = 0; i < n; i++) {
                pSample = PyFloat_FromDouble(audioIn[i]);
                PyList_SetItem(pAudio, i, pSample);
            }
            ArgsTuple = PyTuple_New(1);
            PyTuple_SetItem(ArgsTuple, 0, pAudio);
            // if (pSample != NULL) { NOTE: this distorce the sound.
            //     Py_DECREF(pSample);
            // }
        }
    }
    else {
        ArgsTuple = PyTuple_New(0);
    }

    // WARNING: this can generate errors? How this will work on multithreading? || In PEP 684 this will be per interpreter or global?
    PyObject *objectCapsule = py4pd_add_pd_object(x);
    if (objectCapsule == NULL){
        pd_error(x, "[Python] Failed to add object to Python");
        return (w + 5);
    }
    pValue = PyObject_CallObject(x->function, ArgsTuple);
    
    if (pValue != NULL) {
        if (PyArray_Check(pValue)) {
            PyArrayObject *pArray = PyArray_GETCONTIGUOUS((PyArrayObject *)pValue);
            PyArray_Descr *pArrayType = PyArray_DESCR(pArray);
            int arrayLength = PyArray_SIZE(pArray);
            if (arrayLength == n && PyArray_NDIM(pArray) == 1) {
                if (pArrayType->type_num == NPY_INT) {
                    pd_error(x, "[py4pd] The numpy array must be float, returned int");
                }
                else if (pArrayType->type_num == NPY_FLOAT) {
                    float *audioFloat = (float*)PyArray_DATA(pValue);
                    for (int i = 0; i < n; i++) {
                        audioOut[i] = (t_sample)audioFloat[i];
                    }
                }
                else if (pArrayType->type_num == NPY_DOUBLE) {
                    double *audioDouble = (double*)PyArray_DATA(pValue);
                    for (int i = 0; i < n; i++) {
                        audioOut[i] = (t_sample)audioDouble[i];
                    }
                }
                else {
                    pd_error(x, "[py4pd] The numpy array must be float, returned %d", pArrayType->type_num);
                }
                Py_DECREF(pArray);
                Py_DECREF(pArrayType);
            }
            else {
                pd_error(x, "[py4pd] The numpy array must have the same length of the vecsize and 1 dim. Returned: %d samples and %d dims", arrayLength, PyArray_NDIM(pArray));
                Py_DECREF(pArray);
                Py_DECREF(pArrayType);
            }
        } 
        else{
            pd_error(x, "[Python] Python function must return a numpy array");
        }
    } 
    else {  // if the function returns a error
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[py4pd] Call failed: %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        PyErr_Clear();
    }

    Py_DECREF(ArgsTuple);
    Py_XDECREF(pValue);
    Py_XDECREF(pSample); 
    return (w + 5);
}

// ============================================
/**
 * @brief This function add the py4pd_perform function to the signal chain. If audioOutput (-audioout or -audio) is set, it add an output channel.
 * @param w is the signal vector
 * @return It will return the output the audio output.
 */
static void py4pd_audio_dsp(t_py *x, t_signal **sp) {
    if (x->audioOutput == 0) {
        dsp_add(py4pd_perform, 3, x, sp[0]->s_vec, sp[0]->s_n);
    } 
    else {  // python output is audio
        dsp_add(py4pd_performAudioOutput, 4, x, sp[0]->s_vec, sp[1]->s_vec, sp[0]->s_n);
    }
}

// ============================================
/**
 * @brief This function import the numpy module. We can not use import_array() directly, because it is a macro.
 * @param NULL 
 * @return It will return NULL.
 */
void *py4pdImportNumpy() {
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
static void usenumpy(t_py *x, t_floatarg f) {
    int usenumpy = (int)f;
    if (usenumpy == 1) {
        post("[py4pd] Numpy Array enabled.");
        x->use_NumpyArray = 1;
        if (x->numpyImported == 0) {
            py4pdImportNumpy();
            x->numpyImported = 1;
        }
    } 
    else if (usenumpy == 0) {
        x->use_NumpyArray = 0;
        post("[py4pd] Numpy Array disabled");
    } 
    else {
        pd_error(x, "[py4pd] Numpy status must be 0 (disable) or 1 (enable)");
    }
    return;
}

// ============================================
/**
 * @brief This will enable or disable the numpy array support and start numpy import if it is not imported.
 * @param x is the py4pd object
 * @param f is the status of the numpy array support
 * @return It will return void.
 */
void usepointers(t_py *x, t_floatarg f) {
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

void *py4pd_new(t_symbol *s, int argc, t_atom *argv) {
    int i;
    t_py *x;
    int visMODE = 0;
    int audioOUT = 0;
    int audioIN = 0;
    int libraryMODE = 0;
    int normalMODE = 1;
    t_symbol *scriptName;
    int width = 0;
    int height = 0;
    
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
                    width = atom_getfloatarg(i + 1, argc, argv);
                    height = atom_getfloatarg(i + 2, argc, argv);
                    
                }
            } 
            else if (py4pdArgs == gensym("-audio") || py4pdArgs == gensym("-audioout")) {
                audioOUT = 1;
            }
            else if (py4pdArgs == gensym("-audioin")) {
                audioIN = 1;
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
    if (visMODE == 1 && audioOUT == 0 && audioIN == 0) {
        x = (t_py *)pd_new(py4pd_class_VIS);  // create a new picture object
        if (width > 0 && height > 0) {
            x->x_width = width;
            x->x_height = height;
        }
    } 
    else if (audioIN == 1 && visMODE == 0 && audioOUT == 0) {
        x = (t_py *)pd_new(py4pd_classAudioIn);  // create audio in object
    }
    else if (audioOUT == 1 && visMODE == 0 && audioIN == 0) {
        x = (t_py *)pd_new(py4pd_classAudioOut);  // create audio out object
    } 
    else if (normalMODE == 1 && visMODE == 0 && audioOUT == 0 && audioIN == 0) {
        x = (t_py *)pd_new(py4pd_class);  // create a new py4pd object
    } 
    else if (libraryMODE == 1 && visMODE == 0 && audioOUT == 0 && audioIN == 0) {  // library
        x = (t_py *)pd_new(py4pd_classLibrary);  
        x->x_canvas = canvas_getcurrent();      
        t_canvas *c = x->x_canvas;             
        t_symbol *patch_dir = canvas_getdir(c);
        x->runmode = 0;
        x->object_number = object_count; 
        x->pdPatchFolder = patch_dir;       
        x->pkgPath = patch_dir;  
        set_py4pd_config(x); 
        py4pd_tempfolder(x); 
        findpy4pd_folder(x); 
        libraryLoad(x, argc, argv);
        x->script_name = scriptName;
        object_count++;
        py4pdImportNumpy();
        return (x);
    }
    else {
        pd_error(NULL, "Error in py4pd_new, you can not use more than one flag at the same time.");
        return NULL;
    }

    x->x_canvas = canvas_getcurrent();      
    t_canvas *c = x->x_canvas;             
    t_symbol *patch_dir = canvas_getdir(c);
    x->audioInput = 0;
    x->audioOutput = 0;
    x->visMode = 0;
    x->editorName = NULL;
    x->pyObject = 0;
    py4pd_parser_args(x, c, argc, argv);  // parse arguments
    x->runmode = 0;
    x->object_number = object_count;  // save object number
    x->pdPatchFolder = patch_dir;         // set name of the home path
    x->pkgPath = patch_dir;     // set name of the packages path
    set_py4pd_config(x);  // set the config file (in py4pd.cfg, make this be
    py4pd_tempfolder(x);  // find the py4pd folder
    findpy4pd_folder(x);  // find the py4pd object folder
    if (argc > 1) {       // check if there are two arguments
        set_function(x, s, argc, argv);
        py4pdImportNumpy();
    }
    object_count++;
    return (x);
}

// ============================================
/**
 * @brief Free the memory of the object
 * 
 * @param x 
 * @return void* 
 */
void *py4pd_free(t_py *x) {
    object_count--;
    if (object_count == 0) {
        // Py_Finalize();  BUG: This not work properly with submodules written in C
        object_count = 0;
        #ifdef _WIN64
            char command[1000];
            sprintf(command, "del /q /s %s\\*", x->tempPath->s_name);
            SHELLEXECUTEINFO sei = {0};
            sei.cbSize = sizeof(SHELLEXECUTEINFO);
            sei.fMask = SEE_MASK_NOCLOSEPROCESS;
            sei.lpFile = "cmd.exe";
            sei.lpParameters = command;
            sei.nShow = SW_HIDE;
            ShellExecuteEx(&sei);
            CloseHandle(sei.hProcess);
        #else
            char command[1000];
            sprintf(command, "rm -rf %s", x->tempPath->s_name);
            int commandValue = system(command);
            if (commandValue != 0) {
                pd_error(NULL, "Error to free the temp folder");
            }

        #endif
    }
    if (x->visMode == 1) {
        PY4PD_free(x);
    }
    return (void *)x;
}

// ====================================================
/**
 * @brief Setup the class
 * 
 */

void py4pd_setup(void) {
    py4pd_class =
        class_new(gensym("py4pd"),  // cria o objeto quando escrevemos py4pd
                  (t_newmethod)py4pd_new,  // metodo de criação do objeto
                  (t_method)py4pd_free,    // quando voce deleta o objeto
                  sizeof(t_py),  // quanta memoria precisamos para esse objeto
                  0,             // nao há uma GUI especial para esse objeto???
                  A_GIMME,       // os argumentos são um símbolo
                  0);
    py4pd_class_VIS = class_new(gensym("py4pd"), (t_newmethod)py4pd_new, (t_method)py4pd_free, sizeof(t_py), 0, A_GIMME, 0);
    py4pd_classAudioOut = class_new(gensym("py4pd"), (t_newmethod)py4pd_new, (t_method)py4pd_free, sizeof(t_py), 0, A_GIMME, 0);
    py4pd_classAudioIn = class_new(gensym("py4pd"), (t_newmethod)py4pd_new, (t_method)py4pd_free, sizeof(t_py), 0, A_GIMME, 0);
    py4pd_classLibrary = class_new(gensym("py4pd"), (t_newmethod)py4pd_new, (t_method)py4pd_free, sizeof(t_py), CLASS_NOINLET, A_GIMME, 0);

    // Sound in
    class_addmethod(py4pd_classAudioIn, (t_method)py4pd_audio_dsp, gensym("dsp"), A_CANT, 0);  // add a method to a class
    class_addmethod(py4pd_classAudioOut, (t_method)py4pd_audio_dsp, gensym("dsp"), A_CANT, 0);  // add a method to a class
    CLASS_MAINSIGNALIN(py4pd_classAudioIn, t_py, py4pdAudio);  
    CLASS_MAINSIGNALIN(py4pd_classAudioOut, t_py, py4pdAudio);  

    // Pic related
    class_addmethod(py4pd_class_VIS, (t_method)PY4PD_zoom, gensym("zoom"), A_CANT, 0);

    // this is like have lot of objects with the same name, add all methods for
    class_addmethod(py4pd_class, (t_method)home, gensym("home"), A_GIMME, 0);  // set home path
    class_addmethod(py4pd_class_VIS, (t_method)home, gensym("home"), A_GIMME, 0);  // set home path
    class_addmethod(py4pd_classAudioOut, (t_method)packages, gensym("home"), A_GIMME, 0);  // set packages path
    class_addmethod(py4pd_classAudioIn, (t_method)packages, gensym("packages"), A_GIMME, 0);  // set packages path

    class_addmethod(py4pd_class, (t_method)packages, gensym("packages"), A_GIMME, 0);  // set packages path
    class_addmethod(py4pd_class_VIS, (t_method)packages, gensym("packages"), A_GIMME, 0);  // set packages path
    class_addmethod(py4pd_classAudioOut, (t_method)packages, gensym("packages"), A_GIMME, 0);  // set packages path
    class_addmethod(py4pd_classAudioIn, (t_method)packages, gensym("packages"), A_GIMME, 0);  // set packages path

    // Definitios for the class
    class_addmethod(py4pd_class, (t_method)py4pdThread, gensym("thread"), A_FLOAT, 0);  // on/off threading
    class_addmethod(py4pd_class_VIS, (t_method)py4pdThread, gensym("thread"), A_FLOAT, 0);  // on/off threading
    
    class_addmethod(py4pd_class, (t_method)usepointers, gensym("pointers"), A_FLOAT, 0);  // set home path

    class_addmethod(py4pd_classAudioIn, (t_method)usenumpy, gensym("numpy"), A_FLOAT, 0);  // add a method to a class
    class_addmethod(py4pd_classAudioOut, (t_method)usenumpy, gensym("numpy"), A_FLOAT, 0);  // add a method to a class

    // Coding Methods
    class_addmethod(py4pd_class, (t_method)reload, gensym("reload"), 0, 0);  // reload python script
    class_addmethod(py4pd_class_VIS, (t_method)reload, gensym("reload"), 0, 0);  // reload python script
    class_addmethod(py4pd_classAudioOut, (t_method)reload, gensym("reload"), 0, 0);  // reload python script
    class_addmethod(py4pd_classAudioIn, (t_method)reload, gensym("reload"), 0, 0);  // run python script
    class_addmethod(py4pd_classLibrary, (t_method)reload, gensym("reload"), 0, 0);  // run python script

    class_addmethod(py4pd_class, (t_method)restartPython, gensym("restart"), 0, 0);  // it restart python interpreter
    class_addmethod(py4pd_class_VIS, (t_method)restartPython, gensym("restart"), 0, 0);  // it restart python interpreter
    class_addmethod(py4pd_classAudioOut, (t_method)restartPython, gensym("restart"), 0, 0);  // it restart python interpreter
    class_addmethod(py4pd_classAudioIn, (t_method)restartPython, gensym("restart"), 0, 0);  // it restart python interpreter

    // Object INFO
    class_addmethod(py4pd_class, (t_method)out_py4pdVersion, gensym("version"), 0, 0);  // show version
    class_addmethod(py4pd_class_VIS, (t_method)out_py4pdVersion, gensym("version"), 0, 0);  // show version
    class_addmethod(py4pd_classAudioOut, (t_method)out_py4pdVersion, gensym("version"), 0, 0);  // show version
    class_addmethod(py4pd_classAudioIn, (t_method)out_py4pdVersion, gensym("version"), 0, 0);  // show version

    // Edit Python Code
    class_addmethod(py4pd_class, (t_method)vscode, gensym("vscode"), 0, 0);  // open editor  WARNING: will be removed

    class_addmethod(py4pd_class, (t_method)editor, gensym("editor"), A_GIMME, 0);  // open code
    class_addmethod(py4pd_class_VIS, (t_method)editor, gensym("editor"), A_GIMME, 0);  // open code
    class_addmethod(py4pd_classAudioOut, (t_method)editor, gensym("editor"), A_GIMME, 0);  // open code
    class_addmethod(py4pd_classAudioIn, (t_method)editor, gensym("editor"), A_GIMME, 0);  // open code
    class_addmethod(py4pd_classLibrary, (t_method)editor, gensym("editor"), A_GIMME, 0);  // open code

    class_addmethod(py4pd_class, (t_method)openscript, gensym("open"), A_GIMME, 0); 
    class_addmethod(py4pd_class_VIS, (t_method)openscript, gensym("open"), A_GIMME, 0);
    class_addmethod(py4pd_classAudioOut, (t_method)openscript, gensym("open"), A_GIMME, 0); 
    class_addmethod(py4pd_classAudioIn, (t_method)openscript, gensym("open"), A_GIMME, 0);

    class_addmethod(py4pd_class, (t_method)editor, gensym("click"), 0, 0);  // when click open editor
    class_addmethod(py4pd_classAudioOut, (t_method)editor, gensym("click"), 0, 0);  // when click open editor
    class_addmethod(py4pd_classAudioIn, (t_method)editor, gensym("click"), 0, 0);  // when click open editor

    // User Interface
    class_addmethod(py4pd_class, (t_method)documentation, gensym("doc"), 0, 0);  // open documentation
    class_addmethod(py4pd_class_VIS, (t_method)documentation, gensym("doc"), 0, 0);  // open documentation
    class_addmethod(py4pd_classAudioOut, (t_method)documentation, gensym("doc"), 0, 0);  // open documentation
    class_addmethod(py4pd_classAudioIn, (t_method)documentation, gensym("doc"), 0, 0);  // open documentation
 
    class_addmethod(py4pd_class, (t_method)run, gensym("run"), A_GIMME, 0);  // run function
    class_addmethod(py4pd_class_VIS, (t_method)run, gensym("run"), A_GIMME, 0);  // run function
    
    class_addmethod(py4pd_class, (t_method)set_function, gensym("set"), A_GIMME, 0);  // set function to be called
    class_addmethod(py4pd_class_VIS, (t_method)set_function, gensym("set"), A_GIMME, 0);  // set function to be called
    class_addmethod(py4pd_classAudioOut, (t_method)set_function, gensym("set"), A_GIMME, 0);  // set function to be called
    class_addmethod(py4pd_classAudioIn, (t_method)set_function, gensym("set"), A_GIMME, 0);  // unset function to be called


    class_addmethod(py4pd_class, (t_method)set_param, gensym("key"), A_GIMME, 0);  // set parameter inside py4pd->params
    class_addmethod(py4pd_class_VIS, (t_method)set_param, gensym("key"), A_GIMME, 0);  // set parameter inside py4pd->params
    class_addmethod(py4pd_classAudioOut, (t_method)set_param, gensym("key"), A_GIMME, 0);  // set parameter inside py4pd->params
    class_addmethod(py4pd_classAudioIn, (t_method)set_param, gensym("key"), A_GIMME, 0);  // set parameter inside py4pd->params

    class_addmethod(py4pd_class, (t_method)getmoduleFunction, gensym("functions"), A_GIMME, 0); 
    class_addmethod(py4pd_class_VIS, (t_method)getmoduleFunction, gensym("functions"), A_GIMME, 0);
    class_addmethod(py4pd_classAudioOut, (t_method)getmoduleFunction, gensym("functions"), A_GIMME, 0);
    class_addmethod(py4pd_classAudioIn, (t_method)getmoduleFunction, gensym("functions"), A_GIMME, 0);
}
