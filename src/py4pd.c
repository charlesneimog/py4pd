#include "py4pd.h"
#include "pd_module.h"
#include "py4pd_pic.h"
#include "py4pd_utils.h"

 
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


// ============================================
t_class *py4pd_class;          // For audioin and without audio
t_class *py4pd_class_VIS;      // For visualisation | pic object by pd-else
t_class *py4pd_classAudioOut;  // For audio out

int object_count; // 

// ============================================
// ========= PY4PD METHODS FUNCTIONS ==========
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
        post("[py4pd] The home path is: %s", x->home_path->s_name);
    } else {
        x->home_path = atom_getsymbol(argv);
        post("[py4pd] The home path set to: %s", x->home_path->s_name);
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
        post("[py4pd] The packages path is: %s", x->packages_path->s_name);
        return;  // is this necessary?
    } else {
        if (argc < 2 && argc > 0) {
            if (argv[0].a_type == A_SYMBOL) {
                t_symbol *path = atom_getsymbol(argv);
                // It checks relative path
                if (path->s_name[0] == '.' && path->s_name[1] == '/') {
                    char *new_path = malloc(strlen(x->home_path->s_name) +
                                            strlen(path->s_name) + 1);
                    strcpy(new_path, x->home_path->s_name);
                    strcat(new_path, path->s_name + 1);
                    post("[py4pd] The packages path set to: %s", new_path);
                    x->packages_path = gensym(new_path);
                    free(new_path);
                } else {
                    x->packages_path = atom_getsymbol(argv);
                    post("[py4pd] The packages path set to: %s",
                         x->packages_path->s_name);
                }
            } else {
                pd_error(x, "[py4pd] The packages path must be a string");
                return;
            }
            // check if path exists and is valid
            if (access(x->packages_path->s_name, F_OK) == -1) {
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
// ALWAYS DESCRIPTION OF NEXT FUNCTION
/**
 * @brief print the documentation of the function
 * @param x is the py4pd object
 * @param s is the symbol (message) that was sent to the object
 * @param argc is the number of arguments
 * @param argv is the arguments
 * @return void, but it prints the documentation
 */
static void documentation(t_py *x) {
    PyObject *pFunc;
    if (x->function_called == 0) { 
        pd_error(x, "[py4pd] To see the documentaion you need to set the function first!");
        return;
    }
    pFunc = x->function;
    if (pFunc && PyCallable_Check(pFunc)) {  // Check if the function exists and is callable
        PyObject *pDoc = PyObject_GetAttrString(pFunc, "__doc__");  // Get the documentation of the function
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
        char *command = malloc(strlen(x->home_path->s_name) + strlen(x->script_name->s_name) + 20);
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
    // Not Windows OS
    #else  
        char *command = malloc(strlen(x->home_path->s_name) + strlen(x->script_name->s_name) + 20);
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
        char *command = malloc(strlen(x->home_path->s_name) + strlen(x->script_name->s_name) + 20);
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
static void set_param(t_py *x, t_symbol *s, int argc, t_atom *argv) {
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
        value = PyUnicode_FromString(argv[1].a_w.w_symbol->s_name);
    }
    // check if the value is a float
    else if (argv[1].a_type == A_FLOAT) {
        value = PyFloat_FromDouble(argv[1].a_w.w_float);
    }
    else {
        pd_error(x, "[py4pd] The value must be a symbol or a float!");
        return;
    }
    
    // try to set the value
    if (PyDict_SetItem(x->Dict, key, value) == -1) {
        pd_error(x, "[py4pd] Error setting the parameter!");
        return;
    }
    else {
        post("[py4pd] Parameter set!");
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
static void reload(t_py *x) {
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
    pReload = PyImport_ReloadModule(pModule);
    if (pReload == NULL) {
        pd_error(x, "Error reloading the module!");
        x->function_called = 0;
        Py_DECREF(pFunc);
        Py_DECREF(pModule);
        return;
    } 
    else {
        pFunc = PyObject_GetAttrString(
            pModule,
            x->function_name->s_name);  // Function name inside the script file
        Py_DECREF(pName);
        Py_DECREF(pReload);
        if (pFunc &&
            PyCallable_Check(
                pFunc)) {  // Check if the function exists and is callable
            x->function = pFunc;
            x->function_called = 1;
            post("The module was reloaded!");
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
static void set_function(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s;
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
    snprintf(script_file_path, MAXPDSTRING, "%s/%s.py", x->home_path->s_name, script_file_name->s_name);

    char script_inside_py4pd_path[MAXPDSTRING];
    snprintf(script_inside_py4pd_path, MAXPDSTRING, "%s/%s.py", x->py4pd_scripts->s_name, script_file_name->s_name);

    if (access(script_file_path, F_OK) == -1 && access(script_inside_py4pd_path, F_OK) == -1) {
        pd_error(x, "[py4pd] The script file %s was not found!", script_file_name->s_name);
        Py_XDECREF(x->function);
        return;
    }

    // =====================
    // check number of arguments
    if (argc < 2) {  // check is the number of arguments is correct | set
        pd_error(x, "[py4pd] 'set' message needs two arguments! The 'Script Name' and the 'Function Name'!");
        return;
    }
    // =====================
    PyObject *pModule, *pFunc;  // Create the variables of the python objects

    // =====================
    // create one folder using x->py4pd_folder->s_name + "/pyScripts"
    char *pyScripts_folder = malloc(strlen(x->py4pd_folder->s_name) + 12);
    strcpy(pyScripts_folder, x->py4pd_folder->s_name);
    strcat(pyScripts_folder, "/resources/scripts");
    // =====================
    // Add aditional path to python to work with Pure Data
    PyObject *home_path = PyUnicode_FromString(x->home_path->s_name);  // Place where script file will probably be
    PyObject *site_package = PyUnicode_FromString(x->packages_path->s_name);  // Place where the packages will be
    PyObject *py4pdScripts = PyUnicode_FromString(pyScripts_folder);  // Place where the py4pd scripts will be
    PyObject *sys_path = PySys_GetObject("path");
    PyList_Insert(sys_path, 0, home_path);
    PyList_Insert(sys_path, 0, site_package);
    PyList_Insert(sys_path, 0, py4pdScripts);
    Py_DECREF(home_path);
    Py_DECREF(site_package);
    Py_DECREF(py4pdScripts);

    // =====================
    pModule = PyImport_ImportModule(script_file_name->s_name);  // Import the script file
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
        PyObject *inspect = NULL, *getfullargspec = NULL;
        PyObject *argspec = NULL, *args = NULL;
        inspect = PyImport_ImportModule("inspect");
        getfullargspec = PyObject_GetAttrString(inspect, "getfullargspec");
        argspec = PyObject_CallFunctionObjArgs(getfullargspec, pFunc, NULL);

        args = PyTuple_GetItem(argspec, 0);
        
        //  TODO: way to check if function has *args or **kwargs

        int py_args = PyObject_Size(args);
        if (args == Py_None) {
            x->py_arg_numbers = -1;
            post("[py4pd] The '%s' function has *args or **kwargs!", function_name->s_name);
        } 
        else {
            x->py_arg_numbers = py_args;
            post("[py4pd] The '%s' function has %d arguments!", function_name->s_name, py_args);
        }
        Py_DECREF(inspect);
        Py_DECREF(getfullargspec);
        Py_DECREF(argspec);
        Py_DECREF(args);
        Py_DECREF(pModule);
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
            post("Length of tuple: %i", argCount);
            post("Length of args: %i", x->py_arg_numbers);
            return;
        }
    } 
    else {
        ArgsTuple = PyTuple_New(0);
    }

    // WARNING: If PEP 684 is accepted, check if object capsule will work.

    PyObject *capsule = PyCapsule_New(x, "py4pd", NULL);  // create a capsule to pass the object to the python interpreter
    PyModule_AddObject(PyImport_AddModule("__main__"), "py4pd", capsule);  // add the capsule to the python interpreter

    pValue = PyObject_CallObject(x->function, ArgsTuple);
    if (pValue != NULL) {                // if the function returns a value
        //   TODO: add pointer output when x->Python is 1;

        py4pd_convert_to_pd(x, pValue);  // convert the value to pd
    } 
    else {                             // if the function returns a error
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[Python] Call failed: %s", PyUnicode_AsUTF8(pstr));
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
/*
static void run_function_thread(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;

    if (argc != x->py_arg_numbers) {
        pd_error(x, "[py4pd] Wrong number of arguments!");
        return;
    }
    PyObject *pFunc, *pArgs, *pValue; // pDict, *pModule,
    pFunc = x->function;
    pArgs = PyTuple_New(argc);
    int i;
    if (x->function_called == 0) { // if the set method was not called, then we
can not run the function :) if(pFunc != NULL){
            // create t_atom *argv from x->script_name and x->function_name
            t_atom *newargv = malloc(sizeof(t_atom) * 2);
            SETSYMBOL(newargv, x->script_name);
            SETSYMBOL(newargv+1, x->function_name);
            set_function(x, NULL, 2, newargv);
        } else{
            pd_error(x, "[py4pd] The message need to be formatted like 'set
{script_name} {function_name}'!"); return;
        }
    }

    // CONVERTION TO PYTHON OBJECTS
    // create an array of t_atom to store the list
    t_atom *list = malloc(sizeof(t_atom) * argc);
    for (i = 0; i < argc; ++i) {
        t_atom *argv_i = malloc(sizeof(t_atom));
        *argv_i = argv[i];
        pValue = py4pd_convert_to_python(argv_i);
        if (!pValue) {
            pd_error(x, "[py4pd] Cannot convert argument\n");
            return;
        }
        PyTuple_SetItem(pArgs, i, pValue); // Set the argument in the tuple
    }

    pValue = PyObject_CallObject(pFunc, pArgs); // Call and execute the function
    if (pValue != NULL) {                                // if the function
returns a value
        // convert the python object to a t_atom
        py4pd_convert_to_pd(x, pValue);
    }
    else { // if the function returns a error
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[py4pd] Call failed: %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        Py_DECREF(pvalue);
        Py_DECREF(ptype);
        Py_DECREF(ptraceback);
    }
    return;
}

// ============================================

struct thread_arg_struct {
    t_py x;
    t_symbol s;
    int argc;
    t_atom *argv;
    PyThreadState *interp;
} thread_arg;

// // ============================================

static void *ThreadFunc(void *lpParameter) {
    struct thread_arg_struct *arg = (struct thread_arg_struct *)lpParameter;
    t_py *x = &arg->x;
    t_symbol *s = &arg->s;
    int argc = arg->argc;
    t_atom *argv = arg->argv;
    int object_number = x->object_number;
    thread_status[object_number] = 1;
    // PyGILState_STATE gstate;
    running_some_thread = 1;
    run_function_thread(x, s, argc, argv);
    thread_status[object_number] = 0;
    running_some_thread = 0;
    return 0;
}

// ============================================

static void create_thread(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    struct thread_arg_struct *arg = (struct thread_arg_struct
*)malloc(sizeof(struct thread_arg_struct)); arg->x = *x; arg->argc = argc;
    arg->argv = argv;
    int object_number = x->object_number;
    if (x->function_called == 0) {
        // Pd is crashing when I try to create a thread.
        pd_error(x, "[py4pd] You need to call a function before run!");
        free(arg);
        return;
    } else {
        if (thread_status[object_number] == 0){
            // PyThread is not thread safe, so we need to lock the GIL
            pthread_t thread;
            pthread_create(&thread, NULL, ThreadFunc, arg);
            x->state = 1;
            // check the Thread was created
        } else {
            pd_error(x, "[py4pd] There is a thread running in this Object!");
            free(arg);
        }
    }
    return;
}
*/

// ============================================
/**
 * @brief This function will control were the Python will run, wil PEP 684, I want to make possible using parallelism in Python
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
    if (x->thread == 1) {
        run_function(x, s, argc, argv);
        pd_error(x, "[py4pd] Not implemenented! Wait for approval of PEP 684");
    } 
    else if (x->thread == 0) {
        run_function(x, s, argc, argv);
    } 
    else {
        pd_error(x, "[py4pd] Thread not created");
    }
    return;
}


// ===========================================

static void thread(t_py *x, t_floatarg f) {
    //  TODO: If the run method set before the end of the thread, there is an
    //  error, that close all PureData.
    int thread = (int)f;
    if (thread == 1) {
        post("[py4pd] Threading enabled, wait for approval of PEP 684");
        x->thread = 1;
        return;
    } 
    else if (thread == 0) {
        x->thread = 0;
        post("[py4pd] Threading disabled");
        return;
    } 
    else {
        pd_error(x, "[py4pd] Threading status must be 0 or 1");
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
    if (x->audioInput == 0 && x->audioOutput == 0) {
        return (w + 4);
    }
    t_sample *audioIn =
        (t_sample *)(w[2]);  // this is the input vector (the sound)
    int n = (int)(w[3]);     // this is the vector size (number of samples, for
                             // example 64)
    const npy_intp dims = n;
    PyObject *ArgsTuple, *pValue, *pAudio, *pSample;

    if (x->function_called == 0) {
        pd_error(x, "[py4pd] You need to call a function before run!");
        return (w + 4);
    }

    pSample = NULL;  //
    if (x->use_NumpyArray == 1) {
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

    // WARNING: this can generate errors? How this will work on multithreading?
    PyObject *capsule = PyCapsule_New(
        x, "py4pd",
        NULL);  // create a capsule to pass the object to the python interpreter
    PyModule_AddObject(PyImport_AddModule("__main__"), "py4pd",
                       capsule);  // add the capsule to the python interpreter

    // call the function
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
    //  TODO: Check for memory leaks

    t_py *x = (t_py *)(w[1]);  // this is the object itself
    if (x->audioInput == 0 && x->audioOutput == 0) {
        return (w + 5);
    }
    if (x->function_called == 0) {
        pd_error(x, "[py4pd] You need to call a function before run!");
        return (w + 5);
    }
    t_sample *audioIn = (t_sample *)(w[2]);  // this is the input vector (the sound)
    t_sample *audioOut = (t_sample *)(w[3]);  // this is the output vector (the sound)
    int n = (int)(w[4]);     // this is the vector size (number of samples, for example 64)
    const npy_intp dims = n;
    PyObject *ArgsTuple, *pValue, *pAudio, *pSample;

    pSample = NULL;  // NOTE: This is the way to not distorce the audio output

    if (x->use_NumpyArray == 1) {
        pAudio = PyArray_SimpleNewFromData(1, &dims, NPY_FLOAT, audioIn);
        ArgsTuple = PyTuple_New(1);
        PyTuple_SetItem(ArgsTuple, 0, pAudio);
    } else {
        // pSample = NULL;  NOTE: this change the sound.
        // pAudio = PyList_New(n); cconvert audioIn in tuple
        pAudio = PyTuple_New(n);
        for (int i = 0; i < n; i++) {
            pSample = PyFloat_FromDouble(audioIn[i]);
            PyTuple_SetItem(pAudio, i, pSample);
        }
        ArgsTuple = PyTuple_New(1);
        PyTuple_SetItem(ArgsTuple, 0, pAudio);
        // if (pSample != NULL) { NOTE: this change the sound.
        //     Py_DECREF(pSample);
        // }
    }
    // WARNING: this can generate errors? How this will work on multithreading? || In PEP 684 this will be per interpreter or global?
    PyObject *capsule = PyCapsule_New(x, "py4pd", NULL);  // create a capsule to pass the object to the python interpreter
    PyModule_AddObject(PyImport_AddModule("__main__"), "py4pd", capsule);  // add the capsule to the python interpreter
    pValue = PyObject_CallObject(x->function, ArgsTuple);

    if (pValue != NULL) {
        if (PyList_Check(pValue)) {
            for (int i = 0; i < n; i++) {
                audioOut[i] = PyFloat_AsDouble(PyList_GetItem(pValue, i));
            }
        } else if (PyTuple_Check(pValue)) {
            for (int i = 0; i < n; i++) {
                audioOut[i] = PyFloat_AsDouble(PyTuple_GetItem(pValue, i));
            }
        } else if (x->numpyImported == 1) {
            if (PyArray_Check(pValue)) {
                PyArrayObject *pArray = (PyArrayObject *)pValue;
                for (int i = 0; i < n; i++) {  // TODO: try to add audio support
                                               // without another loop
                    audioOut[i] = PyFloat_AsDouble(
                        PyArray_GETITEM(pArray, PyArray_GETPTR1(pArray, i)));
                }
            } 
            else {
                pd_error(x, "[py4pd] The function must return a list, a tuple or a numpy array, returned: %s", 
                            pValue->ob_type->tp_name);
            }
        } 
        else {
            pd_error(x, "[py4pd] The function must return a list, since numpy array is disabled, returned: %s", 
                            pValue->ob_type->tp_name);
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
    Py_XDECREF(pValue);
    Py_DECREF(ArgsTuple);
    if (pSample != NULL) {
        Py_DECREF(pSample);
    }
    return (w + 5);
}

// ============================================
/**
 * @brief This function add the py4pd_perform function to the signal chain. If audioOutput (-audioout or -audio) is set, it add an output channel.
 * @param w is the signal vector
 * @return It will return the output the audio output.
 */
static void py4pd_dspin(t_py *x, t_signal **sp) {
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
static void *py4pdImportNumpy() {
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
    //  TODO: If the run method set before the end of the thread, there is an
    //  error, that close all PureData.
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
    int normalMODE = 1;

    for (i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            t_symbol *py4pdArgs = atom_getsymbolarg(i, argc, argv);
            if (py4pdArgs == gensym("-picture") ||
                py4pdArgs == gensym("-score") ||
                py4pdArgs == gensym("-canvas")) {
                visMODE = 1;
            } 
            else if (py4pdArgs == gensym("-audio") ||
                       py4pdArgs == gensym("-audioout")) {
                audioOUT = 1;
            }
        }
    }
    if (visMODE == 1 && audioOUT == 0) {
        x = (t_py *)pd_new(py4pd_class_VIS);  // create a new object
    } 
    else if (audioOUT == 1 && visMODE == 0) {
        x = (t_py *)pd_new(py4pd_classAudioOut);  // create a new object
    } 
    else if (normalMODE == 1) {
        x = (t_py *)pd_new(py4pd_class);  // create a new object
    } 
    else {
        post("Error in py4pd_new, please report this error to the developer, this message should not appear.");
        return NULL;
    }

    x->x_canvas = canvas_getcurrent();       // pega o canvas atual
    t_canvas *c = x->x_canvas;               // get the current canvas
    t_symbol *patch_dir = canvas_getdir(c);  // directory of opened patch
    x->audioInput = 0;
    x->audioOutput = 0;
    x->visMode = 0;

    if (!Py_IsInitialized()) {
        object_count = 1;  // To count the numbers of objects, and finalize the
        post("");
        post("[py4pd] by Charles K. Neimog");
        post("[py4pd] Version 0.6.0       ");
        post("[py4pd] Python version %d.%d.%d", PY_MAJOR_VERSION, PY_MINOR_VERSION, PY_MICRO_VERSION);
        post("");
        PyImport_AppendInittab("pd", PyInit_pd);  // Add the pd module to the python interpreter
        Py_Initialize();  // Initialize the Python interpreter. If 1, the signal
    }

    object_count++;  // count the number of objects;
    for (i = 0; i < argc; i++) {
        if (argv[i].a_type == A_SYMBOL) {
            t_symbol *py4pdArgs = atom_getsymbolarg(i, argc, argv);
            if (py4pdArgs == gensym("-picture") ||
                py4pdArgs == gensym("-score") ||
                py4pdArgs == gensym("-canvas")) {
                py4pd_InitVisMode(x, c, py4pdArgs, i, argc, argv);
                x->visMode = 1;
                x->x_outline = 1;
            } 
            else if (py4pdArgs == gensym("-audioout")) {
                // post("[py4pd] Audio Outlets enabled");
                x->audioOutput = 1;
                x->use_NumpyArray = 0;
                x->out_A = outlet_new(
                    &x->x_obj, gensym("signal"));  // create a signal outlet
                int j;
                for (j = i; j < argc; j++) {
                    argv[j] = argv[j + 1];
                }
                argc--;
            } 
            else if (py4pdArgs == gensym("-nvim") ||
                       py4pdArgs == gensym("-vscode") ||
                       py4pdArgs == gensym("-emacs")) {
                x->editorName = gensym(py4pdArgs->s_name + 1);
                int j;
                for (j = i; j < argc; j++) {
                    argv[j] = argv[j + 1];
                }
                argc--;
            } 
            else if (py4pdArgs == gensym("-audioin")) {
                x->audioInput = 1;
                x->use_NumpyArray = 0;
            } 
            else if (py4pdArgs == gensym("-audio")) {
                x->audioInput = 1;
                x->audioOutput = 1;
                x->out_A = outlet_new(
                    &x->x_obj, gensym("signal"));  // create a signal outlet
                x->use_NumpyArray = 0;
                int j;
                for (j = i; j < argc; j++) {
                    argv[j] = argv[j + 1];
                }
                argc--;
            }
        }
    }

    if (x->audioOutput == 0) {
        x->out_A = outlet_new(&x->x_obj, 0);  // cria um outlet caso o objeto nao contenha audio
    }
    x->thread = 0;
    x->object_number = object_count;  // save object number
    x->home_path = patch_dir;         // set name of the home path
    x->packages_path = patch_dir;     // set name of the packages path
    set_py4pd_config(x);  // set the config file (in py4pd.cfg, make this be
    py4pd_tempfolder(x);  // find the py4pd folder
    findpy4pd_folder(x);  // find the py4pd object folder
    if (argc > 1) {       // check if there are two arguments
        set_function(x, s, argc, argv);
        import_array();  // import numpy
        x->numpyImported = 1;
    }
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

    if (object_count == 1) {
        // Py_Finalize(); // BUG: Not possible because it crashes if another
        post("[py4pd] Python interpreter finalized");
        object_count = 0;

        //  TODO: Clear temporary files
        #ifdef _WIN64
            char command[1000];
            sprintf(command, "del /q /s %s\\*", x->temp_folder->s_name);
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
            sprintf(command, "rm -rf %s", x->temp_folder->s_name);
            system(command);
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
                  A_GIMME,       // o argumento é um símbolo
                  0);
    py4pd_class_VIS = class_new(gensym("py4pd"), (t_newmethod)py4pd_new, (t_method)py4pd_free, sizeof(t_py), 0, A_GIMME, 0);
    py4pd_classAudioOut = class_new(gensym("py4pd"), (t_newmethod)py4pd_new, (t_method)py4pd_free, sizeof(t_py), 0, A_GIMME, 0);

    // Sound in
    class_addmethod(py4pd_class, (t_method)py4pd_dspin, gensym("dsp"), A_CANT, 0);  // add a method to a class
    class_addmethod(py4pd_classAudioOut, (t_method)py4pd_dspin, gensym("dsp"), A_CANT, 0);  // add a method to a class
    CLASS_MAINSIGNALIN(py4pd_class, t_py, py4pd_audio);  // TODO: Repensando como fazer isso quando o áudio não for usado.
    CLASS_MAINSIGNALIN(py4pd_classAudioOut, t_py, py4pd_audio);  // TODO: Repensando como fazer isso quando o áudio não for usado.
    class_addmethod(py4pd_class, (t_method)usenumpy, gensym("numpy"), A_FLOAT, 0);  // add a method to a class
    class_addmethod(py4pd_classAudioOut, (t_method)usenumpy, gensym("numpy"), A_FLOAT, 0);  // add a method to a class

    // Pic related
    class_addmethod(py4pd_class_VIS, (t_method)PY4PD_size_callback, gensym("_picsize"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(py4pd_class_VIS, (t_method)PY4PD_mouserelease,gensym("_mouserelease"), 0);
    class_addmethod(py4pd_class_VIS, (t_method)PY4PD_outline, gensym("outline"), A_DEFFLOAT, 0);
    class_addmethod(py4pd_class_VIS, (t_method)PY4PD_zoom, gensym("zoom"), A_CANT, 0);

    // this is like have lot of objects with the same name, add all methods for
    // py4pd_class, py4pd_class_AudioOut and py4pd_class_VIS
    class_addmethod(py4pd_class, (t_method)home, gensym("home"), A_GIMME, 0);  // set home path
    class_addmethod(py4pd_class_VIS, (t_method)home, gensym("home"), A_GIMME, 0);  // set home path
    class_addmethod(py4pd_classAudioOut, (t_method)packages, gensym("home"), A_GIMME, 0);  // set packages path

    class_addmethod(py4pd_class, (t_method)packages, gensym("packages"), A_GIMME, 0);  // set packages path
    class_addmethod(py4pd_class_VIS, (t_method)packages, gensym("packages"), A_GIMME, 0);  // set packages path
    class_addmethod(py4pd_classAudioOut, (t_method)packages, gensym("packages"), A_GIMME, 0);  // set packages path

    class_addmethod(py4pd_class, (t_method)thread, gensym("thread"), A_FLOAT, 0);  // on/off threading
    class_addmethod(py4pd_class_VIS, (t_method)thread, gensym("thread"), A_FLOAT, 0);  // on/off threading
    class_addmethod(py4pd_classAudioOut, (t_method)thread, gensym("thread"), A_FLOAT, 0);  // on/off threading

    class_addmethod(py4pd_class, (t_method)reload, gensym("reload"), 0, 0);  // reload python script
    class_addmethod(py4pd_class_VIS, (t_method)reload, gensym("reload"), 0, 0);  // reload python script
    class_addmethod(py4pd_classAudioOut, (t_method)reload, gensym("reload"), 0, 0);  // reload python script

    class_addmethod(py4pd_class, (t_method)restartPython, gensym("restart"), 0, 0);  // it restart python interpreter
    class_addmethod(py4pd_class_VIS, (t_method)restartPython, gensym("restart"), 0, 0);  // it restart python interpreter
    class_addmethod(py4pd_classAudioOut, (t_method)restartPython, gensym("restart"), 0, 0);  // it restart python interpreter

    // Edit Python Code
    class_addmethod(py4pd_class, (t_method)vscode, gensym("vscode"), 0, 0);  // open editor  WARNING: will be removed

    class_addmethod(py4pd_class, (t_method)editor, gensym("editor"), A_GIMME, 0);  // open code
    class_addmethod(py4pd_class_VIS, (t_method)editor, gensym("editor"), A_GIMME, 0);  // open code
    class_addmethod(py4pd_classAudioOut, (t_method)editor, gensym("editor"), A_GIMME, 0);  // open code

    class_addmethod(py4pd_class, (t_method)openscript, gensym("open"), A_GIMME, 0);  // create file or open it TODO: fix this
    class_addmethod(py4pd_class_VIS, (t_method)openscript, gensym("open"), A_GIMME, 0);  // create file or open it
    class_addmethod(py4pd_classAudioOut, (t_method)openscript, gensym("open"), A_GIMME, 0);  // create file or open it

    class_addmethod(py4pd_class, (t_method)editor, gensym("click"), 0, 0);  // when click open editor
    class_addmethod(py4pd_classAudioOut, (t_method)editor, gensym("click"), 0, 0);  // when click open editor

    // User Interface
    class_addmethod(py4pd_class, (t_method)documentation, gensym("doc"), 0, 0);  // open documentation
    class_addmethod(py4pd_class_VIS, (t_method)documentation, gensym("doc"), 0, 0);  // open documentation
    class_addmethod(py4pd_classAudioOut, (t_method)documentation, gensym("doc"), 0, 0);  // open documentation
 
    class_addmethod(py4pd_class, (t_method)run, gensym("run"), A_GIMME, 0);  // run function
    class_addmethod(py4pd_class_VIS, (t_method)run, gensym("run"), A_GIMME, 0);  // run function
    class_addmethod(py4pd_classAudioOut, (t_method)run, gensym("run"), A_GIMME, 0);  // run function

    class_addmethod(py4pd_class, (t_method)set_function, gensym("set"), A_GIMME, 0);  // set function to be called
    class_addmethod(py4pd_class_VIS, (t_method)set_function, gensym("set"), A_GIMME, 0);  // set function to be called
    class_addmethod(py4pd_classAudioOut, (t_method)set_function, gensym("set"), A_GIMME, 0);  // set function to be called


    class_addmethod(py4pd_class, (t_method)set_param, gensym("key"), A_GIMME, 0);  // set parameter inside py4pd->params
    class_addmethod(py4pd_class_VIS, (t_method)set_param, gensym("key"), A_GIMME, 0);  // set parameter inside py4pd->params
    class_addmethod(py4pd_classAudioOut, (t_method)set_param, gensym("key"), A_GIMME, 0);  // set parameter inside py4pd->params


    //  TODO: Way to set global variables, I think that will be important for things like general path;
    //  TODO: Set some audio parameters to work with py4pd_dspin, 'dspparams', 'dspparams'
    
    class_addmethod(py4pd_class, (t_method)findpy4pd_folder, gensym("debug"), A_GIMME, 0); 
}


