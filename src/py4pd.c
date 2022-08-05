// =================================
// https://github.com/pure-data/pure-data/src/x_gui.c 
#include <m_pd.h>
#include <g_canvas.h>
#include <stdio.h>
#include <string.h>

// If windows 64bits include 
#ifdef _WIN64
#include <windows.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

// Python include
#include <Python.h>

// =================================
// ============ Pd Object code  ====
// =================================

static t_class *py4pd_class; // 

// =====================================
typedef struct _py { // It seems that all the objects are some kind of class.
    
    t_object        x_obj; // convensao no puredata source code
    t_canvas        *x_canvas; // pointer to the canvas
    PyObject        *module; // python object
    PyObject        *function; // function name
    t_float         *function_called; // flag to check if the set function was called
    t_float         *create_inlets; // flag to check if the set function was called
    t_symbol        *packages_path; // packages path 
    t_symbol        *home_path; // home path this always is the path folder (?)
    t_symbol        *function_name; // function name
    t_symbol        *script_name; // script name
    t_outlet        *out_A; // outlet 1.
}t_py;

// // ============================================
// // ============================================
// // ============================================

static void home(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s; // unused but required by pd

    if (argc < 1) {
        post("The home path is: %s", x->home_path->s_name);
        return; // QUESTION: is this necessary?
    } else {
        x->home_path = atom_getsymbol(argv);
        post("The home path set to: %s", x->home_path->s_name);
    }
    // TODO: make it work with path with spaces
}

// // ============================================
// // ============================================
// // ============================================

static void packages(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s; 
    if (argc < 1) {
        post("The packages path is: %s", x->packages_path->s_name);
        return; // is this necessary?
    }
    else {
        if (argc < 2 && argc > 0){
            x->packages_path = atom_getsymbol(argv);
            post("The packages path is now: %s", x->packages_path->s_name);
        }   
        else{
            pd_error(x, "It seems that your package folder has |spaces|. It can not have |spaces|!");
            post("I intend to implement this feature in the future!");
            return;
        }    
    }
}

// ====================================
// ====================================
// ====================================

static void env_install(t_py *x, t_symbol *s, int argc, t_atom *argv){
    // If Windows OS run, if not then warn the user
    (void)s;
    (void)argc;
    (void)argv;
    
    // concat venv_path with the name py4pd
    char *pip_install = malloc(strlen(x->home_path->s_name) + strlen("py4pd") + 20);
    sprintf(pip_install, "/c python -m venv %s/py4pd_packages", x->home_path->s_name);

    // path to venv, 
    char *pip = malloc(strlen(x->home_path->s_name) + strlen("/py4pd_packages/") + 40);
    sprintf(pip, "%s/py4pd_packages/Scripts/pip.exe", x->home_path->s_name);
    // check if pip_path exists
    if (access(pip, F_OK) == -1) {
        SHELLEXECUTEINFO sei = {0};
        sei.cbSize = sizeof(sei);
        sei.fMask = SEE_MASK_NOCLOSEPROCESS;
        sei.lpFile = "cmd.exe ";
        sei.lpParameters = pip_install;
        sei.nShow = SW_HIDE;
        ShellExecuteEx(&sei);
        CloseHandle(sei.hProcess);
        return;
    } else{
        pd_error(x, "The pip already installed!");
    }
}

// ====================================
// ====================================
// ====================================
static void pip_install(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    (void)argc;
    
    char *package = atom_getsymbol(argv+0)->s_name;
    
    char *pip = malloc(strlen(x->home_path->s_name) + strlen("%s/py4pd_packages/Scripts/pip.exe") + 40);
    sprintf(pip, "%s/py4pd_packages/Scripts/pip.exe", x->home_path->s_name);
    if (access(pip, F_OK) == -1) {
        pd_error(x, "The pip path does not exist. Send a message {env_install} to install pip first!");
        return;
    } else{
        char *pip_cmd = malloc(strlen(x->packages_path->s_name) + strlen("py4pd") + 20);
        sprintf(pip_cmd, "/c %s install %s", pip, package);
        post("Installing %s", package);
        SHELLEXECUTEINFO sei = {0};
        sei.cbSize = sizeof(sei);
        sei.fMask = SEE_MASK_NOCLOSEPROCESS;
        sei.lpFile = "cmd.exe ";
        sei.lpParameters = pip_cmd;
        sei.nShow = SW_HIDE;
        ShellExecuteEx(&sei);
        CloseHandle(sei.hProcess);
        post("%s installed!", package);
        return;
    }
}


// ====================================
// ====================================
// ====================================

static void documentation(t_py *x){
    PyObject *pFunc;
    if (x->function_called == 0) { // if the set method was not called, then we can not run the function :)
        pd_error(x, "To see the documentaion you need to set the function first!");
        return;
    }

    pFunc = x->function;
    if (pFunc && PyCallable_Check(pFunc)){ // Check if the function exists and is callable
        PyObject *pDoc = PyObject_GetAttrString(pFunc, "__doc__"); // Get the documentation of the function
        if (pDoc != NULL){
            const char *Doc = PyUnicode_AsUTF8(pDoc); 
            if (Doc != NULL){
                post("");
                post("=== Documentation of the function ==> %s", x->function_name->s_name);
                post("");
                post("%s", Doc);
                post("");
                post("=== End of documentation");
                post("");
            }
            else{
                post("");
                pd_error(x, "No documentation found!");
                post("");
            }
        }
        else{
            post("");
            pd_error(x, "No documentation found!");
            post("");
        }
    }
}

// ====================================
// ====================================
// ====================================

void pd4py_system_func (const char *command){
    int result = system(command);
    if (result == -1){
        post("Error: %s", command);
    }
}

// ============================================
// ============================================

static void create(t_py *x, t_symbol *s, int argc, t_atom *argv){
    // If Windows OS run, if not then warn the user
    (void)s;
    (void)argc;
    
    const char *script_name = argv[0].a_w.w_symbol->s_name;
    post("Opening vscode...");
    #ifdef _WIN64 // ERROR: the endif is missing directive _WIN64

    char *command = malloc(strlen(x->home_path->s_name) + strlen(script_name) + 20);
    sprintf(command, "/c code %s/%s.py", x->home_path->s_name, script_name);
    SHELLEXECUTEINFO sei = {0};
    sei.cbSize = sizeof(sei);
    sei.fMask = SEE_MASK_NOCLOSEPROCESS;
    // sei.lpVerb = "open";
    
    sei.lpFile = "cmd.exe ";
    sei.lpParameters = command;
    sei.nShow = SW_HIDE;
    ShellExecuteEx(&sei);
    CloseHandle(sei.hProcess);
    free(command);
    return;

    // Not Windows OS

    #else // if not windows 64bits
    char *command = malloc(strlen(x->home_path->s_name) + strlen(script_name) + 20);
    sprintf(command, "code %s/%s.py", x->home_path->s_name, script_name);

    pd4py_system_func(command);
    pd_error(x, "Not tested in your Platform, please send me a bug report!");
    return;
    #endif
}

// ====================================
// ====================================
// ====================================

static void vscode(t_py *x){
    // If Windows OS run, if not then warn the user
       
    if (x->function_called == 0) { // if the set method was not called, then we can not run the function :)
        pd_error(x, "To open vscode you need to set the function first!");
        return;
    }
    post("Opening vscode...");
    
    #ifdef _WIN64 // ERROR: the endif is missing directive _WIN64
    char *command = malloc(strlen(x->home_path->s_name) + strlen(x->script_name->s_name) + 20);
    sprintf(command, "/c code %s/%s.py", x->home_path->s_name, x->script_name->s_name);
    SHELLEXECUTEINFO sei = {0};
    sei.cbSize = sizeof(sei);
    sei.fMask = SEE_MASK_NOCLOSEPROCESS;
    // sei.lpVerb = "open";
    sei.lpFile = "cmd.exe ";
    sei.lpParameters = command;
    sei.nShow = SW_HIDE;
    ShellExecuteEx(&sei);
    CloseHandle(sei.hProcess);

    return;

    // Not Windows OS

    #else // if not windows 64bits
    char *command = malloc(strlen(x->home_path->s_name) + strlen(x->script_name->s_name) + 20);
    sprintf(command, "code %s/%s.py", x->home_path->s_name, x->script_name->s_name);
    pd4py_system_func(command);
    pd_error(x, "Not tested in your Platform, please send me a bug report!");
    return;
    #endif
}

// ====================================
// ====================================
// ====================================

static void reload(t_py *x){
    PyObject *pName, *pFunc, *pModule, *pReload;
    if (x->function_called == 0) { // if the set method was not called, then we can not run the function :)
        pd_error(x, "To reload the script you need to set the function first!");
        return;
    }
    pFunc = x->function;
    pModule = x->module;

    // reload the module
    pName = PyUnicode_DecodeFSDefault(x->script_name->s_name); // Name of script file
    pModule = PyImport_Import(pName);
    pReload = PyImport_ReloadModule(pModule);
    if (pReload == NULL) {
        pd_error(x, "Error reloading the module!");
        x->function_called = 0;
        Py_DECREF(x->function);
        Py_DECREF(x->module);
        return;
    } else{
        Py_XDECREF(x->module);
        pFunc = PyObject_GetAttrString(pModule, x->function_name->s_name); // Function name inside the script file
        Py_DECREF(pName); // DOC: Py_DECREF(pName) is necessary?
        if (pFunc && PyCallable_Check(pFunc)){ // Check if the function exists and is callable 
            x->function = pFunc;
            x->module = pModule;
            x->script_name = x->script_name;
            x->function_name = x->function_name; // why 
            x->function_called = malloc(sizeof(int)); // TODO: Better way to solve the warning???
            *(x->function_called) = 1; // 
            post("The module was reloaded!");
            return; 
        }
        else{
            pd_error(x, "Error reloading the module!");
            x->function_called = 0;
            Py_DECREF(x->function);
            Py_DECREF(x->module);
            return;
        }
    } 
}

// ====================================
// ====================================
// ====================================

static void set_function(t_py *x, t_symbol *s, int argc, t_atom *argv){
    t_symbol *script_file_name = atom_gensym(argv+0);
    t_symbol *function_name = atom_gensym(argv+1);
    
    (void)s;
    // Erros handling
    // Check if script has .py extension
    char *extension = strrchr(script_file_name->s_name, '.');
    if (extension != NULL) {
        pd_error(x, "Please dont use extensions in the script file name!");
        return;
    }

    // check if script file exists
    char script_file_path[MAXPDSTRING];
    snprintf(script_file_path, MAXPDSTRING, "%s/%s.py", x->home_path->s_name, script_file_name->s_name);
    if (access(script_file_path, F_OK) == -1) {
        pd_error(x, "The script file %s does not exist!", script_file_path);
        return;
    }
    
    if (x->function_name != NULL){
        int function_is_equal = strcmp(function_name->s_name, x->function_name->s_name);
        if (function_is_equal == 0){    // If the user wants to call the same function again! This is not necessary at first glance. 
            pd_error(x, "WARNING :: The function was already called!");
            pd_error(x, "WARNING :: Calling the function again! This make it slower!");
            post("");
            return;
        }
        else{ // DOC: If the function is different, then we need to delete the old function and create a new one.
            Py_XDECREF(x->function);
            Py_XDECREF(x->module);
            x->function = NULL;
            x->module = NULL;
            x->function_name = NULL;
        }      
    }

    // =====================
    // DOC: check number of arguments
    if (argc < 2) { // check is the number of arguments is correct | set "function_script" "function_name"
        pd_error(x,"py4pd :: The set message needs two arguments! The 'Script name' and the 'function name'!");
        return;
    }
    
    PyObject *pName, *pModule, *pFunc;

    const wchar_t *py_name_ptr;
    // Copilot: Define program name in py_name_ptr
    py_name_ptr = Py_DecodeLocale(script_file_name->s_name, NULL);
    Py_SetProgramName(py_name_ptr); // set program name
    Py_Initialize();
    Py_GetPythonHome();

    // =====================
    const char *home_path_str = x->home_path->s_name;
    char *sys_path_str = malloc(strlen(home_path_str) + strlen("sys.path.append('") + strlen("')") + 1);
    sprintf(sys_path_str, "sys.path.append('%s')", home_path_str);
    PyRun_SimpleString("import sys");
    PyRun_SimpleString(sys_path_str);
    free(sys_path_str); // free the memory allocated for the string, check if this is necessary
    
    // =====================
    // DOC: Set the packages path
    const char *site_path_str = x->packages_path->s_name;
    PyObject* sys = PyImport_ImportModule("sys");
    PyObject* sys_path = PyObject_GetAttrString(sys, "path");
    PyList_Append(sys_path, PyUnicode_FromString(site_path_str));
    Py_DECREF(sys_path);
    Py_DECREF(sys);
    
    // =====================
    pName = PyUnicode_DecodeFSDefault(script_file_name->s_name); // Name of script file
    pModule = PyImport_Import(pName);
    // =====================

    // check if module is NULL
    if (pModule == NULL) {
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "Call failed: %s", PyUnicode_AsUTF8(pstr));
        return;
    }
 
    pFunc = PyObject_GetAttrString(pModule, function_name->s_name); // Function name inside the script file
    Py_DECREF(pName); // DOC: Py_DECREF(pName) is not necessary! 
    if (pFunc && PyCallable_Check(pFunc)){ // Check if the function exists and is callable   
        PyObject *inspect=NULL, *getargspec=NULL, *argspec=NULL, *args=NULL;
        inspect = PyImport_ImportModule("inspect");
        getargspec = PyObject_GetAttrString(inspect, "getargspec");
        argspec = PyObject_CallFunctionObjArgs(getargspec, pFunc, NULL);
        args = PyObject_GetAttrString(argspec, "args");
        int py_args = PyObject_Size(args);
        post("py4pd | function '%s' loaded!", function_name->s_name);
        post("");
        post("It has %i arguments!", py_args);
        post("");
        x->function = pFunc;
        x->module = pModule;
        x->script_name = script_file_name;
        x->function_name = function_name; // why 
        x->function_called = malloc(sizeof(int)); // TODO: Better way to solve the warning???
        *(x->function_called) = 1; // 
        return;

    } else {
        // post PyErr_Print() in pd
        pd_error(x, "py4pd | function %s not loaded!", function_name->s_name);
        x->function_called = 0; // set the flag to 0 because it crash Pd if user try to use args method
        x->function_name = NULL;
        post("");
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "Call failed:\n %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        return;
    }
}

// ============================================
// ============================================
// ============================================

static void run(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    
    if (x->function_called == 0) { // if the set method was not called, then we can not run the function :)
        pd_error(x, "You need to send a message ||| 'set {script} {function}'!");
        return;
    }
    PyObject *pFunc, *pArgs, *pValue; // pDict, *pModule,
    pFunc = x->function;
    pArgs = PyTuple_New(argc);
    int i;

    // DOC: CONVERTION TO PYTHON OBJECTS

    for (i = 0; i < argc; ++i) {
    
        // NUMBERS 
        if (argv[i].a_type == A_FLOAT) { 
            float arg_float = atom_getfloat(argv+i);
            if (arg_float == (int)arg_float){ // DOC: If the float is an integer, then convert to int
                int arg_int = (int)arg_float;
                pValue = PyLong_FromLong(arg_int);
            }
            else{ // If the int is an integer, then convert to int
                pValue = PyFloat_FromDouble(arg_float);
            }

        // STRINGS
        } else if (argv[i].a_type == A_SYMBOL) {
            pValue = PyUnicode_DecodeFSDefault(argv[i].a_w.w_symbol->s_name); // convert to python string
        } else {
            pValue = Py_None;
            Py_INCREF(Py_None);
        }

        // TODO: Lists
        // ERROR IF THE ARGUMENT IS NOT A NUMBER OR A STRING       
        if (!pValue) {
            pd_error(x, "Cannot convert argument\n");
            return;
        }
        // DOC: END OF CONVERTION TO PYTHON OBJECTS

        PyTuple_SetItem(pArgs, i, pValue);
    }

    pValue = PyObject_CallObject(pFunc, pArgs);
    if (pValue != NULL) {                       // DOC: if the function returns a value   
        // check if pValue is a list
        if (PyList_Check(pValue)){ // DOC: If the function return a list list
            int list_size = PyList_Size(pValue);
            // make array with size of list_size
            t_atom *list_array = (t_atom *) malloc(list_size * sizeof(t_atom));            
            for (i = 0; i < list_size; ++i) {
                PyObject *pValue_i = PyList_GetItem(pValue, i);
                if (PyLong_Check(pValue_i)) {
                    long result = PyLong_AsLong(pValue_i);
                    float result_float = (float)result;
                    list_array[i].a_type = A_FLOAT;
                    list_array[i].a_w.w_float = result_float;

                } else if (PyFloat_Check(pValue_i)) {
                    double result = PyFloat_AsDouble(pValue_i);
                    float result_float = (float)result;
                    list_array[i].a_type = A_FLOAT;
                    list_array[i].a_w.w_float = result_float;
                } else if (PyUnicode_Check(pValue_i)) {
                    const char *result = PyUnicode_AsUTF8(pValue_i); // WARNING: initialization discards 'const' qualifier 
                                                                     // from pointer target type [-Wdiscarded-qualifiers]
                    list_array[i].a_type = A_SYMBOL;
                    list_array[i].a_w.w_symbol = gensym(result);
                } else if (PyList_Check(pValue_i)) {
                    // convert_list_inside_list(x, pValue_i, list_array+i, list_size);
                    pd_error(x, "recursive call not implemented yet");
                    //list_array[i].a_type = A_FLOAT;
                    //list_array[i].a_w.w_float = 0;
                } else {
                    pd_error(x, "Cannot convert list item\n");
                    Py_DECREF(pValue);
                    return;
                }
            }
            outlet_list(x->out_A, 0, list_size, list_array); // The loop seems slow :(. TODO: possible do in other way?
        } else {
            if (PyLong_Check(pValue)) {
                long result = PyLong_AsLong(pValue);
                outlet_float(x->out_A, result);
            } else if (PyFloat_Check(pValue)) {
                double result = PyFloat_AsDouble(pValue);
                float result_float = (float)result;
                outlet_float(x->out_A, result_float);
                // outlet_float(x->out_A, result);
            } else if (PyUnicode_Check(pValue)) {
                const char *result = PyUnicode_AsUTF8(pValue); // WARNING: See http://gg.gg/11t8iv
                outlet_symbol(x->out_A, gensym(result)); // adding const in char solve this, but I don't understand why it works! :)
            } else { 
                // check if pValue is a list.    
                // if yes, accumulate and output it using out_A 
                pd_error(x, "Cannot convert list item\n");
                Py_DECREF(pValue);
                return;
                }
        }
        Py_DECREF(pValue);
    }
    else { // DOC: if the function returns a error
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "Call failed: %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        return;
    }
}

// ============================================
// =========== CREATE WIN THREAD ==============
// ============================================

struct thread_arg_struct {
    t_py x;
    t_symbol s;
    int argc;
    t_atom *argv;
} thread_arg;


// ============================================
// ============================================
// ============================================
#ifdef _WIN64

DWORD WINAPI ThreadFunc(LPVOID lpParam) {
    struct thread_arg_struct *arg = (struct thread_arg_struct *)lpParam;
    // define x, s, argc and argv
    t_py *x = &arg->x;
    t_symbol *s = &arg->s;
    int argc = arg->argc;
    t_atom *argv = arg->argv;

    // post argv = argv
    post("argv = %s", argv);
    
    if (x->function_called == 0) { // if the set method was not called, then we can not run the function :)
        pd_error(x, "You need to send a message ||| 'set {script} {function}'!");
        return 0;
    }
    PyObject *pFunc, *pArgs, *pValue; // pDict, *pModule,
    pFunc = x->function;
    pArgs = PyTuple_New(argc);
    int i;

    // DOC: CONVERTION TO PYTHON OBJECTS

    for (i = 0; i < argc; ++i) {
    
        // NUMBERS 
        if (argv[i].a_type == A_FLOAT) { 
            float arg_float = atom_getfloat(argv+i);
            if (arg_float == (int)arg_float){ // DOC: If the float is an integer, then convert to int
                int arg_int = (int)arg_float;
                pValue = PyLong_FromLong(arg_int);
            }
            else{ // If the int is an integer, then convert to int
                pValue = PyFloat_FromDouble(arg_float);
            }

        // STRINGS
        } else if (argv[i].a_type == A_SYMBOL) {
            pValue = PyUnicode_DecodeFSDefault(argv[i].a_w.w_symbol->s_name); // convert to python string
        } else {
            pValue = Py_None;
            Py_INCREF(Py_None);
        }

        // TODO: Lists
        // ERROR IF THE ARGUMENT IS NOT A NUMBER OR A STRING       
        if (!pValue) {
            pd_error(x, "Cannot convert argument\n");
            return 0;
        }
        // DOC: END OF CONVERTION TO PYTHON OBJECTS

        PyTuple_SetItem(pArgs, i, pValue);
    }

    pValue = PyObject_CallObject(pFunc, pArgs);
    if (pValue != NULL) {                       // DOC: if the function returns a value   
        // check if pValue is a list
        if (PyList_Check(pValue)){ // DOC: If the function return a list list
            int list_size = PyList_Size(pValue);
            // make array with size of list_size
            t_atom *list_array = (t_atom *) malloc(list_size * sizeof(t_atom));            
            for (i = 0; i < list_size; ++i) {
                PyObject *pValue_i = PyList_GetItem(pValue, i);
                if (PyLong_Check(pValue_i)) {
                    long result = PyLong_AsLong(pValue_i);
                    float result_float = (float)result;
                    list_array[i].a_type = A_FLOAT;
                    list_array[i].a_w.w_float = result_float;

                } else if (PyFloat_Check(pValue_i)) {
                    double result = PyFloat_AsDouble(pValue_i);
                    float result_float = (float)result;
                    list_array[i].a_type = A_FLOAT;
                    list_array[i].a_w.w_float = result_float;
                } else if (PyUnicode_Check(pValue_i)) {
                    const char *result = PyUnicode_AsUTF8(pValue_i); // WARNING: initialization discards 'const' qualifier 
                                                                     // from pointer target type [-Wdiscarded-qualifiers]
                    list_array[i].a_type = A_SYMBOL;
                    list_array[i].a_w.w_symbol = gensym(result);
                } else if (PyList_Check(pValue_i)) {
                    // convert_list_inside_list(x, pValue_i, list_array+i, list_size);
                    pd_error(x, "recursive call not implemented yet");
                    //list_array[i].a_type = A_FLOAT;
                    //list_array[i].a_w.w_float = 0;
                } else {
                    pd_error(x, "Cannot convert list item\n");
                    Py_DECREF(pValue);
                    return 0;
                }
            }
            outlet_list(x->out_A, 0, list_size, list_array); // The loop seems slow :(. TODO: possible do in other way?
        } else {
            if (PyLong_Check(pValue)) {
                long result = PyLong_AsLong(pValue);
                outlet_float(x->out_A, result);
            } else if (PyFloat_Check(pValue)) {
                double result = PyFloat_AsDouble(pValue);
                float result_float = (float)result;
                outlet_float(x->out_A, result_float);
                // outlet_float(x->out_A, result);
            } else if (PyUnicode_Check(pValue)) {
                const char *result = PyUnicode_AsUTF8(pValue); // WARNING: See http://gg.gg/11t8iv
                outlet_symbol(x->out_A, gensym(result)); // adding const in char solve this, but I don't understand why it works! :)
            } else { 
                // check if pValue is a list.    
                // if yes, accumulate and output it using out_A 
                pd_error(x, "Cannot convert list item\n");
                Py_DECREF(pValue);
                return 0;
                }
        }
        Py_DECREF(pValue);
    }
    else { // DOC: if the function returns a error
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "Call failed: %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        return 0;
    }
}



// ============================================
// ============================================
// ============================================

static void create_thread(t_py *x, t_symbol *s, int argc, t_atom *argv){
    DWORD threadID;
    HANDLE hThread;
    // TODO: This is not working. Why?
    struct thread_arg_struct *arg = (struct thread_arg_struct *)malloc(sizeof(struct thread_arg_struct));
    arg->x = *x;
    arg->argc = argc;
    arg->argv = argv;

    // Pd is crashing when I try to create a thread.

    hThread = CreateThread(NULL, 0, ThreadFunc, arg, 0, &threadID);

    if (hThread == NULL) {
        pd_error(x, "CreateThread failed");
        return;
    } else {
        return;
    }
}

#endif

// ============================================
// =========== SETUP OF OBJECT ================
// ============================================

void *py4pd_new(t_symbol *s, int argc, t_atom *argv){ 

    t_py *x = (t_py *)pd_new(py4pd_class);
    // credits
    post("");
    post("");
    post("");
    post("py4pd by Charles K. Neimog");
    post("version 0.0.1        ");
    post("Based on Python 3.10.5  ");
    post("");
    post("It is inspired by the work of Thomas Grill and SOPI research group.");
    post("");

    // pd things
    // pointer para a classe
    x->x_canvas = canvas_getcurrent(); // pega o canvas atual
    x->out_A = outlet_new(&x->x_obj, &s_anything); // cria um outlet
    
    // ========
    // py things
    t_canvas *c = x->x_canvas; 
    x->home_path = canvas_getdir(c);     // set name 
    x->packages_path = canvas_getdir(c); // set name

    char *pip = malloc(strlen(x->home_path->s_name) + strlen("%s/py4pd_packages/Scripts/pip.exe") + 40);
    sprintf(pip, "%s/py4pd_packages/Scripts/pip.exe", x->home_path->s_name);
    if (access(pip, F_OK) == -1)
        post("Enviroment not found");
    else{
        char *packages = malloc(strlen(x->home_path->s_name) + strlen("%s/Lib/site-packages/") + 40);
        sprintf(packages, "%s/py4pd_packages/Lib/site-packages/", x->home_path->s_name);
        post("packages: %s", packages);
        // set x->packages_path to packages
        x->packages_path = gensym(packages);
        
    }
    free(pip);
    // get arguments and print it
    if (argc == 2) {
        set_function(x, s, argc, argv);
        x->function_called = malloc(sizeof(int)); // TODO: Better way to solve the warning???
        *(x->function_called) = 1;
    } 
    return(x);
}


// ============================================
// =========== REMOVE OBJECT ==================
// ============================================

void py4pd_free(t_py *x){
    PyObject  *pModule, *pFunc; // pDict, *pName,
    pFunc = x->function;
    pModule = x->module;
    if (pModule != NULL) {
        Py_DECREF(pModule);
    }
    if (pFunc != NULL) {
        Py_DECREF(pFunc);
    }
    if (Py_FinalizeEx() < 0) {
        return;
    }
}

// ====================================================
void py4pd_setup(void){
    py4pd_class =     class_new(gensym("py4pd"), // cria o objeto quando escrevemos py4pd
                        (t_newmethod)py4pd_new, // metodo de criação do objeto             
                        (t_method)py4pd_free, // quando voce deleta o objeto
                        sizeof(t_py), // quanta memoria precisamos para esse objeto
                        CLASS_DEFAULT, // nao há uma GUI especial para esse objeto???
                        A_GIMME, // o argumento é um símbolo
                        0); // todos os outros argumentos por exemplo um numero seria A_DEFFLOAT
    
    class_addmethod(py4pd_class, (t_method)home, gensym("home"), A_GIMME, 0);
    class_addmethod(py4pd_class, (t_method)create_thread, gensym("thread"), A_GIMME, 0);
    class_addmethod(py4pd_class, (t_method)vscode, gensym("click"), 0, 0);
    class_addmethod(py4pd_class, (t_method)packages, gensym("packages"), A_GIMME, 0);
    class_addmethod(py4pd_class, (t_method)env_install, gensym("env_install"), 0, 0);
    class_addmethod(py4pd_class, (t_method)pip_install, gensym("pip"), 0, 0);
    class_addmethod(py4pd_class, (t_method)vscode, gensym("vscode"), 0, 0);
    class_addmethod(py4pd_class, (t_method)reload, gensym("reload"), 0, 0);
    class_addmethod(py4pd_class, (t_method)create, gensym("create"), A_GIMME, 0);
    class_addmethod(py4pd_class, (t_method)documentation, gensym("documentation"), 0, 0);
    class_addmethod(py4pd_class, (t_method)set_function, gensym("set"), A_GIMME, 0);
    class_addmethod(py4pd_class, (t_method)run, gensym("run"), A_GIMME, 0); // TODO: better name for this method
    // class_addmethod(py4pd_class, (t_method)amount_of_args, gensym("args"), A_GIMME, 0); // TODO: better name for this method
    }

