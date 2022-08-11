// =================================
// https://github.com/pure-data/pure-data/src/x_gui.c 
#include <m_pd.h>
#include <g_canvas.h>
#include <stdio.h>
#include <string.h>
#include <complex.h>


// If windows 64bits include 
#ifdef _WIN64
#include <windows.h>
#else 
#include <pthread.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif



// Python include
#include <Python.h>

/* 
TODO: Way to set global variables, I think that will be important for things like general path (lilypond, etc)
TODO: Reset the function (like panic for sfont~), In some calls seems that the *function become NULL? WARNING: Possible error of logic
TODO: make function home work with spaces, mainly for Windows OS where the use of lilypond in python need to be specified with spaces
TODO: Return list from python in all run functions
TODO: Add some way to run list how arguments 
TODO: If the run method set before the end of the thread, there is an error, that close all PureData.
TODO: The set function need to run after the load of the object, but before the start of the thread???
        It does not load external modules in the pd_new.

*/

// GLOBAL VARIABLE FOR THREAD, 0 = NO THREAD, 1 = THREAD RUNNING.

static int thread_running = 0; // 0 = NO THREAD, 1 = THREAD RUNNING.

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
    t_float         *thread; // arguments
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
        post("[py4pd] The home path is: %s", x->home_path->s_name);
        return; 
    } else {
        x->home_path = atom_getsymbol(argv);
        post("[py4pd] The home path set to: %s", x->home_path->s_name);
    }
    
}

// // ============================================
// // ============================================
// // ============================================

static void packages(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s; 
    if (argc < 1) {
        post("[py4pd] The packages path is: %s", x->packages_path->s_name);
        return; // is this necessary?
    }
    else {
        if (argc < 2 && argc > 0){
            x->packages_path = atom_getsymbol(argv);
            // probe if the path is valid
            if (access(x->packages_path->s_name, F_OK) != -1) {
                // do nothing
            } else {
                pd_error(x, "The packages path is not valid");
            }
        }   
        else{
            pd_error(x, "It seems that your package folder has |spaces|.");
            return;
        }    
    }
}

// ====================================
// ====================================
// ====================================

#ifdef _WIN64

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
    
    char *package = malloc(strlen(atom_getsymbol(argv+0)->s_name) + 1);
    strcpy(package, atom_getsymbol(argv+0)->s_name);

    char *pip = malloc(strlen(x->home_path->s_name) + strlen("%s/py4pd_packages/Scripts/pip.exe") + 40);
    sprintf(pip, "%s/py4pd_packages/Scripts/pip.exe", x->home_path->s_name);
    if (access(pip, F_OK) == -1) {
        pd_error(x, "The pip path does not exist. Send a message {env_install} to install pip first!");
        return;
    } else{
        char *pip_cmd = malloc(strlen(x->packages_path->s_name) + strlen("py4pd") + 20);
        sprintf(pip_cmd, "/c %s install %s", pip, package);
        post("[py4pd] Installing %s", package);
        SHELLEXECUTEINFO sei = {0};
        sei.cbSize = sizeof(sei);
        sei.fMask = SEE_MASK_NOCLOSEPROCESS;
        sei.lpFile = "cmd.exe ";
        sei.lpParameters = pip_cmd;
        sei.nShow = SW_HIDE;
        ShellExecuteEx(&sei);
        CloseHandle(sei.hProcess);
        post("[py4pd] %s installed!", package);
        return;
    }
}

#endif

// ====================================
// ====================================
// ====================================

static void documentation(t_py *x){
    PyObject *pFunc;
    if (x->function_called == 0) { // if the set method was not called, then we can not run the function :)
        pd_error(x, "[py4pd] To see the documentaion you need to set the function first!");
        return;
    }

    pFunc = x->function;
    if (pFunc && PyCallable_Check(pFunc)){ // Check if the function exists and is callable
        PyObject *pDoc = PyObject_GetAttrString(pFunc, "__doc__"); // Get the documentation of the function
        if (pDoc != NULL){
            const char *Doc = PyUnicode_AsUTF8(pDoc); 
            if (Doc != NULL){
                post("");
                post("[+][+] %s [+][+]", x->function_name->s_name);
                post("");
                post("%s", Doc);
                post("");
            }
            else{

                post("");
                pd_error(x, "[py4pd] No documentation found!");
                post("");
            }
        }
        else{
            post("");
            pd_error(x, "[py4pd] No documentation found!");
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
        post("[py4pd] %s", command);
    }
}

// ============================================
// ============================================

static void create(t_py *x, t_symbol *s, int argc, t_atom *argv){
    // If Windows OS run, if not then warn the user
    (void)s;
    (void)argc;
    
    const char *script_name = argv[0].a_w.w_symbol->s_name;
    post("[py4pd] Opening vscode...");
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
    return;
    #endif
}

// ====================================
// ====================================
// ====================================

static void vscode(t_py *x){
    // If Windows OS run, if not then warn the user
       
    if (x->function_called == 0) { // if the set method was not called, then we can not run the function :)
        pd_error(x, "[py4pd] To open vscode you need to set the function first!");
        return;
    }
    post("[py4pd] Opening vscode...");
    
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
        Py_DECREF(pName); // 
        if (pFunc && PyCallable_Check(pFunc)){ // Check if the function exists and is callable 
            x->function = pFunc;
            x->module = pModule;
            x->script_name = x->script_name;
            x->function_name = x->function_name; // why 
            x->function_called = malloc(sizeof(int)); 
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
    (void)s;

    t_symbol *script_file_name = atom_gensym(argv+0);
    t_symbol *function_name = atom_gensym(argv+1);
    
    // Check if the already was set
    if (x->function_name != NULL){
        int function_is_equal = strcmp(function_name->s_name, x->function_name->s_name);
        if (function_is_equal == 0){    // If the user wants to call the same function again! This is not necessary at first glance. 
            pd_error(x, "[py4pd] The function was already called!");
            Py_XDECREF(x->function);
            Py_XDECREF(x->module);
            x->function = NULL;
            x->module = NULL;
            x->function_name = NULL;
        }
        else{ // DOC: If the function is different, then we need to delete the old function and create a new one.
            Py_XDECREF(x->function);
            Py_XDECREF(x->module);
            x->function = NULL;
            x->module = NULL;
            x->function_name = NULL;
        }      
    }

    // Infos
    post("[py4pd] Script file: %s.py", script_file_name->s_name);
    post("[py4pd] Function name: %s", function_name->s_name);

    // Erro handling
    // Check if script has .py extension
    char *extension = strrchr(script_file_name->s_name, '.');
    if (extension != NULL) {
        pd_error(x, "[py4pd] Don't use extensions in the script file name!");
        return;
    }

    // check if script file exists
    char script_file_path[MAXPDSTRING];
    snprintf(script_file_path, MAXPDSTRING, "%s/%s.py", x->home_path->s_name, script_file_name->s_name);
    if (access(script_file_path, F_OK) == -1) {
        pd_error(x, "The script file %s does not exist!", script_file_path);
        return;
    }
    
    

    // =====================
    // DOC: check number of arguments
    if (argc < 2) { // check is the number of arguments is correct | set "function_script" "function_name"
        pd_error(x,"[py4pd] {set} message needs two arguments! The 'Script name' and the 'function name'!");
        return;
    }
    
    PyObject *pName, *pModule, *pFunc; // DOC: Create the variables of the python objects

    const wchar_t *py_name_ptr;
    py_name_ptr = Py_DecodeLocale(script_file_name->s_name, NULL);
    Py_SetProgramName(py_name_ptr); // set program name
    Py_Initialize();


    // =====================
    // DOC: Set the packages path
    const char *site_path_str = x->packages_path->s_name;
    char *add_packages = malloc(strlen(site_path_str) + strlen("sys.path.append('") + strlen("')") + 1);
    sprintf(add_packages, "sys.path.append('%s')", site_path_str);
    PyRun_SimpleString("import sys");
    PyRun_SimpleString(add_packages);
    free(add_packages);

    // =====================
    const char *home_path_str = x->home_path->s_name;
    char *sys_path_str = malloc(strlen(home_path_str) + strlen("sys.path.append('") + strlen("')") + 1);
    sprintf(sys_path_str, "sys.path.append('%s')", home_path_str);
    PyRun_SimpleString(sys_path_str);
    free(sys_path_str); // free the memory allocated for the string, check if this is necessary
    
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
        pd_error(x, "[py4pd] Call failed: %s", PyUnicode_AsUTF8(pstr));
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
        post("[py4pd] Function loaded!");
        post("[py4pd] It has %i arguments!", py_args);
        x->function = pFunc;
        x->module = pModule;
        x->script_name = script_file_name;
        x->function_name = function_name; 
        x->function_called = malloc(sizeof(int)); 
        *(x->function_called) = 1; // 
        return;

    } else {
        // post PyErr_Print() in pd
        pd_error(x, "Function %s not loaded!", function_name->s_name);
        x->function_called = 0; // set the flag to 0 because it crash Pd if user try to use args method
        x->function_name = NULL;
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

static void run_function(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    
    if (x->function_called == 0) { // if the set method was not called, then we can not run the function :)
        pd_error(x, "[py4pd] The message need to be formatted like 'set {script_name} {function_name}'!");
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
        // ERROR IF THE ARGUMENT IS NOT A NUMBER OR A STRING       
        if (!pValue) {
            pd_error(x, "Cannot convert argument\n");
            return;
        }
        // DOC: END OF CONVERTION TO PYTHON OBJECTS

        PyTuple_SetItem(pArgs, i, pValue);
    }

    pValue = PyObject_CallObject(pFunc, pArgs);
    if (pValue != NULL) {                                // DOC: if the function returns a value   
        if (PyList_Check(pValue)){                       // DOC: If the function return a list list
            int list_size = PyList_Size(pValue);
            t_atom *list_array = (t_atom *) malloc(list_size * sizeof(t_atom));            
            for (i = 0; i < list_size; ++i) {
                PyObject *pValue_i = PyList_GetItem(pValue, i);
                if (PyLong_Check(pValue_i)) {            // DOC: If the function return a list of integers
                    long result = PyLong_AsLong(pValue_i);
                    float result_float = (float)result;
                    list_array[i].a_type = A_FLOAT;
                    list_array[i].a_w.w_float = result_float;

                } else if (PyFloat_Check(pValue_i)) {    // DOC: If the function return a list of floats
                    double result = PyFloat_AsDouble(pValue_i);
                    float result_float = (float)result;
                    list_array[i].a_type = A_FLOAT;
                    list_array[i].a_w.w_float = result_float;
                } else if (PyUnicode_Check(pValue_i)) {  // DOC: If the function return a list of strings
                    const char *result = PyUnicode_AsUTF8(pValue_i); 
                    list_array[i].a_type = A_SYMBOL;
                    list_array[i].a_w.w_symbol = gensym(result);
                } else {
                    pd_error(x, "[py4pd] py4pd just convert int, float and string!\n");
                    pd_error(x, "INFO  [!] The value received is of type %s", Py_TYPE(pValue)->tp_name);
                    Py_DECREF(pValue);
                    return;
                }
            }
            outlet_list(x->out_A, 0, list_size, list_array); // TODO: possible do in other way? Seems slow!
        } else {
            if (PyLong_Check(pValue)) {
                long result = PyLong_AsLong(pValue); // DOC: If the function return a integer
                outlet_float(x->out_A, result);
            } else if (PyFloat_Check(pValue)) {
                double result = PyFloat_AsDouble(pValue); // DOC: If the function return a float
                float result_float = (float)result;
                outlet_float(x->out_A, result_float);
                // outlet_float(x->out_A, result);
            } else if (PyUnicode_Check(pValue)) {
                const char *result = PyUnicode_AsUTF8(pValue); // DOC: If the function return a string
                outlet_symbol(x->out_A, gensym(result)); 
            } else {
                pd_error(x, "[py4pd] py4pd just convert int, float and string!\n");
                pd_error(x, "INFO  [!] The value received is of type %s", Py_TYPE(pValue)->tp_name);
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
        pd_error(x, "[py4pd] Call failed: %s", PyUnicode_AsUTF8(pstr));
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

DWORD WINAPI ThreadFunc(LPVOID lpParam) { // DOC: Thread function in Windows
    struct thread_arg_struct *arg = (struct thread_arg_struct *)lpParam;       
    t_py *x = &arg->x;
    int argc = arg->argc;
    t_symbol *s = &arg->s;
    t_atom *argv = arg->argv;
    thread_running = 1;
    run_function(x, s, argc, argv);
    thread_running = 0;
    return 0;
}
// ============================================
// ============================================
// ============================================

static void create_thread(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    DWORD threadID;
    HANDLE hThread;
    struct thread_arg_struct *arg = (struct thread_arg_struct *)malloc(sizeof(struct thread_arg_struct));
    // arg->x must be equal x not *x
    arg->x = *x;
    arg->argc = argc;
    arg->argv = argv;

    if (x->function_called == 0) {
        // Pd is crashing when I try to create a thread.
        pd_error(x, "[py4pd] You need to call a function before run!");
        free(arg);
        return;
        } 
    else {
        if (thread_running == 0){
            hThread = CreateThread(NULL, 0, ThreadFunc, arg, 0, &threadID);
            if (hThread == NULL) {
                pd_error(x, "[py4pd] CreateThread failed");
                free(arg);
                return;
            }
        } else {
            pd_error(x, "[py4pd] Just one thread can be running at a time!");
            free(arg);
            return;
        }
    }
}

// ============================================
// ============= UNIX =========================
// ============================================

// If OS is Linux or Mac OS then use this function
#else

// what is the linux equivalent for Lvoid Parameter(void *lpParameter)
static void *ThreadFunc(void *lpParameter) {
    struct thread_arg_struct *arg = (struct thread_arg_struct *)lpParameter;
    t_py *x = &arg->x;
    t_symbol *s = &arg->s;
    int argc = arg->argc;
    t_atom *argv = arg->argv;
    thread_running = 1;
    run_function(x, s, argc, argv);
    thread_running = 0;
    return NULL;
}

// create_thread in Linux
static void create_thread(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    struct thread_arg_struct *arg = (struct thread_arg_struct *)malloc(sizeof(struct thread_arg_struct));
    arg->x = *x;
    arg->argc = argc;
    arg->argv = argv;
    if (x->function_called == 0) {
        // Pd is crashing when I try to create a thread.
        pd_error(x, "[py4pd] You need to call a function before run!");
        free(arg);
        return;
    }
    else {
        if (thread_running == 0){
            pthread_t thread;
            pthread_t hThread;
            hThread = pthread_create(&thread, NULL, ThreadFunc, arg);
            // check if hThread is equal to 0
            if (hThread != 0) {
                pd_error(x, "[py4pd] CreateThread failed");
                free(arg);
                return;
            }
        } else {
            pd_error(x, "[py4pd] Just one thread can be running at a time!");
            free(arg);
            return;
        }
    }
}

#endif

// ============================================
// ============================================
// ============================================

static void run(t_py *x, t_symbol *s, int argc, t_atom *argv){
    // convert pointer x->thread to a int
    int thread = *(x->thread);
    if (thread == 1) {
        create_thread(x, s, argc, argv);
    } else if (thread == 2) {
        run_function(x, s, argc, argv);
    } else {
        pd_error(x, "[py4pd] Thread not created");
    }
}

// ============================================
// ============================================
// ============================================

static void thread(t_py *x, t_floatarg f){
    int thread = (int)f;
    if (thread == 1) {
        post("[py4pd] Threading enabled");
        x->thread = malloc(sizeof(int)); 
        *(x->thread) = 1; // 
        return;
    } else if (thread == 0) {
        x->thread = malloc(sizeof(int)); 
        *(x->thread) = 2; // 
        post("[py4pd] Threading disabled");
        return;
    } else {
        pd_error(x, "[py4pd] Threading status must be 0 or 1");
    }
}


// ============================================
// =========== SETUP OF OBJECT ================
// ============================================

void *py4pd_new(t_symbol *s, int argc, t_atom *argv){ 

    t_py *x = (t_py *)pd_new(py4pd_class);
    // credits
    post("");
    post("[py4pd] py4pd by Charles K. Neimog");
    post("[py4pd] version 0.0.1        ");
    post("[py4pd] Based on Python 3.10.5  ");
    post("[py4pd] Inspired by the work of Thomas Grill and SOPI research group.");

    // pd things
    // pointer para a classe
    x->x_canvas = canvas_getcurrent(); // pega o canvas atual
    x->out_A = outlet_new(&x->x_obj, &s_anything); // cria um outlet
    t_canvas *c = x->x_canvas; 
    x->home_path = canvas_getdir(c);     // set name 
    x->packages_path = canvas_getdir(c); // set name
    x->thread = malloc(sizeof(int)); 
    *(x->thread) = 1; // solution but it is weird
    post("[py4pd] Thread by default is enabled");
    post("[py4pd] Home folder is: %s", x->home_path->s_name);

    // check if in x->home_path there is a file py4pd.config
    char *config_path = (char *)malloc(sizeof(char) * (strlen(x->home_path->s_name) + strlen("/py4pd.cfg") + 1));
    strcpy(config_path, x->home_path->s_name);
    strcat(config_path, "/py4pd.cfg");
    if (access(config_path, F_OK) != -1) {
        FILE* file = fopen(config_path, "r"); /* should check the result */
        char line[256];
        while (fgets(line, sizeof(line), file)) {
            if (strstr(line, "packages =") != NULL) {
                char *packages_path = (char *)malloc(sizeof(char) * (strlen(line) - strlen("packages = ") + 1));
                strcpy(packages_path, line + strlen("packages = "));
                packages_path[strlen(packages_path) - 1] = '\0';
                if (strlen(packages_path) > 0) {
                    x->packages_path = gensym(packages_path);
                    
                    // x->packages_path = gensym(packages_path);
                    post("[py4pd] Packages path set to %s", x->packages_path->s_name);
                }
            }
        }
    } else {
        // file does not exist
        post("[py4pd] Could not find py4pd.cfg in home directory");
    }
    // get arguments and print it
    if (argc == 2) {
        set_function(x, s, argc, argv);
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
    // clear all struct

    if (pModule != NULL) {
        Py_DECREF(pModule);
    }
    if (pFunc != NULL) {
        Py_DECREF(pFunc);
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
    
    // add method for bang
    class_addbang(py4pd_class, run);
    class_addmethod(py4pd_class, (t_method)home, gensym("home"), A_GIMME, 0); // set home path
    class_addmethod(py4pd_class, (t_method)vscode, gensym("click"), 0, 0); // when click open vscode
    class_addmethod(py4pd_class, (t_method)packages, gensym("packages"), A_GIMME, 0); // set packages path
    #ifdef _WIN64
    class_addmethod(py4pd_class, (t_method)env_install, gensym("env_install"), 0, 0); // install enviroment
    class_addmethod(py4pd_class, (t_method)pip_install, gensym("pip"), 0, 0); // install packages with pip
    #endif
    class_addmethod(py4pd_class, (t_method)vscode, gensym("vscode"), 0, 0); // open vscode
    class_addmethod(py4pd_class, (t_method)reload, gensym("reload"), 0, 0); // reload python script
    class_addmethod(py4pd_class, (t_method)create, gensym("create"), A_GIMME, 0); // create file or open it
    class_addmethod(py4pd_class, (t_method)documentation, gensym("doc"), 0, 0); // open documentation
    class_addmethod(py4pd_class, (t_method)set_function, gensym("set"), A_GIMME, 0); // set function to be called
    class_addmethod(py4pd_class, (t_method)run, gensym("run"), A_GIMME, 0);  // run function
    class_addmethod(py4pd_class, (t_method)thread, gensym("thread"), A_FLOAT, 0); // on/off threading
    }