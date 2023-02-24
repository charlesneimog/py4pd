#include "pd_module.h"
#include "py4pd.h"
#include "py4pd_utils.h"

t_py *py4pd_object_array[100];
t_class *py4pd_class;
int object_count;

// ===================================================================
// ========================= Pd Object ===============================
// ===================================================================

static void home(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s; // unused but required by pd
    if (argc < 1) {
        post("[py4pd] The home path is: %s", x->home_path->s_name);
    } else {
        x->home_path = atom_getsymbol(argv);
        post("[py4pd] The home path set to: %s", x->home_path->s_name);
    }
    return;
}

// // ============================================

static void packages(t_py *x, t_symbol *s, int argc, t_atom *argv) {
    (void)s; 
    if (argc < 1) {
        post("[py4pd] The packages path is: %s", x->packages_path->s_name);
        return; // is this necessary?
    }
    else {
        if (argc < 2 && argc > 0){
            if (argv[0].a_type == A_SYMBOL) {
                t_symbol *path = atom_getsymbol(argv);

                // It checks relative path

                if (path->s_name[0] == '.' && path->s_name[1] == '/') {
                    char *new_path = malloc(strlen(x->home_path->s_name) + strlen(path->s_name) + 1);
                    strcpy(new_path, x->home_path->s_name);
                    strcat(new_path, path->s_name + 1);
                    post("[py4pd] The packages path set to: %s", new_path);
                    x->packages_path = gensym(new_path);
                    free(new_path);
                } 
                else {
                    x->packages_path = atom_getsymbol(argv);
                    post("[py4pd] The packages path set to: %s", x->packages_path->s_name);
                }
            
            } else{
                pd_error(x, "[py4pd] The packages path must be a string");
                return;
            }
            // check if path exists and is valid
            if (access(x->packages_path->s_name, F_OK) == -1) {
                pd_error(x, "[py4pd] The packages path is not valid");
                return;
            }
        } else{
            pd_error(x, "It seems that your package folder has |spaces|.");
            return;
        }
        return;   
    }
}

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
                post("==== %s documentation ====", x->function_name->s_name);
                post("%s", Doc);
                post("==== %s documentation ====", x->function_name->s_name);
                post("");
                return;
            }
            else{
                pd_error(x, "[py4pd] No documentation found!");
                return;
            }
        }
        else{
            pd_error(x, "[py4pd] No documentation found!");
            return;
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
        return;
    }
}

// ============================================
// ============================================

static void create(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    (void)argc;
    const char *script_name = argv[0].a_w.w_symbol->s_name;
    post("[py4pd] Opening vscode...");

    // Open VsCode in Windows 
    #ifdef _WIN64 
    char *command = malloc(strlen(x->home_path->s_name) + strlen(script_name) + 20);
    sprintf(command, "/c code %s/%s.py", x->home_path->s_name, script_name);
    SHELLEXECUTEINFO sei = {0};
    sei.cbSize = sizeof(sei);
    sei.fMask = SEE_MASK_NOCLOSEPROCESS;
    sei.lpFile = "cmd.exe ";
    sei.lpParameters = command;
    sei.nShow = SW_HIDE;
    ShellExecuteEx(&sei);
    CloseHandle(sei.hProcess);
    free(command);
    return;

    // Open VsCode in Linux and Mac
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

static void editor(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    if (argc != 0){
        x->editorName = atom_getsymbol(argv+0);
        post("[py4pd] Editor set to: %s", x->editorName->s_name);
        return;
    }

    if (x->function_called == 0) { // if the set method was not called, then we can not run the function :)
        pd_error(x, "[py4pd] To open vscode you need to set the function first!");
        return;
    }
    post("[py4pd] Opening editor...");

    // Open VsCode in Windows
    #ifdef _WIN64 // ERROR: the endif is missing directive _WIN64
    char *command = get_editor_command(x);
    command = get_editor_command(x);
    // use get_editor_command
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
    command = get_editor_command(x);
    post("[py4pd] %s", command);
    pd4py_system_func(command);
    #endif

    // If macOS
    #ifdef __APPLE__
    pd_error(x, "Not tested in your Platform, please send me a report!");
    #endif
    return ;

}

// ====================================
// ====================================
// ====================================
static void vscode(t_py *x){
    pd_error(x, "This method is deprecated, please use the editor method instead!");
    editor(x, NULL, 0, NULL);
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
            x->function_called = 1; 
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
    if (x->function_called == 1){
        int function_is_equal = strcmp(function_name->s_name, x->function_name->s_name); // if string is equal strcmp returns 0
        if (function_is_equal == 0){
            pd_error(x, "[py4pd] The function was already set!");
            return;
        }
        else{
            Py_XDECREF(x->function);
            Py_XDECREF(x->module);
            x->function_called = 0;
        }
    }
    
    // Check if there is extension (not to use it)
    char *extension = strrchr(script_file_name->s_name, '.');
    if (extension != NULL) {
        pd_error(x, "[py4pd] Don't use extensions in the script file name!");
        Py_XDECREF(x->function);
        Py_XDECREF(x->module);
        return;
    }

    // check if script file exists
    char script_file_path[MAXPDSTRING];
    snprintf(script_file_path, MAXPDSTRING, "%s/%s.py", x->home_path->s_name, script_file_name->s_name);
    if (access(script_file_path, F_OK) == -1) {
        pd_error(x, "[py4pd] The script file %s does not exist!", script_file_path);
        Py_XDECREF(x->function);
        Py_XDECREF(x->module);
        return;
    }
    
    // =====================
    // check number of arguments
    if (argc < 2) { // check is the number of arguments is correct | set "function_script" "function_name"
        pd_error(x,"[py4pd] 'set' message needs two arguments! The 'Script Name' and the 'Function Name'!");
        return;
    }
    // =====================
    PyObject *pName, *pModule, *pFunc; // Create the variables of the python objects
    
    // =====================
    // Add aditional path to python to work with Pure Data
    PyObject *home_path = PyUnicode_FromString(x->home_path->s_name); // Place where script file will probably be
    PyObject *site_package = PyUnicode_FromString(x->packages_path->s_name); // Place where the packages will be
    PyObject *sys_path = PySys_GetObject("path");
    PyList_Insert(sys_path, 0, home_path);
    PyList_Insert(sys_path, 0, site_package);
    Py_DECREF(home_path);
    Py_DECREF(site_package);

    // =====================
    pName = PyUnicode_DecodeFSDefault(script_file_name->s_name); // Name of script file
    pModule = PyImport_Import(pName);
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
        Py_XDECREF(pName);
        return;
    }
 
    
    pFunc = PyObject_GetAttrString(pModule, function_name->s_name); // Function name inside the script file
    Py_DECREF(pName); // Delete the name of the script file
    if (pFunc && PyCallable_Check(pFunc)){ // Check if the function exists and is callable   
        PyObject *inspect=NULL, *getfullargspec=NULL, *argspec=NULL, *args=NULL;
        inspect = PyImport_ImportModule("inspect");
        getfullargspec = PyObject_GetAttrString(inspect, "getfullargspec");
        argspec = PyObject_CallFunctionObjArgs(getfullargspec, pFunc, NULL);
        args = PyTuple_GetItem(argspec, 0);
        // int isArgs = PyObject_RichCompareBool(args, Py_None, Py_EQ); // TODO: way to check if function has *args or **kwargs
        int py_args = PyObject_Size(args);
        if (args == Py_None){
            x->py_arg_numbers = -1; 
            post("[py4pd] The '%s' function has *args or **kwargs!", function_name->s_name);        
        }
        else {
            x->py_arg_numbers = py_args;
            post("[py4pd] The '%s' function has %d arguments!", function_name->s_name, py_args);
        }        
        x->function = pFunc;
        x->module = pModule;
        x->script_name = script_file_name;
        x->function_name = function_name; 
        x->function_called = 1;

    } else {
        pd_error(x, "[py4pd] Function %s not loaded!", function_name->s_name);
        x->function_called = 0; // set the flag to 0 because it crash Pd if user try to use args method
        x->function_name = NULL;
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
        PyObject *pstr = PyObject_Str(pvalue);
        pd_error(x, "[py4pd] Set function had failed:\n %s", PyUnicode_AsUTF8(pstr));
        Py_DECREF(pstr);
        Py_XDECREF(pModule);
        Py_XDECREF(pFunc);
        Py_XDECREF(pName);
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
        PyErr_Clear();

    }
    return;
}

// ============================================

static void run_function(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    int OpenList_count = 0;
    int CloseList_count = 0;

    PyObject *pValue, *ArgsTuple; 
    if (argc != 0){
        for (int i = 0; i < argc; i++) {
            if (argv[i].a_type == A_SYMBOL){
                if (strchr(argv[i].a_w.w_symbol->s_name, '[') != NULL){
                    CloseList_count++;
                }
                if (strchr(argv[i].a_w.w_symbol->s_name, ']') != NULL){
                    OpenList_count++;
                }
            }
        }
        if (OpenList_count != CloseList_count){
            pd_error(x, "[py4pd] The number of '[' and ']' is not the same!");
            return;
        }
        PyObject *lists[OpenList_count]; // create a list of lists 
        
        ArgsTuple = py4pd_convert_to_py(lists, argc, argv); // convert the arguments to python
        int argCount = PyTuple_Size(ArgsTuple); // get the number of arguments
        if (argCount != x->py_arg_numbers) {
            pd_error(x, "[py4pd] Wrong number of arguments! The function %s needs %i arguments, received %i!", x->function_name->s_name, (int)x->py_arg_numbers, argCount);
            post("Length of tuple: %i", argCount);
            post("Length of args: %i", x->py_arg_numbers);
            return;
        }
    }
    else {
        ArgsTuple = PyTuple_New(0);
    }

    pValue = PyObject_CallObject(x->function, ArgsTuple);
    if (pValue != NULL) {                                // if the function returns a value   
        py4pd_convert_to_pd(x, pValue); // convert the value to pd        
    }
    else { // if the function returns a error
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
    // free lists
    Py_XDECREF(pValue);
    Py_DECREF(ArgsTuple);
    return;
}

// ============================================

// static void run_function_thread(t_py *x, t_symbol *s, int argc, t_atom *argv){
//     (void)s;
//     
//     if (argc != x->py_arg_numbers) {
//         pd_error(x, "[py4pd] Wrong number of arguments!");
//         return;
//     }
//     PyObject *pFunc, *pArgs, *pValue; // pDict, *pModule,
//     pFunc = x->function;
//     pArgs = PyTuple_New(argc);
//     int i;
//     if (x->function_called == 0) { // if the set method was not called, then we can not run the function :)
//         if(pFunc != NULL){
//             // create t_atom *argv from x->script_name and x->function_name
//             t_atom *newargv = malloc(sizeof(t_atom) * 2);
//             SETSYMBOL(newargv, x->script_name);
//             SETSYMBOL(newargv+1, x->function_name);
//             set_function(x, NULL, 2, newargv);
//         } else{
//             pd_error(x, "[py4pd] The message need to be formatted like 'set {script_name} {function_name}'!");
//             return;
//         }
//     }
//     
//     // CONVERTION TO PYTHON OBJECTS
//     // create an array of t_atom to store the list
//     // t_atom *list = malloc(sizeof(t_atom) * argc);
//     for (i = 0; i < argc; ++i) {
//         t_atom *argv_i = malloc(sizeof(t_atom));
//         *argv_i = argv[i];
//         pValue = py4pd_convert_to_python(argv_i);
//         if (!pValue) {
//             pd_error(x, "[py4pd] Cannot convert argument\n");
//             return;
//         }
//         PyTuple_SetItem(pArgs, i, pValue); // Set the argument in the tuple
//     }
//
//     pValue = PyObject_CallObject(pFunc, pArgs); // Call and execute the function
//     if (pValue != NULL) {                                // if the function returns a value   
//         // convert the python object to a t_atom
//         py4pd_convert_to_pd(x, pValue);
//     }
//     else { // if the function returns a error
//         PyObject *ptype, *pvalue, *ptraceback;
//         PyErr_Fetch(&ptype, &pvalue, &ptraceback);
//         PyErr_NormalizeException(&ptype, &pvalue, &ptraceback);
//         PyObject *pstr = PyObject_Str(pvalue);
//         pd_error(x, "[py4pd] Call failed: %s", PyUnicode_AsUTF8(pstr));
//         Py_DECREF(pstr);
//         Py_DECREF(pvalue);
//         Py_DECREF(ptype);
//         Py_DECREF(ptraceback);
//     }
//     return;
// }
//
// // ============================================
//
// struct thread_arg_struct {
//     t_py x;
//     t_symbol s;
//     int argc;
//     t_atom *argv;
//     PyThreadState *interp;
// } thread_arg;
//
// // ============================================
//
// static void *ThreadFunc(void *lpParameter) {
//     struct thread_arg_struct *arg = (struct thread_arg_struct *)lpParameter;
//     t_py *x = &arg->x; 
//     t_symbol *s = &arg->s;
//     int argc = arg->argc;
//     t_atom *argv = arg->argv;
//     int object_number = x->object_number;
//     thread_status[object_number] = 1;
//     // PyGILState_STATE gstate;
//     running_some_thread = 1;
//     run_function_thread(x, s, argc, argv);  
//     thread_status[object_number] = 0;
//     running_some_thread = 0;
//     return 0;
// }
//
// // ============================================
//
// static void create_thread(t_py *x, t_symbol *s, int argc, t_atom *argv){
//     (void)s;
//     struct thread_arg_struct *arg = (struct thread_arg_struct *)malloc(sizeof(struct thread_arg_struct));
//     arg->x = *x;
//     arg->argc = argc;
//     arg->argv = argv;
//     int object_number = x->object_number;
//     if (x->function_called == 0) {
//         // Pd is crashing when I try to create a thread.
//         pd_error(x, "[py4pd] You need to call a function before run!");
//         free(arg);
//         return;
//     } else {
//         if (thread_status[object_number] == 0){
//             // PyThread is not thread safe, so we need to lock the GIL
//             pthread_t thread;
//             pthread_create(&thread, NULL, ThreadFunc, arg);
//             x->state = 1;
//             // check the Thread was created
//         } else {
//             pd_error(x, "[py4pd] There is a thread running in this Object!");
//             free(arg);
//         }
//     }
//     return;
// }

// ============================================
static void run(t_py *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    // check if function was called
    if (x->function_called == 0) {
        pd_error(x, "[py4pd] You need to call a function before run!");
        return;
    }
    int thread = x->thread;
    if (thread == 1) {
        run_function(x, s, argc, argv);
        pd_error(x, "[py4pd] Not implemenented! Wait for approval of PEP 684");
    } else if (thread == 2) {
        run_function(x, s, argc, argv);
        
    } else {
        pd_error(x, "[py4pd] Thread not created");
    }
    return;
}

// ============================================
static void restartPython(t_py *x){
    Py_Finalize();
    x->function_called = 0;
    x->function_name = NULL;
    x->script_name = NULL;
    x->module = NULL;
    x->function = NULL;
    int i;
    for (i = 0; i < 100; i++) {
        t_py *y = py4pd_object_array[i];
        if (y != NULL) {
            y->function_called = 0;
            y->function_name = NULL;
            y->script_name = NULL;
            y->module = NULL;
            y->function = NULL;
            y->packages_path = gensym("./py-modules");
            y->thread = 2;
            y->editorName = gensym("code");
        }
    }
    PyImport_AppendInittab("pd", PyInit_pd); // Add the pd module to the python interpreter
    Py_Initialize();
    post("[py4pd] Python interpreter was restarted!");
    return;
}

// ============================================
// ============================================
// ============================================

static void thread(t_py *x, t_floatarg f){
    int thread = (int)f;
    if (thread == 1) {
        post("[py4pd] Threading enabled, wait for approval of PEP 684");
        x->thread = 1;
        return;
    } else if (thread == 0) {
        x->thread = 2; // 
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
    t_py *x = (t_py *)pd_new(py4pd_class); // create a new object
    if (!Py_IsInitialized()) {
        object_count = 1;   
        post("");
        post("[py4pd] by Charles K. Neimog");
        post("[py4pd] Version 0.5.0       ");
        post("[py4pd] Python version %d.%d.%d", PY_MAJOR_VERSION, PY_MINOR_VERSION, PY_MICRO_VERSION);
        post("");
        PyImport_AppendInittab("pd", PyInit_pd); // Add the pd module to the python interpreter
        Py_Initialize(); // Initialize the Python interpreter. If 1, the signal handler is installed.
    }
    // add a global varible INSIDE_PY4PD == true
    object_count++; // count the number of objects
    x->object_number = object_count; // save object number
    x->out_A = outlet_new(&x->x_obj, 0); // cria um outlet 
    x->x_canvas = canvas_getcurrent(); // pega o canvas atual
    t_canvas *c = x->x_canvas;  // get the current canvas
    t_symbol *patch_dir = canvas_getdir(c); // directory of opened patch
    x->home_path = patch_dir;     // set name of the home path
    x->packages_path = patch_dir; // set name of the packages path
    x->thread = 2; // default is 2 (no threading)
    py4pd_object_array[object_count] = x; // save the object in the array
    set_py4pd_config(x); // set the config file
    if (argc > 1) { // check if there are two arguments
        set_function(x, s, argc, argv); 
    }
    return(x);
}

// ============================================

void py4pd_free(t_py *x){
    PyObject  *pModule, *pFunc; // pDict, *pName,
    pFunc = x->function;
    pModule = x->module;
    object_count--;
    // clear all struct
    if (pModule != NULL) {
        Py_DECREF(pModule);
    }
    if (pFunc != NULL) {
        Py_DECREF(pFunc);
    }
    if (object_count == 1) {
        Py_Finalize();
        object_count = 0;
        post("[py4pd] Python interpreter finalized");
    }
}

// ====================================================

void py4pd_setup(void){
    py4pd_class =       class_new(gensym("py4pd"), // cria o objeto quando escrevemos py4pd
                        (t_newmethod)py4pd_new, // metodo de criação do objeto             
                        (t_method)py4pd_free, // quando voce deleta o objeto
                        sizeof(t_py), // quanta memoria precisamos para esse objeto
                        CLASS_DEFAULT, // nao há uma GUI especial para esse objeto???
                        A_GIMME, // o argumento é um símbolo
                        0); // todos os outros argumentos por exemplo um numero seria A_DEFFLOAT
    
    // Config
    class_addmethod(py4pd_class, (t_method)home, gensym("home"), A_GIMME, 0); // set home path
    class_addmethod(py4pd_class, (t_method)packages, gensym("packages"), A_GIMME, 0); // set packages path
    class_addmethod(py4pd_class, (t_method)thread, gensym("thread"), A_FLOAT, 0); // on/off threading
    class_addmethod(py4pd_class, (t_method)reload, gensym("reload"), 0, 0); // reload python script
    class_addmethod(py4pd_class, (t_method)restartPython, gensym("restart"), 0, 0); // it restart python interpreter

    // Edit py code
    class_addmethod(py4pd_class, (t_method)vscode, gensym("vscode"), 0, 0); // open editor  WARNING: WILL BE DEPRECATED 
    class_addmethod(py4pd_class, (t_method)editor, gensym("editor"), A_GIMME, 0); // open code
    class_addmethod(py4pd_class, (t_method)create, gensym("create"), A_GIMME, 0); // create file or open it
    class_addmethod(py4pd_class, (t_method)editor, gensym("click"), 0, 0); // when click open editor
    
    // User use
    class_addmethod(py4pd_class, (t_method)documentation, gensym("doc"), 0, 0); // open documentation
    class_addmethod(py4pd_class, (t_method)run, gensym("run"), A_GIMME, 0);  // run function
    class_addmethod(py4pd_class, (t_method)set_function, gensym("set"), A_GIMME,  0); // set function to be called
}

// // dll export function
#ifdef _WIN64
__declspec(dllexport) void py4pd_setup(void); // when I add python module, for some reson, pd not see py4pd_setup
#endif

